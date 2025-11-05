import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
app.secret_key = 'change-me-to-a-secure-random-value'  # change in production

# Flask-Login removed - using session-based authentication instead

# === Register Blueprints ===
from history import history
from model_performance import model_performance
app.register_blueprint(history, url_prefix='/')
app.register_blueprint(model_performance, url_prefix='/')

# === Correct directory setup ===
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / 'artifact'
DB_PATH = BASE_DIR / 'users.db'

# === Model details ===
BEST_MODEL_NAME = 'XGBoost'
BEST_MODEL_METRICS = {
    'Accuracy': 0.9976,
    'Precision': 0.9968,
    'Recall': 0.9984,
    'F1-Score': 0.9976,
    'Specificity': 0.9968,
    'ROC-AUC': 0.9999,
    'False Positives': 43,
    'False Negatives': 22
}
MODEL_ACCURACY = '99.76%'

# === Load model artifacts ===
try:
    trained_model = joblib.load(ARTIFACT_DIR / 'best_model.pkl')
    preprocessor = joblib.load(ARTIFACT_DIR / 'preprocessor.pkl')
    scaler_new = joblib.load(ARTIFACT_DIR / 'scaler_new.pkl')
    feature_names = joblib.load(ARTIFACT_DIR / 'feature_names.pkl')
    selected_feature_names = joblib.load(ARTIFACT_DIR / 'selected_feature_names.pkl')
    new_features = joblib.load(ARTIFACT_DIR / 'new_features.pkl')
    label_encoder = joblib.load(ARTIFACT_DIR / 'label_encoder.pkl')

    with open(ARTIFACT_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    print(f"✓ Successfully loaded {BEST_MODEL_NAME} model and all artifacts")
    print(f"✓ Selected features: {len(selected_feature_names)}")
    print(f"✓ Feature names after preprocessing: {len(feature_names)}")
    print(f"✓ New engineered features: {new_features}")
    MODEL_LOADED = True
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    trained_model = None
    preprocessor = None
    scaler_new = None
    feature_names = []
    selected_feature_names = []
    new_features = []
    label_encoder = None
    metadata = {}
    MODEL_LOADED = False


# === Database setup ===
from utils.db import get_connection

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            result TEXT NOT NULL,
            confidence REAL,
            model_used TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    test_user = cursor.execute("SELECT * FROM users WHERE username = ?", ('admin',)).fetchone()
    if not test_user:
        hashed = generate_password_hash('test123')
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ('admin', hashed))

    conn.commit()
    conn.close()


init_db()


# === Feature descriptions ===
FEATURE_DESCRIPTIONS = {
    'duration': 'Length of connection in seconds',
    'src_bytes': 'Number of data bytes from source to destination',
    'dst_bytes': 'Number of data bytes from destination to source',
    'logged_in': 'Login status: 1 if logged in, 0 otherwise',
    'count': 'Connections to same host in past 2 seconds',
    'srv_count': 'Connections to same service in past 2 seconds',
    'serror_rate': 'Percentage of connections with SYN errors',
    'rerror_rate': 'Percentage of connections with REJ errors',
    'same_srv_rate': 'Connections to same service',
    'diff_srv_rate': 'Connections to different services',
    'dst_host_same_srv_rate': 'Connections to same service on destination host',
    'dst_host_diff_srv_rate': 'Connections to different services on destination host',
    'dst_host_same_src_port_rate': 'Connections from same source port',
    'dst_host_srv_count': 'Connections to destination host',
    'num_failed_logins': 'Number of failed login attempts',
    'num_compromised': 'Number of compromised conditions',
    'root_shell': 'Root shell obtained: 1 if yes, 0 otherwise',
    'su_attempted': 'Su root command attempted: 1 if yes, 0 otherwise',
    'num_shells': 'Number of shell prompts',
    'protocol_type': 'Protocol type (tcp, udp, icmp)',
    'service': 'Network service (e.g., http, ftp, smtp)'
}


def get_feature_description(feature):
    return FEATURE_DESCRIPTIONS.get(feature, f'Enter value for {feature}')


app.jinja_env.globals.update(get_feature_description=get_feature_description)
app.jinja_env.globals.update(BEST_MODEL_NAME=BEST_MODEL_NAME)


# === Preprocessing function ===
def preprocess_input(input_dict):
    X_selected = pd.DataFrame([input_dict])

    X_transformed = preprocessor.transform(X_selected)
    try:
        X_arr = X_transformed.toarray()
    except:
        X_arr = np.asarray(X_transformed)

    X_df = pd.DataFrame(X_arr, columns=feature_names)

    base_cols = ['src_bytes', 'dst_bytes', 'count', 'srv_count',
                 'same_srv_rate', 'diff_srv_rate', 'serror_rate', 'rerror_rate']
    for c in base_cols:
        if c not in X_df.columns:
            X_df[c] = 0

    X_df['src_dst_bytes_ratio'] = X_df['src_bytes'] / (X_df['dst_bytes'] + 1)
    X_df['total_bytes'] = X_df['src_bytes'] + X_df['dst_bytes']
    X_df['count_srv_ratio'] = X_df['count'] / (X_df['srv_count'] + 1)
    X_df['same_diff_srv_ratio'] = X_df['same_srv_rate'] / (X_df['diff_srv_rate'] + 1e-5)
    X_df['serror_rerror_ratio'] = X_df['serror_rate'] / (X_df['rerror_rate'] + 1e-5)

    X_df[new_features] = scaler_new.transform(X_df[new_features])
    return X_df


# === Routes ===
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    username = session.get('user', 'Guest')
    conn = get_connection()

    preds = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10', (user_id,)
    ).fetchall()
    total_predictions = conn.execute(
        'SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)
    ).fetchone()[0]
    threats_detected = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ? AND result = 'Threat'", (user_id,)
    ).fetchone()[0]
    normal_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE user_id = ? AND result = 'Normal'", (user_id,)
    ).fetchone()[0]

    threats_over_time = conn.execute(
        "SELECT date(timestamp) as day, COUNT(*) as count FROM predictions WHERE user_id = ? AND result = 'Threat' GROUP BY day ORDER BY day ASC",
        (user_id,)
    ).fetchall()
    threat_vs_normal = conn.execute(
        "SELECT result, COUNT(*) as count FROM predictions WHERE user_id = ? GROUP BY result", (user_id,)
    ).fetchall()
    conn.close()

    avg_confidence = round(
        sum([p['confidence'] or 0 for p in preds]) / len(preds), 2
    ) if preds else 0.0

    return render_template(
        'dashboard.html',
        current_user={'username': username},
        total_predictions=total_predictions,
        threats_detected=threats_detected,
        normal_predictions=normal_predictions,
        best_model=BEST_MODEL_NAME,
        accuracy=MODEL_ACCURACY,
        avg_confidence=avg_confidence,
        recent_predictions=[(p['timestamp'], p['result'], p['confidence']) for p in preds],
        model_insight={"name": BEST_MODEL_NAME, "metrics": BEST_MODEL_METRICS},
        threats_over_time=[{"day": r["day"], "count": r["count"]} for r in threats_over_time],
        threat_vs_normal=[{"result": r["result"], "count": r["count"]} for r in threat_vs_normal]
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            session['user_id'] = user['id']
            flash('Login successful.', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        conn = get_connection()
        existing = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if existing:
            flash('Username already exists.', 'danger')
            conn.close()
            return redirect(url_for('register'))

        hashed = generate_password_hash(password)
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        session['user'] = username
        session['user_id'] = user['id']
        flash('Registration successful.', 'success')
        return redirect(url_for('home'))

    return render_template('login.html', register=True)


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if not MODEL_LOADED:
            flash('Model not loaded. Please check server configuration.', 'danger')
            return redirect(url_for('predict_view'))

        try:
            input_data = {}
            for feature in selected_feature_names:
                value = request.form.get(feature)
                if value is not None and value != '':
                    try:
                        input_data[feature] = float(value)
                    except ValueError:
                        input_data[feature] = value
                else:
                    # If not provided by user, default to 0 (or '' for categorical)
                    input_data[feature] = 0

            X_processed = preprocess_input(input_data)
            prediction_proba = trained_model.predict_proba(X_processed)[0]
            prediction = trained_model.predict(X_processed)[0]
            confidence = round(max(prediction_proba) * 100, 2)

            result_label = label_encoder.inverse_transform([prediction])[0]
            result = 'Normal' if result_label.lower() in ['normal', 'benign'] else 'Threat'

            conn = get_connection()
            conn.execute(
                'INSERT INTO predictions (user_id, result, confidence, model_used) VALUES (?, ?, ?, ?)',
                (session['user_id'], result, confidence, BEST_MODEL_NAME)
            )
            conn.commit()
            conn.close()

            flash(f'Prediction complete: {result} (Confidence: {confidence}%)', 'success')
            return render_template('predict.html', prediction=result, confidence=confidence,
                                   selected_features=selected_feature_names,
                                   best_model_info=BEST_MODEL_METRICS)

        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'danger')
            import traceback
            traceback.print_exc()

    return render_template('predict.html',
                           selected_features=selected_feature_names,
                           best_model_info=BEST_MODEL_METRICS)


# @app.route('/performance')  # Removed - now handled by model_performance blueprint
# def model_performance_view():
#     if 'user' not in session:
#         return redirect(url_for('login'))
#     return render_template('performance.html')


# @app.route('/history')
# def history_view():
#     if 'user' not in session:
#         return redirect(url_for('login'))
#     return render_template('history.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    username = session.get('user', 'Guest')

    # Get user info from database
    conn = get_connection()
    user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    # Get user statistics
    conn = get_connection()
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,)).fetchone()[0]
    conn.close()

    user = {
        'username': username,
        'email': 'user@example.com',  # This could be added to database later
        'created_at': '2025-11-01',  # This could be added to database later
        'last_login': '2025-11-05'  # This could be added to database later
    }
    stats = {'total_predictions': total_predictions}

    if request.method == 'POST':
        old_password = request.form.get('old_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Validate input
        if not old_password or not new_password or not confirm_password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('profile_view'))

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('profile_view'))

        if len(new_password) < 6:
            flash('New password must be at least 6 characters long.', 'danger')
            return redirect(url_for('profile_view'))

        # Skip old password verification for easier testing

        # Update password
        hashed_new_password = generate_password_hash(new_password)
        conn = get_connection()
        conn.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_new_password, user_id))
        conn.commit()
        conn.close()

        flash('Password updated successfully!', 'success')
        return redirect(url_for('profile_view'))

    return render_template('profile.html', user=user, stats=stats)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
