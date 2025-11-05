

from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'change-me-to-a-secure-random-value'  # In production, use a real secret key

# Constants
ARTIFACT_DIR = Path('../notebook/artifacts')
MODEL_PATH = ARTIFACT_DIR / 'best_model.pkl'
TEST_DATA_PATH = Path('../data/raw-data/Test_data.csv')

def load_model():
    """Load the trained model if available"""
    try:
        return joblib.load(MODEL_PATH)
    except:
        return None

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Dummy/test data for dashboard
    user = {'username': session.get('user', 'Guest')}
    return render_template(
        'dashboard.html',
        current_user=user,
        total_predictions=1234,
        threats_detected=42,
        best_model='RandomForest',
        accuracy='98.7%',
        recent_predictions=[
            ('2025-11-05 12:00', 'Normal', 99.1),
            ('2025-11-05 11:58', 'Threat', 87.3),
            ('2025-11-05 11:55', 'Normal', 97.8)
        ]
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Dummy login logic for now
        username = request.form.get('username')
        session['user'] = username
        flash('Login successful.', 'success')
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # TODO: Implement prediction logic
        flash('Prediction feature coming soon!', 'info')
        return redirect(url_for('predict_view'))
    return render_template('predict.html')

@app.route('/performance')
def model_performance_view():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('performance.html')

@app.route('/history')
def history_view():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')

@app.route('/profile')
def profile_view():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Create a user object with the session data
    user = {
        'username': session.get('user', 'Guest'),
        'email': 'user@example.com',  # Placeholder
        'created_at': '2025-11-01',   # Placeholder
        'last_login': '2025-11-05'    # Placeholder
    }
    
    # Add some dummy stats
    stats = {
        'total_predictions': 150  # Placeholder
    }
    
    return render_template('profile.html', user=user, stats=stats)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
