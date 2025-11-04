from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import plotly
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import json
import sys
from pathlib import Path as _Path_for_sys
# Ensure repo root is on sys.path so top-level modules (like `utils`) import correctly
try:
    REPO_ROOT = _Path_for_sys(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
except Exception:
    pass

from utils.db import init_db, verify_user, register_user

app = Flask(__name__)
app.secret_key = "change-me-to-a-secure-random-value"

ARTIFACT_DIR = Path("../notebook/artifacts")
TEST_CSV = Path("../data/raw-data/Test_data.csv")

# Initialize user DB
try:
    init_db()
except Exception:
    # If utils/db can't initialize, continue without crashing; log in console
    print("Warning: could not initialize user DB")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        ok, role = verify_user(username, password)
        if ok:
            session['user'] = username
            session['role'] = role
            flash('Login successful.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username/password', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

def safe_load(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def get_feature_names_from_preprocessor(preprocessor):
    names = []
    try:
        for name, transformer, original_features in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    if name == "cat":
                        names.extend(transformer.get_feature_names_out(original_features))
                    else:
                        names.extend(original_features)
                except Exception:
                    names.extend(original_features)
            else:
                names.extend(original_features)
    except Exception:
        names = []
    return names

@app.route("/", methods=["GET", "POST"])
def home():
    # Require login
    if not session.get('user'):
        return redirect(url_for('login'))
    prediction = None

    # Handle prediction form POST
    if request.method == "POST":
        # Collect form data
        input_data = {
            "src_bytes": int(request.form.get("src_bytes", 0)),
            "dst_bytes": int(request.form.get("dst_bytes", 0)),
            "same_srv_rate": float(request.form.get("same_srv_rate", 0)),
            "dst_host_same_srv_rate": float(request.form.get("dst_host_same_srv_rate", 0)),
            "dst_host_srv_count": int(request.form.get("dst_host_srv_count", 0)),
            "logged_in": int(request.form.get("logged_in", 0)),
            "dst_host_same_src_port_rate": float(request.form.get("dst_host_same_src_port_rate", 0)),
            "dst_host_diff_srv_rate": float(request.form.get("dst_host_diff_srv_rate", 0)),
            "diff_srv_rate": float(request.form.get("diff_srv_rate", 0)),
            "count": int(request.form.get("count", 0)),
            "duration": int(request.form.get("duration", 0)),
            "num_failed_logins": int(request.form.get("num_failed_logins", 0)),
            "num_compromised": int(request.form.get("num_compromised", 0)),
            "root_shell": int(request.form.get("root_shell", 0)),
            "su_attempted": int(request.form.get("su_attempted", 0)),
            "num_shells": int(request.form.get("num_shells", 0)),
            "srv_count": int(request.form.get("srv_count", 0)),
            "serror_rate": float(request.form.get("serror_rate", 0)),
            "rerror_rate": float(request.form.get("rerror_rate", 0)),
            "protocol_type": request.form.get("protocol_type", ""),
            "service": request.form.get("service", "")
        }
        # Send to Netlify function (adjust URL for local/dev as needed)
        try:
            netlify_url = "http://localhost:8888/.netlify/functions/predict"  # Change to prod URL when deployed
            resp = requests.post(netlify_url, json=input_data, timeout=10)
            if resp.status_code == 200:
                prediction = resp.json().get("predictions", ["Error"])[0]
            else:
                prediction = f"Error: {resp.text}"
        except Exception as e:
            prediction = f"Error: {e}"

    # Load artifacts for dashboard metrics/charts
    model = safe_load(ARTIFACT_DIR / "best_model.pkl")
    preprocessor = safe_load(ARTIFACT_DIR / "preprocessor.pkl")
    le = safe_load(ARTIFACT_DIR / "label_encoder.pkl")
    selected_features = safe_load(ARTIFACT_DIR / "selected_feature_names.pkl")
    feature_names_saved = safe_load(ARTIFACT_DIR / "feature_names.pkl")

    # Load test data if present
    df_test = None
    if TEST_CSV.exists():
        try:
            df_test = pd.read_csv(TEST_CSV)
        except Exception:
            df_test = None

    preds = None
    y_test = None
    X_test_raw = None
    if df_test is not None and model is not None and preprocessor is not None:
        if selected_features and all([c in df_test.columns for c in selected_features]):
            X_test_raw = df_test[selected_features].copy()
        else:
            cols = [c for c in df_test.columns if c != "class"]
            X_test_raw = df_test[cols].copy()
        X_test_raw = X_test_raw.fillna(0)
        try:
            Xp = preprocessor.transform(X_test_raw)
            if hasattr(Xp, "toarray"):
                Xp_arr = Xp.toarray()
            else:
                Xp_arr = np.asarray(Xp)
            preds = model.predict(Xp_arr)
            if "class" in df_test.columns and le is not None:
                try:
                    y_test = le.transform(df_test["class"])
                except Exception:
                    y_test = None
        except Exception:
            preds = None
            y_test = None

    # Metrics
    total_preds = int(len(preds)) if preds is not None else "N/A"
    anomalies = int(np.sum(preds == 1)) if preds is not None else "N/A"
    acc_val = "N/A"
    if y_test is not None and preds is not None:
        try:
            acc_val = f"{accuracy_score(y_test, preds):.3f}"
        except Exception:
            acc_val = "N/A"

    # Prediction Distribution Chart
    pred_dist_plot = None
    if preds is not None:
        vals, counts = np.unique(preds, return_counts=True)
        labels = [str(int(v)) for v in vals]
        fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color=["#1f77b4", "#e45756"])])
        fig.update_layout(title="Predicted Classes (test)", xaxis_title="Predicted label", yaxis_title="Count")
        pred_dist_plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Confusion Matrix & Metrics
    cm_plot = None
    f1_val = None
    roc_auc_val = None
    if y_test is not None and preds is not None:
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            cm_fig = go.Figure(data=go.Heatmap(
                z=[[int(tn), int(fp)], [int(fn), int(tp)]],
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                colorscale="Blues"
            ))
            cm_fig.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
            cm_plot = json.dumps(cm_fig, cls=plotly.utils.PlotlyJSONEncoder)
            f1_val = f"{f1_score(y_test, preds, zero_division=0):.4f}"
            if hasattr(model, "predict_proba"):
                try:
                    Xp = preprocessor.transform(X_test_raw)
                    if hasattr(Xp, "toarray"):
                        Xp_arr = Xp.toarray()
                    else:
                        Xp_arr = np.asarray(Xp)
                    proba = model.predict_proba(Xp_arr)[:, 1]
                    roc_auc_val = f"{roc_auc_score(y_test, proba):.4f}"
                except Exception:
                    roc_auc_val = None
        except Exception:
            cm_plot = None

    # Feature Importances
    feat_imp_plot = None
    if model is not None and hasattr(model, "feature_importances_"):
        try:
            f_names = get_feature_names_from_preprocessor(preprocessor)
            if not f_names and feature_names_saved:
                f_names = feature_names_saved
            if f_names:
                imps = np.asarray(model.feature_importances_)
                imps = imps[: len(f_names)]
                df_imp = pd.DataFrame({"feature": f_names[: len(imps)], "importance": imps})
                df_imp = df_imp.sort_values("importance", ascending=False).head(20)
                fig_imp = go.Figure(data=[go.Bar(x=df_imp["importance"][::-1], y=df_imp["feature"][::-1], orientation="h")])
                fig_imp.update_layout(title="Top feature importances", xaxis_title="importance")
                feat_imp_plot = json.dumps(fig_imp, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception:
            feat_imp_plot = None

    return render_template(
        "dashboard.html",
        total_preds=total_preds,
        anomalies=anomalies,
        acc_val=acc_val,
        pred_dist_plot=pred_dist_plot,
        cm_plot=cm_plot,
        f1_val=f1_val,
        roc_auc_val=roc_auc_val,
        feat_imp_plot=feat_imp_plot,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
