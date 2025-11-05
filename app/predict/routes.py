import os
import joblib
import json
import pandas as pd
from flask import render_template, request, flash
from flask_login import login_required, current_user
from . import predict
from utils.db import save_prediction

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifact')

@predict.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_view():
    # Load feature names for form
    selected_features = joblib.load(os.path.join(ARTIFACT_DIR, 'selected_feature_names.pkl'))
    feature_names = joblib.load(os.path.join(ARTIFACT_DIR, 'feature_names.pkl'))

    prediction = None
    confidence = None
    model_used = None
    input_data = None

    # Feature tooltips/descriptions for form fields
    feature_tooltips = {
        "duration": "Length (in seconds) of the connection",
        "protocol_type": "Type of protocol (e.g., tcp, udp, icmp)",
        "service": "Network service on the destination (e.g., http, telnet, ftp)",
        "flag": "Status flag of the connection",
        "src_bytes": "Number of data bytes from source to destination",
        "dst_bytes": "Number of data bytes from destination to source",
        "land": "1 if connection is from/to the same host/port; 0 otherwise",
        "wrong_fragment": "Number of wrong fragments",
        "urgent": "Number of urgent packets",
        "hot": "Number of 'hot' indicators",
        "num_failed_logins": "Number of failed login attempts",
        "logged_in": "1 if successfully logged in; 0 otherwise",
        "num_compromised": "Number of compromised conditions",
        "root_shell": "1 if root shell is obtained; 0 otherwise",
        "su_attempted": "1 if 'su root' command attempted; 0 otherwise",
        "num_root": "Number of root accesses",
        "num_file_creations": "Number of file creation operations",
        "num_shells": "Number of shell prompts",
        "num_access_files": "Number of operations on access control files",
        "num_outbound_cmds": "Number of outbound commands in an ftp session",
        "is_host_login": "1 if the login belongs to the host list; 0 otherwise",
        "is_guest_login": "1 if the login is a guest login; 0 otherwise",
        "count": "Number of connections to the same host as the current connection in the past two seconds",
        "srv_count": "Number of connections to the same service as the current connection in the past two seconds",
        "serror_rate": "Percentage of connections that have 'SYN' errors",
        "srv_serror_rate": "Percentage of connections to the same service that have 'SYN' errors",
        "rerror_rate": "Percentage of connections that have 'REJ' errors",
        "srv_rerror_rate": "Percentage of connections to the same service that have 'REJ' errors",
        "same_srv_rate": "Percentage of connections to the same service",
        "diff_srv_rate": "Percentage of connections to different services",
        "srv_diff_host_rate": "Percentage of connections to different hosts",
        # Add more as needed...
    }

    # Load best model info for display
    metadata_path = os.path.join(ARTIFACT_DIR, 'metadata.json')
    best_model_info = None
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            best_model_name = metadata.get('best_model')
            for model in metadata.get('model_performance', []):
                if model.get('Model') == best_model_name:
                    best_model_info = model
                    break
    except Exception:
        best_model_info = None

    if request.method == 'POST':
        # Collect input data from form
        input_data = {f: request.form.get(f, '') for f in selected_features}
        try:
            # Load artifacts
            preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, 'preprocessor.pkl'))
            scaler_new = joblib.load(os.path.join(ARTIFACT_DIR, 'scaler_new.pkl'))
            model = joblib.load(os.path.join(ARTIFACT_DIR, 'best_model.pkl'))
            new_features = joblib.load(os.path.join(ARTIFACT_DIR, 'new_features.pkl'))
            label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, 'label_encoder.pkl'))

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            X_selected = input_df[selected_features]

            # Preprocessing
            X_transformed = preprocessor.transform(X_selected)
            X_df = pd.DataFrame(X_transformed, columns=feature_names)

            # Add engineered features (example, adjust as needed)
            X_df['src_dst_bytes_ratio'] = X_df['src_bytes'] / (X_df['dst_bytes'] + 1)
            X_df['total_bytes'] = X_df['src_bytes'] + X_df['dst_bytes']
            X_df['count_srv_ratio'] = X_df['count'] / (X_df['srv_count'] + 1)
            X_df['same_diff_srv_ratio'] = X_df['same_srv_rate'] / (X_df['diff_srv_rate'] + 1e-5)
            X_df['serror_rerror_ratio'] = X_df['serror_rate'] / (X_df['rerror_rate'] + 1e-5)

            # Scale new features
            X_df[new_features] = scaler_new.transform(X_df[new_features])

            # Predict
            pred = model.predict(X_df)[0]
            proba = model.predict_proba(X_df)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            confidence = float(max(proba) * 100)
            model_used = type(model).__name__
            prediction = pred_label

            # Save prediction to database
            save_prediction(
                user_id=current_user.id,
                input_data=json.dumps(input_data),
                prediction=prediction,
                confidence=confidence
            )

        except Exception as e:
            flash(f'Prediction failed: {e}', 'danger')

    return render_template(
        'predict.html',
        selected_features=selected_features,
        prediction=prediction,
        confidence=confidence,
        model_used=model_used,
        input_data=input_data,
        feature_tooltips=feature_tooltips,
        best_model_info=best_model_info
    )
