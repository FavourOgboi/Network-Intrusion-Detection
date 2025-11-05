import os
import joblib
import json
import pandas as pd
from flask import render_template, request, flash
from flask_login import login_required, current_user
from . import predict

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

            # TODO: Save to predictions table

        except Exception as e:
            flash(f'Prediction failed: {e}', 'danger')

    return render_template(
        'predict.html',
        selected_features=selected_features,
        prediction=prediction,
        confidence=confidence,
        model_used=model_used,
        input_data=input_data
    )
