import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def handler(event, context):
    # Load artifacts (relative to this file or absolute)
    ARTIFACT_DIR = Path(__file__).parent.parent.parent / "artifact"
    model = joblib.load(ARTIFACT_DIR / "best_model.pkl")
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.pkl")
    le = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    selected_features = joblib.load(ARTIFACT_DIR / "selected_feature_names.pkl")

    # Parse input
    try:
        body = json.loads(event["body"])
        # Accepts either a dict (single sample) or list of dicts (batch)
        if isinstance(body, dict):
            data = [body]
        elif isinstance(body, list):
            data = body
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Input must be a dict or list of dicts."})
            }
        df = pd.DataFrame(data)
        # Keep only selected features
        if selected_features:
            df = df.reindex(columns=selected_features, fill_value=0)
        # Preprocess
        Xp = preprocessor.transform(df)
        if hasattr(Xp, "toarray"):
            Xp = Xp.toarray()
        # Predict
        preds = model.predict(Xp)
        labels = le.inverse_transform(preds) if le is not None else preds.tolist()
        return {
            "statusCode": 200,
            "body": json.dumps({"predictions": labels})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
