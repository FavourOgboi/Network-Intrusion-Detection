import os
import json
import pandas as pd
from flask import render_template
from flask_login import login_required, current_user
from . import dashboard
from .. import db
from sqlalchemy import text
from collections import Counter
from datetime import datetime

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifact')

@dashboard.route('/')
@login_required
def dashboard_home():
    from sqlalchemy import text
    from datetime import datetime

    # Query all predictions for the current user
    result = db.session.execute(
        text("SELECT prediction, confidence, model_used, timestamp FROM predictions WHERE user_id = :uid ORDER BY timestamp ASC"),
        {"uid": current_user.id}
    )
    predictions = [dict(prediction=row[0], confidence=row[1], model_used=row[2], timestamp=row[3]) for row in result]

    if not predictions:
        return render_template(
            'dashboard.html',
            total_predictions=0,
            threats_detected=0,
            avg_confidence=0.0,
            best_model="N/A",
            accuracy="N/A",
            class_labels=['No Data'],
            class_counts=[0],
            time_labels=[],
            time_counts=[]
        )

    # Compute stats
    total_predictions = len(predictions)
    threat_count = sum(1 for p in predictions if str(p['prediction']).strip().upper() == 'THREAT')
    avg_confidence = round(sum(float(p['confidence']) for p in predictions) / total_predictions, 2)
    model_names = [p['model_used'] for p in predictions if p['model_used']]
    best_model = max(set(model_names), key=model_names.count) if model_names else "N/A"
    accuracy = "N/A"  # You can compute this if you have true labels

    # Chart 1: Class distribution
    safe_count = total_predictions - threat_count
    class_labels = ['Safe', 'Threat']
    class_counts = [safe_count, threat_count]

    # Chart 2: Predictions over time
    time_labels = []
    time_counts = []
    for idx, p in enumerate(predictions, 1):
        try:
            dt = p['timestamp']
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            label = dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            label = str(p['timestamp'])
        time_labels.append(label)
        time_counts.append(idx)

    return render_template(
        'dashboard.html',
        total_predictions=total_predictions,
        threats_detected=threat_count,
        avg_confidence=avg_confidence,
        best_model=best_model,
        accuracy=accuracy,
        class_labels=class_labels,
        class_counts=class_counts,
        time_labels=time_labels,
        time_counts=time_counts
    )
