from flask import render_template
from flask_login import login_required, current_user
from . import history
from .. import db
from sqlalchemy import desc
import json

@history.route('/history')
@login_required
def history_view():
    # Query predictions for current user
    result = db.session.execute(
        "SELECT id, timestamp, input_data, prediction, confidence, model_used FROM predictions WHERE user_id = :uid ORDER BY timestamp DESC LIMIT 50",
        {"uid": current_user.id}
    )
    predictions = []
    for row in result:
        predictions.append({
            "id": row[0],
            "timestamp": row[1],
            "input_data": json.loads(row[2]) if row[2] else {},
            "prediction": row[3],
            "confidence": row[4],
            "model_used": row[5]
        })

    return render_template(
        'history.html',
        predictions=predictions
    )
