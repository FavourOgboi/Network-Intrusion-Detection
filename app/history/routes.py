from flask import render_template, session, redirect, url_for, make_response
from flask_login import login_required, current_user
from . import history
# from app import db
from sqlalchemy import desc
import json
import csv
import io

@history.route('/history')
def history_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    from sqlalchemy import text
    from collections import Counter

    # Query predictions for current user
    from utils.db import get_connection
    conn = get_connection()
    user_id = session['user_id']
    # Get the top 10 most recent predictions for the current user
    rows = conn.execute(
        "SELECT timestamp, result, confidence, model_used FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
        (user_id,)
    ).fetchall()
    conn.close()
    predictions = []
    for row in rows:
        predictions.append({
            "timestamp": row["timestamp"],
            "prediction": row["result"],
            "confidence": row["confidence"],
            "model_used": row["model_used"]
        })

    return render_template(
        'history.html',
        predictions=predictions
    )

@history.route('/history/download')
def download_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    from utils.db import get_connection
    conn = get_connection()
    user_id = session['user_id']
    # Get all predictions for the current user
    rows = conn.execute(
        "SELECT timestamp, result, confidence, model_used FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Prediction', 'Confidence', 'Model Used'])
    for row in rows:
        writer.writerow([row['timestamp'], row['result'], row['confidence'], row['model_used']])

    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response
