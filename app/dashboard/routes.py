import os
import json
import pandas as pd
from flask import render_template
from flask_login import login_required
from . import dashboard

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifact')

@dashboard.route('/')
@login_required
def dashboard_home():
    # Load metadata.json
    metadata_path = os.path.join(ARTIFACT_DIR, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    best_model = metadata.get('best_model', 'N/A')
    saved_time = metadata.get('saved_time', 'N/A')

    # Load model_performance_comparison.csv
    perf_path = os.path.join(ARTIFACT_DIR, 'model_performance_comparison.csv')
    perf_df = pd.read_csv(perf_path)
    best_model_row = perf_df[perf_df['Model'] == best_model].iloc[0] if best_model in perf_df['Model'].values else perf_df.iloc[0]
    accuracy = best_model_row.get('Accuracy', 'N/A')
    true_positives = best_model_row.get('True Positives', 'N/A')
    false_positives = best_model_row.get('False Positives', 'N/A')

    # TODO: Query predictions table for total predictions and threats detected

    return render_template(
        'dashboard.html',
        best_model=best_model,
        saved_time=saved_time,
        accuracy=accuracy,
        true_positives=true_positives,
        false_positives=false_positives,
        # Add more as needed
    )
