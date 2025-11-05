import os
import pandas as pd
from flask import render_template
from flask_login import login_required
from . import model_performance

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'artifact')

@model_performance.route('/model-performance')
@login_required
def model_performance_view():
    # Load model_performance_comparison.csv
    perf_path = os.path.join(ARTIFACT_DIR, 'model_performance_comparison.csv')
    perf_df = pd.read_csv(perf_path)
    # Convert DataFrame to list of dicts for table rendering
    model_table = perf_df.to_dict(orient='records')

    # TODO: Load confusion matrices and feature importance if available

    return render_template(
        'model_performance.html',
        model_table=model_table,
        columns=perf_df.columns
    )
