import os
import pandas as pd
import json
from flask import render_template, session, redirect, url_for
from . import model_performance

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'artifact')

@model_performance.route('/model-performance')
def model_performance_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Hardcoded metadata as provided
    metadata = {
        "selected_features": [
            "src_bytes", "dst_bytes", "same_srv_rate", "dst_host_same_srv_rate",
            "dst_host_srv_count", "logged_in", "dst_host_same_src_port_rate",
            "dst_host_diff_srv_rate", "diff_srv_rate", "count", "duration",
            "num_failed_logins", "num_compromised", "root_shell", "su_attempted",
            "num_shells", "srv_count", "serror_rate", "rerror_rate",
            "protocol_type", "service"
        ],
        "feature_names": [
            "src_bytes", "dst_bytes", "same_srv_rate", "dst_host_same_srv_rate",
            "dst_host_srv_count", "logged_in", "dst_host_same_src_port_rate",
            "dst_host_diff_srv_rate", "diff_srv_rate", "count", "duration",
            "num_failed_logins", "num_compromised", "root_shell", "su_attempted",
            "num_shells", "srv_count", "serror_rate", "rerror_rate",
            "protocol_type_icmp", "protocol_type_tcp", "protocol_type_udp",
            "service_IRC", "service_X11", "service_Z39_50", "service_auth",
            "service_bgp", "service_courier", "service_csnet_ns", "service_ctf",
            "service_daytime", "service_discard", "service_domain",
            "service_domain_u", "service_echo", "service_eco_i", "service_ecr_i",
            "service_efs", "service_exec", "service_finger", "service_ftp",
            "service_ftp_data", "service_gopher", "service_hostnames",
            "service_http", "service_http_443", "service_http_8001",
            "service_imap4", "service_iso_tsap", "service_klogin",
            "service_kshell", "service_ldap", "service_link", "service_login",
            "service_mtp", "service_name", "service_netbios_dgm",
            "service_netbios_ns", "service_netbios_ssn", "service_netstat",
            "service_nnsp", "service_nntp", "service_ntp_u", "service_other",
            "service_pm_dump", "service_pop_2", "service_pop_3",
            "service_printer", "service_private", "service_red_i",
            "service_remote_job", "service_rje", "service_shell",
            "service_smtp", "service_sql_net", "service_ssh", "service_sunrpc",
            "service_supdup", "service_systat", "service_telnet",
            "service_tim_i", "service_time", "service_urh_i", "service_urp_i",
            "service_uucp", "service_uucp_path", "service_vmnet", "service_whois"
        ],
        "new_features": [
            "src_dst_bytes_ratio", "total_bytes", "count_srv_ratio",
            "same_diff_srv_ratio", "serror_rerror_ratio"
        ],
        "best_model": "XGBoost",
        "saved_time": "2025-11-04 18:19:22",
        "model_performance": [
            {
                "Model": "XGBoost",
                "Accuracy": "0.9976",
                "Precision": "0.9968",
                "Recall": "0.9984",
                "Specificity": "0.9968",
                "F1-Score": "0.9976",
                "ROC-AUC": "0.9999",
                "True Positives": 13427,
                "True Negatives": 13406,
                "False Positives": 43,
                "False Negatives": 22,
                "FP Rate": "0.0032",
                "FN Rate": "0.0016",
                "CV Accuracy": 0.997583463454532
            },
            {
                "Model": "LightGBM",
                "Accuracy": "0.9975",
                "Precision": "0.9965",
                "Recall": "0.9986",
                "Specificity": "0.9965",
                "F1-Score": "0.9975",
                "ROC-AUC": "0.9999",
                "True Positives": 13430,
                "True Negatives": 13402,
                "False Positives": 47,
                "False Negatives": 19,
                "FP Rate": "0.0035",
                "FN Rate": "0.0014",
                "CV Accuracy": 0.997546285969217
            },
            {
                "Model": "Random Forest",
                "Accuracy": "0.9975",
                "Precision": "0.9964",
                "Recall": "0.9986",
                "Specificity": "0.9964",
                "F1-Score": "0.9975",
                "ROC-AUC": "1.0000",
                "True Positives": 13430,
                "True Negatives": 13400,
                "False Positives": 49,
                "False Negatives": 19,
                "FP Rate": "0.0036",
                "FN Rate": "0.0014",
                "CV Accuracy": 0.9974719309985872
            },
            {
                "Model": "Neural Network",
                "Accuracy": "0.9915",
                "Precision": "0.9904",
                "Recall": "0.9926",
                "Specificity": "0.9904",
                "F1-Score": "0.9915",
                "ROC-AUC": "0.9993",
                "True Positives": 13349,
                "True Negatives": 13320,
                "False Positives": 129,
                "False Negatives": 100,
                "FP Rate": "0.0096",
                "FN Rate": "0.0074",
                "CV Accuracy": 0.9914863558628895
            },
            {
                "Model": "Logistic Regression",
                "Accuracy": "0.9679",
                "Precision": "0.9590",
                "Recall": "0.9775",
                "Specificity": "0.9582",
                "F1-Score": "0.9682",
                "ROC-AUC": "0.9917",
                "True Positives": 13147,
                "True Negatives": 12887,
                "False Positives": 562,
                "False Negatives": 302,
                "FP Rate": "0.0418",
                "FN Rate": "0.0225",
                "CV Accuracy": 0.9678786526879322
            }
        ]
    }

    # Use the model_performance from metadata
    model_table = metadata['model_performance']
    columns = list(model_table[0].keys()) if model_table else []

    return render_template(
        'performance.html',
        metadata=metadata,
        model_table=model_table,
        columns=columns
    )
