%%writefile build_and_save_model.py

"""
Enhanced model building script with comprehensive performance comparison table.
Run with: python notebook/build_and_save_model.py
"""

from pathlib import Path
import os
import json
import time
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Optional imports
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    import optuna
except Exception:
    optuna = None

from imblearn.over_sampling import SMOTE

REQUIRED_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_shells', 'count',
    'srv_count', 'serror_rate', 'rerror_rate', 'protocol_type', 'service'
]

NEW_FEATURES = ['src_dst_bytes_ratio', 'total_bytes', 'count_srv_ratio', 
                'same_diff_srv_ratio', 'serror_rerror_ratio']


def evaluate_model_comprehensive(model, X, y, model_name, cv=5):
    """Comprehensive model evaluation with all metrics"""
    print(f"  Evaluating {model_name}...")
    
    # Cross-validation predictions
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    
    # Get probability predictions if available
    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
    except:
        y_pred_proba = None
    
    # Calculate all metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # ROC-AUC
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except:
            roc_auc = np.nan
    else:
        roc_auc = np.nan
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Calculate rates
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    return {
        'Model': model_name,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'Specificity': f"{specificity:.4f}",
        'F1-Score': f"{f1:.4f}",
        'ROC-AUC': f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A",
        'True Positives': int(tp),
        'True Negatives': int(tn),
        'False Positives': int(fp),
        'False Negatives': int(fn),
        'FP Rate': f"{fpr:.4f}",
        'FN Rate': f"{fnr:.4f}",
        'CV Accuracy': accuracy  # Keep numeric for sorting
    }


def print_performance_table(results_df):
    """Print beautiful formatted performance table"""
    print("\n" + "="*150)
    print("📊 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("="*150)
    
    # Main metrics table
    print("\n🎯 PERFORMANCE METRICS:")
    print("-"*150)
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
    print(results_df[display_cols].to_string(index=False))
    
    # Confusion Matrix table
    print("\n\n📋 CONFUSION MATRIX COUNTS:")
    print("-"*150)
    cm_cols = ['Model', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    print(results_df[cm_cols].to_string(index=False))
    
    # Error rates table
    print("\n\n⚠️  ERROR RATES:")
    print("-"*150)
    error_cols = ['Model', 'FP Rate', 'FN Rate']
    print(results_df[error_cols].to_string(index=False))
    
    # Summary
    print("\n\n🏆 SUMMARY:")
    print("-"*150)
    best_acc = results_df.loc[0, 'Model']
    best_f1 = results_df.loc[results_df['F1-Score'].astype(float).idxmax(), 'Model']
    lowest_fn = results_df.loc[results_df['False Negatives'].idxmin(), 'Model']
    lowest_fp = results_df.loc[results_df['False Positives'].idxmin(), 'Model']
    
    print(f"✓ Best Overall Model (Accuracy): {best_acc}")
    print(f"✓ Best F1-Score: {best_f1}")
    print(f"✓ Lowest False Negatives: {lowest_fn} ({results_df['False Negatives'].min()} errors)")
    print(f"✓ Lowest False Positives: {lowest_fp} ({results_df['False Positives'].min()} errors)")
    print("="*150 + "\n")


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def feature_selection(df_train, n_top=10):
    numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    low_var = [c for c in numerical_cols if df_train[c].nunique() <= 1]
    cand = [c for c in numerical_cols if c not in low_var]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X = df_train[cand]
    y = df_train['class']
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=cand)
    top = importances.sort_values(ascending=False).head(n_top).index.tolist()
    selected = list(top)
    
    for feat in REQUIRED_FEATURES:
        if feat in df_train.columns and feat not in selected:
            selected.append(feat)
    return selected


def build_preprocessor(X_train_selected):
    numerical_features = X_train_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train_selected.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor, numerical_features, categorical_features


def transform_to_df(preprocessor, X_selected):
    Xt = preprocessor.transform(X_selected)
    feature_names = []
    
    for name, transformer, original_features in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                if name == 'cat':
                    feature_names.extend(transformer.get_feature_names_out(original_features))
                else:
                    feature_names.extend(original_features)
            except:
                feature_names.extend(original_features)
        else:
            feature_names.extend(original_features)
    
    try:
        Xt_arr = Xt.toarray()
    except:
        Xt_arr = np.asarray(Xt)
    
    X_df = pd.DataFrame(Xt_arr, columns=feature_names)
    return X_df, feature_names


def add_engineered_features(X_df):
    base_cols = ['src_bytes', 'dst_bytes', 'count', 'srv_count', 
                 'same_srv_rate', 'diff_srv_rate', 'serror_rate', 'rerror_rate']
    for c in base_cols:
        if c not in X_df.columns:
            X_df[c] = 0
    
    X_df['src_dst_bytes_ratio'] = X_df['src_bytes'] / (X_df['dst_bytes'] + 1)
    X_df['total_bytes'] = X_df['src_bytes'] + X_df['dst_bytes']
    X_df['count_srv_ratio'] = X_df['count'] / (X_df['srv_count'] + 1)
    X_df['same_diff_srv_ratio'] = X_df['same_srv_rate'] / (X_df['diff_srv_rate'] + 1e-5)
    X_df['serror_rerror_ratio'] = X_df['serror_rate'] / (X_df['rerror_rate'] + 1e-5)
    return X_df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='/content/sample_data/Train_data.csv')
    p.add_argument('--test', default='/content/sample_data/Test_data.csv')
    p.add_argument('--out', default=os.path.join('notebook', 'artifacts'))
    p.add_argument('--optuna-trials', type=int, default=20)
    args = p.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    train_path = Path(args.train)
    test_path = Path(args.test)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('🚀 Loading data...')
    if not train_path.exists():
        train_path = repo_root / args.train
    if not test_path.exists():
        test_path = repo_root / args.test

    df_train, df_test = load_data(train_path, test_path)
    print(f'   Train shape: {df_train.shape}, Test shape: {df_test.shape}')

    if 'class' not in df_train.columns:
        raise RuntimeError("Train file must contain 'class' column")

    print('🔍 Selecting features...')
    selected_feature_names = feature_selection(df_train, n_top=10)
    print(f'   Selected {len(selected_feature_names)} features')

    X_train_selected = df_train[selected_feature_names]
    X_test_selected = df_test[selected_feature_names] if all(f in df_test.columns for f in selected_feature_names) else pd.DataFrame(columns=selected_feature_names)

    preprocessor, num_feats, cat_feats = build_preprocessor(X_train_selected)
    print('⚙️  Fitting preprocessor...')
    preprocessor.fit(X_train_selected)

    print('🔄 Transforming data...')
    X_train_df, feature_names = transform_to_df(preprocessor, X_train_selected)
    X_test_df, _ = transform_to_df(preprocessor, X_test_selected) if not X_test_selected.empty else (pd.DataFrame(columns=feature_names), feature_names)

    print('🛠️  Adding engineered features...')
    X_train_df = add_engineered_features(X_train_df)
    X_test_df = add_engineered_features(X_test_df)

    print('📏 Scaling features...')
    scaler_new = StandardScaler()
    X_train_df[NEW_FEATURES] = scaler_new.fit_transform(X_train_df[NEW_FEATURES])
    if len(X_test_df) > 0:
        try:
            X_test_df[NEW_FEATURES] = scaler_new.transform(X_test_df[NEW_FEATURES])
        except:
            X_test_df[NEW_FEATURES] = scaler_new.transform(X_test_df[NEW_FEATURES].fillna(0))

    # Labels
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['class'])

    # Balance with SMOTE
    print('⚖️  Applying SMOTE...')
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_df, y_train)
    print(f'   After SMOTE: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}')

    # Train models
    print('\n🤖 Training models...')
    models = []
    
    # Logistic Regression
    try:
        logreg = LogisticRegression(max_iter=300, random_state=42)
        logreg.fit(X_train_bal, y_train_bal)
        models.append(('Logistic Regression', logreg))
    except Exception as e:
        print(f'   ⚠️  LogReg failed: {e}')

    # Random Forest (with Optuna if available)
    if optuna is not None:
        print(f'   🔧 Tuning Random Forest with Optuna ({args.optuna_trials} trials)...')
        def objective(trial):
            n_est = trial.suggest_int('n_estimators', 50, 400, step=50)
            max_d = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
            min_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, 
                                          min_samples_split=min_split, random_state=42, n_jobs=-1)
            try:
                from sklearn.model_selection import cross_val_score
                score = cross_val_score(model, X_train_bal, y_train_bal, cv=3, scoring='accuracy', n_jobs=-1).mean()
            except:
                score = 0.0
            return score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=False)
        rf_best = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
        rf_best.fit(X_train_bal, y_train_bal)
        print(f'   ✓ Best RF params: {study.best_params}')
    else:
        rf_best = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_best.fit(X_train_bal, y_train_bal)
    models.append(('Random Forest', rf_best))

    # XGBoost
    if XGBClassifier is not None:
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb.fit(X_train_bal, y_train_bal)
            models.append(('XGBoost', xgb))
        except Exception as e:
            print(f'   ⚠️  XGBoost failed: {e}')

    # LightGBM
    if LGBMClassifier is not None:
        try:
            lgbm = LGBMClassifier(random_state=42, verbose=-1)
            lgbm.fit(X_train_bal, y_train_bal)
            models.append(('LightGBM', lgbm))
        except Exception as e:
            print(f'   ⚠️  LightGBM failed: {e}')

    # Neural Network
    try:
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        mlp.fit(X_train_bal, y_train_bal)
        models.append(('Neural Network', mlp))
    except Exception as e:
        print(f'   ⚠️  MLP failed: {e}')

    # Evaluate all models comprehensively
    print('\n📊 Evaluating models (5-fold CV)...')
    results = []
    for name, model in models:
        result = evaluate_model_comprehensive(model, X_train_bal, y_train_bal, name, cv=5)
        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('CV Accuracy', ascending=False).reset_index(drop=True)

    # Print performance table
    print_performance_table(results_df)

    # Print raw table format (easy to copy/paste)
    print("\n\n" + "="*150)
    print("📋 RAW TABLE FORMAT (Copy-Paste Ready)")
    print("="*150)
    print(results_df.to_string(index=False))
    print("="*150)
    
    # Save results to CSV
    csv_path = Path(out_dir) / 'model_performance_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Performance table saved to: {csv_path}")

    # Select best model
    best_name = results_df.iloc[0]['Model']
    best_model = dict(models)[best_name]
    print(f"\n🏆 Selected best model: {best_name}")

    # Save best model
    try:
        best_clone = clone(best_model)
        best_clone.fit(X_train_bal, y_train_bal)
        out_model_path = Path(out_dir) / 'best_model.pkl'
        joblib.dump(best_clone, out_model_path)
        print(f'✓ Saved best model -> {out_model_path}')
    except Exception as e:
        print(f'⚠️  Failed to save best model: {e}')

    # Save artifacts
    try:
        joblib.dump(preprocessor, Path(out_dir) / 'preprocessor.pkl')
        joblib.dump(scaler_new, Path(out_dir) / 'scaler_new.pkl')
        joblib.dump(feature_names, Path(out_dir) / 'feature_names.pkl')
        joblib.dump(selected_feature_names, Path(out_dir) / 'selected_feature_names.pkl')
        joblib.dump(NEW_FEATURES, Path(out_dir) / 'new_features.pkl')
        joblib.dump(le, Path(out_dir) / 'label_encoder.pkl')
        print(f'✓ Saved preprocessing artifacts to {out_dir}')
    except Exception as e:
        print(f'⚠️  Failed to save artifacts: {e}')

    # Save metadata
    meta = {
        'selected_features': selected_feature_names,
        'feature_names': feature_names,
        'new_features': NEW_FEATURES,
        'best_model': best_name,
        'saved_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_performance': results_df.to_dict('records')
    }
    with open(Path(out_dir) / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('✓ Wrote metadata.json')

    print(f'\n✅ Done! All artifacts saved to: {out_dir}')


if __name__ == '__main__':
    main()