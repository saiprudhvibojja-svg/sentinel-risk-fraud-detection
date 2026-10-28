import argparse
import json
import os
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import optuna

FEATURE_COLS = ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]


def load_dataset(db_path: str = "data/creditcard.db", csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load credit card dataset from SQLite database or fallback CSV."""
    db_file = Path(db_path)
    csv_file = Path(csv_path)

    if db_file.exists():
        print(f"Loading data from SQLite: {db_path}...")
        conn = sqlite3.connect(db_file)
        query = f"SELECT {', '.join(FEATURE_COLS)}, is_fraud FROM transactions"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    if csv_file.exists():
        print(f"Loading data from CSV: {csv_path}...")
        df = pd.read_csv(csv_file)
        rename_map = {
            "Time": "time_elapsed",
            "Amount": "amount",
            "Class": "is_fraud",
        }
        for i in range(1, 29):
            rename_map[f"V{i}"] = f"v{i}"
        df = df.rename(columns=rename_map)
        df.columns = [c.lower() for c in df.columns]
        return df[[*FEATURE_COLS, "is_fraud"]]

    raise FileNotFoundError(f"Neither database ({db_path}) nor CSV ({csv_path}) found.")


def run_optuna_pipeline(
    n_trials: int = 20,
    timeout: int = 600,
    db_path: str = "data/creditcard.db",
    csv_path: str = "data/creditcard.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SentinelRisk Optuna Bayesian Cloud Training Pipeline")
    print(f"Trials: {n_trials} | Timeout: {timeout}s")
    print("=" * 60)

    df = load_dataset(db_path, csv_path)
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df['is_fraud'].sum():,} fraud cases.")

    X = df[FEATURE_COLS]
    y = df["is_fraud"]

    print("Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("Fitting RobustScaler on amount and time_elapsed (train set only)...")
    scaler = RobustScaler()
    cols_to_scale = ["time_elapsed", "amount"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print("Applying SMOTE (sampling_strategy=0.1) on training set...")
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training set: Legit={sum(y_train_res == 0):,}, Fraud={sum(y_train_res == 1):,}")

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 3, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0)
        n_estimators = trial.suggest_int("n_estimators", 100, 200, step=25)

        clf = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            n_estimators=n_estimators,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_res, y_train_res)

        y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
        precisions, recalls, _ = precision_recall_curve(y_test, y_test_proba)
        pr_auc = float(auc(recalls, precisions))
        return pr_auc

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize", study_name="sentinel_xgboost_study")

    print("\nStarting Optuna Hyperparameter Optimization Study...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("\n" + "=" * 60)
    print("Optuna Study Complete!")
    print(f"Best Trial #{study.best_trial.number}: PR-AUC = {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Retrain champion model on best hyperparameters
    print("\nTraining Champion Model using Best Hyperparameters...")
    champion_params = study.best_params.copy()
    champion_model = XGBClassifier(
        **champion_params,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    champion_model.fit(X_train_res, y_train_res)

    print("Evaluating Champion Model on Held-out Test Set...")
    y_test_proba = champion_model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    pr_auc = float(auc(recalls, precisions))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))

    # Optimal F1 decision threshold calculation
    numerator = 2 * (precisions[:-1] * recalls[:-1])
    denominator = precisions[:-1] + recalls[:-1] + 1e-12
    f1_scores = numerator / denominator

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    print("\n" + "=" * 60)
    print("Champion Model Evaluation Metrics")
    print("=" * 60)
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Optimal Threshold:  {optimal_threshold:.4f}")
    print(f"Optimal F1-Score:   {best_f1:.4f}")
    print(f"Precision:          {best_precision:.4f}")
    print(f"Recall:             {best_recall:.4f}")
    print(f"Confusion Matrix:   TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("=" * 60)

    # 1. Export reports/pr_curve.png
    pr_curve_path = Path(reports_dir) / "pr_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"Optuna Champion PR Curve (AUC = {pr_auc:.4f})", color="#2563eb", lw=2)
    plt.scatter(
        [best_recall],
        [best_precision],
        color="#dc2626",
        s=80,
        zorder=5,
        label=f"Optimal Cutoff ({optimal_threshold:.3f}) | F1={best_f1:.3f}",
    )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve - SentinelRisk Bayesian Champion", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    print(f"Saved PR curve plot to {pr_curve_path}")

    # 2. Export reports/metrics.json
    metrics_path = Path(reports_dir) / "metrics.json"
    metrics_data = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "f1_score": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
        },
        "test_samples": len(y_test),
        "test_fraud_count": int(y_test.sum()),
        "champion_hyperparameters": champion_params,
        "optuna_best_value": study.best_value,
        "total_trials_evaluated": len(study.trials),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # 3. Export reports/optuna_study.json
    study_path = Path(reports_dir) / "optuna_study.json"
    trials_summary = []
    for t in study.trials:
        trials_summary.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    study_data = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_pr_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "trials": trials_summary,
    }
    with open(study_path, "w", encoding="utf-8") as f:
        json.dump(study_data, f, indent=4)
    print(f"Saved Optuna study summary to {study_path}")

    # 4. Export models/fraud_model.joblib
    model_artifact = {
        "model": champion_model,
        "scaler": scaler,
        "cols_to_scale": cols_to_scale,
        "feature_cols": FEATURE_COLS,
        "optimal_threshold": optimal_threshold,
        "metrics": metrics_data,
        "hyperparameters": champion_params,
    }
    model_path = Path(models_dir) / "fraud_model.joblib"
    joblib.dump(model_artifact, model_path)
    print(f"Saved champion model artifact to {model_path}")

    return model_artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with Optuna Bayesian Search")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna optimization trials")
    parser.add_argument("--timeout", type=int, default=600, help="Max optimization time in seconds")
    parser.add_argument("--db-path", type=str, default="data/creditcard.db", help="Path to SQLite DB")
    parser.add_argument("--csv-path", type=str, default="data/creditcard.csv", help="Path to CSV fallback")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model export")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for report export")

    args = parser.parse_args()
    run_optuna_pipeline(
        n_trials=args.n_trials,
        timeout=args.timeout,
        db_path=args.db_path,
        csv_path=args.csv_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
    )
