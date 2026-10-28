import os
import json
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

FEATURE_COLS = ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]


def load_data_from_db(db_path="data/creditcard.db"):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    conn = sqlite3.connect(db_path)
    query = f"SELECT {', '.join(FEATURE_COLS)}, is_fraud FROM transactions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train_model(db_path="data/creditcard.db", models_dir="models", reports_dir="reports"):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {db_path}...")
    df = load_data_from_db(db_path)
    print(f"Loaded dataset: shape={df.shape}, fraud_count={df['is_fraud'].sum()}")

    X = df[FEATURE_COLS]
    y = df["is_fraud"]

    # Stratified 80/20 train-test split
    print("Performing stratified 80/20 train-test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scale 'amount' and 'time_elapsed' using RobustScaler fit ONLY on training set
    print("Fitting RobustScaler on amount and time_elapsed...")
    scaler = RobustScaler()
    cols_to_scale = ["time_elapsed", "amount"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Apply SMOTE (sampling_strategy=0.1) only on the training set
    print("Applying SMOTE (sampling_strategy=0.1) on training set...")
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training distribution: class_0={sum(y_train_res == 0)}, class_1={sum(y_train_res == 1)}")

    # Train XGBClassifier (eval_metric='aucpr', scale_pos_weight=5)
    print("Training XGBClassifier (eval_metric='aucpr', scale_pos_weight=5)...")
    model = XGBClassifier(
        eval_metric="aucpr",
        scale_pos_weight=5,
        random_state=42,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate on test set
    print("Evaluating on test set...")
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate Precision-Recall curve & PR-AUC
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    pr_auc = float(auc(recalls, precisions))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))

    # F1-maximization for optimal decision threshold
    # thresholds array has length len(precisions) - 1
    numerator = 2 * (precisions[:-1] * recalls[:-1])
    denominator = precisions[:-1] + recalls[:-1] + 1e-12
    f1_scores = numerator / denominator

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    # Predictions at optimal threshold
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Optimal Threshold:  {optimal_threshold:.4f}")
    print(f"Best F1-Score:      {best_f1:.4f}")
    print(f"Precision:          {best_precision:.4f}")
    print(f"Recall:             {best_recall:.4f}")
    print(f"Confusion Matrix:   TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("=" * 50)

    # Plot & Save Precision-Recall curve
    pr_curve_path = Path(reports_dir) / "pr_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"PR Curve (AUC = {pr_auc:.4f})", color="royalblue", lw=2)
    plt.scatter(
        [best_recall],
        [best_precision],
        color="crimson",
        zorder=5,
        label=f"Optimal Threshold ({optimal_threshold:.3f}) | F1={best_f1:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Credit Card Fraud Detection")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    print(f"Saved PR curve plot to {pr_curve_path}")

    # Save metrics.json
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
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # Save model artifact
    model_artifact = {
        "model": model,
        "scaler": scaler,
        "cols_to_scale": cols_to_scale,
        "feature_cols": FEATURE_COLS,
        "optimal_threshold": optimal_threshold,
        "metrics": metrics_data,
    }
    model_path = Path(models_dir) / "fraud_model.joblib"
    joblib.dump(model_artifact, model_path)
    print(f"Saved model pipeline to {model_path}")

    return model_artifact


if __name__ == "__main__":
    train_model()
