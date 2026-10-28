"""
SentinelRisk - MLOps Population Stability Index (PSI) & Drift Detection Engine.
Monitors incoming transaction distributions against the training baseline,
quantifying statistical divergence using decile PSI and Kolmogorov-Smirnov statistics.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("sentinel.drift")

BASELINE_FILE = Path("models/baseline_distribution.json")

# In-memory storage for the latest batch drift evaluation
_LATEST_DRIFT_REPORT: Optional[Dict[str, Any]] = None


def calculate_psi(
    expected_proportions: List[float],
    actual_proportions: List[float],
    epsilon: float = 1e-4
) -> float:
    """
    Calculates the Population Stability Index (PSI) between baseline (expected)
    and current batch (actual) frequency distributions.

    Formula:
        PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    """
    actual = np.array(actual_proportions, dtype=float)
    expected = np.array(expected_proportions, dtype=float)

    # Clean zero or negative values with smoothing epsilon
    actual = np.where(actual <= 0, epsilon, actual)
    expected = np.where(expected <= 0, epsilon, expected)

    # Renormalize probabilities to sum to 1.0
    actual = actual / np.sum(actual)
    expected = expected / np.sum(expected)

    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(np.round(psi, 6))


def load_baseline_profile() -> Dict[str, Any]:
    """Loads baseline decile profile or generates an in-memory fallback."""
    if BASELINE_FILE.exists():
        try:
            with open(BASELINE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {BASELINE_FILE}: {e}. Using fallback baseline.")

    # In-memory realistic fallback if baseline JSON is missing
    fallback_features = ["V4", "V10", "V12", "V14", "V17", "Amount"]
    features_dict = {}
    for feat in fallback_features:
        # Standard deciles for standard normal (or scaled amount)
        std_scale = 100.0 if feat == "Amount" else 1.0
        q = np.linspace(-1.5 * std_scale, 1.5 * std_scale, 9).tolist()
        features_dict[feat] = {
            "feature_key": feat.lower(),
            "mean": 0.0 if feat != "Amount" else 88.0,
            "std": 1.0 if feat != "Amount" else 250.0,
            "min": -5.0 if feat != "Amount" else 0.0,
            "max": 5.0 if feat != "Amount" else 5000.0,
            "interior_cutpoints": q,
            "baseline_proportions": [0.10] * 10,
            "num_bins": 10
        }

    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": 284807,
            "version": "1.0.0-fallback",
            "binning_strategy": "decile_10_bins",
            "features_tracked": fallback_features
        },
        "features": features_dict
    }


def assess_batch_drift(batch_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates Population Stability Index (PSI) and Kolmogorov-Smirnov statistics
    for high-importance risk drivers in an incoming batch DataFrame.
    """
    global _LATEST_DRIFT_REPORT

    baseline = load_baseline_profile()
    features_config = baseline.get("features", {})

    # Map column names case-insensitively
    col_lookup = {c.lower(): c for c in batch_df.columns}

    feature_results: Dict[str, Any] = {}
    psi_values: List[float] = []

    for display_name, feat_meta in features_config.items():
        key = feat_meta.get("feature_key", display_name.lower())
        col_name = col_lookup.get(key) or col_lookup.get(display_name.lower())

        if col_name is None:
            continue

        raw_series = pd.to_numeric(batch_df[col_name], errors="coerce").dropna().values
        if len(raw_series) == 0:
            continue

        interior_cutpoints = feat_meta["interior_cutpoints"]
        baseline_props = feat_meta["baseline_proportions"]

        # Bin actual batch values into the same decile edges
        edges = np.r_[-np.inf, interior_cutpoints, np.inf]
        counts, _ = np.histogram(raw_series, bins=edges)
        actual_props = (counts / len(raw_series)).tolist()

        psi = calculate_psi(baseline_props, actual_props)
        psi_values.append(psi)

        # Empirical Kolmogorov-Smirnov statistic on binned empirical CDFs
        actual_cdf = np.cumsum(actual_props)
        baseline_cdf = np.cumsum(baseline_props)
        ks_stat = float(np.round(np.max(np.abs(actual_cdf - baseline_cdf)), 4))

        # Classify drift severity
        if psi >= 0.25:
            status = "DRIFT_DETECTED"
            severity = "SIGNIFICANT"
            interpretation = "Significant Distribution Shift (Retraining Recommended)"
        elif psi >= 0.10:
            status = "WARNING"
            severity = "MODERATE"
            interpretation = "Moderate Shift (Monitor Closely)"
        else:
            status = "STABLE"
            severity = "NEGLIGIBLE"
            interpretation = "Negligible Drift (Stable Distribution)"

        feature_results[display_name] = {
            "psi": psi,
            "status": status,
            "severity": severity,
            "interpretation": interpretation,
            "ks_statistic": ks_stat,
            "batch_mean": float(np.round(np.mean(raw_series), 4)),
            "baseline_mean": float(np.round(feat_meta["mean"], 4)),
            "batch_std": float(np.round(np.std(raw_series), 4)),
            "baseline_std": float(np.round(feat_meta["std"], 4)),
            "actual_distribution": [round(p, 4) for p in actual_props],
            "baseline_distribution": [round(p, 4) for p in baseline_props]
        }

    if psi_values:
        max_psi = float(np.round(max(psi_values), 4))
        mean_psi = float(np.round(np.mean(psi_values), 4))
        max_feature = max(feature_results.keys(), key=lambda k: feature_results[k]["psi"])
    else:
        max_psi = 0.0
        mean_psi = 0.0
        max_feature = "None"

    if max_psi >= 0.25:
        overall_status = "DRIFT_DETECTED"
        retrain_recommended = True
        recommendation = f"Severe distribution shift detected in feature '{max_feature}' (PSI: {max_psi:.4f}). Immediate model retraining advised."
    elif max_psi >= 0.10:
        overall_status = "WARNING"
        retrain_recommended = False
        recommendation = f"Moderate distribution shift detected in feature '{max_feature}' (PSI: {max_psi:.4f}). Monitor transaction stream closely."
    else:
        overall_status = "STABLE"
        retrain_recommended = False
        recommendation = "All monitored risk features remain statistically stable with negligible drift."

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_size": len(batch_df),
        "overall_status": overall_status,
        "overall_psi": max_psi,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "max_drifted_feature": max_feature,
        "retrain_recommended": retrain_recommended,
        "recommendation": recommendation,
        "baseline_metadata": baseline.get("metadata", {}),
        "features": feature_results
    }

    _LATEST_DRIFT_REPORT = report
    return report


def get_latest_drift_report() -> Dict[str, Any]:
    """Returns the most recent drift report or creates a clean baseline status."""
    global _LATEST_DRIFT_REPORT
    if _LATEST_DRIFT_REPORT is not None:
        return _LATEST_DRIFT_REPORT

    # If no batch evaluated yet, return initial baseline status
    baseline = load_baseline_profile()
    features = {}
    for display_name, meta in baseline.get("features", {}).items():
        features[display_name] = {
            "psi": 0.0,
            "status": "STABLE",
            "severity": "NEGLIGIBLE",
            "interpretation": "Baseline Reference (Awaiting Batch Stream)",
            "ks_statistic": 0.0,
            "batch_mean": round(meta["mean"], 4),
            "baseline_mean": round(meta["mean"], 4),
            "batch_std": round(meta["std"], 4),
            "baseline_std": round(meta["std"], 4),
            "actual_distribution": meta["baseline_proportions"],
            "baseline_distribution": meta["baseline_proportions"]
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_size": 0,
        "overall_status": "STABLE",
        "overall_psi": 0.0,
        "mean_psi": 0.0,
        "max_psi": 0.0,
        "max_drifted_feature": "None",
        "retrain_recommended": False,
        "recommendation": "Engine initialized with reference training baseline. Ready to audit incoming batches.",
        "baseline_metadata": baseline.get("metadata", {}),
        "features": features
    }
