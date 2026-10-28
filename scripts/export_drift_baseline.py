"""
SentinelRisk - Baseline Distribution Exporter for MLOps Drift Detection.
Calculates decile bin edges, baseline frequency proportions, and summary statistics
for key fraud risk drivers (V4, V10, V12, V14, V17, Amount) and saves to
models/baseline_distribution.json.
"""

import os
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

DB_PATH = Path("data/creditcard.db")
CSV_PATH = Path("data/creditcard.csv")
OUTPUT_PATH = Path("models/baseline_distribution.json")

KEY_FEATURES = ["v4", "v10", "v12", "v14", "v17", "amount"]


def load_dataset() -> pd.DataFrame:
    """Loads feature data from SQLite DB, CSV, or synthetic fallback."""
    if DB_PATH.exists():
        print(f"[Baseline Exporter] Loading dataset from SQLite DB: {DB_PATH}")
        with sqlite3.connect(DB_PATH) as conn:
            cols = ", ".join(KEY_FEATURES)
            df = pd.read_sql_query(f"SELECT {cols} FROM transactions", conn)
        return df

    if CSV_PATH.exists():
        print(f"[Baseline Exporter] Loading dataset from CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.lower() for c in df.columns]
        return df[KEY_FEATURES]

    print("[Baseline Exporter] Warning: No local DB or CSV found. Generating realistic baseline sample.")
    np.random.seed(42)
    n = 10000
    data = {
        "v4": np.random.normal(0.0, 1.4, n),
        "v10": np.random.normal(0.0, 1.1, n),
        "v12": np.random.normal(0.0, 1.0, n),
        "v14": np.random.normal(0.0, 0.95, n),
        "v17": np.random.normal(0.0, 0.85, n),
        "amount": np.random.exponential(88.0, n),
    }
    return pd.DataFrame(data)


def export_baseline():
    df = load_dataset()
    total_samples = len(df)
    print(f"[Baseline Exporter] Processing {total_samples:,} records for baseline profile...")

    profile = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_samples,
            "version": "1.0.0",
            "binning_strategy": "decile_10_bins",
            "features_tracked": [f.upper() if f != "amount" else "Amount" for f in KEY_FEATURES]
        },
        "features": {}
    }

    for feat in KEY_FEATURES:
        col_series = df[feat].dropna().values
        # Deciles from 10% to 90% (9 interior cut points for 10 bins)
        quantiles = np.linspace(10, 90, 9)
        interior_cutpoints = np.percentile(col_series, quantiles)

        # Ensure cutpoints are strictly monotonic for unique bins
        unique_cutpoints = []
        for cp in interior_cutpoints:
            val = float(cp)
            if not unique_cutpoints or val > unique_cutpoints[-1]:
                unique_cutpoints.append(val)
            else:
                # Add slight epsilon if identical (e.g. repeated 0s in amounts)
                unique_cutpoints.append(unique_cutpoints[-1] + 1e-4)

        # Construct full bin edges with infinite bounds
        full_edges = np.r_[-np.inf, unique_cutpoints, np.inf]
        counts, _ = np.histogram(col_series, bins=full_edges)
        proportions = (counts / len(col_series)).tolist()

        display_name = feat.upper() if feat != "amount" else "Amount"
        profile["features"][display_name] = {
            "feature_key": feat,
            "mean": float(np.mean(col_series)),
            "std": float(np.std(col_series)),
            "min": float(np.min(col_series)),
            "max": float(np.max(col_series)),
            "interior_cutpoints": unique_cutpoints,
            "baseline_proportions": proportions,
            "num_bins": len(proportions)
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"[Baseline Exporter] Successfully exported baseline distribution to {OUTPUT_PATH}")


if __name__ == "__main__":
    export_baseline()
