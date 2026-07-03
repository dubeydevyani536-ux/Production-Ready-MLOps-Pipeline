"""
Monitoring and drift check script for the Production-Ready MLOps Pipeline.

Usage:
    python monitor_and_retrain.py --batch_csv new_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift monitoring for versioned ML models")
    parser.add_argument("--batch_csv", type=str, required=True, help="Path to new batch CSV")
    parser.add_argument("--drift_threshold", type=float, default=0.35, help="Overall drift threshold")
    return parser.parse_args()


def get_latest_model_dir(root: str = "models") -> Path:
    candidates = sorted(Path(root).glob("model_v_*"))
    if not candidates:
        raise FileNotFoundError("No versioned model directory found under 'models/'.")
    return candidates[-1]


def numeric_drift(current_mean: float, baseline_mean: float, baseline_std: float) -> float:
    std = baseline_std if baseline_std and baseline_std > 0 else 1.0
    return abs(current_mean - baseline_mean) / std


def categorical_distance(current_dist: dict[str, float], baseline_dist: dict[str, float]) -> float:
    keys = set(current_dist.keys()) | set(baseline_dist.keys())
    return 0.5 * sum(abs(current_dist.get(k, 0.0) - baseline_dist.get(k, 0.0)) for k in keys)


def main() -> None:
    args = parse_args()

    latest_dir = get_latest_model_dir()
    with open(latest_dir / "baseline_profile.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    batch = pd.read_csv(args.batch_csv)

    numeric_report = {}
    categorical_report = {}
    drift_values = []

    for col, stats in baseline.get("numeric", {}).items():
        if col not in batch.columns:
            numeric_report[col] = {"status": "missing_in_batch", "drift_score": None}
            continue

        series = pd.to_numeric(batch[col], errors="coerce")
        drift_score = numeric_drift(
            current_mean=float(series.mean()),
            baseline_mean=float(stats["mean"]),
            baseline_std=float(stats["std"]),
        )
        numeric_report[col] = {
            "current_mean": float(series.mean()),
            "baseline_mean": float(stats["mean"]),
            "baseline_std": float(stats["std"]),
            "drift_score": float(drift_score),
        }
        drift_values.append(float(drift_score))

    for col, stats in baseline.get("categorical", {}).items():
        if col not in batch.columns:
            categorical_report[col] = {"status": "missing_in_batch", "distance": None}
            continue

        current_dist = batch[col].astype(str).fillna("MISSING").value_counts(normalize=True).to_dict()
        distance = categorical_distance(current_dist, stats["distribution"])
        categorical_report[col] = {
            "distance": float(distance),
            "current_distribution": {k: float(v) for k, v in current_dist.items()},
        }
        drift_values.append(float(distance))

    overall_drift = float(sum(drift_values) / max(len(drift_values), 1))
    retrain_recommended = overall_drift >= args.drift_threshold

    report = {
        "model_version": latest_dir.name,
        "overall_drift_score": overall_drift,
        "drift_threshold": args.drift_threshold,
        "retrain_recommended": retrain_recommended,
        "numeric_report": numeric_report,
        "categorical_report": categorical_report,
    }

    out_dir = Path("monitoring_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latest_drift_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nMonitoring complete")
    print("-------------------")
    print(f"Saved report to: {out_path.resolve()}")
    print(f"Overall drift score : {overall_drift:.4f}")
    print(f"Retrain recommended : {retrain_recommended}")


if __name__ == "__main__":
    main()
