"""
Production-Ready MLOps Pipeline - Training Script
-------------------------------------------------
Features:
1. CSV or synthetic training data
2. Automatic preprocessing for numeric/categorical columns
3. Model training and evaluation
4. MLflow experiment tracking
5. Local versioned model artifacts
6. Baseline data profiling for monitoring

Usage:
    python train_mlops_pipeline.py --simulate --n_samples 5000 --target_col target
    python train_mlops_pipeline.py --csv data.csv --target_col target
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-style MLOps training pipeline")
    parser.add_argument("--csv", type=str, default="", help="Path to CSV dataset")
    parser.add_argument("--simulate", action="store_true", help="Train on synthetic demo data")
    parser.add_argument("--n_samples", type=int, default=5000, help="Rows for synthetic data")
    parser.add_argument("--target_col", type=str, default="target", help="Target column")
    parser.add_argument("--experiment_name", type=str, default="Production_Ready_MLOps_Pipeline", help="MLflow experiment name")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def simulate_dataset(n_samples: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    cities = np.array(["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune"])
    channels = np.array(["web", "mobile", "partner"])
    segments = np.array(["A", "B", "C"])

    age = rng.integers(21, 61, size=n_samples)
    income = rng.normal(65000, 18000, size=n_samples).clip(15000, None)
    tenure_months = rng.integers(1, 60, size=n_samples)
    city = rng.choice(cities, size=n_samples, replace=True)
    channel = rng.choice(channels, size=n_samples, replace=True, p=[0.45, 0.45, 0.10])
    segment = rng.choice(segments, size=n_samples, replace=True, p=[0.35, 0.40, 0.25])

    score = (
        -3.2
        + 0.015 * (age - 30)
        + 0.00002 * (income - 50000)
        + 0.04 * (tenure_months > 10)
        + 0.30 * (channel == "mobile").astype(int)
        + 0.25 * (segment == "A").astype(int)
        - 0.20 * (city == "Pune").astype(int)
    )
    prob = 1 / (1 + np.exp(-score))
    target = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "age": age,
            "income": np.round(income, 2),
            "tenure_months": tenure_months,
            "city": city,
            "channel": channel,
            "segment": segment,
            "target": target,
        }
    )


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.simulate:
        return simulate_dataset(n_samples=args.n_samples, random_state=args.random_state)

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.target_col not in df.columns:
            raise ValueError(f"Target column '{args.target_col}' not found in CSV.")
        return df

    raise ValueError("Provide either --simulate or --csv.")


def infer_feature_types(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def create_baseline_profile(df: pd.DataFrame, target_col: str) -> dict:
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    profile = {"numeric": {}, "categorical": {}}

    for col in numeric_cols:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        profile["numeric"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0) if series.std(ddof=0) and not np.isnan(series.std(ddof=0)) else 1.0),
            "missing_rate": float(series.isna().mean()),
        }

    for col in categorical_cols:
        vc = feature_df[col].astype(str).fillna("MISSING").value_counts(normalize=True)
        profile["categorical"][col] = {
            "distribution": {k: float(v) for k, v in vc.to_dict().items()},
            "missing_rate": float(feature_df[col].isna().mean()),
        }

    return profile


def save_versioned_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    baseline_profile: dict,
    feature_columns: list[str],
    target_col: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = Path("models") / f"model_v_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, version_dir / "model.joblib")

    metadata = {
        "version": version_dir.name,
        "created_at": timestamp,
        "target_col": target_col,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "model_type": "RandomForestClassifier",
    }

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(version_dir / "baseline_profile.json", "w", encoding="utf-8") as f:
        json.dump(baseline_profile, f, indent=2)

    with open(version_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_columns}, f, indent=2)

    return version_dir


def main() -> None:
    args = parse_args()
    df = load_data(args)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found.")

    numeric_cols, categorical_cols = infer_feature_types(df, args.target_col)
    feature_columns = [c for c in df.columns if c != args.target_col]

    X = df[feature_columns].copy()
    y = df[args.target_col].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    pipeline = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob, threshold=0.5)

    baseline_profile = create_baseline_profile(df=pd.concat([X_train, pd.Series(y_train, name=args.target_col)], axis=1), target_col=args.target_col)
    version_dir = save_versioned_artifacts(
        pipeline=pipeline,
        metrics=metrics,
        baseline_profile=baseline_profile,
        feature_columns=feature_columns,
        target_col=args.target_col,
    )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=version_dir.name):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("numeric_features", len(numeric_cols))
        mlflow.log_param("categorical_features", len(categorical_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.log_artifacts(str(version_dir))
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn_model")

    print("\nTraining complete")
    print("-----------------")
    print(f"Saved versioned model to: {version_dir.resolve()}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")


if __name__ == "__main__":
    main()
