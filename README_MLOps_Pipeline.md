# Production-Ready MLOps Pipeline

A portfolio-ready MLOps starter project that automates data preprocessing, model training, experiment tracking, versioning, artifact logging, REST API deployment, CI/CD-style checks, and monitoring-driven retraining decisions.

## Project Overview

This project demonstrates an end-to-end **production-style MLOps workflow** for tabular machine learning systems. It is designed to show how a model can move from data preprocessing and experiment tracking to containerized deployment and monitoring.

It aligns with the following project claims:

- Built a **production-style MLOps pipeline** automating **data preprocessing, model training, versioning, and deployment workflows**
- Used **MLflow** and **DVC-style artifact management concepts** for experiment tracking, model registry, data versioning, artifact logging, and reproducible workflows
- Containerized model serving with **FastAPI**, **Docker**, and **REST APIs** for real-time inference and modular deployment
- Added **CI/CD and monitoring workflows** for automated testing, performance checks, drift detection, and retraining triggers

## What This Repository Includes

### 1. Training pipeline
The training script:
- loads a CSV dataset or creates a synthetic demo dataset
- detects numeric and categorical features
- builds a preprocessing + model pipeline
- trains a classifier
- evaluates it on a held-out test set
- logs metrics and artifacts to **MLflow**
- versions the model locally under a timestamped model directory

### 2. Model versioning
Each training run creates a versioned folder such as:

```text
models/model_v_20260704_153500/
```

Each version stores:
- `model.joblib`
- `metadata.json`
- `baseline_profile.json`
- `feature_columns.json`

### 3. FastAPI model serving
The serving app exposes:
- `/health`
- `/model-info`
- `/predict`

The `/predict` endpoint accepts JSON records and returns:
- predictions
- probabilities
- model version

### 4. Monitoring and drift checks
The monitoring script:
- compares a new batch against the training baseline
- checks numeric drift using standardized mean shifts
- checks categorical drift using distribution distance
- writes a drift report
- flags whether retraining is recommended

### 5. Docker deployment
A Dockerfile is included for serving the API in a containerized environment.

### 6. CI/CD starter workflow
A GitHub Actions workflow is included as a starter for:
- dependency installation
- training smoke tests
- FastAPI import checks

## Repository Structure

```text
.
├── README_MLOps_Pipeline.md
├── train_mlops_pipeline.py
├── app_fastapi.py
├── monitor_and_retrain.py
├── requirements_mlops.txt
├── Dockerfile.mlops
└── github_actions_mlops_ci.yml
```

## Installation

```bash
pip install -r requirements_mlops.txt
```

## How to Train

### Train on synthetic demo data
```bash
python train_mlops_pipeline.py --simulate --n_samples 5000 --target_col target
```

### Train on your own CSV
```bash
python train_mlops_pipeline.py --csv data.csv --target_col target
```

This creates:
- a versioned model folder in `models/`
- metrics in MLflow
- baseline profiling files for monitoring

## Launch MLflow UI

```bash
mlflow ui
```

By default, runs are logged to the local `mlruns/` directory.

## Run the FastAPI Service

```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"age": 34, "income": 65000, "tenure_months": 12, "city": "Delhi", "channel": "web"},
      {"age": 28, "income": 52000, "tenure_months": 5, "city": "Mumbai", "channel": "mobile"}
    ]
  }'
```

## Run Monitoring

```bash
python monitor_and_retrain.py --batch_csv new_data.csv
```

This generates:
- `monitoring_reports/latest_drift_report.json`

If the drift score crosses the configured threshold, the script recommends retraining.

## Docker Usage

Build the image:

```bash
docker build -f Dockerfile.mlops -t mlops-fastapi .
```

Run the container:

```bash
docker run -p 8000:8000 mlops-fastapi
```

## CI/CD Starter

The included GitHub Actions workflow can be placed into:

```text
.github/workflows/mlops-ci.yml
```

It performs:
- checkout
- Python setup
- dependency installation
- training smoke test
- API import validation

## Why this project is strong

This project is strong because it goes beyond model training and shows the **operational lifecycle** of ML systems:

- reproducible training
- experiment tracking
- artifact versioning
- REST deployment
- drift monitoring
- retraining readiness
- containerization and CI/CD hooks

## Resume-ready Summary

Built a production-ready MLOps pipeline automating preprocessing, model training, experiment tracking, versioning, FastAPI deployment, Docker-based serving, drift monitoring, and CI/CD-style validation for reproducible ML workflows.

## Notes

This is a clean, portfolio-friendly starter implementation. If you want, it can be adapted to your fraud, forecasting, or A/B testing projects so the MLOps pipeline directly serves one of your real models.
