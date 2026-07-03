"""
FastAPI serving app for the Production-Ready MLOps Pipeline.

Run:
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def get_latest_model_dir(root: str = "models") -> Path:
    root_path = Path(root)
    candidates = sorted(root_path.glob("model_v_*"))
    if not candidates:
        raise FileNotFoundError("No versioned model directory found under 'models/'.")
    return candidates[-1]


LATEST_MODEL_DIR = get_latest_model_dir()
MODEL = joblib.load(LATEST_MODEL_DIR / "model.joblib")

with open(LATEST_MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
    METADATA = json.load(f)

FEATURE_COLUMNS = METADATA["feature_columns"]


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of feature dictionaries")


app = FastAPI(title="Production-Ready MLOps Pipeline API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": METADATA["version"]}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_version": METADATA["version"],
        "model_type": METADATA["model_type"],
        "target_col": METADATA["target_col"],
        "feature_columns": FEATURE_COLUMNS,
        "metrics": METADATA["metrics"],
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    df = pd.DataFrame(request.records)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLUMNS]

    preds = MODEL.predict(df).tolist()

    response = {
        "model_version": METADATA["version"],
        "predictions": preds,
    }

    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(df)[:, 1].tolist()
        response["probabilities"] = probs

    return response
