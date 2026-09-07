"""FastAPI service for first-24h ICU mortality risk.

Run locally:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    FEATURE_CONFIG,
    PIPELINE_PATH,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)

# Fixed, documented cut-points. These are NOT clinically validated thresholds;
# they exist so the response is readable without a separate lookup.
RISK_TIERS = [(0.10, "low"), (0.30, "moderate"), (1.01, "high")]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request."""
    if not PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {PIPELINE_PATH}. "
            "Run `python -m src.models.train_final` first."
        )
    state["pipeline"] = joblib.load(PIPELINE_PATH)
    state["feature_order"] = (
        FEATURE_CONFIG["numeric_features"] + FEATURE_CONFIG["categorical_features"]
    )
    yield
    state.clear()


app = FastAPI(
    title="ICU Mortality Risk API",
    description=(
        "Predicts in-hospital mortality from the first 24 hours of an ICU stay. "
        "Trained on MIMIC-IV v3.1. Research and demonstration only — not for clinical use."
    ),
    version=FEATURE_CONFIG["model_version"],
    lifespan=lifespan,
)


def _risk_tier(probability: float) -> str:
    for cutoff, label in RISK_TIERS:
        if probability < cutoff:
            return label
    return "high"

@app.get("/")
def root():
    return {"message": "ICU Mortality Risk API", "docs": "/docs", "health": "/health"}


    
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if "pipeline" in state else "model_not_loaded",
        model_name=FEATURE_CONFIG["model_name"],
        model_version=FEATURE_CONFIG["model_version"],
        n_features=len(state.get("feature_order", [])),
        test_auroc=FEATURE_CONFIG["test_auroc"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    provided = patient.model_dump()
    # One-row DataFrame in the exact column order the pipeline was fitted on.
    row = pd.DataFrame([{col: provided.get(col) for col in state["feature_order"]}])

    try:
        probability = float(state["pipeline"].predict_proba(row)[0, 1])
    except Exception as exc:  # surface preprocessing errors as 400, not 500
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    n_provided = sum(v is not None for v in provided.values())
    return PredictionResponse(
        mortality_probability=round(probability, 4),
        risk_tier=_risk_tier(probability),
        model_name=FEATURE_CONFIG["model_name"],
        model_version=FEATURE_CONFIG["model_version"],
        n_features_provided=n_provided,
        n_features_missing=len(state["feature_order"]) - n_provided,
    )
