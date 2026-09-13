"""FastAPI service for first-24h ICU mortality risk.

Run locally:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.feature_store import fetch_features, get_database_url, store_is_reachable
from app.schemas import (
    FEATURE_CONFIG,
    PIPELINE_PATH,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
    StayPredictionResponse,
)

# Fixed, documented cut-points. These are NOT clinically validated thresholds;
# they exist so the response is readable without a separate lookup.
RISK_TIERS = [(0.10, "low"), (0.30, "moderate"), (1.01, "high")]

state: dict = {}


def _load_model() -> tuple[object, str]:
    """Load the model, preferring an MLflow registry version if one is configured.

    Set MLFLOW_MODEL_URI (e.g. "models:/icu-mortality-xgboost/3") to serve a
    specific registered version; MLFLOW_TRACKING_URI points at the backend. With
    neither set, the local joblib artifact baked into the image is used — that is
    what the container does, so the image stays self-contained and needs no
    network at startup.

    Returns the loaded pipeline and a short string naming where it came from,
    which /health reports so a running service can always be traced to a model.
    """
    model_uri = os.environ.get("MLFLOW_MODEL_URI")
    if model_uri:
        import mlflow.sklearn  # imported only when actually used

        return mlflow.sklearn.load_model(model_uri), model_uri

    if not PIPELINE_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {PIPELINE_PATH} and MLFLOW_MODEL_URI is unset. "
            "Run `python -m src.models.train_final` first."
        )
    return joblib.load(PIPELINE_PATH), str(PIPELINE_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not per request."""
    state["pipeline"], state["model_source"] = _load_model()
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


def _feature_store_status() -> str:
    """Report the feature store as configured / reachable / unreachable."""
    database_url = get_database_url()
    if not database_url:
        return "not_configured"
    return "connected" if store_is_reachable(database_url) else "unreachable"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if "pipeline" in state else "model_not_loaded",
        model_name=FEATURE_CONFIG["model_name"],
        model_version=FEATURE_CONFIG["model_version"],
        n_features=len(state.get("feature_order", [])),
        test_auroc=FEATURE_CONFIG["test_auroc"],
        model_source=state.get("model_source", "not_loaded"),
        feature_store=_feature_store_status(),
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


@app.get("/predict/{stay_id}", response_model=StayPredictionResponse)
def predict_by_stay(stay_id: int) -> StayPredictionResponse:
    """Score a stay whose features already live in the feature store.

    This is the shape a real deployment takes: the caller sends an identifier,
    not 66 values. Only the model's feature columns are read, so identifiers,
    timestamps and the outcome never reach the model, and none of the patient's
    feature values appear in the response.
    """
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    database_url = get_database_url()
    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="Feature store not configured. Set DATABASE_URL to enable this endpoint.",
        )

    try:
        row = fetch_features(stay_id, state["feature_order"], database_url)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store unavailable: {exc}") from exc

    if row is None:
        raise HTTPException(status_code=404, detail=f"stay_id {stay_id} not found")

    probability = float(state["pipeline"].predict_proba(row)[0, 1])
    n_provided = int(row.iloc[0].notna().sum())

    return StayPredictionResponse(
        stay_id=stay_id,
        mortality_probability=round(probability, 4),
        risk_tier=_risk_tier(probability),
        model_name=FEATURE_CONFIG["model_name"],
        model_version=FEATURE_CONFIG["model_version"],
        n_features_provided=n_provided,
        n_features_missing=len(state["feature_order"]) - n_provided,
    )
