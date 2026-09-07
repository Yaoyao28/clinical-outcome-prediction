"""Request / response schemas for the prediction API.

The input schema is generated from ``models/final_feature_config.json`` at
import time, so the API always accepts exactly the columns the model was
trained on. Every feature is optional: missing values are legitimate clinical
input (a lab that was not drawn) and the preprocessor imputes them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

# MODEL_DIR lets Docker / cloud / tests point the app at a different artifact
# folder without touching code. Default = <project>/models.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
CONFIG_PATH = MODEL_DIR / "final_feature_config.json"
PIPELINE_PATH = MODEL_DIR / "final_xgboost_pipeline.joblib"


def load_feature_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature config not found at {path}. "
            "Run `python -m src.models.train_final` or set MODEL_DIR."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_patient_features_model(config: dict) -> type[BaseModel]:
    """Create a Pydantic model with one optional field per feature."""
    fields: dict[str, tuple] = {}
    for name in config["numeric_features"]:
        fields[name] = (Optional[float], Field(default=None))
    for name in config["categorical_features"]:
        fields[name] = (Optional[str], Field(default=None))

    return create_model(
        "PatientFeatures",
        __config__=ConfigDict(extra="forbid"),  # reject unknown columns loudly
        **fields,
    )


FEATURE_CONFIG = load_feature_config()
PatientFeatures = build_patient_features_model(FEATURE_CONFIG)


class PredictionResponse(BaseModel):
    mortality_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str = Field(..., description="low / moderate / high, from fixed probability cut-points")
    model_name: str
    model_version: str
    n_features_provided: int
    n_features_missing: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    n_features: int
    test_auroc: float
