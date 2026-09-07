"""API tests. They build a tiny model on synthetic data so no MIMIC artifact is needed.

The app reads MODEL_DIR at import time, so the fixture writes the artifacts,
sets the env var, then imports the app fresh.
"""

import importlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.preprocessing import build_preprocessor
from src.models.logistic import build_logistic_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "anchor_age": rng.normal(65, 12, n),
            "creatinine_first": rng.lognormal(0, 0.4, n),
            "gender": rng.choice(["F", "M"], n),
            "admission_type": rng.choice(["EW EMER.", "ELECTIVE"], n),
        }
    )
    y = (df["anchor_age"] + 20 * df["creatinine_first"] + rng.normal(0, 10, n) > 95).astype(int)

    numeric = ["anchor_age", "creatinine_first"]
    categorical = ["gender", "admission_type"]
    pipe = build_logistic_model(build_preprocessor(numeric, categorical))
    pipe.fit(df, y)
    joblib.dump(pipe, tmp / "final_xgboost_pipeline.joblib")

    config = {
        "model_name": "test_model",
        "model_version": "test",
        "test_auroc": 0.9,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
    (tmp / "final_feature_config.json").write_text(json.dumps(config))

    os.environ["MODEL_DIR"] = str(tmp)
    for mod in ("app.main", "app.schemas"):
        sys.modules.pop(mod, None)
    main = importlib.import_module("app.main")

    with TestClient(main.app) as c:
        yield c

    os.environ.pop("MODEL_DIR", None)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 4


def test_predict_full_row(client):
    r = client.post(
        "/predict",
        json={"anchor_age": 80, "creatinine_first": 3.0, "gender": "M", "admission_type": "EW EMER."},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["mortality_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "moderate", "high"}
    assert body["n_features_missing"] == 0


def test_predict_with_missing_values(client):
    r = client.post("/predict", json={"anchor_age": 40})
    assert r.status_code == 200
    assert r.json()["n_features_missing"] == 3


def test_predict_rejects_unknown_column(client):
    r = client.post("/predict", json={"anchor_age": 40, "not_a_feature": 1})
    assert r.status_code == 422


def test_predict_rejects_wrong_type(client):
    r = client.post("/predict", json={"anchor_age": "eighty"})
    assert r.status_code == 422
