"""Tests for the feature-store endpoint.

These use an on-disk SQLite database rather than Postgres: the access layer goes
through SQLAlchemy, so the code path under test is the same one the container
runs, and the suite stays runnable in CI without a database service.
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
from sqlalchemy import create_engine

from src.features.preprocessing import build_preprocessor
from src.models.logistic import build_logistic_model

NUMERIC = ["anchor_age", "creatinine_first"]
CATEGORICAL = ["gender", "admission_type"]


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "stay_id": np.arange(30000, 30000 + n),
            "subject_id": np.arange(1, n + 1),
            "anchor_age": rng.normal(65, 12, n),
            "creatinine_first": rng.lognormal(0, 0.4, n),
            "gender": rng.choice(["F", "M"], n),
            "admission_type": rng.choice(["EW EMER.", "ELECTIVE"], n),
            "hospital_expire_flag": rng.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="module")
def client(tmp_path_factory, cohort):
    tmp = tmp_path_factory.mktemp("stack")

    pipe = build_logistic_model(build_preprocessor(NUMERIC, CATEGORICAL))
    pipe.fit(cohort[NUMERIC + CATEGORICAL], cohort["hospital_expire_flag"])
    joblib.dump(pipe, tmp / "final_xgboost_pipeline.joblib")

    (tmp / "final_feature_config.json").write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "model_version": "test",
                "test_auroc": 0.9,
                "numeric_features": NUMERIC,
                "categorical_features": CATEGORICAL,
            }
        )
    )

    db_path = tmp / "cohort.db"
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
    cohort.to_sql("modeling_cohort", engine, if_exists="replace", index=False)

    os.environ["MODEL_DIR"] = str(tmp)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    for module in ("app.main", "app.schemas", "app.feature_store"):
        sys.modules.pop(module, None)
    main = importlib.import_module("app.main")

    with TestClient(main.app) as c:
        yield c

    os.environ.pop("MODEL_DIR", None)
    os.environ.pop("DATABASE_URL", None)


def test_health_reports_connected_store(client):
    body = client.get("/health").json()
    assert body["feature_store"] == "connected"


def test_predict_by_stay_id(client, cohort):
    stay_id = int(cohort["stay_id"].iloc[0])
    response = client.get(f"/predict/{stay_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["stay_id"] == stay_id
    assert 0.0 <= body["mortality_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "moderate", "high"}
    assert body["n_features_provided"] == len(NUMERIC + CATEGORICAL)


def test_stay_prediction_matches_post_predict(client, cohort):
    """The two endpoints must agree — same row, same model, same probability."""
    row = cohort.iloc[0]
    by_id = client.get(f"/predict/{int(row['stay_id'])}").json()
    by_body = client.post(
        "/predict",
        json={column: row[column] for column in NUMERIC + CATEGORICAL},
    ).json()

    assert by_id["mortality_probability"] == pytest.approx(
        by_body["mortality_probability"], abs=1e-4
    )


def test_unknown_stay_id_returns_404(client):
    assert client.get("/predict/999999999").status_code == 404


def test_response_leaks_no_feature_values(client, cohort):
    """The response carries a risk estimate, never the patient's feature values."""
    body = client.get(f"/predict/{int(cohort['stay_id'].iloc[0])}").json()
    assert set(body) == {
        "stay_id",
        "mortality_probability",
        "risk_tier",
        "model_name",
        "model_version",
        "n_features_provided",
        "n_features_missing",
    }
