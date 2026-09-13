"""Tests for the MLflow tracking helpers.

These run against a temporary file-based tracking directory, so they need no
server and leave nothing behind.
"""

import numpy as np
import pandas as pd
import pytest

from src.tracking import (
    DEFAULT_TRACKING_URI,
    EXPERIMENT_NAME,
    get_tracking_uri,
    git_commit,
    log_cv_results,
    start_run,
)

mlflow = pytest.importorskip("mlflow")


def test_tracking_uri_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert get_tracking_uri() == DEFAULT_TRACKING_URI
    assert get_tracking_uri().startswith("sqlite:///")


def test_tracking_uri_respects_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid:5000")
    assert get_tracking_uri() == "http://example.invalid:5000"


def test_git_commit_is_a_hash_or_none():
    commit = git_commit()
    assert commit is None or (4 <= len(commit) <= 40 and commit.isalnum())


def test_start_run_logs_params_and_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    with start_run("unit-test-run", tags={"stage": "test"}) as run:
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("test_auroc", 0.892)
        run_id = run.info.run_id

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    logged = client.get_run(run_id)

    assert logged.data.params["max_depth"] == "4"
    assert logged.data.metrics["test_auroc"] == pytest.approx(0.892)
    assert logged.data.tags["stage"] == "test"
    assert logged.data.tags["project"] == "clinical-outcome-prediction"

    experiment = client.get_experiment(logged.info.experiment_id)
    assert experiment.name == EXPERIMENT_NAME


def test_log_cv_results_records_mean_and_sd(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    rng = np.random.default_rng(0)
    cv_results = pd.DataFrame(
        {
            "auroc": rng.normal(0.89, 0.005, 6),
            "auprc": rng.normal(0.60, 0.01, 6),
            "brier": rng.normal(0.066, 0.002, 6),
        }
    )

    with start_run("cv-run") as run:
        log_cv_results(cv_results, model_name="xgboost")
        run_id = run.info.run_id

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    logged = client.get_run(run_id)

    assert logged.data.params["cv_n_fits"] == "6"
    assert logged.data.metrics["cv_auroc_mean"] == pytest.approx(cv_results["auroc"].mean())
    assert logged.data.metrics["cv_auroc_sd"] == pytest.approx(cv_results["auroc"].std(ddof=1))
    assert logged.data.tags["model"] == "xgboost"
    assert len(client.get_metric_history(run_id, "cv_auroc_fold")) == 6
