"""Tests for bootstrap confidence intervals and repeated grouped CV."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.resampling import (
    bootstrap_metric_ci,
    bootstrap_metrics_table,
    format_estimate_with_ci,
    repeated_grouped_cv,
    summarize_cv_results,
)
from src.features.preprocessing import build_preprocessor
from src.models.logistic import build_logistic_model


def _synthetic_scores(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # informative but noisy probabilities
    p = np.clip(0.6 * y + rng.normal(0.2, 0.25, size=n), 0.001, 0.999)
    return y, p


def test_bootstrap_ci_brackets_point_estimate():
    y, p = _synthetic_scores()
    result = bootstrap_metric_ci(y, p, roc_auc_score, n_bootstrap=200)

    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert 0.5 < result["estimate"] < 1.0
    assert result["n_valid"] == 200


def test_bootstrap_ci_is_reproducible():
    y, p = _synthetic_scores()
    a = bootstrap_metric_ci(y, p, roc_auc_score, n_bootstrap=100, random_state=1)
    b = bootstrap_metric_ci(y, p, roc_auc_score, n_bootstrap=100, random_state=1)
    assert a == b


def test_bootstrap_ci_narrows_with_more_data():
    y_small, p_small = _synthetic_scores(n=60, seed=3)
    y_large, p_large = _synthetic_scores(n=3000, seed=3)

    small = bootstrap_metric_ci(y_small, p_small, roc_auc_score, n_bootstrap=200)
    large = bootstrap_metric_ci(y_large, p_large, roc_auc_score, n_bootstrap=200)

    width_small = small["ci_high"] - small["ci_low"]
    width_large = large["ci_high"] - large["ci_low"]
    assert width_large < width_small


def test_bootstrap_ci_handles_single_class_resamples():
    # Two positives out of 20: many resamples will contain no positives.
    y = np.array([0] * 18 + [1] * 2)
    p = np.linspace(0.05, 0.95, 20)
    result = bootstrap_metric_ci(y, p, roc_auc_score, n_bootstrap=100)
    assert result["n_valid"] < 100
    assert not np.isnan(result["estimate"])


def test_bootstrap_ci_rejects_length_mismatch():
    with pytest.raises(ValueError):
        bootstrap_metric_ci([0, 1, 1], [0.2, 0.8], roc_auc_score)


def test_bootstrap_metrics_table_shape():
    y, p = _synthetic_scores()
    table = bootstrap_metrics_table(
        y,
        p,
        {"auroc": roc_auc_score},
        n_bootstrap=50,
    )
    assert list(table.columns) == [
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_bootstrap",
        "n_valid",
    ]
    assert table.loc[0, "metric"] == "auroc"


def test_format_estimate_with_ci():
    assert format_estimate_with_ci(0.8712, 0.855, 0.887) == "0.871 (0.855–0.887)"
    assert format_estimate_with_ci(0.5, np.nan, np.nan) == "0.500 (CI n/a)"


def test_repeated_grouped_cv_never_splits_a_patient(synthetic_clinical_df):
    df = synthetic_clinical_df.copy()
    # Give every two rows the same patient so grouping matters.
    df["subject_id"] = np.arange(len(df)) // 2

    feature_cols = [
        "anchor_age",
        "creatinine_first",
        "lactate_first",
        "gender",
        "admission_type",
    ]
    preprocessor = build_preprocessor(
        numeric_features=["anchor_age", "creatinine_first", "lactate_first"],
        categorical_features=["gender", "admission_type"],
    )
    pipeline = build_logistic_model(preprocessor)

    results = repeated_grouped_cv(
        pipeline,
        df[feature_cols],
        df["hospital_expire_flag"],
        df["subject_id"],
        n_splits=3,
        n_repeats=2,
        model_name="logistic",
    )

    assert len(results) == 6
    assert set(results["model"]) == {"logistic"}
    assert results["auroc"].between(0, 1).all()
    assert (results["n_train"] + results["n_valid"] == len(df)).all()

    summary = summarize_cv_results(results)
    assert summary.loc[0, "n_fits"] == 6
    assert "auroc_mean" in summary.columns


def test_repeated_grouped_cv_rejects_length_mismatch():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        repeated_grouped_cv(None, X, [0, 1], [1, 2, 3])
