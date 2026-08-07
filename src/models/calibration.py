"""Probability-calibration utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def build_prefit_calibrator(
    fitted_model,
    *,
    method: str,
):
    """Wrap an already fitted model for sigmoid or isotonic calibration.

    Uses FrozenEstimator when available and falls back to cv="prefit"
    for older scikit-learn versions.
    """
    if method not in {
        "sigmoid",
        "isotonic",
    }:
        raise ValueError(
            "method must be 'sigmoid' or 'isotonic'."
        )

    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(
                fitted_model
            ),
            method=method,
        )

    except (
        ImportError,
        ModuleNotFoundError,
    ):
        return CalibratedClassifierCV(
            estimator=fitted_model,
            method=method,
            cv="prefit",
        )


def fit_sigmoid_calibrator(
    fitted_model,
    X_calibration,
    y_calibration,
):
    """Fit sigmoid (Platt) calibration."""
    calibrator = build_prefit_calibrator(
        fitted_model,
        method="sigmoid",
    )

    calibrator.fit(
        X_calibration,
        y_calibration,
    )

    return calibrator


def fit_isotonic_calibrator(
    fitted_model,
    X_calibration,
    y_calibration,
):
    """Fit isotonic calibration."""
    calibrator = build_prefit_calibrator(
        fitted_model,
        method="isotonic",
    )

    calibrator.fit(
        X_calibration,
        y_calibration,
    )

    return calibrator


def _probability_metrics(
    y_true,
    probabilities,
) -> dict:
    probabilities = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        1e-8,
        1 - 1e-8,
    )

    return {
        "auroc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "auprc": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "mean_predicted_risk": float(
            probabilities.mean()
        ),
        "observed_event_rate": float(
            np.mean(y_true)
        ),
    }


def compare_calibration_methods(
    fitted_model,
    sigmoid_model,
    isotonic_model,
    X,
    y,
) -> pd.DataFrame:
    """Compare uncalibrated, sigmoid, and isotonic probabilities."""
    method_probabilities = {
        "Uncalibrated": (
            fitted_model
            .predict_proba(X)[:, 1]
        ),
        "Sigmoid": (
            sigmoid_model
            .predict_proba(X)[:, 1]
        ),
        "Isotonic": (
            isotonic_model
            .predict_proba(X)[:, 1]
        ),
    }

    records = []

    for method_name, probabilities in method_probabilities.items():
        records.append(
            {
                "method": method_name,
                **_probability_metrics(
                    y,
                    probabilities,
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "brier_score",
                "log_loss",
            ],
            ascending=True,
        )
        .reset_index(drop=True)
    )
