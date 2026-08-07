"""Reusable evaluation plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.data.validation import validate_probability_vector


def _save_figure(
    path: str | Path | None,
    *,
    dpi: int = 300,
) -> None:
    if path is None:
        return

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )


def plot_roc_curve(
    y_true,
    probabilities,
    *,
    name: str = "Model",
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
    ax=None,
):
    """Plot ROC curve."""
    validate_probability_vector(
        probabilities
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7, 6)
        )

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name=name,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    _save_figure(save_path)

    return ax


def plot_precision_recall_curve(
    y_true,
    probabilities,
    *,
    name: str = "Model",
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
    ax=None,
):
    """Plot precision-recall curve."""
    validate_probability_vector(
        probabilities
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7, 6)
        )

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name=name,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    _save_figure(save_path)

    return ax


def plot_calibration_curve(
    y_true,
    probabilities,
    *,
    name: str = "Model",
    n_bins: int = 10,
    strategy: str = "quantile",
    title: str = "Calibration Curve",
    save_path: str | Path | None = None,
    ax=None,
):
    """Plot observed event rate against predicted probability."""
    validate_probability_vector(
        probabilities
    )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7, 6)
        )

    CalibrationDisplay.from_predictions(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
        name=name,
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()
    _save_figure(save_path)

    return ax


def plot_confusion_matrix(
    y_true,
    probabilities,
    *,
    threshold: float = 0.50,
    display_labels=(
        "Survived",
        "Died",
    ),
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    ax=None,
):
    """Plot confusion matrix after applying a probability threshold."""
    validate_probability_vector(
        probabilities
    )

    predictions = (
        np.asarray(probabilities)
        >= threshold
    ).astype(int)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(6, 6)
        )

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=display_labels,
        values_format="d",
        ax=ax,
    )

    ax.set_title(
        f"{title} (threshold={threshold:.2f})"
    )
    plt.tight_layout()
    _save_figure(save_path)

    return ax


def plot_probability_distribution(
    y_true,
    probabilities,
    *,
    title: str = "Predicted Probability Distribution",
    save_path: str | Path | None = None,
    bins: int = 15,
    ax=None,
):
    """Plot predicted risks for survivors and mortality events."""
    validate_probability_vector(
        probabilities
    )

    probabilities = np.asarray(
        probabilities
    )
    y_array = np.asarray(y_true)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 6)
        )

    ax.hist(
        probabilities[
            y_array == 0
        ],
        bins=bins,
        alpha=0.6,
        label="Survived",
    )

    ax.hist(
        probabilities[
            y_array == 1
        ],
        bins=bins,
        alpha=0.6,
        label="Died",
    )

    ax.set_title(title)
    ax.set_xlabel(
        "Predicted mortality probability"
    )
    ax.set_ylabel("Number of admissions")
    ax.legend()

    plt.tight_layout()
    _save_figure(save_path)

    return ax
