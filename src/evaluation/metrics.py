"""Classification, discrimination, and probability-quality metrics."""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.validation import (
    validate_probability_vector,
)


def discrimination_metrics(
    y_true,
    probabilities,
) -> dict:
    """Return AUROC and AUPRC.

    Both metrics require predicted probabilities rather than hard labels.
    """
    validate_probability_vector(
        probabilities
    )

    y_array = np.asarray(y_true)

    if np.unique(y_array).size < 2:
        auroc = np.nan
    else:
        auroc = roc_auc_score(
            y_array,
            probabilities,
        )

    if (y_array == 1).sum() == 0:
        auprc = np.nan
    else:
        auprc = average_precision_score(
            y_array,
            probabilities,
        )

    return {
        "auroc": auroc,
        "auprc": auprc,
    }


def probability_metrics(
    y_true,
    probabilities,
) -> dict:
    """Return metrics that assess predicted probability quality."""
    validate_probability_vector(
        probabilities
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    clipped = np.clip(
        probabilities,
        1e-8,
        1 - 1e-8,
    )

    return {
        **discrimination_metrics(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            clipped,
        ),
        "log_loss": log_loss(
            y_true,
            clipped,
            labels=[0, 1],
        ),
        "mean_predicted_risk": float(
            clipped.mean()
        ),
        "observed_event_rate": float(
            np.mean(y_true)
        ),
    }


def classification_metrics(
    y_true,
    probabilities,
    *,
    threshold: float = 0.50,
) -> dict:
    """Return discrimination plus threshold-based classification metrics."""
    validate_probability_vector(
        probabilities
    )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        **discrimination_metrics(
            y_true,
            probabilities,
        ),
        "threshold": float(threshold),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision_ppv": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall_sensitivity": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "patients_flagged": int(
            predictions.sum()
        ),
        "flagged_percentage": float(
            predictions.mean()
        ),
    }
