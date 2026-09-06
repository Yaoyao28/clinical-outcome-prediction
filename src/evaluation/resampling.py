"""Resampling utilities: bootstrap confidence intervals and repeated grouped CV.

Why this module exists
----------------------
A single train/validation/test split gives one number per metric and no
sense of how much that number would move if the patients had been split
differently. Two tools fix that:

* ``bootstrap_metric_ci`` — resample the *evaluation set* with replacement
  many times and recompute the metric each time. The spread of those values
  is the sampling uncertainty of the metric on this population.
* ``repeated_grouped_cv`` — refit the *whole pipeline* on many different
  patient-level folds. The spread across folds tells you how stable the
  model (not just the metric) is, and lets you compare models fairly.

Both are patient-aware: bootstrap resamples rows, which is fine when each
row is one ICU stay; CV uses ``StratifiedGroupKFold`` so no patient appears
in both train and test folds.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.config import RANDOM_STATE
from src.evaluation.metrics import discrimination_metrics, probability_metrics


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_metric_ci(
    y_true,
    probabilities,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Return a point estimate and percentile bootstrap CI for one metric.

    Parameters
    ----------
    y_true:
        Binary labels (0/1).
    probabilities:
        Predicted probabilities for the positive class.
    metric_fn:
        Any function ``metric_fn(y_true, probabilities) -> float``,
        e.g. ``sklearn.metrics.roc_auc_score``.
    n_bootstrap:
        Number of resamples. 1000 is a reasonable default; 200 is enough
        for a quick look.
    confidence_level:
        Width of the interval (0.95 -> 2.5th and 97.5th percentiles).

    Returns
    -------
    dict with keys ``estimate``, ``ci_low``, ``ci_high``, ``n_bootstrap``,
    ``n_valid`` (resamples where the metric could be computed).

    Notes
    -----
    Resamples that contain only one class are skipped because AUROC is
    undefined there. With very few positives this can happen often —
    ``n_valid`` tells you how many resamples actually contributed.
    """
    y_array = np.asarray(y_true)
    p_array = np.asarray(probabilities)

    if y_array.shape[0] != p_array.shape[0]:
        raise ValueError("y_true and probabilities must have the same length.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    rng = np.random.default_rng(random_state)
    n = y_array.shape[0]

    estimate = float(metric_fn(y_array, p_array))

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_sample = y_array[idx]
        if np.unique(y_sample).size < 2:
            continue
        scores.append(float(metric_fn(y_sample, p_array[idx])))

    if len(scores) == 0:
        return {
            "estimate": estimate,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_bootstrap": n_bootstrap,
            "n_valid": 0,
        }

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(scores, [alpha, 1.0 - alpha])

    return {
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_bootstrap": n_bootstrap,
        "n_valid": len(scores),
    }


def bootstrap_metrics_table(
    y_true,
    probabilities,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Apply ``bootstrap_metric_ci`` to several metrics; return one row each.

    Example
    -------
    >>> from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    >>> bootstrap_metrics_table(
    ...     y_test, p_test,
    ...     {"auroc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss},
    ... )
    """
    rows = []
    for name, fn in metrics.items():
        result = bootstrap_metric_ci(
            y_true,
            probabilities,
            fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        result["metric"] = name
        rows.append(result)

    columns = ["metric", "estimate", "ci_low", "ci_high", "n_bootstrap", "n_valid"]
    return pd.DataFrame(rows)[columns]


def format_estimate_with_ci(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    decimals: int = 3,
) -> str:
    """Return ``'0.871 (0.855–0.887)'`` style strings for README tables."""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return f"{estimate:.{decimals}f} (CI n/a)"
    return f"{estimate:.{decimals}f} ({ci_low:.{decimals}f}–{ci_high:.{decimals}f})"


# ---------------------------------------------------------------------------
# Repeated, patient-grouped cross-validation
# ---------------------------------------------------------------------------
def repeated_grouped_cv(
    pipeline,
    X: pd.DataFrame,
    y,
    groups,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Fit ``pipeline`` on repeated StratifiedGroupKFold splits.

    Each repeat reshuffles the patient assignment, so across
    ``n_splits * n_repeats`` fits you see how much the metrics move with
    a different draw of patients — the honest way to compare models.

    Parameters
    ----------
    pipeline:
        An unfitted sklearn ``Pipeline`` (e.g. from ``build_logistic_model``).
        It is cloned before every fit, so the object you pass is not modified.
    X, y:
        Features and binary target for the *training pool* (train + validation
        combined is the usual choice; never include the frozen test set).
    groups:
        Patient identifier per row (``subject_id``). Guarantees no patient is
        split across train and validation folds.
    n_splits, n_repeats:
        5 x 5 = 25 fits is a common default. Use 3 x 2 for a quick check.
    model_name:
        Optional label written into the output, handy when concatenating
        results from several models.

    Returns
    -------
    DataFrame with one row per fold: ``repeat``, ``fold``, ``n_train``,
    ``n_valid``, ``n_positive_valid``, ``auroc``, ``auprc``, ``brier``,
    ``log_loss`` and ``model``.
    """
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)

    if len(X) != y_array.shape[0] or len(X) != groups_array.shape[0]:
        raise ValueError("X, y and groups must have the same number of rows.")

    rows = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat,
        )
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X, y_array, groups_array)
        ):
            model = clone(pipeline)
            model.fit(X.iloc[train_idx], y_array[train_idx])
            probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]

            y_valid = y_array[valid_idx]
            disc = discrimination_metrics(y_valid, probabilities)
            prob = probability_metrics(y_valid, probabilities)

            rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "n_train": int(train_idx.size),
                    "n_valid": int(valid_idx.size),
                    "n_positive_valid": int(y_valid.sum()),
                    "auroc": disc["auroc"],
                    "auprc": disc["auprc"],
                    "brier": prob.get("brier_score", np.nan),
                    "log_loss": prob.get("log_loss", np.nan),
                }
            )

    return pd.DataFrame(rows)


def summarize_cv_results(
    cv_results: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("auroc", "auprc", "brier"),
) -> pd.DataFrame:
    """Collapse fold-level results to mean, SD and 2.5/97.5 percentiles per model."""
    summaries = []
    for model_name, group in cv_results.groupby("model", dropna=False):
        row = {"model": model_name, "n_fits": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_p2.5"] = values.quantile(0.025)
            row[f"{metric}_p97.5"] = values.quantile(0.975)
        summaries.append(row)
    return pd.DataFrame(summaries)
