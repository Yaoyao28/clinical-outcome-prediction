"""Threshold and clinical review-capacity analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    classification_metrics,
)


DEFAULT_THRESHOLDS = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
)


def evaluate_threshold(
    y_true,
    probabilities,
    threshold: float,
) -> dict:
    """Evaluate one classification threshold."""
    return classification_metrics(
        y_true,
        probabilities,
        threshold=threshold,
    )


def evaluate_thresholds(
    y_true,
    probabilities,
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate multiple probability thresholds."""
    rows = [
        evaluate_threshold(
            y_true,
            probabilities,
            threshold,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(rows)


def review_capacity_analysis(
    y_true,
    probabilities,
    *,
    capacities: Iterable[float] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ),
) -> pd.DataFrame:
    """Evaluate top-risk review-capacity scenarios.

    Example: if clinicians can review only the highest-risk 10% of
    admissions, what percentage of deaths would be captured?
    """
    y_array = np.asarray(
        y_true
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_array) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have equal length."
        )

    ranked_indices = np.argsort(
        -probabilities
    )

    total_deaths = int(
        (y_array == 1).sum()
    )

    rows = []

    for capacity in capacities:
        if not 0 < capacity <= 1:
            raise ValueError(
                "Review capacities must be in (0, 1]."
            )

        review_count = max(
            1,
            int(
                np.ceil(
                    capacity
                    * len(y_array)
                )
            ),
        )

        reviewed_indices = (
            ranked_indices[
                :review_count
            ]
        )

        deaths_captured = int(
            y_array[
                reviewed_indices
            ].sum()
        )

        death_capture_rate = (
            deaths_captured
            / total_deaths
            if total_deaths > 0
            else np.nan
        )

        precision_reviewed = (
            deaths_captured
            / review_count
        )

        rows.append(
            {
                "review_capacity_percentage": (
                    float(capacity)
                ),
                "patients_reviewed": int(
                    review_count
                ),
                "deaths_captured": (
                    deaths_captured
                ),
                "total_deaths": (
                    total_deaths
                ),
                "death_capture_rate": (
                    death_capture_rate
                ),
                "precision_among_reviewed": (
                    precision_reviewed
                ),
                "false_alerts_among_reviewed": (
                    review_count
                    - deaths_captured
                ),
            }
        )

    return pd.DataFrame(rows)
