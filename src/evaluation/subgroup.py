"""Subgroup performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    classification_metrics,
    discrimination_metrics,
)


def subgroup_analysis(
    dataframe: pd.DataFrame,
    *,
    subgroup_column: str,
    target_column: str,
    probability_column: str,
    threshold: float = 0.50,
    minimum_sample_size: int = 20,
    minimum_events: int = 5,
) -> pd.DataFrame:
    """Compute exploratory model performance by subgroup.

    AUROC is reported as missing if a subgroup contains only one class.
    AUPRC is reported as missing if the subgroup contains zero events.
    """
    required_columns = {
        subgroup_column,
        target_column,
        probability_column,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing subgroup-analysis columns: "
            f"{sorted(missing_columns)}"
        )

    records = []

    for subgroup_value, group in dataframe.groupby(
        subgroup_column,
        dropna=False,
    ):
        y_group = group[
            target_column
        ]

        probabilities = group[
            probability_column
        ].to_numpy()

        metrics = classification_metrics(
            y_group,
            probabilities,
            threshold=threshold,
        )

        event_count = int(
            y_group.sum()
        )

        records.append(
            {
                "subgroup_variable": (
                    subgroup_column
                ),
                "subgroup_value": (
                    subgroup_value
                ),
                "sample_size": int(
                    len(group)
                ),
                "deaths": event_count,
                "event_rate": float(
                    y_group.mean()
                ),
                **metrics,
                "small_sample_warning": (
                    len(group)
                    < minimum_sample_size
                ),
                "few_event_warning": (
                    event_count
                    < minimum_events
                ),
            }
        )

    result = pd.DataFrame(
        records
    )

    if not result.empty:
        result[
            "interpretation_note"
        ] = np.where(
            result[
                "small_sample_warning"
            ]
            | result[
                "few_event_warning"
            ],
            (
                "Exploratory only: "
                "small sample or few events."
            ),
            "Interpret with caution.",
        )

    return result
