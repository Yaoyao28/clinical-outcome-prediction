"""Validation utilities for clinical modeling data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config import ID_COLUMNS, TARGET_COLUMN


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Raise an error when required columns are missing."""
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataframe_name} is missing required columns: "
            f"{missing_columns}"
        )


def validate_binary_target(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    dataframe_name: str = "dataframe",
) -> None:
    """Validate that the target is non-missing and binary 0/1."""
    validate_required_columns(
        df,
        [target_column],
        dataframe_name=dataframe_name,
    )

    if df[target_column].isna().any():
        raise ValueError(
            f"{dataframe_name}.{target_column} contains missing values."
        )

    invalid_values = set(
        df.loc[
            ~df[target_column].isin([0, 1]),
            target_column,
        ].unique()
    )

    if invalid_values:
        raise ValueError(
            f"{dataframe_name}.{target_column} contains "
            f"non-binary values: {sorted(invalid_values)}"
        )


def validate_unique_stay(
    df: pd.DataFrame,
    *,
    stay_id_column: str = "stay_id",
    dataframe_name: str = "dataframe",
) -> None:
    """Validate one row per ICU stay."""
    validate_required_columns(
        df,
        [stay_id_column],
        dataframe_name=dataframe_name,
    )

    if df[stay_id_column].isna().any():
        raise ValueError(
            f"{dataframe_name}.{stay_id_column} contains missing values."
        )

    if not df[stay_id_column].is_unique:
        raise ValueError(
            f"{dataframe_name}.{stay_id_column} is not unique."
        )


def validate_cohort(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
    dataframe_name: str = "cohort",
) -> None:
    """Run core cohort integrity checks."""
    id_columns = list(id_columns)

    validate_required_columns(
        df,
        [*id_columns, target_column],
        dataframe_name=dataframe_name,
    )

    for column in id_columns:
        if df[column].isna().any():
            raise ValueError(
                f"{dataframe_name}.{column} contains missing values."
            )

    validate_unique_stay(
        df,
        dataframe_name=dataframe_name,
    )

    validate_binary_target(
        df,
        target_column=target_column,
        dataframe_name=dataframe_name,
    )


def validate_patient_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    patient_id_column: str = "subject_id",
) -> None:
    """Confirm that no patient appears in more than one split."""
    for df, name in [
        (train_df, "train"),
        (validation_df, "validation"),
        (test_df, "test"),
    ]:
        validate_required_columns(
            df,
            [patient_id_column],
            dataframe_name=name,
        )

    train_ids = set(train_df[patient_id_column])
    validation_ids = set(validation_df[patient_id_column])
    test_ids = set(test_df[patient_id_column])

    overlaps = {
        "train-validation": train_ids & validation_ids,
        "train-test": train_ids & test_ids,
        "validation-test": validation_ids & test_ids,
    }

    problems = [
        f"{name}: {len(values)} patients"
        for name, values in overlaps.items()
        if values
    ]

    if problems:
        raise ValueError(
            "Patient leakage detected: "
            + "; ".join(problems)
        )


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Validate that all required modeling features are present."""
    validate_required_columns(
        df,
        list(feature_columns),
        dataframe_name=dataframe_name,
    )


def validate_probability_vector(
    probabilities,
    *,
    vector_name: str = "probabilities",
) -> None:
    """Validate a one-dimensional probability vector in [0, 1]."""
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            f"{vector_name} must be 1D; "
            f"received shape {probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{vector_name} contains NaN or infinite values."
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise ValueError(
            f"{vector_name} contains values outside [0, 1]."
        )


def cohort_summary(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    patient_id_column: str = "subject_id",
    stay_id_column: str = "stay_id",
) -> pd.Series:
    """Return a compact cohort summary."""
    validate_required_columns(
        df,
        [
            patient_id_column,
            stay_id_column,
            target_column,
        ],
        dataframe_name="cohort",
    )

    return pd.Series(
        {
            "rows": len(df),
            "unique_patients": df[
                patient_id_column
            ].nunique(),
            "unique_stays": df[
                stay_id_column
            ].nunique(),
            "deaths": int(
                df[target_column].sum()
            ),
            "survivors": int(
                (df[target_column] == 0).sum()
            ),
            "mortality_rate": float(
                df[target_column].mean()
            ),
        },
        name="value",
    )
