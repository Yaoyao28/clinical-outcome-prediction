"""Survival endpoint construction, validation, and cohort summaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    dataframe_name: str = "dataframe",
) -> None:
    """Raise ValueError when required columns are missing."""
    required = set(columns)
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def build_survival_dataset(
    cohort: pd.DataFrame,
    *,
    start_col: str = "intime",
    death_time_col: str = "deathtime",
    discharge_time_col: str = "dischtime",
    source_event_col: str = "hospital_expire_flag",
    stay_id_col: str = "stay_id",
    allow_death_time_fallback: bool = True,
) -> pd.DataFrame:
    """
    Construct a time-to-event dataset for in-hospital mortality.

    Definitions
    -----------
    Time origin:
        ICU admission (`start_col`).

    Event:
        In-hospital death, using `source_event_col == 1`.

    Event time:
        `death_time_col`.

    Censoring:
        Hospital discharge alive, using `discharge_time_col`.

    If a death is recorded but `death_time_col` is missing, discharge time
    can optionally be used as a transparent fallback. The returned dataframe
    records this in `death_time_missing` and `end_time_source`.

    Returns
    -------
    pd.DataFrame
        Copy of the cohort with these added columns:
        - event
        - death_time_missing
        - end_time_source
        - end_time
        - duration_days
    """
    _require_columns(
        cohort,
        [
            start_col,
            death_time_col,
            discharge_time_col,
            source_event_col,
            stay_id_col,
        ],
        dataframe_name="cohort",
    )

    df = cohort.copy()

    if df[stay_id_col].duplicated().any():
        raise ValueError(
            f"{stay_id_col!r} must be unique before survival construction."
        )

    source_event = pd.to_numeric(
        df[source_event_col],
        errors="coerce",
    )

    nonmissing_event = source_event.dropna()

    if not nonmissing_event.isin([0, 1]).all():
        invalid = sorted(
            nonmissing_event.loc[
                ~nonmissing_event.isin([0, 1])
            ].unique().tolist()
        )
        raise ValueError(
            f"{source_event_col!r} must be binary 0/1. "
            f"Invalid values: {invalid}"
        )

    df["event"] = source_event.astype("Int64")

    for column in [
        start_col,
        death_time_col,
        discharge_time_col,
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    df["death_time_missing"] = (
        (df["event"] == 1)
        & df[death_time_col].isna()
    )

    if (
        not allow_death_time_fallback
        and bool(df["death_time_missing"].any())
    ):
        raise ValueError(
            "At least one death record has a missing death timestamp."
        )

    if allow_death_time_fallback:
        death_end_time = (
            df[death_time_col]
            .fillna(df[discharge_time_col])
        )
    else:
        death_end_time = df[death_time_col]

    df["end_time"] = df[discharge_time_col].copy()

    death_mask = df["event"] == 1
    df.loc[
        death_mask,
        "end_time",
    ] = death_end_time.loc[
        death_mask
    ]

    df["end_time_source"] = np.where(
        df["event"] == 1,
        np.where(
            df[death_time_col].notna(),
            death_time_col,
            f"{discharge_time_col}_fallback_for_death",
        ),
        f"{discharge_time_col}_alive",
    )

    df["duration_days"] = (
        (
            df["end_time"]
            - df[start_col]
        )
        .dt.total_seconds()
        / 86400.0
    )

    return df


def validate_survival_dataset(
    survival_df: pd.DataFrame,
    *,
    stay_id_col: str = "stay_id",
    duration_col: str = "duration_days",
    event_col: str = "event",
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """
    Validate a survival dataset.

    Valid rows require:
    - unique ICU stay identifier;
    - non-missing positive duration;
    - event indicator equal to 0 or 1.

    If `drop_invalid=True`, invalid duration/event rows are removed.
    Duplicate stay IDs always raise an error.
    """
    _require_columns(
        survival_df,
        [
            stay_id_col,
            duration_col,
            event_col,
        ],
        dataframe_name="survival_df",
    )

    df = survival_df.copy()

    if df[stay_id_col].duplicated().any():
        duplicate_count = int(
            df[stay_id_col].duplicated(
                keep=False
            ).sum()
        )
        raise ValueError(
            f"{stay_id_col!r} is not unique; "
            f"{duplicate_count} rows belong to duplicated IDs."
        )

    numeric_duration = pd.to_numeric(
        df[duration_col],
        errors="coerce",
    )

    numeric_event = pd.to_numeric(
        df[event_col],
        errors="coerce",
    )

    valid_duration = (
        numeric_duration.notna()
        & np.isfinite(numeric_duration)
        & (numeric_duration > 0)
    )

    valid_event = (
        numeric_event.notna()
        & numeric_event.isin([0, 1])
    )

    valid_rows = (
        valid_duration
        & valid_event
    )

    if not valid_rows.all():
        invalid_count = int(
            (~valid_rows).sum()
        )

        if not drop_invalid:
            raise ValueError(
                f"{invalid_count} invalid survival rows found. "
                "Use drop_invalid=True to remove them explicitly."
            )

        df = df.loc[
            valid_rows
        ].copy()

        numeric_duration = (
            numeric_duration.loc[
                valid_rows
            ]
        )
        numeric_event = (
            numeric_event.loc[
                valid_rows
            ]
        )

    df[duration_col] = (
        numeric_duration.astype(float)
    )

    df[event_col] = (
        numeric_event.astype(int)
    )

    return df


def summarize_survival_cohort(
    survival_df: pd.DataFrame,
    *,
    subject_id_col: str = "subject_id",
    duration_col: str = "duration_days",
    event_col: str = "event",
) -> pd.DataFrame:
    """Create a compact survival-cohort summary table."""
    _require_columns(
        survival_df,
        [
            subject_id_col,
            duration_col,
            event_col,
        ],
        dataframe_name="survival_df",
    )

    event = pd.to_numeric(
        survival_df[event_col],
        errors="coerce",
    )

    duration = pd.to_numeric(
        survival_df[duration_col],
        errors="coerce",
    )

    return pd.DataFrame(
        {
            "metric": [
                "ICU stays",
                "Unique patients",
                "Deaths",
                "Discharged alive / censored",
                "Event rate",
                "Median observed duration (days)",
                "Minimum duration (days)",
                "Maximum duration (days)",
            ],
            "value": [
                int(len(survival_df)),
                int(
                    survival_df[
                        subject_id_col
                    ].nunique()
                ),
                int(event.sum()),
                int((event == 0).sum()),
                float(event.mean()),
                float(duration.median()),
                float(duration.min()),
                float(duration.max()),
            ],
        }
    )


def create_age_groups(
    df: pd.DataFrame,
    *,
    age_col: str = "anchor_age",
    output_col: str = "age_group",
    bins: Iterable[float] = (
        -np.inf,
        49,
        69,
        np.inf,
    ),
    labels: Iterable[str] = (
        "18-49",
        "50-69",
        "70+",
    ),
) -> pd.DataFrame:
    """Create categorical age groups on a copy of a dataframe."""
    _require_columns(
        df,
        [age_col],
        dataframe_name="df",
    )

    bins = list(bins)
    labels = list(labels)

    if len(labels) != len(bins) - 1:
        raise ValueError(
            "Number of labels must equal len(bins) - 1."
        )

    result = df.copy()

    result[output_col] = pd.cut(
        pd.to_numeric(
            result[age_col],
            errors="coerce",
        ),
        bins=bins,
        labels=labels,
    )

    return result
