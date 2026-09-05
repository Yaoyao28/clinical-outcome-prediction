"""Cox proportional hazards dataset preparation, fitting, and diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter


def choose_parsimonious_covariates(
    df: pd.DataFrame,
    candidate_covariates: Iterable[str],
    *,
    event_col: str = "event",
    events_per_covariate: int = 5,
    minimum_covariates: int = 1,
) -> list[str]:
    """
    Select a conservative number of available Cox covariates.

    This helper is intended for small exploratory cohorts such as the
    MIMIC-IV demo dataset. It is not a universal model-selection rule.
    """
    if event_col not in df.columns:
        raise ValueError(
            f"{event_col!r} not found."
        )

    if events_per_covariate <= 0:
        raise ValueError(
            "events_per_covariate must be positive."
        )

    available = [
        column
        for column in candidate_covariates
        if column in df.columns
        and df[column].notna().sum() > 0
    ]

    if not available:
        return []

    n_events = int(
        pd.to_numeric(
            df[event_col],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    max_covariates = max(
        minimum_covariates,
        n_events // events_per_covariate,
    )

    max_covariates = min(
        max_covariates,
        len(available),
    )

    return available[
        :max_covariates
    ]


def prepare_cox_dataset(
    df: pd.DataFrame,
    covariates: Iterable[str],
    *,
    duration_col: str = "duration_days",
    event_col: str = "event",
    median_impute: bool = True,
    drop_zero_variance: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare a numeric dataframe for Cox proportional hazards regression.

    All requested covariates are converted to numeric. Missing values are
    median-imputed by default. Zero-variance covariates are removed.

    Returns
    -------
    cox_df, retained_covariates
    """
    covariates = list(covariates)

    if not covariates:
        raise ValueError(
            "At least one Cox covariate is required."
        )

    required = {
        duration_col,
        event_col,
        *covariates,
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing Cox columns: {missing}"
        )

    cox_df = df[
        [
            duration_col,
            event_col,
            *covariates,
        ]
    ].copy()

    cox_df[duration_col] = pd.to_numeric(
        cox_df[duration_col],
        errors="coerce",
    )

    cox_df[event_col] = pd.to_numeric(
        cox_df[event_col],
        errors="coerce",
    )

    valid_core = (
        cox_df[duration_col].notna()
        & np.isfinite(
            cox_df[duration_col]
        )
        & (
            cox_df[duration_col] > 0
        )
        & cox_df[event_col].isin(
            [0, 1]
        )
    )

    cox_df = cox_df.loc[
        valid_core
    ].copy()

    if cox_df.empty:
        raise ValueError(
            "No valid Cox rows remain after duration/event validation."
        )

    retained = []

    for column in covariates:
        cox_df[column] = pd.to_numeric(
            cox_df[column],
            errors="coerce",
        )

        if median_impute:
            median_value = (
                cox_df[column]
                .median()
            )

            if pd.isna(median_value):
                continue

            cox_df[column] = (
                cox_df[column]
                .fillna(
                    median_value
                )
            )

        retained.append(
            column
        )

    if not median_impute:
        cox_df = cox_df.dropna(
            subset=retained
        )

    if drop_zero_variance:
        zero_variance = [
            column
            for column in retained
            if (
                cox_df[column]
                .nunique(
                    dropna=True
                )
                <= 1
            )
        ]

        if zero_variance:
            cox_df = cox_df.drop(
                columns=zero_variance
            )

            retained = [
                column
                for column in retained
                if column not in zero_variance
            ]

    if not retained:
        raise ValueError(
            "No usable Cox covariates remain after preprocessing."
        )

    if cox_df[
        [
            duration_col,
            event_col,
            *retained,
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing values remain in the prepared Cox dataset."
        )

    cox_df[event_col] = (
        cox_df[event_col]
        .astype(int)
    )

    return (
        cox_df[
            [
                duration_col,
                event_col,
                *retained,
            ]
        ].copy(),
        retained,
    )


def fit_cox_model(
    cox_df: pd.DataFrame,
    *,
    duration_col: str = "duration_days",
    event_col: str = "event",
    penalizer_fallback: float = 0.1,
) -> tuple[CoxPHFitter, dict]:
    """
    Fit Cox PH regression with an optional penalized fallback.

    Returns
    -------
    model, metadata

    metadata contains:
    - penalizer_fallback_used
    - unpenalized_error
    - n_rows
    - n_events
    """
    if duration_col not in cox_df.columns:
        raise ValueError(
            f"{duration_col!r} not found."
        )

    if event_col not in cox_df.columns:
        raise ValueError(
            f"{event_col!r} not found."
        )

    if len(
        [
            c
            for c in cox_df.columns
            if c not in {
                duration_col,
                event_col,
            }
        ]
    ) == 0:
        raise ValueError(
            "Cox dataset must contain at least one covariate."
        )

    metadata = {
        "penalizer_fallback_used": False,
        "unpenalized_error": None,
        "n_rows": int(
            len(cox_df)
        ),
        "n_events": int(
            cox_df[event_col].sum()
        ),
    }

    try:
        model = CoxPHFitter()

        model.fit(
            cox_df,
            duration_col=duration_col,
            event_col=event_col,
        )

    except Exception as exc:
        if penalizer_fallback is None:
            raise

        metadata[
            "penalizer_fallback_used"
        ] = True

        metadata[
            "unpenalized_error"
        ] = repr(exc)

        model = CoxPHFitter(
            penalizer=penalizer_fallback
        )

        model.fit(
            cox_df,
            duration_col=duration_col,
            event_col=event_col,
        )

    return model, metadata


def extract_hazard_ratios(
    model: CoxPHFitter,
) -> pd.DataFrame:
    """Return a compact Cox coefficient / hazard-ratio table."""
    summary = (
        model.summary
        .reset_index()
        .copy()
    )

    rename_map = {
        "exp(coef)": "hazard_ratio",
        "exp(coef) lower 95%":
            "hr_lower_95",
        "exp(coef) upper 95%":
            "hr_upper_95",
        "p": "p_value",
    }

    summary = summary.rename(
        columns=rename_map
    )

    wanted = [
        "covariate",
        "coef",
        "hazard_ratio",
        "hr_lower_95",
        "hr_upper_95",
        "p_value",
    ]

    available = [
        column
        for column in wanted
        if column in summary.columns
    ]

    return summary[
        available
    ].copy()


def plot_hazard_ratios(
    hazard_ratio_table: pd.DataFrame,
    *,
    covariate_col: str = "covariate",
    hr_col: str = "hazard_ratio",
    lower_col: str = "hr_lower_95",
    upper_col: str = "hr_upper_95",
    title: str = "Cox Proportional Hazards Model",
    save_path: Optional[str | Path] = None,
):
    """Create a hazard-ratio forest plot on a logarithmic x-axis."""
    required = {
        covariate_col,
        hr_col,
        lower_col,
        upper_col,
    }

    missing = sorted(
        required.difference(
            hazard_ratio_table.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing forest-plot columns: {missing}"
        )

    plot_df = (
        hazard_ratio_table
        .dropna(
            subset=[
                hr_col,
                lower_col,
                upper_col,
            ]
        )
        .copy()
    )

    finite_mask = np.isfinite(
        plot_df[
            [
                hr_col,
                lower_col,
                upper_col,
            ]
        ]
    ).all(axis=1)

    plot_df = plot_df.loc[
        finite_mask
    ].copy()

    positive_mask = (
        (plot_df[hr_col] > 0)
        & (plot_df[lower_col] > 0)
        & (plot_df[upper_col] > 0)
    )

    plot_df = plot_df.loc[
        positive_mask
    ].copy()

    if plot_df.empty:
        raise ValueError(
            "No finite positive hazard-ratio estimates are available."
        )

    plot_df = (
        plot_df
        .sort_values(
            hr_col
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(plot_df)
    )

    lower_error = (
        plot_df[hr_col]
        - plot_df[lower_col]
    )

    upper_error = (
        plot_df[upper_col]
        - plot_df[hr_col]
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                4,
                len(plot_df) * 0.8,
            ),
        )
    )

    ax.errorbar(
        plot_df[hr_col],
        y,
        xerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        1.0,
        linestyle="--",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_df[
            covariate_col
        ]
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Hazard ratio (log scale)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


def check_proportional_hazards(
    model: CoxPHFitter,
    cox_df: pd.DataFrame,
    *,
    p_value_threshold: float = 0.05,
    show_plots: bool = True,
) -> dict:
    """
    Run lifelines proportional-hazards diagnostics.

    Returns a metadata dictionary rather than hiding diagnostic failures.
    """
    result = {
        "status": "not_run",
        "error": None,
    }

    try:
        model.check_assumptions(
            cox_df,
            p_value_threshold=(
                p_value_threshold
            ),
            show_plots=show_plots,
        )

        result[
            "status"
        ] = (
            "completed_without_exception"
        )

    except Exception as exc:
        result[
            "status"
        ] = (
            "diagnostic_failed"
        )

        result[
            "error"
        ] = repr(exc)

    return result
