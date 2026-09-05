"""Kaplan-Meier estimation, subgroup summaries, plots, and log-rank testing."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = sorted(
        set(columns).difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def fit_kaplan_meier(
    df: pd.DataFrame,
    *,
    duration_col: str = "duration_days",
    event_col: str = "event",
    label: str = "Overall cohort",
) -> KaplanMeierFitter:
    """Fit and return a Kaplan-Meier estimator."""
    _require_columns(
        df,
        [
            duration_col,
            event_col,
        ],
    )

    if df.empty:
        raise ValueError(
            "Cannot fit Kaplan-Meier on an empty dataframe."
        )

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=df[duration_col],
        event_observed=df[event_col],
        label=label,
    )

    return kmf


def survival_probabilities_at_times(
    kmf: KaplanMeierFitter,
    times: Iterable[float],
    *,
    time_column_name: str = "day",
) -> pd.DataFrame:
    """Evaluate the fitted survival function at selected times."""
    times = list(times)

    if not times:
        raise ValueError(
            "At least one time point is required."
        )

    return pd.DataFrame(
        {
            time_column_name: times,
            "estimated_survival_probability": [
                float(
                    kmf.predict(
                        time
                    )
                )
                for time in times
            ],
        }
    )


def summarize_survival_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    duration_col: str = "duration_days",
    event_col: str = "event",
    id_col: str = "stay_id",
) -> pd.DataFrame:
    """
    Summarize sample size, events, event rate, and median duration by group.
    """
    _require_columns(
        df,
        [
            group_col,
            duration_col,
            event_col,
            id_col,
        ],
    )

    return (
        df
        .groupby(
            group_col,
            observed=True,
            dropna=False,
        )
        .agg(
            patients=(
                id_col,
                "count",
            ),
            deaths=(
                event_col,
                "sum",
            ),
            event_rate=(
                event_col,
                "mean",
            ),
            median_duration_days=(
                duration_col,
                "median",
            ),
        )
        .reset_index()
    )


def plot_kaplan_meier(
    kmf: KaplanMeierFitter,
    *,
    title: str = "Kaplan–Meier Survival Curve",
    xlabel: str = "Days since ICU admission",
    ylabel: str = "Estimated survival probability",
    show_censors: bool = True,
    ci_show: bool = True,
    figsize: tuple[float, float] = (8, 6),
    save_path: Optional[str | Path] = None,
):
    """Plot one fitted Kaplan-Meier survival curve."""
    fig, ax = plt.subplots(
        figsize=figsize
    )

    kmf.plot_survival_function(
        ax=ax,
        show_censors=show_censors,
        ci_show=ci_show,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

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


def plot_kaplan_meier_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    duration_col: str = "duration_days",
    event_col: str = "event",
    title: Optional[str] = None,
    xlabel: str = "Days since ICU admission",
    ylabel: str = "Estimated survival probability",
    ci_show: bool = False,
    show_censors: bool = True,
    figsize: tuple[float, float] = (9, 7),
    save_path: Optional[str | Path] = None,
):
    """Plot one Kaplan-Meier curve for each non-missing group."""
    _require_columns(
        df,
        [
            group_col,
            duration_col,
            event_col,
        ],
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    plotted = 0

    for value in (
        df[group_col]
        .dropna()
        .unique()
    ):
        group_df = df.loc[
            df[group_col] == value
        ].copy()

        if group_df.empty:
            continue

        kmf = KaplanMeierFitter()

        kmf.fit(
            durations=group_df[
                duration_col
            ],
            event_observed=group_df[
                event_col
            ],
            label=(
                f"{value} "
                f"(n={len(group_df)}, "
                f"events={int(group_df[event_col].sum())})"
            ),
        )

        kmf.plot_survival_function(
            ax=ax,
            ci_show=ci_show,
            show_censors=show_censors,
        )

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError(
            f"No usable groups were found in {group_col!r}."
        )

    ax.set_title(
        title
        or f"Kaplan–Meier Survival by {group_col}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

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


def run_logrank_test(
    df: pd.DataFrame,
    *,
    group_col: str,
    group_a,
    group_b,
    duration_col: str = "duration_days",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Run a two-group log-rank test.

    When one group has zero events, the function returns a table containing
    NaN test statistics plus an explanatory note instead of failing silently.
    """
    _require_columns(
        df,
        [
            group_col,
            duration_col,
            event_col,
        ],
    )

    a = df.loc[
        df[group_col] == group_a
    ].copy()

    b = df.loc[
        df[group_col] == group_b
    ].copy()

    if a.empty or b.empty:
        raise ValueError(
            "Both comparison groups must contain at least one observation."
        )

    events_a = int(
        a[event_col].sum()
    )
    events_b = int(
        b[event_col].sum()
    )

    common = {
        "comparison": [
            f"{group_a} vs {group_b}"
        ],
        "n_group_a": [
            len(a)
        ],
        "events_group_a": [
            events_a
        ],
        "n_group_b": [
            len(b)
        ],
        "events_group_b": [
            events_b
        ],
    }

    if events_a == 0 or events_b == 0:
        return pd.DataFrame(
            {
                **common,
                "test_statistic": [
                    np.nan
                ],
                "p_value": [
                    np.nan
                ],
                "note": [
                    "Insufficient events in at least one group."
                ],
            }
        )

    result = logrank_test(
        a[duration_col],
        b[duration_col],
        event_observed_A=a[
            event_col
        ],
        event_observed_B=b[
            event_col
        ],
    )

    return pd.DataFrame(
        {
            **common,
            "test_statistic": [
                float(
                    result.test_statistic
                )
            ],
            "p_value": [
                float(
                    result.p_value
                )
            ],
        }
    )
