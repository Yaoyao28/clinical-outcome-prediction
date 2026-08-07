"""Global SHAP and Random Forest feature-importance utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_shap_importance(
    shap_values,
    feature_names,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP contribution."""
    shap_values = np.asarray(
        shap_values,
        dtype=float,
    )

    feature_names = list(
        feature_names
    )

    if shap_values.ndim != 2:
        raise ValueError(
            "shap_values must be a 2D array."
        )

    if (
        shap_values.shape[1]
        != len(feature_names)
    ):
        raise ValueError(
            "Number of SHAP columns does not match feature_names."
        )

    importance = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": (
                importance
            ),
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result[
        "importance_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


def random_forest_builtin_importance(
    rf_classifier,
    feature_names,
) -> pd.DataFrame:
    """Return Random Forest impurity-based feature importance."""
    if not hasattr(
        rf_classifier,
        "feature_importances_",
    ):
        raise TypeError(
            "Classifier does not expose feature_importances_."
        )

    feature_names = list(
        feature_names
    )

    importance = np.asarray(
        rf_classifier
        .feature_importances_,
        dtype=float,
    )

    if len(importance) != len(
        feature_names
    ):
        raise ValueError(
            "Random Forest importance length does not match feature_names."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "random_forest_builtin_importance": (
                importance
            ),
        }
    )

    result[
        "rf_builtin_rank"
    ] = (
        result[
            "random_forest_builtin_importance"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result


def compare_feature_importance(
    shap_importance: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Merge SHAP and Random Forest built-in importance rankings."""
    required_shap = {
        "feature",
        "mean_absolute_shap",
        "importance_rank",
    }

    required_rf = {
        "feature",
        "random_forest_builtin_importance",
        "rf_builtin_rank",
    }

    missing_shap = (
        required_shap
        - set(
            shap_importance.columns
        )
    )

    missing_rf = (
        required_rf
        - set(
            rf_importance.columns
        )
    )

    if missing_shap:
        raise ValueError(
            "SHAP importance table is missing: "
            f"{sorted(missing_shap)}"
        )

    if missing_rf:
        raise ValueError(
            "RF importance table is missing: "
            f"{sorted(missing_rf)}"
        )

    comparison = (
        shap_importance
        .merge(
            rf_importance,
            on="feature",
            how="left",
        )
    )

    comparison[
        "rank_difference"
    ] = (
        comparison[
            "importance_rank"
        ]
        - comparison[
            "rf_builtin_rank"
        ]
    )

    return comparison
