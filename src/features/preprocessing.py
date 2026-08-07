"""Reusable preprocessing utilities for tabular clinical data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    feature_columns:
        Columns intended for modeling.

    Returns
    -------
    (numeric_features, categorical_features)
    """
    feature_columns = list(feature_columns)

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    return (
        numeric_features,
        categorical_features,
    )


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    *,
    add_numeric_missing_indicators: bool = True,
    scale_numeric_features: bool = True,
    categorical_imputation_strategy: str = "most_frequent",
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Build the project's sklearn preprocessing pipeline.

    Numeric features:
    - median imputation
    - optional missingness indicators
    - optional standardization

    Categorical features:
    - most-frequent imputation
    - one-hot encoding
    - unknown categories ignored
    """
    numeric_features = list(
        numeric_features
    )

    categorical_features = list(
        categorical_features
    )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=(
                    add_numeric_missing_indicators
                ),
            ),
        )
    ]

    if scale_numeric_features:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=(
                        categorical_imputation_strategy
                    )
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit preprocessing using training data only."""
    preprocessor.fit(
        X_train
    )

    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    *,
    return_dataframe: bool = True,
) -> pd.DataFrame | np.ndarray:
    """Transform features with an already fitted preprocessor."""
    transformed = preprocessor.transform(
        X
    )

    if not return_dataframe:
        return transformed

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=X.index,
    )


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Return names after imputation, scaling, and one-hot encoding."""
    if not hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        raise TypeError(
            "The provided preprocessor does not support "
            "get_feature_names_out()."
        )

    return (
        preprocessor
        .get_feature_names_out()
    )
