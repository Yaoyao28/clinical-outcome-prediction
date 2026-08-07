"""XGBoost model utilities."""

from __future__ import annotations

import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "xgboost is not installed. "
        "Install it with `pip install xgboost`."
    ) from error

from src.config import RANDOM_STATE


def calculate_scale_pos_weight(
    y_train,
) -> float:
    """Calculate negatives / positives for imbalanced binary outcomes."""
    y_array = np.asarray(
        y_train
    )

    negative_count = int(
        (y_array == 0).sum()
    )

    positive_count = int(
        (y_array == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Training target contains zero positive cases."
        )

    return (
        negative_count
        / positive_count
    )


def build_xgboost(
    preprocessor,
    *,
    scale_pos_weight: float = 1.0,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    min_child_weight: float = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    n_jobs: int = -1,
) -> Pipeline:
    """Create the XGBoost pipeline."""
    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=(
            colsample_bytree
        ),
        reg_lambda=reg_lambda,
        scale_pos_weight=(
            scale_pos_weight
        ),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def train_xgboost(
    preprocessor,
    X_train,
    y_train,
    *,
    scale_pos_weight: float | None = None,
    **model_kwargs,
) -> Pipeline:
    """Build and fit XGBoost."""
    if scale_pos_weight is None:
        scale_pos_weight = (
            calculate_scale_pos_weight(
                y_train
            )
        )

    model = build_xgboost(
        preprocessor,
        scale_pos_weight=(
            scale_pos_weight
        ),
        **model_kwargs,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model
