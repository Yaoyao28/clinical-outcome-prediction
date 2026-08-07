"""Logistic Regression model utilities."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def build_logistic_model(
    preprocessor,
    *,
    max_iter: int = 2000,
    class_weight: str | dict | None = "balanced",
    C: float = 1.0,
) -> Pipeline:
    """Create the Logistic Regression pipeline."""
    classifier = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        C=C,
        random_state=RANDOM_STATE,
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


def train_logistic_model(
    preprocessor,
    X_train,
    y_train,
    **model_kwargs,
) -> Pipeline:
    """Build and fit Logistic Regression."""
    model = build_logistic_model(
        preprocessor,
        **model_kwargs,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model
