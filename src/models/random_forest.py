"""Random Forest model utilities."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE


def build_random_forest(
    preprocessor,
    *,
    n_estimators: int = 500,
    max_depth: int | None = 8,
    min_samples_leaf: int = 5,
    max_features="sqrt",
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
) -> Pipeline:
    """Create the Random Forest pipeline."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
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


def train_random_forest(
    preprocessor,
    X_train,
    y_train,
    **model_kwargs,
) -> Pipeline:
    """Build and fit Random Forest."""
    model = build_random_forest(
        preprocessor,
        **model_kwargs,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model
