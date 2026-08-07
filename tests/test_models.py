import numpy as np

from src.features import (
    build_preprocessor,
    infer_feature_types,
)
from src.models import (
    train_logistic_model,
    train_random_forest,
)


def _prepare_training_objects(
    train_df,
    feature_columns,
):
    numeric_features, categorical_features = (
        infer_feature_types(
            train_df,
            feature_columns,
        )
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        sparse_output=False,
    )

    X_train = train_df[
        feature_columns
    ].copy()

    y_train = train_df[
        "hospital_expire_flag"
    ].copy()

    return (
        preprocessor,
        X_train,
        y_train,
    )


def _assert_probability_vector(
    model,
    X,
):
    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    assert len(
        probabilities
    ) == len(X)

    assert np.isfinite(
        probabilities
    ).all()

    assert (
        (
            probabilities
            >= 0
        )
        & (
            probabilities
            <= 1
        )
    ).all()


def test_logistic_regression_pipeline(
    split_dataframes,
    feature_columns,
):
    train_df, _, test_df = (
        split_dataframes
    )

    (
        preprocessor,
        X_train,
        y_train,
    ) = _prepare_training_objects(
        train_df,
        feature_columns,
    )

    model = train_logistic_model(
        preprocessor,
        X_train,
        y_train,
        max_iter=500,
    )

    _assert_probability_vector(
        model,
        test_df[
            feature_columns
        ],
    )


def test_random_forest_pipeline(
    split_dataframes,
    feature_columns,
):
    train_df, _, test_df = (
        split_dataframes
    )

    (
        preprocessor,
        X_train,
        y_train,
    ) = _prepare_training_objects(
        train_df,
        feature_columns,
    )

    model = train_random_forest(
        preprocessor,
        X_train,
        y_train,
        n_estimators=50,
        max_depth=4,
        min_samples_leaf=2,
        n_jobs=1,
    )

    _assert_probability_vector(
        model,
        test_df[
            feature_columns
        ],
    )


def test_model_pipeline_contains_expected_steps(
    split_dataframes,
    feature_columns,
):
    train_df, _, _ = (
        split_dataframes
    )

    (
        preprocessor,
        X_train,
        y_train,
    ) = _prepare_training_objects(
        train_df,
        feature_columns,
    )

    model = train_logistic_model(
        preprocessor,
        X_train,
        y_train,
        max_iter=500,
    )

    assert (
        "preprocessor"
        in model.named_steps
    )

    assert (
        "classifier"
        in model.named_steps
    )
