import numpy as np
import pandas as pd

from src.features import (
    build_preprocessor,
    fit_preprocessor,
    get_transformed_feature_names,
    infer_feature_types,
    transform_features,
)


def test_infer_feature_types(
    synthetic_clinical_df,
    feature_columns,
):
    numeric_features, categorical_features = (
        infer_feature_types(
            synthetic_clinical_df,
            feature_columns,
        )
    )

    assert set(
        numeric_features
    ) == {
        "anchor_age",
        "creatinine_first",
        "lactate_first",
    }

    assert set(
        categorical_features
    ) == {
        "gender",
        "admission_type",
    }


def test_preprocessor_fits_and_transforms(
    synthetic_clinical_df,
    feature_columns,
):
    numeric_features, categorical_features = (
        infer_feature_types(
            synthetic_clinical_df,
            feature_columns,
        )
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        add_numeric_missing_indicators=True,
        scale_numeric_features=True,
        sparse_output=False,
    )

    X = synthetic_clinical_df[
        feature_columns
    ].copy()

    fit_preprocessor(
        preprocessor,
        X,
    )

    transformed = (
        transform_features(
            preprocessor,
            X,
            return_dataframe=True,
        )
    )

    assert isinstance(
        transformed,
        pd.DataFrame,
    )

    assert transformed.shape[0] == X.shape[0]

    assert (
        transformed
        .isna()
        .sum()
        .sum()
        == 0
    )

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    assert (
        len(feature_names)
        == transformed.shape[1]
    )


def test_unknown_category_does_not_crash(
    synthetic_clinical_df,
    feature_columns,
):
    train_df = (
        synthetic_clinical_df
        .iloc[:60]
        .copy()
    )

    test_df = (
        synthetic_clinical_df
        .iloc[60:]
        .copy()
    )

    test_df.loc[
        test_df.index[0],
        "admission_type",
    ] = "NEW_CATEGORY"

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

    preprocessor.fit(
        train_df[
            feature_columns
        ]
    )

    transformed = (
        preprocessor.transform(
            test_df[
                feature_columns
            ]
        )
    )

    assert transformed.shape[0] == len(
        test_df
    )
