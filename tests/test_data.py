import numpy as np
import pandas as pd
import pytest

from src.data import (
    cohort_summary,
    validate_binary_target,
    validate_cohort,
    validate_feature_columns,
    validate_patient_split,
    validate_probability_vector,
    validate_required_columns,
    validate_unique_stay,
)


def test_validate_cohort_passes(
    synthetic_clinical_df,
):
    validate_cohort(
        synthetic_clinical_df,
        target_column="hospital_expire_flag",
        dataframe_name="synthetic",
    )


def test_unique_stay_failure(
    synthetic_clinical_df,
):
    df = synthetic_clinical_df.copy()

    df.loc[
        1,
        "stay_id",
    ] = df.loc[
        0,
        "stay_id",
    ]

    with pytest.raises(
        ValueError
    ):
        validate_unique_stay(
            df,
            dataframe_name="duplicate_stay",
        )


def test_binary_target_failure(
    synthetic_clinical_df,
):
    df = synthetic_clinical_df.copy()

    df.loc[
        0,
        "hospital_expire_flag",
    ] = 3

    with pytest.raises(
        ValueError
    ):
        validate_binary_target(
            df,
            target_column="hospital_expire_flag",
        )


def test_patient_splits_do_not_overlap(
    split_dataframes,
):
    train_df, validation_df, test_df = (
        split_dataframes
    )

    validate_patient_split(
        train_df,
        validation_df,
        test_df,
    )


def test_patient_overlap_is_detected(
    split_dataframes,
):
    train_df, validation_df, test_df = (
        split_dataframes
    )

    validation_df = (
        validation_df.copy()
    )

    validation_df.loc[
        0,
        "subject_id",
    ] = train_df.loc[
        0,
        "subject_id",
    ]

    with pytest.raises(
        ValueError
    ):
        validate_patient_split(
            train_df,
            validation_df,
            test_df,
        )


def test_probability_vector_passes():
    probabilities = np.array(
        [
            0.01,
            0.20,
            0.50,
            0.90,
        ]
    )

    validate_probability_vector(
        probabilities
    )


def test_probability_vector_rejects_out_of_range():
    probabilities = np.array(
        [
            0.2,
            1.2,
        ]
    )

    with pytest.raises(
        ValueError
    ):
        validate_probability_vector(
            probabilities
        )


def test_feature_columns_validation(
    synthetic_clinical_df,
    feature_columns,
):
    validate_feature_columns(
        synthetic_clinical_df,
        feature_columns,
    )


def test_cohort_summary(
    synthetic_clinical_df,
):
    summary = cohort_summary(
        synthetic_clinical_df,
    )

    assert (
        summary["rows"]
        == len(
            synthetic_clinical_df
        )
    )

    assert (
        summary[
            "unique_stays"
        ]
        == synthetic_clinical_df[
            "stay_id"
        ].nunique()
    )
