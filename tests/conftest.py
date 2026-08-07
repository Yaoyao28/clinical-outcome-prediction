from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


@pytest.fixture
def synthetic_clinical_df():
    """Small mixed-type binary-classification dataset."""
    rng = np.random.default_rng(42)

    n = 80

    age = rng.integers(
        18,
        90,
        size=n,
    )

    creatinine = rng.normal(
        1.1,
        0.4,
        size=n,
    )

    lactate = rng.normal(
        1.8,
        0.8,
        size=n,
    )

    gender = rng.choice(
        ["F", "M"],
        size=n,
    )

    admission_type = rng.choice(
        [
            "EMERGENCY",
            "URGENT",
            "ELECTIVE",
        ],
        size=n,
    )

    # Create a signal so models have something learnable.
    linear_score = (
        -5.0
        + 0.035 * age
        + 0.9 * creatinine
        + 0.7 * lactate
        + 0.4 * (admission_type == "EMERGENCY")
    )

    probability = (
        1
        / (
            1
            + np.exp(
                -linear_score
            )
        )
    )

    outcome = rng.binomial(
        1,
        probability,
    )

    # Guarantee both classes exist.
    outcome[:4] = [
        0,
        1,
        0,
        1,
    ]

    df = pd.DataFrame(
        {
            "subject_id": np.arange(
                1000,
                1000 + n,
            ),
            "hadm_id": np.arange(
                2000,
                2000 + n,
            ),
            "stay_id": np.arange(
                3000,
                3000 + n,
            ),
            "anchor_age": age.astype(float),
            "creatinine_first": creatinine,
            "lactate_first": lactate,
            "gender": gender,
            "admission_type": admission_type,
            "hospital_expire_flag": outcome,
        }
    )

    # Add controlled missingness.
    df.loc[
        [5, 10, 15],
        "creatinine_first",
    ] = np.nan

    df.loc[
        [8, 22],
        "gender",
    ] = None

    return df


@pytest.fixture
def feature_columns():
    return [
        "anchor_age",
        "creatinine_first",
        "lactate_first",
        "gender",
        "admission_type",
    ]


@pytest.fixture
def split_dataframes(
    synthetic_clinical_df,
):
    """Patient-disjoint train/validation/test splits."""
    df = synthetic_clinical_df.copy()

    train_df = (
        df.iloc[:50]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        df.iloc[50:65]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[65:]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_df,
        validation_df,
        test_df,
    )
