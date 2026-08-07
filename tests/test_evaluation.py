import numpy as np
import pandas as pd

from src.evaluation import (
    classification_metrics,
    discrimination_metrics,
    evaluate_threshold,
    evaluate_thresholds,
    probability_metrics,
    review_capacity_analysis,
    subgroup_analysis,
)


def test_classification_metrics_known_example():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.60,
            0.70,
            0.20,
        ]
    )

    metrics = classification_metrics(
        y_true,
        probabilities,
        threshold=0.50,
    )

    assert metrics[
        "true_negatives"
    ] == 1

    assert metrics[
        "false_positives"
    ] == 1

    assert metrics[
        "false_negatives"
    ] == 1

    assert metrics[
        "true_positives"
    ] == 1

    assert metrics[
        "recall_sensitivity"
    ] == 0.5

    assert metrics[
        "specificity"
    ] == 0.5


def test_discrimination_metrics_range():
    y_true = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
        ]
    )

    metrics = discrimination_metrics(
        y_true,
        probabilities,
    )

    assert (
        0 <= metrics["auroc"] <= 1
    )

    assert (
        0 <= metrics["auprc"] <= 1
    )


def test_probability_metrics():
    y_true = np.array(
        [
            0,
            1,
            0,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.8,
            0.2,
            0.9,
        ]
    )

    metrics = probability_metrics(
        y_true,
        probabilities,
    )

    assert metrics[
        "brier_score"
    ] >= 0

    assert metrics[
        "log_loss"
    ] >= 0


def test_evaluate_thresholds_returns_all_rows():
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.3,
            0.6,
            0.8,
        ]
    )

    thresholds = [
        0.2,
        0.5,
        0.7,
    ]

    result = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds=thresholds,
    )

    assert len(result) == len(
        thresholds
    )

    assert list(
        result["threshold"]
    ) == thresholds


def test_review_capacity_analysis():
    y_true = np.array(
        [
            1,
            0,
            0,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.9,
            0.8,
            0.2,
            0.7,
            0.1,
        ]
    )

    result = review_capacity_analysis(
        y_true,
        probabilities,
        capacities=[
            0.2,
            0.4,
        ],
    )

    assert len(result) == 2

    assert (
        result[
            "patients_reviewed"
        ]
        >= 1
    ).all()

    assert (
        (
            result[
                "death_capture_rate"
            ]
            >= 0
        )
        & (
            result[
                "death_capture_rate"
            ]
            <= 1
        )
    ).all()


def test_subgroup_analysis():
    df = pd.DataFrame(
        {
            "gender": [
                "F",
                "F",
                "M",
                "M",
                "M",
                "F",
            ],
            "outcome": [
                0,
                1,
                0,
                1,
                0,
                0,
            ],
            "probability": [
                0.1,
                0.8,
                0.2,
                0.7,
                0.3,
                0.4,
            ],
        }
    )

    result = subgroup_analysis(
        df,
        subgroup_column="gender",
        target_column="outcome",
        probability_column="probability",
        threshold=0.50,
        minimum_sample_size=1,
        minimum_events=1,
    )

    assert set(
        result[
            "subgroup_value"
        ]
    ) == {
        "F",
        "M",
    }

    assert (
        "sample_size"
        in result.columns
    )

    assert (
        "recall_sensitivity"
        in result.columns
    )
