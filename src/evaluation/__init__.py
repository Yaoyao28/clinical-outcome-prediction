"""Evaluation utilities for clinical outcome prediction."""

from .metrics import (
    classification_metrics,
    discrimination_metrics,
    probability_metrics,
)
from .plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
)
from .threshold import (
    evaluate_threshold,
    evaluate_thresholds,
    review_capacity_analysis,
)
from .subgroup import (
    subgroup_analysis,
)

__all__ = [
    "classification_metrics",
    "discrimination_metrics",
    "probability_metrics",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_probability_distribution",
    "evaluate_threshold",
    "evaluate_thresholds",
    "review_capacity_analysis",
    "subgroup_analysis",
]
