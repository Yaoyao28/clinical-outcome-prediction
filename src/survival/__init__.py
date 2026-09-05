"""Reusable survival-analysis utilities for the clinical outcome project."""

from .data import (
    build_survival_dataset,
    validate_survival_dataset,
    summarize_survival_cohort,
    create_age_groups,
)

from .kaplan_meier import (
    fit_kaplan_meier,
    survival_probabilities_at_times,
    summarize_survival_by_group,
    plot_kaplan_meier,
    plot_kaplan_meier_by_group,
    run_logrank_test,
)

from .cox_model import (
    choose_parsimonious_covariates,
    prepare_cox_dataset,
    fit_cox_model,
    extract_hazard_ratios,
    plot_hazard_ratios,
    check_proportional_hazards,
)

__all__ = [
    "build_survival_dataset",
    "validate_survival_dataset",
    "summarize_survival_cohort",
    "create_age_groups",
    "fit_kaplan_meier",
    "survival_probabilities_at_times",
    "summarize_survival_by_group",
    "plot_kaplan_meier",
    "plot_kaplan_meier_by_group",
    "run_logrank_test",
    "choose_parsimonious_covariates",
    "prepare_cox_dataset",
    "fit_cox_model",
    "extract_hazard_ratios",
    "plot_hazard_ratios",
    "check_proportional_hazards",
]
