"""Tree SHAP utilities with binary-class compatibility handling."""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError as error:
    raise ImportError(
        "shap is not installed. "
        "Install it with `pip install shap`."
    ) from error


def extract_positive_class_shap_values(
    shap_output,
    *,
    n_rows: int,
    n_features: int,
) -> np.ndarray:
    """Standardize binary SHAP output to rows x features."""
    if isinstance(
        shap_output,
        list,
    ):
        values = np.asarray(
            shap_output[
                1
                if len(shap_output) > 1
                else 0
            ]
        )

    else:
        array = np.asarray(
            shap_output
        )

        if array.ndim == 2:
            values = array

        elif (
            array.ndim == 3
            and array.shape
            == (
                n_rows,
                n_features,
                2,
            )
        ):
            values = array[
                :,
                :,
                1,
            ]

        elif (
            array.ndim == 3
            and array.shape
            == (
                2,
                n_rows,
                n_features,
            )
        ):
            values = array[
                1
            ]

        elif (
            array.ndim == 3
            and array.shape
            == (
                n_rows,
                2,
                n_features,
            )
        ):
            values = array[
                :,
                1,
                :,
            ]

        else:
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{array.shape}"
            )

    values = np.asarray(
        values,
        dtype=float,
    )

    expected_shape = (
        n_rows,
        n_features,
    )

    if values.shape != expected_shape:
        raise ValueError(
            f"SHAP values have shape {values.shape}; "
            f"expected {expected_shape}."
        )

    return values


def positive_class_base_value(
    explainer,
) -> float:
    """Return the Tree SHAP expected value for class 1."""
    expected = np.asarray(
        explainer.expected_value
    ).ravel()

    if expected.size == 0:
        raise ValueError(
            "SHAP explainer has no expected value."
        )

    return float(
        expected[
            1
            if expected.size > 1
            else 0
        ]
    )


def compute_tree_shap(
    classifier,
    X_transformed,
):
    """Compute positive-class Tree SHAP values.

    Returns
    -------
    explainer, shap_values, base_value
    """
    explainer = shap.TreeExplainer(
        classifier
    )

    raw_output = (
        explainer.shap_values(
            X_transformed
        )
    )

    values = (
        extract_positive_class_shap_values(
            raw_output,
            n_rows=(
                X_transformed.shape[0]
            ),
            n_features=(
                X_transformed.shape[1]
            ),
        )
    )

    base_value = (
        positive_class_base_value(
            explainer
        )
    )

    return (
        explainer,
        values,
        base_value,
    )


def create_local_explanation(
    shap_values,
    feature_values,
    feature_names,
    *,
    row_position: int,
    base_value: float,
):
    """Create a SHAP Explanation object for one patient."""
    shap_values = np.asarray(
        shap_values
    )

    feature_values = np.asarray(
        feature_values
    )

    return shap.Explanation(
        values=shap_values[
            row_position
        ],
        base_values=base_value,
        data=feature_values[
            row_position
        ],
        feature_names=list(
            feature_names
        ),
    )
