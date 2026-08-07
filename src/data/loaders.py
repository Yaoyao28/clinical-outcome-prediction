"""Data-loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    DATE_COLUMNS,
    FEATURE_CONFIG_PATH,
    PROCESSED_DATA_DIR,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
)


def _convert_existing_datetime_columns(
    df: pd.DataFrame,
    date_columns: Iterable[str] = DATE_COLUMNS,
) -> pd.DataFrame:
    """Convert configured date columns only when they exist."""
    df = df.copy()

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def load_csv(
    path: str | Path,
    *,
    convert_dates: bool = True,
    date_columns: Iterable[str] = DATE_COLUMNS,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV file and optionally convert known datetime columns."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file was not found: {path.resolve()}"
        )

    df = pd.read_csv(path, **read_csv_kwargs)

    if convert_dates:
        df = _convert_existing_datetime_columns(
            df,
            date_columns=date_columns,
        )

    return df


def load_modeling_split(
    split_name: str,
    *,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load train, validation, or test modeling data."""
    split_name = split_name.strip().lower()

    filename_map = {
        "train": TRAIN_FILENAME,
        "validation": VALIDATION_FILENAME,
        "valid": VALIDATION_FILENAME,
        "val": VALIDATION_FILENAME,
        "test": TEST_FILENAME,
    }

    if split_name not in filename_map:
        raise ValueError(
            "split_name must be train, validation/valid/val, or test."
        )

    return load_csv(
        Path(processed_dir)
        / filename_map[split_name]
    )


def load_train_data(
    *,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load the training dataset."""
    return load_modeling_split(
        "train",
        processed_dir=processed_dir,
    )


def load_validation_data(
    *,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load the validation dataset."""
    return load_modeling_split(
        "validation",
        processed_dir=processed_dir,
    )


def load_test_data(
    *,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Load the test dataset."""
    return load_modeling_split(
        "test",
        processed_dir=processed_dir,
    )


def load_modeling_splits(
    *,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    return (
        load_train_data(processed_dir=processed_dir),
        load_validation_data(processed_dir=processed_dir),
        load_test_data(processed_dir=processed_dir),
    )


def load_feature_config(
    path: str | Path = FEATURE_CONFIG_PATH,
) -> dict:
    """Load feature_config.json and validate key fields."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature config was not found: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = {
        "target_column",
        "retained_feature_columns",
    }

    missing_keys = required_keys - set(config)

    if missing_keys:
        raise ValueError(
            "Feature config is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return config
