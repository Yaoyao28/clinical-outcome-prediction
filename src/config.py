"""Central configuration for the clinical outcome prediction project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

SQL_DIR = PROJECT_DIR / "sql"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
TESTS_DIR = PROJECT_DIR / "tests"

RANDOM_STATE = 42
TARGET_COLUMN = "hospital_expire_flag"

ID_COLUMNS = ["subject_id", "hadm_id", "stay_id"]
DATE_COLUMNS = ["intime", "prediction_time", "outtime"]

TRAIN_FILENAME = "train_modeling_raw.csv"
VALIDATION_FILENAME = "validation_modeling_raw.csv"
TEST_FILENAME = "test_modeling_raw.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / TRAIN_FILENAME
VALIDATION_PATH = PROCESSED_DATA_DIR / VALIDATION_FILENAME
TEST_PATH = PROCESSED_DATA_DIR / TEST_FILENAME

FEATURE_CONFIG_PATH = MODELS_DIR / "feature_config.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
SELECTED_MODEL_PATH = MODELS_DIR / "selected_model.joblib"
SELECTED_MODEL_CONFIG_PATH = MODELS_DIR / "selected_model_config.json"


def create_project_output_directories() -> None:
    """Create standard output directories if missing."""
    for directory in [
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        PREDICTIONS_DIR,
        EXPLANATIONS_DIR,
        ANALYSIS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
