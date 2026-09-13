"""Fit the final model on the training pool, log it to MLflow, and save API artifacts.

Run from the project root, after `src/data/extract_bigquery.py` and notebook 07b
(which writes the frozen test-subject IDs):

    python -m src.models.train_final

Outputs
-------
models/final_xgboost_pipeline.joblib   fitted preprocessor + XGBoost (git-ignored)
models/final_feature_config.json       feature lists + version metadata (tracked)
mlruns/                                MLflow run: params, metrics, model (git-ignored)

The JSON is what `app/` uses to build its input schema, so the API and the model
can never disagree about which columns exist. The MLflow run additionally gives
the model a registry version the API can load by name (`MLFLOW_MODEL_URI`).

Pass --no-mlflow to skip tracking (useful if mlflow is not installed).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import MODELS_DIR, PROJECT_DIR, TARGET_COLUMN
from src.features.preprocessing import build_preprocessor, infer_feature_types
from src.models.xgboost_model import build_xgboost

COHORT_PATH = PROJECT_DIR / "data/raw/full/modeling_cohort_full.csv"
TEST_IDS_PATH = PROJECT_DIR / "data/processed/full/test_subject_ids.csv"
PIPELINE_PATH = MODELS_DIR / "final_xgboost_pipeline.joblib"
CONFIG_PATH = MODELS_DIR / "final_feature_config.json"

ID_AND_LEAKAGE = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "prediction_time",
    TARGET_COLUMN,
]


def fit_final_model(df: pd.DataFrame, test_subjects: pd.Series):
    """Fit the unweighted XGBoost pipeline on the pool; return it plus metadata."""
    is_test = df["subject_id"].isin(test_subjects)
    pool, test = df[~is_test], df[is_test]

    feature_columns = [c for c in pool.columns if c not in ID_AND_LEAKAGE]
    numeric_features, categorical_features = infer_feature_types(pool, feature_columns)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = build_xgboost(preprocessor)  # scale_pos_weight defaults to 1.0 (unweighted)
    pipeline.fit(pool[feature_columns], pool[TARGET_COLUMN])

    probs = pipeline.predict_proba(test[feature_columns])[:, 1]
    metrics = {
        "test_auroc": float(roc_auc_score(test[TARGET_COLUMN], probs)),
        "test_auprc": float(average_precision_score(test[TARGET_COLUMN], probs)),
        "test_brier": float(brier_score_loss(test[TARGET_COLUMN], probs)),
    }

    config = {
        "model_name": "xgboost_unweighted",
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_on": "MIMIC-IV v3.1, first ICU stay, adults",
        "n_train_rows": int(len(pool)),
        "n_train_positives": int(pool[TARGET_COLUMN].sum()),
        "n_test_rows": int(len(test)),
        "n_test_positives": int(test[TARGET_COLUMN].sum()),
        "test_auroc": round(metrics["test_auroc"], 4),
        "test_auprc": round(metrics["test_auprc"], 4),
        "test_brier": round(metrics["test_brier"], 4),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COLUMN,
        "prediction_window_hours": 24,
    }
    return pipeline, config, metrics, pool[feature_columns].head(5)


def _log_to_mlflow(pipeline, config: dict, metrics: dict, input_example) -> str | None:
    """Log params, metrics and the fitted pipeline; register a new model version."""
    import mlflow
    import mlflow.sklearn

    from src.tracking import REGISTERED_MODEL_NAME, start_run

    classifier = pipeline.named_steps["classifier"]
    params = {
        "model_type": "xgboost",
        "scale_pos_weight": classifier.get_params().get("scale_pos_weight"),
        "n_estimators": classifier.get_params().get("n_estimators"),
        "learning_rate": classifier.get_params().get("learning_rate"),
        "max_depth": classifier.get_params().get("max_depth"),
        "min_child_weight": classifier.get_params().get("min_child_weight"),
        "subsample": classifier.get_params().get("subsample"),
        "colsample_bytree": classifier.get_params().get("colsample_bytree"),
        "n_numeric_features": len(config["numeric_features"]),
        "n_categorical_features": len(config["categorical_features"]),
        "n_train_rows": config["n_train_rows"],
        "n_train_positives": config["n_train_positives"],
    }

    with start_run("final-xgboost", tags={"stage": "final", "model": "xgboost_unweighted"}):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "feature_config.json")
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            # MLflow 3.x defaults to skops, which refuses to serialize XGBoost
            # objects unless every type is declared trusted. cloudpickle handles
            # the whole Pipeline (preprocessor + booster) as one object; the
            # artifact is only ever loaded from our own registry.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version)) if versions else None
    return latest.version if latest else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    args = parser.parse_args()

    df = pd.read_csv(COHORT_PATH)
    test_subjects = pd.read_csv(TEST_IDS_PATH)["subject_id"]

    pipeline, config, metrics, input_example = fit_final_model(df, test_subjects)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"saved {PIPELINE_PATH.name}  "
        f"test AUROC {metrics['test_auroc']:.4f}  "
        f"AUPRC {metrics['test_auprc']:.4f}  "
        f"Brier {metrics['test_brier']:.4f}"
    )
    print(
        f"saved {CONFIG_PATH.name}  "
        f"({len(config['numeric_features'])} numeric, "
        f"{len(config['categorical_features'])} categorical)"
    )

    if args.no_mlflow:
        return

    try:
        version = _log_to_mlflow(pipeline, config, metrics, input_example)
    except ImportError:
        print("mlflow not installed — skipped tracking (pip install mlflow)")
    except Exception as exc:  # tracking must never block producing the artifact
        print(f"MLflow logging failed: {exc}")
    else:
        from src.tracking import REGISTERED_MODEL_NAME, get_tracking_uri

        print(f"logged to MLflow ({get_tracking_uri()})")
        if version:
            print(f"registered {REGISTERED_MODEL_NAME} version {version}")
            print(f'  serve it with: $env:MLFLOW_MODEL_URI = "models:/{REGISTERED_MODEL_NAME}/{version}"')


if __name__ == "__main__":
    main()
