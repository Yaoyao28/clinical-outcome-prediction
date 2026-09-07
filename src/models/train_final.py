"""Fit the final model on the training pool and save the artifacts the API needs.

Run from the project root after `src/data/extract_bigquery.py` and notebook 07b
(which writes the frozen test-subject IDs):

    python -m src.models.train_final

Outputs (both git-ignored except the JSON):
    models/final_xgboost_pipeline.joblib   fitted preprocessor + XGBoost
    models/final_feature_config.json       feature lists + version metadata

The JSON is what `app/` uses to build its input schema, so the API and the
model can never disagree about which columns exist.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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


def main() -> None:
    df = pd.read_csv(COHORT_PATH)
    test_subjects = pd.read_csv(TEST_IDS_PATH)["subject_id"]
    is_test = df["subject_id"].isin(test_subjects)
    pool, test = df[~is_test], df[is_test]

    feature_columns = [c for c in pool.columns if c not in ID_AND_LEAKAGE]
    numeric_features, categorical_features = infer_feature_types(pool, feature_columns)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipeline = build_xgboost(preprocessor)  # scale_pos_weight defaults to 1.0 (unweighted)
    pipeline.fit(pool[feature_columns], pool[TARGET_COLUMN])

    test_probs = pipeline.predict_proba(test[feature_columns])[:, 1]
    auroc = roc_auc_score(test[TARGET_COLUMN], test_probs)
    auprc = average_precision_score(test[TARGET_COLUMN], test_probs)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    config = {
        "model_name": "xgboost_unweighted",
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_on": "MIMIC-IV v3.1, first ICU stay, adults",
        "n_train_rows": int(len(pool)),
        "n_train_positives": int(pool[TARGET_COLUMN].sum()),
        "test_auroc": round(float(auroc), 4),
        "test_auprc": round(float(auprc), 4),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET_COLUMN,
        "prediction_window_hours": 24,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"saved {PIPELINE_PATH.name}  test AUROC {auroc:.4f}  AUPRC {auprc:.4f}")
    print(f"saved {CONFIG_PATH.name}  ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")


if __name__ == "__main__":
    main()
