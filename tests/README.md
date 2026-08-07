# Test Suite

These tests validate the reusable Phase 1 `src/` package using synthetic data.

They do **not** require access to MIMIC-IV raw data.

## Run from the project root

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Expected structure:

```text
clinical-outcome-prediction/
├── src/
├── notebooks/
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── README.md
└── requirements.txt
```

## What is tested

- cohort integrity and binary outcomes;
- patient-level split leakage;
- probability range validation;
- numerical/categorical feature inference;
- preprocessing with missing values;
- unknown categorical levels;
- Logistic Regression pipeline;
- Random Forest pipeline;
- AUROC/AUPRC and classification metrics;
- threshold analysis;
- clinical review-capacity analysis;
- subgroup analysis.

The basic test suite intentionally avoids expensive XGBoost tuning and SHAP computations so it stays fast enough for local development and future GitHub Actions.
