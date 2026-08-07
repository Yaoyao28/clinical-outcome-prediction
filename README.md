# Clinical Outcome Prediction Platform

An end-to-end clinical machine learning and real-world data analytics project using the MIMIC-IV Clinical Database.

The current implementation predicts **in-hospital mortality using information available during the first 24 hours after ICU admission**, with emphasis on patient-level data splitting, leakage prevention, model calibration, explainability, threshold analysis, and reusable machine learning code.

> **Important:** The current development version uses the MIMIC-IV Clinical Database Demo. Results are exploratory and are not intended for clinical use.

---

## Project Overview

Electronic health records contain structured information such as demographics, laboratory measurements, vital signs, admission characteristics, and ICU encounter data that can be used to develop clinical risk prediction models.

This project implements a reproducible machine learning workflow for ICU outcome prediction using MIMIC-IV.

The current version focuses on:

- ICU cohort construction
- first-24-hour feature engineering
- leakage-safe patient-level splitting
- preprocessing pipelines
- baseline and tree-based machine learning models
- grouped cross-validation
- hyperparameter tuning
- probability calibration
- final test evaluation
- SHAP interpretation
- threshold and clinical utility analysis
- false-positive and false-negative review
- subgroup analysis
- reusable `src/` modules
- automated unit testing with `pytest`

The same pipeline is intended to be rerun later on the full MIMIC-IV Clinical Database.

---

## Clinical Question

> Can structured clinical information available during the first 24 hours after ICU admission be used to estimate the probability of in-hospital mortality?

### Prediction target

```text
hospital_expire_flag

0 = survived hospitalization
1 = died during hospitalization
```

### Prediction timeline

```text
ICU admission
     ↓
First 24 hours of clinical data
     ↓
Prediction time
     ↓
In-hospital outcome
```

Only information available within the defined prediction window is used as model input.

---

## Dataset

### Current development dataset

**MIMIC-IV Clinical Database Demo**

Source: PhysioNet

Core data sources include:

- patient demographics
- hospital admissions
- ICU stays
- laboratory measurements
- vital signs
- ICU care-unit information

### Planned full-scale analysis

The finalized workflow is designed to be rerun on the full **MIMIC-IV Clinical Database** after development and validation on the demo dataset.

---

## Cohort Construction

The modeling cohort is constructed at the ICU-stay level.

Key cohort design decisions include:

- retain the first ICU stay for each hospital admission;
- use one modeling observation per ICU stay;
- preserve `subject_id`, `hadm_id`, and `stay_id` for validation and leakage control;
- validate the binary mortality target;
- define a first-24-hour feature window;
- exclude post-prediction information from model predictors;
- maintain patient-level independence across train, validation, and test sets.

SQL and Python are both used for cohort construction and feature preparation.

---

## Machine Learning Workflow

```text
MIMIC-IV Raw Data
        │
        ▼
SQL Cohort Construction
        │
        ▼
Data Validation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
First-24-Hour Feature Engineering
        │
        ▼
Patient-Level Train / Validation / Test Split
        │
        ▼
Preprocessing Pipeline
        │
        ├── Missing-value imputation
        ├── Missingness indicators
        ├── Standardization
        └── One-hot encoding
        │
        ▼
Model Development
        │
        ├── Dummy Baseline
        ├── Logistic Regression
        ├── Random Forest
        └── XGBoost
        │
        ▼
Grouped Cross-Validation
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Validation-Based Model Selection
        │
        ▼
Probability Calibration
        │
        ├── Sigmoid
        └── Isotonic
        │
        ▼
Final Test Evaluation
        │
        ▼
SHAP Interpretation
        │
        ▼
Threshold / Clinical Utility / Error Analysis
```

---

## Models

The following classification models are compared:

| Model | Purpose |
|---|---|
| Dummy Classifier | Reference baseline |
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Nonlinear ensemble model |
| XGBoost | Gradient-boosted tree model |
| Tuned Random Forest | Cross-validated RF candidate |
| Tuned XGBoost | Cross-validated XGBoost candidate |

The primary model-selection metric is **validation AUPRC**, with AUROC used as a secondary discrimination metric.

---

## Current Results

### Validation model selection

The Random Forest was selected based on validation performance.

| Model stage | AUROC | AUPRC |
|---|---:|---:|
| Selected Random Forest — Validation | 0.944 | 0.750 |

### Final test performance

The untouched test cohort showed substantially lower performance:

| Model stage | AUROC | AUPRC |
|---|---:|---:|
| Random Forest — Test | 0.647 | 0.216 |

This drop illustrates why an untouched test set is essential, especially when working with a very small development dataset.

### Calibration findings

Calibration methods were evaluated using validation Brier score and log loss.

In the small validation cohort, isotonic calibration appeared to improve calibration substantially, but it did not generalize to the test cohort.

This was treated as evidence of **calibration overfitting caused by the extremely small calibration sample**.

For downstream SHAP and threshold analysis, the project therefore retains the underlying uncalibrated Random Forest probabilities.

---

## Evaluation Metrics

### Discrimination

- AUROC
- AUPRC

### Threshold-based classification

- Accuracy
- Precision / PPV
- Recall / Sensitivity
- Specificity
- Negative Predictive Value
- F1 Score
- True Positives
- True Negatives
- False Positives
- False Negatives

### Probability quality

- Brier Score
- Log Loss
- Calibration curves
- Mean predicted risk
- Observed event rate

### Clinical operating characteristics

- probability-threshold trade-offs
- number of patients flagged
- false-positive burden
- false-negative burden
- risk ranking
- review-capacity analysis
- death-capture rate

---

## Probability Calibration

The project evaluates:

- uncalibrated model probabilities
- sigmoid calibration
- isotonic calibration

Calibration is fitted using validation data only.

The untouched test set is not used to select the calibration method.

Because the MIMIC-IV demo cohort is extremely small, calibration estimates are unstable and are interpreted cautiously.

---

## Explainable AI

SHAP is used to examine model behavior at both global and patient-specific levels.

Analyses include:

- mean absolute SHAP feature importance
- SHAP summary / beeswarm plots
- SHAP dependence plots
- patient-level waterfall plots
- comparison with Random Forest built-in feature importance

Example high-ranking features in the current demo analysis include:

- creatinine measurements
- ICU admission timing
- bicarbonate
- age
- BUN
- potassium
- lactate

SHAP values are interpreted as **predictive contributions**, not causal effects.

---

## Threshold and Clinical Utility Analysis

A clinical prediction system requires a probability threshold to convert predicted risk into an actionable classification.

The project evaluates multiple thresholds and examines the trade-off between:

```text
Lower threshold
     ↓
Higher sensitivity
     ↓
Fewer missed deaths
     ↓
More false-positive alerts
     ↓
Higher clinical review burden
```

and

```text
Higher threshold
     ↓
Lower alert burden
     ↓
Higher specificity
     ↓
More missed mortality events
```

The current demo analysis shows that lower thresholds can capture more mortality events but may flag a large proportion of the cohort.

Because the test cohort contains very few deaths, no threshold is presented as clinically optimal.

---

## Error Analysis

### False negatives

False-negative cases are reviewed because they represent mortality events with predicted probabilities below the operating threshold.

The analysis examines:

- predicted mortality probability
- distance from the selected threshold
- demographics
- ICU characteristics
- admission type
- available laboratory and vital-sign features

### False positives

High-risk survivors are also reviewed.

A false-positive prediction does not necessarily imply that the patient was clinically low risk. A patient may have been severely ill but survived after treatment, or the model may have identified high-risk physiology that was not deterministically associated with death.

---

## Subgroup Analysis

Exploratory subgroup analyses are performed when data are available for:

- sex
- age group
- race
- insurance
- ICU care unit
- admission type

Because the demo test cohort is extremely small, subgroup estimates are explicitly flagged when sample size or event counts are too low for reliable interpretation.

These analyses do **not** establish model fairness or subgroup-specific clinical validity.

---

## Software Engineering

Reusable implementation code is separated from exploratory notebooks.

```text
src/
├── __init__.py
├── config.py
│
├── data/
│   ├── __init__.py
│   ├── loaders.py
│   └── validation.py
│
├── features/
│   ├── __init__.py
│   └── preprocessing.py
│
├── models/
│   ├── __init__.py
│   ├── logistic.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   └── calibration.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── plots.py
│   ├── threshold.py
│   └── subgroup.py
│
└── interpretation/
    ├── __init__.py
    ├── shap_utils.py
    └── feature_importance.py
```

This separates:

```text
notebooks/  → experiments, analysis, plots, interpretation
src/        → reusable implementation
sql/        → cohort and feature extraction
tests/      → automated validation
models/     → serialized model artifacts
results/    → numerical and analysis outputs
reports/    → project reports
```

---

## Repository Structure

```text
clinical-outcome-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── sql/
│
├── src/
│
├── tests/
│
├── models/
│
├── results/
│
├── reports/
│
├── app/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Testing

Reusable project modules are covered by automated tests using `pytest`.

The test suite validates:

- cohort integrity
- binary target validation
- unique ICU stay identifiers
- patient-level split leakage
- valid probability ranges
- numerical and categorical feature inference
- preprocessing with missing values
- handling of unknown categorical values
- Logistic Regression pipeline
- Random Forest pipeline
- AUROC / AUPRC calculations
- probability metrics
- threshold analysis
- review-capacity analysis
- subgroup analysis

### Current test status

```text
21 passed
```

Run the test suite from the project root:

```bash
python -m pytest -v
```

On Windows using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Installation

Clone the repository and create a virtual environment.

```bash
git clone <repository-url>
cd clinical-outcome-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -v
```

---

## Core Dependencies

```text
numpy
pandas
scipy
scikit-learn
xgboost
shap
matplotlib
duckdb
joblib
jupyter
ipykernel
pytest
```

---

## Technologies

### Data and analytics

- Python
- pandas
- NumPy
- SciPy
- SQL
- DuckDB

### Machine learning

- scikit-learn
- XGBoost

### Explainability

- SHAP

### Visualization

- Matplotlib

### Engineering

- joblib
- pytest
- Git
- Jupyter

---

## Project Status

### Completed

- cohort construction
- data validation
- missingness analysis
- first-24-hour feature engineering
- patient-level train / validation / test split
- preprocessing pipeline
- Dummy baseline
- Logistic Regression
- Random Forest
- XGBoost
- grouped cross-validation
- hyperparameter tuning
- validation-based model selection
- probability calibration
- untouched test evaluation
- SHAP interpretation
- threshold analysis
- clinical review-capacity analysis
- false-positive analysis
- false-negative analysis
- subgroup analysis
- reusable `src/` package
- automated `pytest` test suite

### Current development stage

The core predictive modeling phase is complete.

The next stage expands the project from predictive modeling into **real-world evidence and clinical trial analytics**.

---

## Planned Extensions

### Real-World Evidence / Trial Analytics

- Kaplan–Meier survival analysis
- Cox proportional hazards modeling
- trial-style cohort builder
- configurable inclusion / exclusion criteria
- propensity score estimation
- propensity score matching
- inverse probability treatment weighting
- standardized mean difference diagnostics
- covariate balance / Love plots
- observational treatment-effect analysis

### Machine Learning

- PyTorch MLP baseline
- full MIMIC-IV rerun
- temporal validation
- external validation

### Deployment

- FastAPI prediction service
- Streamlit dashboard
- Docker
- CI/CD
- cloud deployment

---

## Limitations

The current implementation uses the small MIMIC-IV demo dataset.

Important limitations include:

- very small cohort size;
- very few mortality events;
- unstable validation and test estimates;
- calibration overfitting risk;
- unstable subgroup metrics;
- limited generalizability;
- internal evaluation only;
- no external validation;
- predictive associations should not be interpreted causally.

The current model and analysis are intended for research, education, and portfolio demonstration only.

**This project is not intended for clinical use.**

---

## Reproducibility

The project is designed so that reusable modeling logic lives in `src/`, while notebooks focus on analysis and interpretation.

A clean reproducibility workflow is:

```text
01 Data overview
        ↓
02 Cohort construction
        ↓
03 Patient split / missingness
        ↓
04 Feature engineering
        ↓
05 Preprocessing
        ↓
06 Baseline + Logistic Regression
        ↓
07 Tree models + model selection
        ↓
08 Calibration + final evaluation
        ↓
09 SHAP interpretation
        ↓
10 Threshold + subgroup + error analysis
```

Before publishing a release, all notebooks should be restarted and rerun from a clean kernel and the complete test suite should pass.

---

## Future Project Direction

The long-term goal is to evolve this repository from an ICU mortality prediction project into a broader:

> **Real-World Clinical Outcome Prediction and Treatment Effect Analysis Platform**

The predictive modeling workflow will serve as the foundation for survival analysis, trial-style cohort construction, and observational causal inference using MIMIC-IV.
