"""MLflow experiment tracking and model registry helpers.

Why this module exists
----------------------
Model comparison produced a lot of numbers spread across notebook outputs and
CSV files: three models, two class-weighting settings, repeated CV folds, and a
frozen test evaluation. Which hyper-parameters produced which AUROC was
recoverable only by reading the notebook. MLflow makes that mechanical — every
fit records its parameters, metrics, and the fitted artifact, and the resulting
model gets a version number the API can ask for by name.

Tracking backend
----------------
``MLFLOW_TRACKING_URI`` selects where runs are stored. Unset, it defaults to a
local SQLite database (``mlflow.db``, git-ignored) with artifacts under
``mlruns/``. SQLite rather than the older plain-file store because the file
store is in maintenance mode and the model registry needs a database backend.
No server is required:

    mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000

To use a remote or managed tracking server instead, set the variable — no code
change is needed:

    $env:MLFLOW_TRACKING_URI = "https://<server>"
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from typing import Any, Iterator

from src.config import PROJECT_DIR

EXPERIMENT_NAME = "icu-mortality"
REGISTERED_MODEL_NAME = "icu-mortality-xgboost"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_ROOT = (PROJECT_DIR / "mlruns").as_uri()


def get_tracking_uri() -> str:
    """Tracking URI from the environment, falling back to a local SQLite file."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def git_commit() -> str | None:
    """Short commit hash of the working tree, so a run points back at its code."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


@contextmanager
def start_run(
    run_name: str,
    *,
    experiment: str = EXPERIMENT_NAME,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Open an MLflow run with the project's experiment and standard tags.

    Usage
    -----
    >>> with start_run("final-xgboost") as run:
    ...     mlflow.log_param("max_depth", 4)

    Import of ``mlflow`` is deferred so that the inference image, which does not
    install it, can still import the rest of ``src``.
    """
    import mlflow

    mlflow.set_tracking_uri(get_tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=DEFAULT_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)

    run_tags = {"project": "clinical-outcome-prediction"}
    commit = git_commit()
    if commit:
        run_tags["git_commit"] = commit
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
        yield run


def log_cv_results(cv_results, model_name: str, metrics: tuple[str, ...] = ("auroc", "auprc", "brier")) -> None:
    """Log per-fold CV metrics as mean/SD plus a per-fold step series.

    ``cv_results`` is the DataFrame returned by
    :func:`src.evaluation.resampling.repeated_grouped_cv` for one model.
    """
    import mlflow

    mlflow.log_param("cv_n_fits", len(cv_results))
    for metric in metrics:
        values = cv_results[metric].dropna()
        if values.empty:
            continue
        mlflow.log_metric(f"cv_{metric}_mean", float(values.mean()))
        mlflow.log_metric(f"cv_{metric}_sd", float(values.std(ddof=1)))
        for step, value in enumerate(values):
            mlflow.log_metric(f"cv_{metric}_fold", float(value), step=step)
    mlflow.set_tag("model", model_name)
