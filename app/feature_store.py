"""Feature store access for the prediction service.

Why this module exists
----------------------
``POST /predict`` asks the caller to supply every feature, which is fine for a
demo but not how a clinical system works: the features already exist in the
warehouse, keyed by ICU stay. This module gives the API a second path —
``GET /predict/{stay_id}`` — that reads the stored first-24h feature row and
scores it, so the caller only needs an identifier.

Database-agnostic by design: everything goes through SQLAlchemy, so the same
code runs against the Postgres container in ``docker-compose.yml`` and against
an in-process SQLite database in the tests.

Configuration
-------------
``DATABASE_URL`` selects the backend, e.g.

    postgresql+psycopg2://icu:icu@localhost:5432/icu     # docker-compose
    sqlite:///./cohort.db                                # local experiment

If it is unset the feature-store endpoint is simply disabled; ``POST /predict``
keeps working, which is what the self-contained inference image relies on.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

TABLE_NAME = "modeling_cohort"
ID_COLUMN = "stay_id"


def get_database_url() -> str | None:
    """Connection string from the environment, or None when no store is configured."""
    return os.environ.get("DATABASE_URL") or None


@lru_cache(maxsize=1)
def get_engine(database_url: str):
    """One pooled engine per process, created lazily on first use.

    ``pool_pre_ping`` makes the pool check a connection before handing it out,
    so a container that outlives a database restart recovers instead of serving
    errors from stale connections.
    """
    from sqlalchemy import create_engine

    return create_engine(database_url, pool_pre_ping=True, future=True)


def fetch_features(stay_id: int, feature_columns: list[str], database_url: str) -> pd.DataFrame | None:
    """Return a one-row frame of features for ``stay_id``, or None if absent.

    Only the columns the model expects are selected — the cohort table also
    holds identifiers, timestamps and the outcome, none of which may reach the
    model at prediction time.
    """
    from sqlalchemy import bindparam, text

    quoted = ", ".join(f'"{column}"' for column in feature_columns)
    statement = text(
        f'SELECT {quoted} FROM {TABLE_NAME} WHERE "{ID_COLUMN}" = :stay_id LIMIT 1'
    ).bindparams(bindparam("stay_id"))

    with get_engine(database_url).connect() as connection:
        frame = pd.read_sql(statement, connection, params={"stay_id": int(stay_id)})

    return frame if not frame.empty else None


def store_is_reachable(database_url: str) -> bool:
    """True if the database answers a trivial query — used by /health."""
    from sqlalchemy import text

    try:
        with get_engine(database_url).connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True
