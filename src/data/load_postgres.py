"""Load the modeling cohort into the feature store (Postgres or any SQLAlchemy URL).

Start the database and load it:

    docker compose up -d db
    $env:DATABASE_URL = "postgresql+psycopg2://icu:icu@localhost:5432/icu"
    python -m src.data.load_postgres

The table is replaced on each run, so re-running after a cohort refresh is safe.
An index on stay_id is created afterwards because the API looks rows up by it.

Note: this loads credentialed MIMIC-IV data into a local container. The compose
file binds Postgres to localhost only and the volume lives outside version
control; do not point this at a shared or public database.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from src.config import PROJECT_DIR

COHORT_PATH = PROJECT_DIR / "data/raw/full/modeling_cohort_full.csv"
TABLE_NAME = "modeling_cohort"
CHUNK_SIZE = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="SQLAlchemy URL; defaults to $DATABASE_URL",
    )
    parser.add_argument("--csv", default=str(COHORT_PATH), help="cohort CSV to load")
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit(
            "No database URL. Set DATABASE_URL or pass --database-url, e.g.\n"
            '  $env:DATABASE_URL = "postgresql+psycopg2://icu:icu@localhost:5432/icu"'
        )

    from sqlalchemy import create_engine, text

    df = pd.read_csv(args.csv)
    engine = create_engine(args.database_url, future=True)

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )

    with engine.begin() as connection:
        connection.execute(
            text(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_stay_id "
                 f'ON {TABLE_NAME} ("stay_id")')
        )
        rows = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()

    print(f"loaded {rows:,} rows x {df.shape[1]} columns into {TABLE_NAME}")
    print(f"example stay_id: {df['stay_id'].iloc[0]}")


if __name__ == "__main__":
    main()
