"""Extract the full MIMIC-IV modeling cohort from BigQuery to a local CSV."""

from pathlib import Path

from google.cloud import bigquery

PROJECT = "mimic-dev-507717"
SQL_PATH = Path("sql/05_full_cohort_bigquery.sql")
OUT_PATH = Path("data/raw/full/modeling_cohort_full.csv")


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    df = client.query(SQL_PATH.read_text(encoding="utf-8")).to_dataframe()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"rows: {len(df):,}  cols: {df.shape[1]}")
    print(f"mortality: {df['hospital_expire_flag'].mean():.4f}")
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()