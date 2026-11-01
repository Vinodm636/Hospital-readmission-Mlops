"""
run_pipeline.py
---------------
Master script to run the FULL Part 1 pipeline end-to-end from
the project root — without needing Airflow.

Steps:
    1. Ingest CSV → data/staging/
    2. Clean data → data/processed/
    3. Feature engineering → data/curated/
    4. Validate cleaned data
    5. Load all data → PostgreSQL

Run:
    python run_pipeline.py

Prerequisites:
    - data/raw/diabetic_data.csv must exist
    - PostgreSQL must be running
    - .env must have POSTGRES_PASSWORD set
    - python setup_database.py must have been run once
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_step(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"STEP {n}: {title}")
    print('='*60)


def main():
    print("\n" + "="*60)
    print("HOSPITAL READMISSION — FULL ETL PIPELINE")
    print("="*60)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    print_step(1, "Data Ingestion")
    from src.ingestion.ingest import run_ingestion
    ingest_result = run_ingestion(save_to_db=False)
    print(f"  Status  : {ingest_result['metadata']['status']}")
    print(f"  Rows    : {ingest_result['metadata']['row_count']:,}")
    print(f"  Columns : {ingest_result['metadata']['column_count']}")

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    print_step(2, "Data Cleaning")
    from src.preprocessing.clean_data import run_cleaning
    clean_result = run_cleaning()
    s = clean_result["stats"]
    print(f"  Cleaned rows : {s['cleaned_row_count']:,}")
    print(f"  Rejected     : {s['rejected_count']}")
    print(f"  Retention    : {s['retention_rate_pct']}%")

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print_step(3, "Feature Engineering")
    from src.transformation.feature_engineering import run_feature_engineering
    fe_result = run_feature_engineering()
    s = fe_result["stats"]
    print(f"  Total columns   : {s['total_columns']}")
    print(f"  New features    : {s['new_features']}")
    print(f"  Readmission rate: {s['readmission_rate_pct']}%")

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    print_step(4, "Data Validation (cleaned dataset)")
    from src.validation.data_quality import run_validation
    results = run_validation(dataset_label="cleaned", halt_on_critical_failure=True)
    passed  = sum(1 for r in results if r["status"] == "PASS")
    warned  = sum(1 for r in results if r["status"] == "WARN")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  PASS: {passed}  |  WARN: {warned}  |  FAIL: {failed}")
    if failed:
        print("  CRITICAL FAILURES — stopping pipeline.")
        sys.exit(1)

    # ── Step 5: Load to PostgreSQL ────────────────────────────────────────────
    print_step(5, "Loading to PostgreSQL")
    try:
        from src.database.loader import run_full_load
        load_result = run_full_load()
        print("  Tables loaded:")
        for table, count in load_result.items():
            print(f"    {table:<30}: {count:,} rows")
    except Exception as exc:
        print(f"  WARNING: PostgreSQL load failed — {exc}")
        print("  Run setup_database.py first, then re-run this script.")
        print("  The curated CSV at data/curated/diabetic_curated.csv is ready.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print('='*60)
    print(f"  Raw data     : data/raw/diabetic_data.csv")
    print(f"  Staging      : data/staging/diabetic_staging.csv")
    print(f"  Cleaned      : data/processed/diabetic_cleaned.csv")
    print(f"  Curated      : data/curated/diabetic_curated.csv")
    print(f"  Rejected log : data/rejected/")
    print(f"  Validation   : data/logs/")
    print()
    print("Next steps:")
    print("  1. Start dashboard : streamlit run dashboard/app.py")
    print("  2. Train ML models : python ml/run_ml_pipeline.py")
    print()


if __name__ == "__main__":
    main()
