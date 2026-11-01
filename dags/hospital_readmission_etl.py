"""
hospital_readmission_etl.py
---------------------------
Apache Airflow DAG for the Hospital Readmission ETL pipeline.

What this DAG does:
    Orchestrates the full data pipeline from raw CSV ingestion through to
    populating the PostgreSQL analytical data mart.

Why Airflow:
    Airflow provides scheduling, task dependency management, retries,
    logging, and a web UI to monitor pipeline runs.  A DAG (Directed
    Acyclic Graph) defines tasks and the order they must run in.

DAG task order:
    extract
        ↓
    load_raw           (load staging CSV → raw.encounters)
        ↓
    validate_raw       (data quality checks on staging data)
        ↓
    transform          (cleaning + feature engineering)
        ↓
    validate_cleaned   (data quality checks on processed data)
        ↓
    load_postgres      (load curated data → all PostgreSQL layers)
        ↓
    create_analytics   (populate dimension + fact tables)
        ↓
    pipeline_summary   (log final statistics)

Schedule:
    Manual trigger (schedule_interval=None) — run on demand.
    Can be changed to '@daily' for scheduled execution.

Viva questions:
    Q: What is an Airflow DAG?
    A: A DAG (Directed Acyclic Graph) is a collection of tasks with defined
       dependencies. 'Directed' means tasks have a one-way order.
       'Acyclic' means there are no circular dependencies (no loops).

    Q: What are XComs in Airflow?
    A: XComs (cross-communications) allow tasks to pass small pieces of
       data to each other. For example, the 'extract' task can pass the
       run_id to the 'transform' task via XCom.

    Q: What happens if a task fails?
    A: Airflow retries the task (up to retries=2 here). If it still fails,
       downstream tasks are not executed. The failure is visible in the UI.
"""

import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path so imports work from Airflow
# When running via Airflow, the dags/ folder is added automatically
# but src/ may not be. We add the project root explicitly.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Default DAG arguments
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "student",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

# ─────────────────────────────────────────────────────────────────────────────
# Task callables
# Each function is the Python code that runs for one DAG task.
# We import the modules here (inside the function) so Airflow does not
# need them at DAG parse time — only at execution time.
# ─────────────────────────────────────────────────────────────────────────────

def task_extract(**context) -> str:
    """
    Task 1: Extract — Ingest the raw CSV dataset.

    Calls src/ingestion/ingest.py:run_ingestion()
    Pushes run_id to XCom so downstream tasks can use it.

    Returns:
        str: The pipeline run_id.
    """
    from src.ingestion.ingest import run_ingestion

    result = run_ingestion()
    actual_run_id = result["run_id"]
    context["task_instance"].xcom_push(key="run_id", value=actual_run_id)

    print(f"Extraction complete. run_id={actual_run_id}")
    print(f"  Rows: {result['metadata']['row_count']}")
    print(f"  Status: {result['metadata']['status']}")
    return actual_run_id


def task_load_raw(**context) -> int:
    """
    Task 2: Load Raw — Load staging CSV into raw.encounters.

    Returns:
        int: Rows loaded.
    """
    from src.database.loader import load_raw_encounters

    run_id = context["task_instance"].xcom_pull(key="run_id", task_ids="extract")
    staging_csv = str(PROJECT_ROOT / "data" / "staging" / "diabetic_staging.csv")

    rows = load_raw_encounters(staging_csv, run_id=run_id)
    print(f"Raw load complete: {rows} rows → raw.encounters")
    return rows


def task_validate_raw(**context) -> dict:
    """
    Task 3: Validate Raw — Run data-quality checks on staging data.

    Fails the task (raises) if critical checks fail.

    Returns:
        dict: Validation summary.
    """
    from src.validation.data_quality import run_validation

    run_id = context["task_instance"].xcom_pull(key="run_id", task_ids="extract")
    staging_csv = str(PROJECT_ROOT / "data" / "staging" / "diabetic_staging.csv")

    results = run_validation(
        csv_path=staging_csv,
        dataset_label="raw",
        run_id=run_id,
        halt_on_critical_failure=True,
    )

    summary = {
        "pass": sum(1 for r in results if r["status"] == "PASS"),
        "fail": sum(1 for r in results if r["status"] == "FAIL"),
        "warn": sum(1 for r in results if r["status"] == "WARN"),
    }
    print(f"Raw validation: {summary}")
    return summary


def task_transform(**context) -> dict:
    """
    Task 4: Transform — Run cleaning and feature engineering.

    Steps:
        a) Clean the staging data (clean_data.py)
        b) Engineer features (feature_engineering.py)

    Returns:
        dict: Summary of transformation outputs.
    """
    from src.preprocessing.clean_data import run_cleaning
    from src.transformation.feature_engineering import run_feature_engineering

    run_id = context["task_instance"].xcom_pull(key="run_id", task_ids="extract")

    # Step a: Clean
    clean_result = run_cleaning(run_id=run_id)
    print(f"Cleaning complete: {clean_result['stats']}")

    # Step b: Feature engineering
    fe_result = run_feature_engineering(run_id=run_id)
    print(f"Feature engineering complete: {fe_result['stats']}")

    return {
        "cleaned_path": clean_result["cleaned_path"],
        "curated_path": fe_result["curated_path"],
        "rejected_path": clean_result["rejected_path"],
        "clean_stats": clean_result["stats"],
        "fe_stats": fe_result["stats"],
    }


def task_validate_cleaned(**context) -> dict:
    """
    Task 5: Validate Cleaned — Run data-quality checks on cleaned data.

    Returns:
        dict: Validation summary.
    """
    from src.validation.data_quality import run_validation

    run_id = context["task_instance"].xcom_pull(key="run_id", task_ids="extract")
    cleaned_csv = str(PROJECT_ROOT / "data" / "processed" / "diabetic_cleaned.csv")

    results = run_validation(
        csv_path=cleaned_csv,
        dataset_label="cleaned",
        run_id=run_id,
        halt_on_critical_failure=True,
    )

    summary = {
        "pass": sum(1 for r in results if r["status"] == "PASS"),
        "fail": sum(1 for r in results if r["status"] == "FAIL"),
        "warn": sum(1 for r in results if r["status"] == "WARN"),
    }
    print(f"Cleaned validation: {summary}")
    return summary


def task_load_postgres(**context) -> dict:
    """
    Task 6: Load PostgreSQL — Load curated data into all DB layers.

    Returns:
        dict: Row counts per table.
    """
    from src.database.loader import run_full_load

    run_id = context["task_instance"].xcom_pull(key="run_id", task_ids="extract")
    result = run_full_load(run_id=run_id)
    print(f"PostgreSQL load complete: {result}")
    return result


def task_create_analytics(**context) -> str:
    """
    Task 7: Create Analytics — Verify analytics views exist and log counts.

    This is a verification task that queries the analytics tables to
    confirm data was loaded successfully.

    Returns:
        str: Summary message.
    """
    from src.database.connection import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        # Check row counts in all analytics tables
        tables = [
            "analytics.dim_admission",
            "analytics.dim_diagnosis",
            "analytics.fact_patient_summary",
            "analytics.fact_readmission",
        ]
        counts = {}
        for tbl in tables:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {tbl}")
            ).scalar()
            counts[tbl] = count
            print(f"  {tbl}: {count} rows")

        # Verify the readmission rate view
        rate = conn.execute(
            text("SELECT readmission_rate_pct FROM analytics.vw_readmission_rate")
        ).scalar()
        print(f"  Overall readmission rate: {rate}%")

    summary = f"Analytics tables verified. Counts: {counts}. Readmission rate: {rate}%"
    return summary


def task_pipeline_summary(**context) -> str:
    """
    Task 8: Pipeline Summary — Log the final end-to-end run summary.

    Pulls results from all upstream tasks via XCom and writes a summary.

    Returns:
        str: Human-readable pipeline summary.
    """
    ti = context["task_instance"]
    run_id = ti.xcom_pull(key="run_id", task_ids="extract")
    raw_rows = ti.xcom_pull(task_ids="load_raw") or 0
    transform_result = ti.xcom_pull(task_ids="transform") or {}
    pg_result = ti.xcom_pull(task_ids="load_postgres") or {}

    summary_lines = [
        "=" * 60,
        "PIPELINE SUMMARY",
        "=" * 60,
        f"Run ID          : {run_id}",
        f"Completed at    : {datetime.now().isoformat()}",
        f"Raw rows loaded : {raw_rows}",
    ]

    if transform_result:
        stats = transform_result.get("clean_stats", {})
        summary_lines.extend([
            f"Cleaned rows    : {stats.get('cleaned_row_count', 'N/A')}",
            f"Rejected rows   : {stats.get('rejected_count', 'N/A')}",
            f"Retention rate  : {stats.get('retention_rate_pct', 'N/A')}%",
        ])

    if pg_result:
        summary_lines.append(f"DB tables loaded: {list(pg_result.keys())}")

    summary_lines.append("=" * 60)
    summary = "\n".join(summary_lines)
    print(summary)

    # Save summary to a log file
    summary_file = PROJECT_ROOT / "logs" / f"pipeline_summary_{datetime.now().strftime('%Y-%m-%d')}.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "a") as f:
        f.write(summary + "\n\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="hospital_readmission_etl",
    default_args=DEFAULT_ARGS,
    description=(
        "ETL pipeline for the Hospital Readmission Analytics project. "
        "Ingests UCI Diabetes dataset, cleans, validates, and loads "
        "into PostgreSQL analytical data mart."
    ),
    schedule_interval=None,   # Manual trigger (change to '@daily' for scheduling)
    catchup=False,
    tags=["hospital", "readmission", "etl", "healthcare"],
    doc_md="""
## Hospital Readmission ETL DAG

This DAG implements the full ETL pipeline for the Hospital Readmission
Analytics and Prediction project.

### Pipeline Steps
1. **extract** — Ingest raw CSV from data/raw/
2. **load_raw** — Load staging CSV into raw.encounters
3. **validate_raw** — Run 8 data-quality checks on raw data
4. **transform** — Clean data + engineer 10 features
5. **validate_cleaned** — Re-validate cleaned data
6. **load_postgres** — Populate all PostgreSQL layers
7. **create_analytics** — Verify dimensional model tables
8. **pipeline_summary** — Write run summary log

### Dataset
UCI Diabetes 130-US Hospitals Dataset (de-identified, public)

### Prerequisite
Place `diabetic_data.csv` in `data/raw/` before triggering this DAG.
    """,
) as dag:

    # ── Task definitions ──────────────────────────────────────────────────────

    start = EmptyOperator(task_id="start")

    extract = PythonOperator(
        task_id="extract",
        python_callable=task_extract,
        doc_md="Ingest raw CSV and preserve staging copy.",
    )

    load_raw = PythonOperator(
        task_id="load_raw",
        python_callable=task_load_raw,
        doc_md="Load raw staging data into raw.encounters (text format).",
    )

    validate_raw = PythonOperator(
        task_id="validate_raw",
        python_callable=task_validate_raw,
        doc_md="Run 8 data-quality checks on the raw staging dataset.",
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_transform,
        doc_md="Clean data and engineer 10 analytical/ML features.",
        execution_timeout=timedelta(minutes=30),
    )

    validate_cleaned = PythonOperator(
        task_id="validate_cleaned",
        python_callable=task_validate_cleaned,
        doc_md="Re-run data-quality checks on the cleaned dataset.",
    )

    load_postgres = PythonOperator(
        task_id="load_postgres",
        python_callable=task_load_postgres,
        doc_md="Load curated data into all PostgreSQL schemas.",
        execution_timeout=timedelta(minutes=30),
    )

    create_analytics = PythonOperator(
        task_id="create_analytics",
        python_callable=task_create_analytics,
        doc_md="Verify analytics dimensional tables and report row counts.",
    )

    pipeline_summary = PythonOperator(
        task_id="pipeline_summary",
        python_callable=task_pipeline_summary,
        doc_md="Write end-to-end pipeline execution summary.",
        trigger_rule="all_done",   # Run even if some upstream tasks warn
    )

    end = EmptyOperator(task_id="end")

    # ── Task dependencies ─────────────────────────────────────────────────────
    # The >> operator sets downstream dependencies in Airflow

    (
        start
        >> extract
        >> load_raw
        >> validate_raw
        >> transform
        >> validate_cleaned
        >> load_postgres
        >> create_analytics
        >> pipeline_summary
        >> end
    )
