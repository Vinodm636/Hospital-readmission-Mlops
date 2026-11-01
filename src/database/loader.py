"""
loader.py
---------
PostgreSQL data loader for the Hospital Readmission project.

What this module does:
    Reads the curated CSV and loads it into all PostgreSQL layers:
    - raw.encounters       (bulk raw text copy)
    - staging.encounters   (typed staging copy)
    - curated.encounters   (full cleaned + engineered data)
    - analytics dimension + fact tables (dimensional model)

    Also loads pipeline metadata (ingestion, validation, rejected records)
    from JSON log files into the corresponding analytics tables.

Why we need it:
    PostgreSQL is the central store for all analytical queries and the
    Streamlit dashboard.  The loader bridges the file-based ETL outputs
    and the database.

Input:
    data/raw/diabetic_staging.csv    → raw.encounters
    data/processed/diabetic_cleaned.csv → staging.encounters
    data/curated/diabetic_curated.csv   → curated.encounters + analytics.*

Output:
    Populated PostgreSQL tables (all schemas)

Viva questions:
    Q: What is a bulk insert and why is it faster than row-by-row inserts?
    A: Pandas to_sql() with method='multi' sends multiple rows per SQL
       statement.  Row-by-row inserts have network overhead for each row.
       For 100k rows this can be 10-100x faster.

    Q: What does if_exists='append' vs 'replace' mean?
    A: 'replace' drops and recreates the table on every run (loses schema
       constraints). 'append' adds rows to the existing table, preserving
       the schema we defined in schema.sql.  We use 'append' here.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Project root path setup ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(_PROJECT_ROOT / ".env")
from src.database.connection import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Generic bulk loader
# ─────────────────────────────────────────────────────────────────────────────

def bulk_load_dataframe(
    df: pd.DataFrame,
    schema: str,
    table: str,
    if_exists: str = "append",
    chunksize: int = 5000,
) -> int:
    """
    Load a DataFrame into a PostgreSQL table using the fast psycopg2
    COPY protocol (CSV buffer), falling back to pandas to_sql on error.

    Args:
        df (pd.DataFrame): Data to load.
        schema (str): Target schema name.
        table (str): Target table name.
        if_exists (str): 'append' or 'replace' (only affects fallback path).
        chunksize (int): Rows per INSERT batch (only used in fallback).

    Returns:
        int: Number of rows loaded.

    Raises:
        Exception: Re-raises any database error after logging.
    """
    import io
    import psycopg2

    engine = get_engine()
    rows = len(df)

    # Attempt fast COPY via raw psycopg2 connection
    try:
        raw_conn = engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                # Serialise to CSV in memory (NaN → empty string for NULL)
                buf = io.StringIO()
                df.to_csv(buf, index=False, na_rep="", header=True)
                buf.seek(0)
                # Skip the header line — copy_expert expects data rows only
                buf.readline()
                buf_data = io.StringIO(buf.read())

                copy_sql = (
                    f"COPY {schema}.{table} "
                    f"({', '.join(df.columns)}) "
                    f"FROM STDIN WITH CSV NULL '' "
                )
                cur.copy_expert(copy_sql, buf_data)
            raw_conn.commit()
        finally:
            raw_conn.close()

        logger.info("Loaded %d rows → %s.%s", rows, schema, table)
        return rows

    except Exception as exc:
        logger.warning(
            "Fast COPY failed for %s.%s (%s) — falling back to to_sql",
            schema, table, exc,
        )
        # Fallback: pandas to_sql
        try:
            df.to_sql(
                name=table,
                con=engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
                method="multi",
            )
            logger.info("Loaded %d rows → %s.%s (fallback)", rows, schema, table)
            return rows
        except Exception as exc2:
            logger.error("Failed to load %s.%s: %s", schema, table, exc2)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Load raw encounters
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_encounters(
    staging_csv: str, run_id: str, truncate_first: bool = True
) -> int:
    """
    Load the raw staging CSV into raw.encounters (all TEXT columns).

    Args:
        staging_csv (str): Path to data/staging/diabetic_staging.csv.
        run_id (str): Pipeline run identifier.
        truncate_first (bool): Truncate the table before loading.

    Returns:
        int: Rows loaded.
    """
    logger.info("Loading raw encounters from: %s", staging_csv)
    df = pd.read_csv(staging_csv, low_memory=False, dtype=str)
    df["ingestion_run_id"] = run_id
    df["ingestion_timestamp"] = datetime.now().isoformat()

    # Normalise column names: lowercase, strip whitespace, hyphens → underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"[\-\s]+", "_", regex=True)
    )

    # Rename 'change' column (SQL reserved word)
    if "change" in df.columns:
        df = df.rename(columns={"change": "change_col"})

    engine = get_engine()
    if truncate_first:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE raw.encounters RESTART IDENTITY"))
            conn.commit()
        logger.info("raw.encounters truncated")

    return bulk_load_dataframe(df, schema="raw", table="encounters")


# ─────────────────────────────────────────────────────────────────────────────
# Load curated encounters
# ─────────────────────────────────────────────────────────────────────────────

def load_curated_encounters(
    curated_csv: str, run_id: str, truncate_first: bool = True
) -> int:
    """
    Load the curated (cleaned + engineered) CSV into curated.encounters.

    Args:
        curated_csv (str): Path to data/curated/diabetic_curated.csv.
        run_id (str): Pipeline run identifier.
        truncate_first (bool): Truncate before loading.

    Returns:
        int: Rows loaded.
    """
    logger.info("Loading curated encounters from: %s", curated_csv)
    df = pd.read_csv(curated_csv, low_memory=False)
    df["pipeline_run_id"] = run_id
    df["created_at"] = datetime.now().isoformat()

    # Select only columns that match the schema
    schema_cols = [
        "pipeline_run_id", "created_at",
        "encounter_surrogate_id", "patient_surrogate_id",
        "encounter_id", "patient_nbr",
        "race", "gender", "age", "age_midpoint",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "payer_code", "medical_specialty",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "diag_1", "diag_2", "diag_3", "number_diagnoses",
        "max_glu_serum", "a1cresult",
        "metformin", "repaglinide", "nateglinide", "glimepiride",
        "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
        "insulin", "diabetesmed",
        "readmitted", "readmission_binary",
        "diagnosis_count", "medication_count", "prior_admission_count",
        "is_prior_admission", "length_of_stay", "is_long_stay",
        "primary_diagnosis_category", "has_diabetes_drug",
        "total_service_utilisation",
    ]
    cols_to_load = [c for c in schema_cols if c in df.columns]
    df_load = df[cols_to_load].copy()

    engine = get_engine()
    if truncate_first:
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE curated.encounters RESTART IDENTITY"))
            conn.commit()
        logger.info("curated.encounters truncated")

    return bulk_load_dataframe(df_load, schema="curated", table="encounters")


# ─────────────────────────────────────────────────────────────────────────────
# Load analytics dimensional tables
# ─────────────────────────────────────────────────────────────────────────────

def load_analytics_tables(curated_csv: str, truncate_first: bool = True) -> dict:
    """
    Populate all analytics dimension and fact tables from the curated dataset.

    Tables populated:
        analytics.dim_admission
        analytics.dim_diagnosis
        analytics.fact_patient_summary
        analytics.fact_readmission

    Args:
        curated_csv (str): Path to data/curated/diabetic_curated.csv.
        truncate_first (bool): Truncate analytics tables before loading.

    Returns:
        dict: Row counts per table.
    """
    logger.info("Loading analytics tables from: %s", curated_csv)
    df = pd.read_csv(curated_csv, low_memory=False)
    engine = get_engine()

    counts: dict = {}

    if truncate_first:
        tables_to_truncate = [
            "analytics.fact_readmission",
            "analytics.fact_patient_summary",
            "analytics.dim_diagnosis",
            "analytics.dim_admission",
        ]
        with engine.connect() as conn:
            for tbl in tables_to_truncate:
                conn.execute(text(f"TRUNCATE TABLE {tbl} RESTART IDENTITY"))
            conn.commit()
        logger.info("Analytics tables truncated")

    # ── dim_admission ─────────────────────────────────────────────────────────
    adm_cols = [
        "encounter_surrogate_id", "patient_surrogate_id",
        "race", "gender", "age", "age_midpoint",
        "admission_type_id", "admission_source_id",
        "payer_code", "medical_specialty",
    ]
    df_adm = df[[c for c in adm_cols if c in df.columns]].copy()
    df_adm = df_adm.rename(columns={"age": "age_group"})
    df_adm["created_at"] = datetime.now().isoformat()
    counts["dim_admission"] = bulk_load_dataframe(
        df_adm, schema="analytics", table="dim_admission"
    )

    # ── dim_diagnosis ─────────────────────────────────────────────────────────
    diag_cols = [
        "encounter_surrogate_id",
        "diag_1", "diag_2", "diag_3",
        "primary_diagnosis_category", "number_diagnoses", "diagnosis_count",
    ]
    df_diag = df[[c for c in diag_cols if c in df.columns]].copy()
    df_diag["created_at"] = datetime.now().isoformat()
    counts["dim_diagnosis"] = bulk_load_dataframe(
        df_diag, schema="analytics", table="dim_diagnosis"
    )

    # ── fact_patient_summary ──────────────────────────────────────────────────
    fps_cols = [
        "encounter_surrogate_id", "patient_surrogate_id",
        "time_in_hospital", "length_of_stay", "is_long_stay",
        "num_lab_procedures", "num_procedures", "num_medications",
        "medication_count", "number_outpatient", "number_emergency",
        "number_inpatient", "prior_admission_count", "is_prior_admission",
        "total_service_utilisation", "diagnosis_count", "has_diabetes_drug",
        "max_glu_serum", "a1cresult", "diabetesmed", "discharge_disposition_id",
    ]
    df_fps = df[[c for c in fps_cols if c in df.columns]].copy()
    df_fps["created_at"] = datetime.now().isoformat()
    counts["fact_patient_summary"] = bulk_load_dataframe(
        df_fps, schema="analytics", table="fact_patient_summary"
    )

    # ── fact_readmission ──────────────────────────────────────────────────────
    fra_cols = [
        "encounter_surrogate_id", "patient_surrogate_id",
        "readmitted", "readmission_binary",
        "age", "age_midpoint", "gender", "race",
        "primary_diagnosis_category", "admission_type_id",
        "discharge_disposition_id", "length_of_stay",
        "num_medications", "prior_admission_count", "has_diabetes_drug",
    ]
    df_fra = df[[c for c in fra_cols if c in df.columns]].copy()
    df_fra = df_fra.rename(columns={"age": "age_group"})
    df_fra["created_at"] = datetime.now().isoformat()
    counts["fact_readmission"] = bulk_load_dataframe(
        df_fra, schema="analytics", table="fact_readmission"
    )

    logger.info("Analytics tables loaded: %s", counts)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Load pipeline metadata from JSON logs
# ─────────────────────────────────────────────────────────────────────────────

def load_validation_results_to_db(
    validation_json: str,
    run_id: str,
    dataset_label: str,
) -> int:
    """
    Load validation check results from a JSON file into
    analytics.validation_results.

    Args:
        validation_json (str): Path to the validation results JSON.
        run_id (str): Pipeline run ID.
        dataset_label (str): 'raw' or 'cleaned'.

    Returns:
        int: Number of records loaded.
    """
    json_path = Path(validation_json)
    if not json_path.exists():
        logger.warning("Validation JSON not found: %s", validation_json)
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checks = data.get("checks", [])
    if not checks:
        return 0

    records = []
    for check in checks:
        records.append({
            "run_id": run_id,
            "dataset_label": dataset_label,
            "check_name": check.get("check_name"),
            "status": check.get("status"),
            "description": check.get("description"),
            "failed_count": check.get("failed_count", 0),
            "details": check.get("details"),
            "check_timestamp": check.get("timestamp"),
        })

    df = pd.DataFrame(records)
    return bulk_load_dataframe(
        df, schema="analytics", table="validation_results"
    )


def load_rejected_records_to_db(rejected_csv: str) -> int:
    """
    Load rejected records from a CSV file into analytics.rejected_records.

    Args:
        rejected_csv (str): Path to the rejected records CSV.

    Returns:
        int: Number of records loaded.
    """
    csv_path = Path(rejected_csv)
    if not csv_path.exists():
        logger.warning("Rejected records CSV not found: %s", rejected_csv)
        return 0

    df = pd.read_csv(rejected_csv)
    if df.empty:
        logger.info("No rejected records to load.")
        return 0

    return bulk_load_dataframe(
        df, schema="analytics", table="rejected_records"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Master loader — runs all loading steps
# ─────────────────────────────────────────────────────────────────────────────

def run_full_load(
    config_path: str = None,
    run_id: Optional[str] = None,
) -> dict:
    """
    Execute the complete database loading pipeline.

    Steps:
        1. Load raw staging data → raw.encounters
        2. Load curated data → curated.encounters
        3. Populate analytics dimensional tables
        4. Load validation results and rejected records

    Args:
        config_path (str): Path to config.yaml.
        run_id (str | None): Pipeline run ID. Generated if None.

    Returns:
        dict: Row counts for each table loaded.
    """
    config = load_config(config_path)
    if run_id is None:
        run_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("DATABASE LOAD STARTED — run_id: %s", run_id)
    logger.info("=" * 60)

    results: dict = {}
    today = datetime.now().strftime("%Y-%m-%d")

    # Resolve file paths
    staging_csv = str(
        Path(config["paths"]["data_staging"]) / "diabetic_staging.csv"
    )
    curated_csv = str(
        Path(config["paths"]["data_curated"]) / "diabetic_curated.csv"
    )
    rejected_csv = str(
        Path(config["paths"]["data_rejected"]) / f"rejected_records_{today}.csv"
    )
    val_raw_json = str(
        Path(config["paths"]["data_logs"]) / f"validation_results_raw_{today}.json"
    )
    val_clean_json = str(
        Path(config["paths"]["data_logs"]) / f"validation_results_cleaned_{today}.json"
    )

    # Step 1: Raw
    if Path(staging_csv).exists():
        results["raw_encounters"] = load_raw_encounters(staging_csv, run_id)
    else:
        logger.warning("Staging CSV not found, skipping raw.encounters load")
        results["raw_encounters"] = 0

    # Step 2: Curated
    if Path(curated_csv).exists():
        results["curated_encounters"] = load_curated_encounters(curated_csv, run_id)
    else:
        raise FileNotFoundError(
            f"Curated CSV not found: {curated_csv}\n"
            "Run the ETL pipeline steps first."
        )

    # Step 3: Analytics dimensional model
    analytics_counts = load_analytics_tables(curated_csv)
    results.update(analytics_counts)

    # Step 4: Validation results and rejected records
    results["validation_raw"] = load_validation_results_to_db(
        val_raw_json, run_id, "raw"
    )
    results["validation_cleaned"] = load_validation_results_to_db(
        val_clean_json, run_id, "cleaned"
    )
    results["rejected_records"] = load_rejected_records_to_db(rejected_csv)

    logger.info("=" * 60)
    logger.info("DATABASE LOAD COMPLETE: %s", results)
    logger.info("=" * 60)

    return results


# Optional type hint import
from typing import Optional


if __name__ == "__main__":
    result = run_full_load()
    print("\n=== DATABASE LOAD RESULT ===")
    for table, count in result.items():
        print(f"  {table:<30}: {count} rows")
    print("============================\n")
