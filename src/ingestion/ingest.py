"""
ingest.py
---------
Dataset ingestion module for the Hospital Readmission project.

What this module does:
    Reads the raw UCI Diabetes CSV file, preserves an unmodified copy in
    data/raw/, records ingestion metadata (timestamp, row count, status),
    and produces a light staging copy for downstream processing.

Why we need it:
    The project requirement states we must maintain a raw copy of extracted
    data, record extraction date/source/status/row count, handle errors, and
    never manually edit the analytical dataset.

Input:
    data/raw/diabetic_data.csv  (downloaded by student from UCI/Kaggle)

Processing:
    1. Validate the file exists and is readable
    2. Load the CSV with Pandas
    3. Record ingestion metadata
    4. Save metadata to JSON log and PostgreSQL (if available)
    5. Write a staging copy to data/staging/

Output:
    data/staging/diabetic_staging.csv
    data/logs/ingestion_metadata_<date>.json

Connects to:
    src/preprocessing/clean_data.py  (next pipeline step)

Viva questions:
    Q: Why do we keep a raw copy?
    A: So we can always re-run the pipeline from the original source without
       re-downloading, and to prove we haven't manually altered the data.

    Q: What is ingestion metadata?
    A: A record of WHEN data was ingested, from WHERE, HOW MANY rows, and
       whether the ingestion succeeded or failed — useful for auditing.
"""

import json
import os
import sys
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Add project root to sys.path so `src.*` imports work when running
#    this file directly (python src/ingestion/ingest.py) or as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(_PROJECT_ROOT / ".env")

# Import the shared logger
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load the central YAML configuration file."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion metadata
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(
    run_id: str,
    source_name: str,
    file_name: str,
    status: str,
    row_count: int,
    column_count: int,
    error_message: Optional[str] = None,
) -> dict:
    """
    Build a metadata dictionary for one ingestion run.

    Args:
        run_id (str): Unique identifier for this pipeline run.
        source_name (str): Human-readable name of the data source.
        file_name (str): Name of the ingested file.
        status (str): 'SUCCESS' or 'FAILED'.
        row_count (int): Number of rows in the ingested file.
        column_count (int): Number of columns in the ingested file.
        error_message (str | None): Error detail if status is 'FAILED'.

    Returns:
        dict: Metadata dictionary ready for JSON serialisation.
    """
    return {
        "run_id": run_id,
        "source_name": source_name,
        "extraction_timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "status": status,
        "row_count": row_count,
        "column_count": column_count,
        "error_message": error_message,
    }


def save_metadata_to_json(metadata: dict, log_dir: str = "data/logs") -> str:
    """
    Save ingestion metadata to a dated JSON file.

    Args:
        metadata (dict): The metadata dictionary to save.
        log_dir (str): Directory for log files.

    Returns:
        str: Path to the written JSON file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_path / f"ingestion_metadata_{today}.json"

    # Append to existing records if file already exists
    records: list = []
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []

    records.append(metadata)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info("Ingestion metadata saved: %s", log_file)
    return str(log_file)


def save_metadata_to_postgres(metadata: dict) -> None:
    """
    Save ingestion metadata to the PostgreSQL ingestion_metadata table.
    This is optional — if the DB is not available it logs a warning and
    continues (the JSON log is always written as the primary record).

    Args:
        metadata (dict): Ingestion metadata dictionary.
    """
    try:
        # Import here to avoid circular import and allow offline usage
        from src.database.connection import get_engine
        from sqlalchemy import text

        engine = get_engine()
        insert_sql = text("""
            INSERT INTO analytics.ingestion_metadata
                (run_id, source_name, extraction_timestamp, file_name,
                 status, row_count, column_count, error_message)
            VALUES
                (:run_id, :source_name, :extraction_timestamp, :file_name,
                 :status, :row_count, :column_count, :error_message)
            ON CONFLICT (run_id) DO NOTHING
        """)
        with engine.connect() as conn:
            conn.execute(insert_sql, metadata)
            conn.commit()
        logger.info("Ingestion metadata saved to PostgreSQL.")
    except Exception as exc:
        # Non-fatal: log the warning but do not abort ingestion
        logger.warning(
            "Could not save metadata to PostgreSQL (DB may not be ready): %s", exc
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core ingestion
# ─────────────────────────────────────────────────────────────────────────────

def preserve_raw_copy(source_path: Path, raw_dir: str = "data/raw") -> Path:
    """
    Ensure a raw backup of the original file exists in data/raw/.
    If the file is already in data/raw/ this is a no-op.

    Args:
        source_path (Path): Path to the source CSV.
        raw_dir (str): Target raw data directory.

    Returns:
        Path: Path to the raw copy.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    destination = raw_path / source_path.name

    if destination.resolve() == source_path.resolve():
        logger.info("Source file is already in raw directory: %s", destination)
        return destination

    shutil.copy2(source_path, destination)
    logger.info("Raw copy preserved: %s → %s", source_path, destination)
    return destination


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load the UCI Diabetes CSV into a Pandas DataFrame.

    The UCI dataset uses '?' as a missing-value marker throughout.
    We load it as-is at this stage and let the cleaning step handle
    replacement so we preserve a faithful raw copy.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with original column names.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV cannot be parsed.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"DATASET NOT FOUND: {file_path}\n"
            f"{'='*60}\n"
            "Please download the UCI Diabetes 130-US Hospitals dataset:\n"
            "  URL : https://www.kaggle.com/datasets/brandao/diabetes\n"
            "  File: diabetic_data.csv\n"
            f"  Place it in: data/raw/diabetic_data.csv\n"
            f"{'='*60}"
        )

    logger.info("Loading CSV: %s", file_path)

    try:
        # UCI Diabetes dataset is tab-separated ('\t'), not comma-separated.
        # We detect the separator automatically using Python's csv.Sniffer,
        # falling back to tab if detection fails.
        import csv
        with open(file_path, "r", encoding="utf-8") as sample_file:
            sample = sample_file.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            sep = dialect.delimiter
        except csv.Error:
            sep = "\t"  # UCI dataset default
        logger.info("Detected CSV separator: %r", sep)
        df = pd.read_csv(file_path, sep=sep, low_memory=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse CSV file '{file_path}': {exc}"
        ) from exc

    logger.info("CSV loaded: %d rows × %d columns", len(df), len(df.columns))
    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Generate a lightweight data profile of the raw DataFrame.

    Args:
        df (pd.DataFrame): The raw dataset.

    Returns:
        dict: Profile summary (shapes, dtypes, null counts, sample values).
    """
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "question_mark_counts": {
            col: int((df[col] == "?").sum())
            for col in df.select_dtypes(include="str").columns
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 2),
    }

    # Log the most important stats
    logger.info(
        "Profile — rows: %d | cols: %d | duplicates: %d | memory: %.1f MB",
        profile["row_count"],
        profile["column_count"],
        profile["duplicate_rows"],
        profile["memory_usage_mb"],
    )

    # Log columns with '?' missing values
    for col, count in profile["question_mark_counts"].items():
        if count > 0:
            pct = count / len(df) * 100
            logger.debug("  Column '%s' has %d '?' values (%.1f%%)", col, count, pct)

    return profile


def save_staging_copy(df: pd.DataFrame, staging_dir: str = "data/staging") -> Path:
    """
    Save a staging copy of the raw data (unmodified) for pipeline use.
    This is the file that downstream steps will read — the raw copy in
    data/raw/ is never touched again.

    Args:
        df (pd.DataFrame): The raw DataFrame.
        staging_dir (str): Directory for staging files.

    Returns:
        Path: Path to the staging CSV.
    """
    staging_path = Path(staging_dir)
    staging_path.mkdir(parents=True, exist_ok=True)

    output_file = staging_path / "diabetic_staging.csv"
    df.to_csv(output_file, index=False)
    logger.info("Staging copy saved: %s (%d rows)", output_file, len(df))
    return output_file


def save_profile(profile: dict, log_dir: str = "data/logs") -> str:
    """Save the data profile to a JSON file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    profile_file = log_path / f"data_profile_{today}.json"

    with open(profile_file, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)

    logger.info("Data profile saved: %s", profile_file)
    return str(profile_file)


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    raw_csv_path: Optional[str] = None,
    config_path: str = None,
    save_to_db: bool = True,
) -> dict:
    """
    Run the full ingestion pipeline.

    Steps:
        1. Load configuration
        2. Resolve file path
        3. Load CSV
        4. Preserve raw copy
        5. Profile the data
        6. Save staging copy
        7. Record and persist metadata

    Args:
        raw_csv_path (str | None): Path to the CSV. If None, uses config default.
        config_path (str): Path to config.yaml.
        save_to_db (bool): Whether to also save metadata to PostgreSQL.

    Returns:
        dict: Contains 'metadata', 'profile', 'staging_path' keys.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the CSV cannot be parsed.
    """
    config = load_config(config_path)
    run_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("INGESTION STARTED — run_id: %s", run_id)
    logger.info("=" * 60)

    # Resolve dataset file path
    if raw_csv_path is None:
        dataset_file = config["dataset"]["primary_file"]
        raw_dir = config["paths"]["data_raw"]
        raw_csv_path = str(Path(raw_dir) / dataset_file)

    file_path = Path(raw_csv_path)
    source_name = config["dataset"]["name"]

    status = "SUCCESS"
    error_message = None
    df = pd.DataFrame()
    profile: dict = {}
    staging_path = ""

    try:
        # ── Step 1: Load CSV ────────────────────────────────────────────────
        df = load_csv(file_path)

        # ── Step 2: Preserve raw copy (if not already in raw/) ─────────────
        preserve_raw_copy(file_path, raw_dir=config["paths"]["data_raw"])

        # ── Step 3: Profile the raw data ───────────────────────────────────
        profile = profile_dataframe(df)

        # ── Step 4: Save staging copy ──────────────────────────────────────
        staging_file = save_staging_copy(df, staging_dir=config["paths"]["data_staging"])
        staging_path = str(staging_file)

        # ── Step 5: Save data profile ──────────────────────────────────────
        save_profile(profile, log_dir=config["paths"]["data_logs"])

        logger.info("Ingestion completed successfully.")

    except FileNotFoundError as exc:
        status = "FAILED"
        error_message = str(exc)
        logger.error("INGESTION FAILED — file not found: %s", exc)
        raise  # re-raise so the DAG task fails properly

    except ValueError as exc:
        status = "FAILED"
        error_message = str(exc)
        logger.error("INGESTION FAILED — CSV parse error: %s", exc)
        raise

    except Exception as exc:
        status = "FAILED"
        error_message = f"Unexpected error: {exc}"
        logger.error("INGESTION FAILED — unexpected error: %s", exc)
        raise

    finally:
        # ── Always record metadata ──────────────────────────────────────────
        metadata = build_metadata(
            run_id=run_id,
            source_name=source_name,
            file_name=file_path.name,
            status=status,
            row_count=len(df) if not df.empty else 0,
            column_count=len(df.columns) if not df.empty else 0,
            error_message=error_message,
        )

        log_file = save_metadata_to_json(
            metadata, log_dir=config["paths"]["data_logs"]
        )
        logger.info("Metadata log: %s", log_file)

        if save_to_db:
            save_metadata_to_postgres(metadata)

        logger.info("=" * 60)
        logger.info("INGESTION %s — rows: %d", status, len(df))
        logger.info("=" * 60)

    return {
        "run_id": run_id,
        "metadata": metadata,
        "profile": profile,
        "staging_path": staging_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_ingestion()
    print("\n=== INGESTION RESULT ===")
    print(f"  Run ID       : {result['run_id']}")
    print(f"  Status       : {result['metadata']['status']}")
    print(f"  Rows         : {result['metadata']['row_count']}")
    print(f"  Columns      : {result['metadata']['column_count']}")
    print(f"  Staging file : {result['staging_path']}")
    print("========================\n")
