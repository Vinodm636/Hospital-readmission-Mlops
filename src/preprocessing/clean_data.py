"""
clean_data.py
-------------
Data cleaning pipeline for the Hospital Readmission project.

What this module does:
    Takes the raw staging dataset and applies a series of cleaning steps:
    - Replaces '?' and other special missing-value codes with NaN
    - Standardises column names (lowercase, no spaces)
    - Removes exact duplicate rows
    - Removes invalid/incomplete encounters (missing required fields)
    - Standardises categorical field values
    - Creates patient-safe surrogate keys
    - Logs every record rejection with a reason

Why we need it:
    The UCI dataset uses '?' as a missing-value marker and has various
    data-quality issues. We must clean this data before analysis or
    machine-learning, and we must document every record we discard.

Input:
    data/staging/diabetic_staging.csv

Processing:
    1. Load staging CSV
    2. Replace special missing-value codes
    3. Standardise column names
    4. Remove duplicate rows
    5. Remove invalid encounters
    6. Standardise categorical values
    7. Create surrogate keys
    8. Save cleaned data to data/processed/
    9. Save rejected records to data/rejected/

Output:
    data/processed/diabetic_cleaned.csv
    data/rejected/rejected_records_<date>.csv

Connects to:
    src/transformation/feature_engineering.py (next step)

Viva questions:
    Q: What are surrogate keys and why do we create them?
    A: Surrogate keys are system-generated identifiers (e.g. hashed IDs)
       that replace the original identifiers. We do this to avoid
       exposing any patient identifier that might link to real persons,
       even in a de-identified dataset.

    Q: Why do we log rejected records instead of silently dropping them?
    A: Data provenance — we need to explain why the final dataset size
       is different from the raw data, and identify any systematic bias
       in what was excluded.
"""

import hashlib
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ── Project root path setup ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Replace special missing-value codes
# ─────────────────────────────────────────────────────────────────────────────

def replace_missing_value_codes(df: pd.DataFrame, missing_codes: list) -> pd.DataFrame:
    """
    Replace special missing-value codes (e.g. '?') with np.nan.

    The UCI Diabetes dataset uses '?' throughout to indicate a missing or
    unknown value.  We replace it with NaN so Pandas missing-value methods
    work correctly.

    Args:
        df (pd.DataFrame): Input DataFrame.
        missing_codes (list): List of strings to treat as missing (e.g. ['?']).

    Returns:
        pd.DataFrame: DataFrame with missing codes replaced by np.nan.
    """
    before = df.isnull().sum().sum()

    for code in missing_codes:
        df = df.replace(code, np.nan)

    after = df.isnull().sum().sum()
    logger.info(
        "Missing-value codes replaced: %d new NaN values created", after - before
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Standardise column names
# ─────────────────────────────────────────────────────────────────────────────

def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names: lowercase, strip whitespace, replace spaces/hyphens
    with underscores.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardised column names.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    logger.info("Column names standardised: %s", list(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Remove duplicate rows
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(
    df: pd.DataFrame,
    run_id: str,
    rejected_records: list,
) -> pd.DataFrame:
    """
    Remove exact duplicate rows and log them as rejected records.

    Args:
        df (pd.DataFrame): Input DataFrame.
        run_id (str): Current pipeline run identifier.
        rejected_records (list): Accumulator list for rejected records.

    Returns:
        pd.DataFrame: DataFrame without duplicates.
    """
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        duplicate_df = df[duplicate_mask].copy()
        for _, row in duplicate_df.iterrows():
            rejected_records.append({
                "run_id": run_id,
                "record_id": str(row.get("encounter_id", "unknown")),
                "rejection_reason": "DUPLICATE_ROW",
                "rejection_timestamp": datetime.now().isoformat(),
                "details": "Exact duplicate of an earlier record",
            })
        df = df[~duplicate_mask].reset_index(drop=True)
        logger.info("Removed %d duplicate rows", duplicate_count)
    else:
        logger.info("No duplicate rows found")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Remove invalid encounters
# ─────────────────────────────────────────────────────────────────────────────

def remove_invalid_encounters(
    df: pd.DataFrame,
    run_id: str,
    rejected_records: list,
    required_columns: list,
) -> pd.DataFrame:
    """
    Remove encounters that are missing required fields or have impossible values.

    Rejection criteria:
        1. Missing encounter_id or patient_nbr (cannot identify the record)
        2. Missing readmitted label (target variable — cannot use for analysis)
        3. Gender is not Male or Female (UCI has 'Unknown/Invalid' for some)
        4. time_in_hospital <= 0 (impossible)
        5. num_medications < 0 (impossible)

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        run_id (str): Pipeline run identifier.
        rejected_records (list): Accumulator for rejected records.
        required_columns (list): Columns that must be non-null.

    Returns:
        pd.DataFrame: DataFrame with invalid encounters removed.
    """
    original_count = len(df)

    def reject_rows(mask: pd.Series, reason: str, details: str) -> None:
        """Helper: add rows matching mask to rejected list."""
        rows = df[mask]
        for _, row in rows.iterrows():
            rejected_records.append({
                "run_id": run_id,
                "record_id": str(row.get("encounter_id", "unknown")),
                "rejection_reason": reason,
                "rejection_timestamp": datetime.now().isoformat(),
                "details": details,
            })

    # ── 1. Missing key identifier columns ──────────────────────────────────
    for col in ["encounter_id", "patient_nbr", "readmitted"]:
        if col in df.columns:
            mask = df[col].isnull()
            if mask.any():
                reject_rows(mask, "MISSING_REQUIRED_FIELD",
                            f"Required column '{col}' is null")
                df = df[~mask].reset_index(drop=True)

    # ── 2. Invalid gender (UCI uses 'Unknown/Invalid') ──────────────────────
    if "gender" in df.columns:
        valid_gender = {"Male", "Female"}
        mask = ~df["gender"].isin(valid_gender) & df["gender"].notna()
        if mask.any():
            reject_rows(mask, "INVALID_GENDER",
                        f"Gender not in {valid_gender}")
            df = df[~mask].reset_index(drop=True)

    # ── 3. Impossible numeric values ────────────────────────────────────────
    if "time_in_hospital" in df.columns:
        mask = df["time_in_hospital"].notna() & (df["time_in_hospital"] <= 0)
        if mask.any():
            reject_rows(mask, "INVALID_TIME_IN_HOSPITAL",
                        "time_in_hospital must be > 0")
            df = df[~mask].reset_index(drop=True)

    if "num_medications" in df.columns:
        mask = df["num_medications"].notna() & (df["num_medications"] < 0)
        if mask.any():
            reject_rows(mask, "INVALID_MEDICATION_COUNT",
                        "num_medications cannot be negative")
            df = df[~mask].reset_index(drop=True)

    removed = original_count - len(df)
    logger.info(
        "Invalid encounter removal: %d rows removed, %d rows retained",
        removed,
        len(df),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Standardise categorical fields
# ─────────────────────────────────────────────────────────────────────────────

def standardise_categorical_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise categorical fields for consistency:
    - Strip whitespace from string columns
    - Standardise the readmitted column to consistent labels
    - Map age brackets to midpoint numeric values for ML (kept as separate col)

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardised categoricals.
    """
    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Standardise the readmitted column values
    if "readmitted" in df.columns:
        # The UCI dataset uses '<30', '>30', 'NO'
        # Standardise any case variations
        df["readmitted"] = df["readmitted"].str.upper()
        df["readmitted"] = df["readmitted"].replace({
            "<30": "<30",
            ">30": ">30",
            "NO": "NO",
        })
        logger.info(
            "readmitted value counts: %s",
            df["readmitted"].value_counts().to_dict(),
        )

    # Create a numeric age midpoint column
    age_midpoint_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    if "age" in df.columns:
        df["age_midpoint"] = df["age"].map(age_midpoint_map)
        logger.info("Age midpoint column created")

    # Standardise discharge_disposition_id to integer where possible
    for col in ["admission_type_id", "discharge_disposition_id",
                "admission_source_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Create patient-safe surrogate keys
# ─────────────────────────────────────────────────────────────────────────────

def create_surrogate_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient-safe surrogate keys to replace the original identifiers.

    Why:
        Even though the UCI dataset is de-identified, we add an extra layer
        of privacy by replacing patient_nbr and encounter_id with hashed
        surrogate keys.  This is a standard data engineering practice for
        healthcare data.

    Approach:
        We use SHA-256 hashing of the original ID combined with a fixed
        project-level salt.  The same original ID always produces the same
        surrogate key (deterministic), which allows joining across tables
        without exposing the original value.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with surrogate key columns added.
    """
    # Project-level salt (not a secret — just makes keys project-specific)
    SALT = "hospital_readmission_project_v1"

    def hash_id(value: str) -> str:
        """Create a short deterministic surrogate key using SHA-256."""
        raw = f"{SALT}:{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]  # 16-char hex prefix

    if "encounter_id" in df.columns:
        df["encounter_surrogate_id"] = df["encounter_id"].astype(str).apply(hash_id)

    if "patient_nbr" in df.columns:
        df["patient_surrogate_id"] = df["patient_nbr"].astype(str).apply(hash_id)

    logger.info(
        "Surrogate keys created: encounter_surrogate_id, patient_surrogate_id"
    )
    logger.info(
        "Sample surrogate: encounter_id=%s → surrogate=%s",
        str(df["encounter_id"].iloc[0]) if "encounter_id" in df.columns else "N/A",
        df["encounter_surrogate_id"].iloc[0]
        if "encounter_surrogate_id" in df.columns
        else "N/A",
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Handle missing values in remaining columns
# ─────────────────────────────────────────────────────────────────────────────

def handle_remaining_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply targeted missing-value handling after the initial NaN replacement.

    Strategy per column type:
        - race: fill with 'Unknown' (common in healthcare analytics)
        - payer_code: fill with 'Unknown'
        - medical_specialty: fill with 'Unknown'
        - weight: high missingness in UCI — fill with 'Unknown' (kept as categorical)
        - diag_1/2/3: fill with '0' (no recorded diagnosis code)
        - numeric columns: leave as NaN (handled by ML imputer later)

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with targeted missing-value fills.
    """
    categorical_fill_unknown = ["race", "payer_code", "medical_specialty", "weight"]
    for col in categorical_fill_unknown:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col] = df[col].fillna("Unknown")
                logger.info("Filled %d NaN in '%s' with 'Unknown'", null_count, col)

    # Diagnosis codes — fill missing with '0' (no diagnosis)
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            null_count = df[diag_col].isnull().sum()
            if null_count > 0:
                df[diag_col] = df[diag_col].fillna("0")
                logger.info("Filled %d NaN in '%s' with '0'", null_count, diag_col)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_cleaned_data(
    df: pd.DataFrame, processed_dir: str = "data/processed"
) -> Path:
    """Save the cleaned DataFrame to the processed directory."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    output_file = processed_path / "diabetic_cleaned.csv"
    df.to_csv(output_file, index=False)
    logger.info(
        "Cleaned data saved: %s (%d rows × %d cols)",
        output_file, len(df), len(df.columns),
    )
    return output_file


def save_rejected_records(
    rejected_records: list, rejected_dir: str = "data/rejected"
) -> Path:
    """
    Save all rejected records to a dated CSV file.

    Args:
        rejected_records (list): List of rejection dictionaries.
        rejected_dir (str): Directory for rejected records.

    Returns:
        Path: Path to the written CSV file.
    """
    rejected_path = Path(rejected_dir)
    rejected_path.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = rejected_path / f"rejected_records_{today}.csv"

    if rejected_records:
        rejected_df = pd.DataFrame(rejected_records)
        rejected_df.to_csv(output_file, index=False)
        logger.info(
            "Rejected records saved: %s (%d records)", output_file, len(rejected_records)
        )
    else:
        # Create an empty file to confirm the step ran
        pd.DataFrame(
            columns=["run_id", "record_id", "rejection_reason",
                     "rejection_timestamp", "details"]
        ).to_csv(output_file, index=False)
        logger.info("No records rejected — empty file written: %s", output_file)

    return output_file


# ─────────────────────────────────────────────────────────────────────────────
# Main cleaning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning(
    staging_csv: Optional[str] = None,
    config_path: str = None,
    run_id: Optional[str] = None,
) -> dict:
    """
    Execute the full cleaning pipeline.

    Args:
        staging_csv (str | None): Path to staging CSV. Uses config default if None.
        config_path (str): Path to config.yaml.
        run_id (str | None): Pipeline run identifier. Generated if None.

    Returns:
        dict: Summary with 'cleaned_path', 'rejected_path', 'stats'.
    """
    config = load_config(config_path)
    if run_id is None:
        run_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("CLEANING STARTED — run_id: %s", run_id)
    logger.info("=" * 60)

    # Resolve staging file path
    if staging_csv is None:
        staging_csv = str(
            Path(config["paths"]["data_staging"]) / "diabetic_staging.csv"
        )

    staging_path = Path(staging_csv)
    if not staging_path.exists():
        raise FileNotFoundError(
            f"Staging file not found: {staging_csv}\n"
            "Run the ingestion step first: python src/ingestion/ingest.py"
        )

    logger.info("Loading staging data: %s", staging_path)
    df = pd.read_csv(staging_path, low_memory=False)
    raw_count = len(df)
    logger.info("Loaded %d rows from staging", raw_count)

    rejected_records: list = []

    # ── Apply cleaning steps in sequence ────────────────────────────────────
    df = replace_missing_value_codes(df, config["etl"]["missing_value_codes"])
    df = standardise_column_names(df)
    df = remove_duplicates(df, run_id, rejected_records)
    df = remove_invalid_encounters(
        df, run_id, rejected_records, config["etl"]["required_columns"]
    )
    df = standardise_categorical_fields(df)
    df = create_surrogate_keys(df)
    df = handle_remaining_missing_values(df)

    # ── Save outputs ─────────────────────────────────────────────────────────
    cleaned_file = save_cleaned_data(df, config["paths"]["data_processed"])
    rejected_file = save_rejected_records(rejected_records, config["paths"]["data_rejected"])

    stats = {
        "raw_row_count": raw_count,
        "cleaned_row_count": len(df),
        "rejected_count": len(rejected_records),
        "retention_rate_pct": round(len(df) / raw_count * 100, 2) if raw_count else 0,
    }

    logger.info("=" * 60)
    logger.info(
        "CLEANING COMPLETE — %d rows retained / %d rejected (%.1f%% retention)",
        stats["cleaned_row_count"],
        stats["rejected_count"],
        stats["retention_rate_pct"],
    )
    logger.info("=" * 60)

    return {
        "run_id": run_id,
        "cleaned_path": str(cleaned_file),
        "rejected_path": str(rejected_file),
        "stats": stats,
    }


# Optional import guard for type hints
from typing import Optional


if __name__ == "__main__":
    result = run_cleaning()
    print("\n=== CLEANING RESULT ===")
    print(f"  Run ID        : {result['run_id']}")
    print(f"  Cleaned file  : {result['cleaned_path']}")
    print(f"  Rejected file : {result['rejected_path']}")
    for k, v in result["stats"].items():
        print(f"  {k:<22}: {v}")
    print("=======================\n")
