"""
data_quality.py
---------------
Automated data-quality validation for the Hospital Readmission project.

What this module does:
    Runs a suite of checks against both the raw/staging dataset and the
    cleaned/processed dataset.  Each check produces a structured result
    (check_name, status, description, failed_count, timestamp) that is
    saved to a JSON log and optionally to PostgreSQL.

Why we need it:
    Data quality issues can silently corrupt analysis and model training.
    Automated checks ensure we detect problems early and document them.
    The project requirement states: "Apply data-quality checks and produce
    a rejected-record or error log."

Input:
    data/staging/diabetic_staging.csv   (raw validation)
    data/processed/diabetic_cleaned.csv (post-cleaning validation)

Processing:
    1. Check required columns exist
    2. Check null percentages per column
    3. Check for duplicate rows
    4. Check numeric range validity
    5. Check categorical value validity
    6. Check row count is non-zero
    7. Check schema consistency
    8. Check readmission label distribution

Output:
    data/logs/validation_results_<date>.json
    Returns a list of check result dicts

Viva questions:
    Q: What is data validation and why is it separate from cleaning?
    A: Cleaning modifies the data (fills, removes, transforms).
       Validation CHECKS the data against rules and reports findings —
       it does not change the data.  Running validation after cleaning
       confirms the cleaning worked correctly.

    Q: What happens when a validation check fails?
    A: The check is recorded with status='FAIL' and failed_count > 0.
       The pipeline can be configured to either warn or halt depending
       on the severity (e.g. missing required column = halt).
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
# Configuration — defined first so it can be called by all functions below
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Check result builder
# ─────────────────────────────────────────────────────────────────────────────

def make_check_result(
    check_name: str,
    status: str,
    description: str,
    failed_count: int = 0,
    details: Optional[str] = None,
) -> dict:
    """
    Build a standardised check result dictionary.

    Args:
        check_name (str): Short name for the check (e.g. 'required_columns').
        status (str): 'PASS', 'FAIL', or 'WARN'.
        description (str): Human-readable explanation of what was checked.
        failed_count (int): Number of records/columns that failed.
        details (str | None): Additional detail for failures.

    Returns:
        dict: Standardised check result.
    """
    return {
        "check_name": check_name,
        "status": status,
        "description": description,
        "failed_count": failed_count,
        "details": details,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_required_columns(df: pd.DataFrame, required_cols: list) -> dict:
    """
    Verify that all required columns are present in the DataFrame.
    This is a hard failure — the pipeline should not continue without them.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return make_check_result(
            check_name="required_columns",
            status="FAIL",
            description=f"Required columns are missing from the dataset",
            failed_count=len(missing),
            details=f"Missing columns: {missing}",
        )
    return make_check_result(
        check_name="required_columns",
        status="PASS",
        description=f"All {len(required_cols)} required columns are present",
    )


def check_row_count(df: pd.DataFrame, min_rows: int = 1000) -> dict:
    """
    Verify the DataFrame has more than the minimum expected number of rows.
    Prevents running the pipeline on an empty or truncated file.
    """
    actual = len(df)
    if actual < min_rows:
        return make_check_result(
            check_name="row_count",
            status="FAIL",
            description=f"Dataset has fewer than {min_rows} rows",
            failed_count=min_rows - actual,
            details=f"Actual row count: {actual}",
        )
    return make_check_result(
        check_name="row_count",
        status="PASS",
        description=f"Row count {actual} meets minimum {min_rows}",
    )


def check_null_percentages(
    df: pd.DataFrame, threshold_pct: float = 50.0
) -> dict:
    """
    Check if any column has more than threshold_pct% null values.
    A column with >50% nulls may be unusable for analysis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold_pct (float): Percentage above which a column is flagged.
    """
    null_pcts = (df.isnull().mean() * 100).round(2)
    high_null_cols = null_pcts[null_pcts > threshold_pct].to_dict()

    if high_null_cols:
        return make_check_result(
            check_name="null_percentages",
            status="WARN",
            description=(
                f"Columns with >{threshold_pct}% null values found"
            ),
            failed_count=len(high_null_cols),
            details=str(high_null_cols),
        )
    return make_check_result(
        check_name="null_percentages",
        status="PASS",
        description=(
            f"No column exceeds {threshold_pct}% null values"
        ),
    )


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Check for duplicate rows in the DataFrame."""
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        return make_check_result(
            check_name="duplicate_rows",
            status="FAIL",
            description="Duplicate rows found in the dataset",
            failed_count=dup_count,
            details=f"{dup_count} duplicate rows detected",
        )
    return make_check_result(
        check_name="duplicate_rows",
        status="PASS",
        description="No duplicate rows found",
    )


def check_numeric_ranges(df: pd.DataFrame, numeric_ranges: dict) -> dict:
    """
    Validate that numeric columns fall within defined min/max ranges.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_ranges (dict): Maps column names to {'min': x, 'max': y}.
    """
    violations: dict = {}
    for col, bounds in numeric_ranges.items():
        if col not in df.columns:
            continue
        col_data = pd.to_numeric(df[col], errors="coerce")
        out_of_range = (
            (col_data < bounds["min"]) | (col_data > bounds["max"])
        ).sum()
        if out_of_range > 0:
            violations[col] = {
                "out_of_range_count": int(out_of_range),
                "expected_min": bounds["min"],
                "expected_max": bounds["max"],
                "actual_min": float(col_data.min()),
                "actual_max": float(col_data.max()),
            }

    if violations:
        return make_check_result(
            check_name="numeric_ranges",
            status="WARN",
            description="Some numeric columns contain out-of-range values",
            failed_count=sum(v["out_of_range_count"] for v in violations.values()),
            details=str(violations),
        )
    return make_check_result(
        check_name="numeric_ranges",
        status="PASS",
        description="All numeric columns are within expected ranges",
    )


def check_categorical_values(df: pd.DataFrame) -> dict:
    """
    Validate key categorical columns contain only expected values.

    Checks:
        - gender: Must be 'Male' or 'Female' (or NaN)
        - readmitted: Must be '<30', '>30', or 'NO' (or NaN)
        - age: Must be one of the ten UCI age brackets
    """
    violations: list = []

    valid_gender = {"Male", "Female"}
    if "gender" in df.columns:
        invalid_gender = ~df["gender"].isin(valid_gender) & df["gender"].notna()
        if invalid_gender.sum() > 0:
            violations.append(
                f"gender: {invalid_gender.sum()} invalid values "
                f"(expected {valid_gender})"
            )

    valid_readmitted = {"<30", ">30", "NO"}
    if "readmitted" in df.columns:
        invalid_ra = (
            ~df["readmitted"].isin(valid_readmitted) & df["readmitted"].notna()
        )
        if invalid_ra.sum() > 0:
            violations.append(
                f"readmitted: {invalid_ra.sum()} invalid values "
                f"(expected {valid_readmitted})"
            )

    valid_ages = {
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
    }
    if "age" in df.columns:
        invalid_age = ~df["age"].isin(valid_ages) & df["age"].notna()
        if invalid_age.sum() > 0:
            violations.append(
                f"age: {invalid_age.sum()} invalid brackets"
            )

    if violations:
        return make_check_result(
            check_name="categorical_values",
            status="FAIL",
            description="Invalid categorical values found",
            failed_count=len(violations),
            details="; ".join(violations),
        )
    return make_check_result(
        check_name="categorical_values",
        status="PASS",
        description="All key categorical columns contain valid values",
    )


def check_readmission_distribution(df: pd.DataFrame) -> dict:
    """
    Check that the readmitted column has all expected classes and report
    class distribution.  This is informational — warns if a class is missing.
    """
    if "readmitted" not in df.columns:
        return make_check_result(
            check_name="readmission_distribution",
            status="FAIL",
            description="Column 'readmitted' not found",
            failed_count=1,
        )

    dist = df["readmitted"].value_counts().to_dict()
    expected_classes = {"<30", ">30", "NO"}
    missing_classes = expected_classes - set(dist.keys())

    status = "WARN" if missing_classes else "PASS"
    return make_check_result(
        check_name="readmission_distribution",
        status=status,
        description=f"Readmission class distribution: {dist}",
        failed_count=len(missing_classes),
        details=f"Missing classes: {missing_classes}" if missing_classes else None,
    )


def check_schema_consistency(
    df: pd.DataFrame, expected_min_columns: int = 40
) -> dict:
    """
    Check that the DataFrame has at least the expected number of columns.
    The UCI dataset has 50 columns; significantly fewer suggests a parsing error.
    """
    actual = len(df.columns)
    if actual < expected_min_columns:
        return make_check_result(
            check_name="schema_consistency",
            status="FAIL",
            description=(
                f"DataFrame has only {actual} columns "
                f"(expected at least {expected_min_columns})"
            ),
            failed_count=expected_min_columns - actual,
            details=f"Columns found: {list(df.columns)}",
        )
    return make_check_result(
        check_name="schema_consistency",
        status="PASS",
        description=f"Schema has {actual} columns — meets minimum {expected_min_columns}",
    )


def check_surrogate_keys(df: pd.DataFrame) -> dict:
    """
    Verify surrogate keys were created and are unique (no collisions).
    Only runs on the cleaned dataset after cleaning step.
    """
    results = []
    for key_col in ["encounter_surrogate_id", "patient_surrogate_id"]:
        if key_col not in df.columns:
            results.append(f"{key_col}: column not found")
            continue
        null_count = df[key_col].isnull().sum()
        dup_count = df[key_col].duplicated().sum()

        # encounter_surrogate_id should be unique (one row per encounter)
        if key_col == "encounter_surrogate_id" and dup_count > 0:
            results.append(
                f"{key_col}: {dup_count} duplicate surrogate keys (collision risk)"
            )
        if null_count > 0:
            results.append(f"{key_col}: {null_count} null values")

    if results:
        return make_check_result(
            check_name="surrogate_keys",
            status="WARN",
            description="Surrogate key issues detected",
            failed_count=len(results),
            details="; ".join(results),
        )
    return make_check_result(
        check_name="surrogate_keys",
        status="PASS",
        description="Surrogate keys are present and unique",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    csv_path: Optional[str] = None,
    dataset_label: str = "raw",
    config_path: str = None,
    run_id: Optional[str] = None,
    halt_on_critical_failure: bool = True,
) -> List[dict]:
    """
    Run the full suite of data-quality checks on a dataset file.

    Args:
        csv_path (str | None): Path to CSV. If None, uses config default.
        dataset_label (str): 'raw' or 'cleaned' — controls which checks run.
        config_path (str): Path to config.yaml.
        run_id (str | None): Pipeline run identifier.
        halt_on_critical_failure (bool): If True, raises on FAIL status for
            required-column checks.

    Returns:
        List[dict]: All check results.
    """
    config = load_config(config_path)
    if run_id is None:
        run_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("VALIDATION STARTED — label: %s | run_id: %s", dataset_label, run_id)
    logger.info("=" * 60)

    # Resolve file path
    if csv_path is None:
        if dataset_label == "raw":
            csv_path = str(
                Path(config["paths"]["data_staging"]) / "diabetic_staging.csv"
            )
        else:
            csv_path = str(
                Path(config["paths"]["data_processed"]) / "diabetic_cleaned.csv"
            )

    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Validation input file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded %d rows × %d cols for validation", len(df), len(df.columns))

    results: list = []

    # ── Run checks ────────────────────────────────────────────────────────────
    results.append(check_row_count(df, min_rows=1000))
    results.append(
        check_required_columns(df, config["etl"]["required_columns"])
    )
    results.append(check_schema_consistency(df))
    results.append(check_duplicate_rows(df))
    results.append(check_null_percentages(df))
    results.append(
        check_numeric_ranges(df, config["etl"]["numeric_ranges"])
    )
    results.append(check_categorical_values(df))
    results.append(check_readmission_distribution(df))

    # Extra checks on cleaned dataset
    if dataset_label == "cleaned":
        results.append(check_surrogate_keys(df))

    # ── Log summary ───────────────────────────────────────────────────────────
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")

    logger.info(
        "Validation summary — PASS: %d | FAIL: %d | WARN: %d",
        pass_count, fail_count, warn_count,
    )
    for r in results:
        level = logger.error if r["status"] == "FAIL" else (
            logger.warning if r["status"] == "WARN" else logger.info
        )
        level(
            "  [%s] %s — %s (failed_count=%d)",
            r["status"], r["check_name"], r["description"], r["failed_count"],
        )

    # ── Save results to JSON ──────────────────────────────────────────────────
    save_validation_results(results, run_id, dataset_label, config)

    # ── Critical failure check ────────────────────────────────────────────────
    if halt_on_critical_failure:
        critical_fails = [
            r for r in results
            if r["status"] == "FAIL"
            and r["check_name"] in ("required_columns", "row_count", "schema_consistency")
        ]
        if critical_fails:
            raise ValueError(
                f"Critical data-quality failures detected: "
                f"{[r['check_name'] for r in critical_fails]}. "
                "Pipeline halted. Review the validation report."
            )

    return results


def save_validation_results(
    results: list,
    run_id: str,
    dataset_label: str,
    config: dict,
) -> str:
    """Save validation results to a dated JSON log file."""
    log_dir = Path(config["paths"]["data_logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"validation_results_{dataset_label}_{today}.json"

    output = {
        "run_id": run_id,
        "dataset_label": dataset_label,
        "generated_at": datetime.now().isoformat(),
        "total_checks": len(results),
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "fail_count": sum(1 for r in results if r["status"] == "FAIL"),
        "warn_count": sum(1 for r in results if r["status"] == "WARN"),
        "checks": results,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Validation results saved: %s", log_file)
    return str(log_file)


# ─────────────────────────────────────────────────────────────────────────────
# Optional import guard
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional


if __name__ == "__main__":
    import sys

    label = sys.argv[1] if len(sys.argv) > 1 else "raw"
    print(f"\nRunning validation on {label} dataset...")
    results = run_validation(dataset_label=label)
    print(f"\n=== VALIDATION SUMMARY ({label.upper()}) ===")
    for r in results:
        icon = "✓" if r["status"] == "PASS" else ("✗" if r["status"] == "FAIL" else "!")
        print(f"  [{r['status']:4}] {icon} {r['check_name']}: {r['description']}")
    print("=" * 50 + "\n")
