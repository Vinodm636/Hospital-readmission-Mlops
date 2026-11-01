"""
feature_engineering.py
-----------------------
Feature engineering pipeline for the Hospital Readmission project.

What this module does:
    Creates new analytical and machine-learning features from the cleaned
    UCI Diabetes dataset.  Every engineered feature is documented with its
    definition and source columns.

Why we need it:
    The raw dataset columns are useful but limited. For example, we need
    to know if a patient was previously admitted (prior_admission_count),
    how many unique diagnoses they had (diagnosis_count), and what their
    readmission risk binary label is (readmission_binary).  Machine learning
    models also perform better when features are clearly defined.

Input:
    data/processed/diabetic_cleaned.csv

Feature Engineering:
    1.  readmission_binary         — target variable (1 = readmitted <30 days)
    2.  diagnosis_count            — number of non-zero diagnosis codes
    3.  medication_count           — alias for num_medications (for clarity)
    4.  prior_admission_count      — total prior admissions (outpatient + emergency + inpatient)
    5.  is_prior_admission         — binary flag for any prior admission
    6.  length_of_stay             — alias for time_in_hospital
    7.  is_long_stay               — flag for stays > 7 days
    8.  primary_diagnosis_category — ICD-9 code category for diag_1
    9.  has_diabetes_drug          — flag if any diabetes medication was prescribed
    10. total_service_utilisation  — outpatient + emergency + inpatient visits
    11. age_midpoint               — numeric midpoint of age bracket (already created in cleaning)

Output:
    data/curated/diabetic_curated.csv

Connects to:
    src/database/ (loading into PostgreSQL)
    ml/preprocessing.py (ML feature pipeline)

Viva questions:
    Q: What is feature engineering?
    A: Creating new informative columns from existing data to improve
       analysis and model performance. Example: 'diagnosis_count' is
       more useful than three separate diag_1/diag_2/diag_3 columns.

    Q: What is the readmission_binary target variable?
    A: It is 1 when a patient was readmitted within 30 days ('<30'),
       and 0 for '>30' or 'NO'. This converts the 3-class readmission
       label into a binary prediction problem — which is clinically most
       important.

    Q: Why create is_prior_admission?
    A: Prior hospital admissions are a known risk factor for readmission.
       Having a binary flag makes it easy to use in decision trees and
       logistic regression.
"""

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
# Feature 1: Binary readmission target
# ─────────────────────────────────────────────────────────────────────────────

def create_readmission_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary readmission target variable.

    Definition:
        readmission_binary = 1 if readmitted == '<30' (within 30 days)
        readmission_binary = 0 if readmitted == '>30' or 'NO'

    This is the standard binary classification target for the UCI dataset.
    Note: The dataset label is 'readmitted within 30 days' not a strict
    clinical 30-day readmission window — documented in the data dictionary.

    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'readmitted' column.

    Returns:
        pd.DataFrame: DataFrame with 'readmission_binary' column added.
    """
    if "readmitted" not in df.columns:
        raise ValueError("Column 'readmitted' not found — cannot create target variable.")

    df["readmission_binary"] = (df["readmitted"] == "<30").astype(int)

    rate = df["readmission_binary"].mean() * 100
    logger.info(
        "readmission_binary created: %d positive (%.1f%%), %d negative",
        df["readmission_binary"].sum(),
        rate,
        (df["readmission_binary"] == 0).sum(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Diagnosis count
# ─────────────────────────────────────────────────────────────────────────────

def create_diagnosis_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of non-missing, non-zero diagnosis codes per encounter.

    The UCI dataset provides three diagnosis columns (diag_1, diag_2, diag_3).
    We count how many are recorded for each encounter.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'diagnosis_count' column added.
    """
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    if not diag_cols:
        df["diagnosis_count"] = 0
        return df

    def count_valid_diags(row: pd.Series) -> int:
        """Count non-null, non-zero diagnosis codes in a row."""
        count = 0
        for col in diag_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() not in ("", "0", "nan"):
                count += 1
        return count

    df["diagnosis_count"] = df[diag_cols].apply(
        lambda row: sum(
            1 for v in row
            if pd.notna(v) and str(v).strip() not in ("", "0", "nan")
        ),
        axis=1,
    )
    logger.info(
        "diagnosis_count created — mean: %.2f, max: %d",
        df["diagnosis_count"].mean(),
        df["diagnosis_count"].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: Medication count (semantic alias)
# ─────────────────────────────────────────────────────────────────────────────

def create_medication_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clearly named medication_count feature.

    The UCI dataset has 'num_medications' (number of distinct medications
    administered).  We create a semantic alias for clarity in dashboards
    and ML feature lists.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'medication_count' column added.
    """
    if "num_medications" in df.columns:
        df["medication_count"] = df["num_medications"]
    else:
        df["medication_count"] = np.nan
    logger.info("medication_count created from num_medications")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 & 5: Prior admission count and binary flag
# ─────────────────────────────────────────────────────────────────────────────

def create_prior_admission_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prior-admission count and binary indicator.

    Definition:
        prior_admission_count = number_outpatient + number_emergency + number_inpatient
            (total hospital visits in the year prior to the current encounter)
        is_prior_admission = 1 if prior_admission_count > 0, else 0

    Prior admissions are a known clinical predictor of readmission risk.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with prior admission features added.
    """
    prior_cols = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"]
                  if c in df.columns]

    if prior_cols:
        df["prior_admission_count"] = df[prior_cols].fillna(0).sum(axis=1)
    else:
        df["prior_admission_count"] = 0

    df["is_prior_admission"] = (df["prior_admission_count"] > 0).astype(int)

    logger.info(
        "prior_admission_count: mean=%.2f | is_prior_admission rate=%.1f%%",
        df["prior_admission_count"].mean(),
        df["is_prior_admission"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 & 7: Length of stay alias and long-stay flag
# ─────────────────────────────────────────────────────────────────────────────

def create_length_of_stay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create length-of-stay feature and long-stay binary flag.

    Definitions:
        length_of_stay = time_in_hospital (number of days, 1–14 in UCI dataset)
        is_long_stay   = 1 if length_of_stay > 7 days, else 0

    Length of stay is a proxy for severity of illness and resource use.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with length-of-stay features.
    """
    if "time_in_hospital" in df.columns:
        df["length_of_stay"] = df["time_in_hospital"]
    else:
        df["length_of_stay"] = np.nan

    df["is_long_stay"] = (df["length_of_stay"] > 7).astype(int)

    logger.info(
        "length_of_stay: mean=%.2f days | is_long_stay rate=%.1f%%",
        df["length_of_stay"].mean() if "length_of_stay" in df.columns else 0,
        df["is_long_stay"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8: Primary diagnosis category (ICD-9)
# ─────────────────────────────────────────────────────────────────────────────

def create_primary_diagnosis_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the primary diagnosis (diag_1) ICD-9 code to a broad clinical category.

    This is an important feature because different disease groups have
    different readmission patterns.

    ICD-9 Category Mapping (simplified, standard clinical groupings):
        001–139  : Infectious & Parasitic Diseases
        140–239  : Neoplasms
        240–279  : Endocrine/Metabolic (includes Diabetes)
        280–289  : Blood Diseases
        290–319  : Mental Disorders
        320–389  : Nervous System
        390–459  : Circulatory System
        460–519  : Respiratory System
        520–579  : Digestive System
        580–629  : Genitourinary System
        630–679  : Pregnancy/Childbirth
        680–709  : Skin Diseases
        710–739  : Musculoskeletal
        740–759  : Congenital Anomalies
        760–779  : Perinatal Conditions
        780–799  : Symptoms/Signs
        800–999  : Injury & Poisoning
        E/V codes: External causes / Supplementary
        250.xx   : Diabetes (common in this dataset)

    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'diag_1' column.

    Returns:
        pd.DataFrame: DataFrame with 'primary_diagnosis_category' column.
    """
    def map_icd9_category(code: str) -> str:
        """Map an ICD-9 code string to a broad disease category."""
        if pd.isna(code) or str(code).strip() in ("0", ""):
            return "Unknown"
        code_str = str(code).strip().upper()

        # Handle E and V codes
        if code_str.startswith("E"):
            return "External_Causes"
        if code_str.startswith("V"):
            return "Supplementary"

        # Extract numeric prefix
        try:
            numeric_code = float(code_str.split(".")[0])
        except ValueError:
            return "Unknown"

        if 1 <= numeric_code <= 139:
            return "Infectious_Parasitic"
        elif 140 <= numeric_code <= 239:
            return "Neoplasms"
        elif 240 <= numeric_code <= 279:
            return "Endocrine_Metabolic"
        elif 280 <= numeric_code <= 289:
            return "Blood_Diseases"
        elif 290 <= numeric_code <= 319:
            return "Mental_Disorders"
        elif 320 <= numeric_code <= 389:
            return "Nervous_System"
        elif 390 <= numeric_code <= 459:
            return "Circulatory_System"
        elif 460 <= numeric_code <= 519:
            return "Respiratory_System"
        elif 520 <= numeric_code <= 579:
            return "Digestive_System"
        elif 580 <= numeric_code <= 629:
            return "Genitourinary"
        elif 630 <= numeric_code <= 679:
            return "Pregnancy_Childbirth"
        elif 680 <= numeric_code <= 709:
            return "Skin_Diseases"
        elif 710 <= numeric_code <= 739:
            return "Musculoskeletal"
        elif 740 <= numeric_code <= 759:
            return "Congenital"
        elif 760 <= numeric_code <= 779:
            return "Perinatal"
        elif 780 <= numeric_code <= 799:
            return "Symptoms_Signs"
        elif 800 <= numeric_code <= 999:
            return "Injury_Poisoning"
        else:
            return "Other"

    if "diag_1" in df.columns:
        df["primary_diagnosis_category"] = df["diag_1"].apply(map_icd9_category)
        logger.info(
            "primary_diagnosis_category distribution: %s",
            df["primary_diagnosis_category"].value_counts().head(5).to_dict(),
        )
    else:
        df["primary_diagnosis_category"] = "Unknown"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 9: Diabetes medication flag
# ─────────────────────────────────────────────────────────────────────────────

def create_diabetes_drug_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary flag indicating whether ANY diabetes-specific medication
    was prescribed during the encounter.

    Diabetes drug columns in the UCI dataset (all have values:
    'No', 'Down', 'Steady', 'Up'):
        metformin, repaglinide, nateglinide, chlorpropamide, glimepiride,
        acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone,
        rosiglitazone, acarbose, miglitol, troglitazone, tolazamide,
        examide, citoglipton, insulin, glyburide-metformin, glipizide-metformin,
        glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'has_diabetes_drug' column.
    """
    diabetes_drugs = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
        "miglitol", "troglitazone", "tolazamide", "insulin",
    ]
    # Use only columns that exist in the DataFrame
    drug_cols_present = [c for c in diabetes_drugs if c in df.columns]

    if drug_cols_present:
        # A drug is "prescribed" if the value is not 'No' and not null
        df["has_diabetes_drug"] = df[drug_cols_present].apply(
            lambda row: int(
                any(
                    str(v).strip() not in ("No", "nan", "")
                    for v in row
                    if pd.notna(v)
                )
            ),
            axis=1,
        )
    else:
        df["has_diabetes_drug"] = 0

    logger.info(
        "has_diabetes_drug: %d (%.1f%%) patients have a diabetes drug",
        df["has_diabetes_drug"].sum(),
        df["has_diabetes_drug"].mean() * 100,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 10: Total service utilisation
# ─────────────────────────────────────────────────────────────────────────────

def create_total_service_utilisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a total service utilisation score.

    Definition:
        total_service_utilisation = number_outpatient + number_emergency
                                    + number_inpatient + time_in_hospital

    This captures the overall intensity of healthcare use across all
    encounter types, not just the current hospitalisation.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'total_service_utilisation' column.
    """
    utilisation_cols = [
        c for c in
        ["number_outpatient", "number_emergency", "number_inpatient", "time_in_hospital"]
        if c in df.columns
    ]
    if utilisation_cols:
        df["total_service_utilisation"] = df[utilisation_cols].fillna(0).sum(axis=1)
    else:
        df["total_service_utilisation"] = 0

    logger.info(
        "total_service_utilisation: mean=%.2f, max=%.0f",
        df["total_service_utilisation"].mean(),
        df["total_service_utilisation"].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main feature engineering pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    cleaned_csv: Optional[str] = None,
    config_path: str = None,
    run_id: Optional[str] = None,
) -> dict:
    """
    Execute the full feature engineering pipeline.

    Args:
        cleaned_csv (str | None): Path to cleaned CSV. Uses config default if None.
        config_path (str): Path to config.yaml.
        run_id (str | None): Pipeline run identifier.

    Returns:
        dict: Summary with 'curated_path', 'feature_list', 'stats'.
    """
    config = load_config(config_path)
    if run_id is None:
        run_id = str(uuid.uuid4())

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING STARTED — run_id: %s", run_id)
    logger.info("=" * 60)

    # Resolve input file
    if cleaned_csv is None:
        cleaned_csv = str(
            Path(config["paths"]["data_processed"]) / "diabetic_cleaned.csv"
        )

    input_path = Path(cleaned_csv)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {cleaned_csv}\n"
            "Run the cleaning step first: python src/preprocessing/clean_data.py"
        )

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info("Loaded %d rows × %d cols for feature engineering", len(df), len(df.columns))

    initial_cols = set(df.columns)

    # ── Apply feature engineering in sequence ────────────────────────────────
    df = create_readmission_binary(df)
    df = create_diagnosis_count(df)
    df = create_medication_count(df)
    df = create_prior_admission_features(df)
    df = create_length_of_stay_features(df)
    df = create_primary_diagnosis_category(df)
    df = create_diabetes_drug_flag(df)
    df = create_total_service_utilisation(df)

    new_cols = list(set(df.columns) - initial_cols)
    logger.info("New features created (%d): %s", len(new_cols), sorted(new_cols))

    # ── Save curated output ───────────────────────────────────────────────────
    curated_dir = Path(config["paths"]["data_curated"])
    curated_dir.mkdir(parents=True, exist_ok=True)
    output_file = curated_dir / "diabetic_curated.csv"
    df.to_csv(output_file, index=False)

    logger.info(
        "Curated dataset saved: %s (%d rows × %d cols)",
        output_file, len(df), len(df.columns),
    )

    stats = {
        "input_rows": len(df),
        "total_columns": len(df.columns),
        "new_features": len(new_cols),
        "readmission_positive_count": int(df["readmission_binary"].sum())
        if "readmission_binary" in df.columns else 0,
        "readmission_rate_pct": round(
            df["readmission_binary"].mean() * 100, 2
        ) if "readmission_binary" in df.columns else 0,
    }

    logger.info("=" * 60)
    logger.info(
        "FEATURE ENGINEERING COMPLETE — %d new features, readmission rate: %.2f%%",
        len(new_cols),
        stats["readmission_rate_pct"],
    )
    logger.info("=" * 60)

    return {
        "run_id": run_id,
        "curated_path": str(output_file),
        "feature_list": sorted(new_cols),
        "stats": stats,
    }


# Optional type hint import
from typing import Optional


if __name__ == "__main__":
    result = run_feature_engineering()
    print("\n=== FEATURE ENGINEERING RESULT ===")
    print(f"  Run ID        : {result['run_id']}")
    print(f"  Curated file  : {result['curated_path']}")
    print(f"  New features  : {result['feature_list']}")
    for k, v in result["stats"].items():
        print(f"  {k:<30}: {v}")
    print("===================================\n")
