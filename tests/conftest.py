"""
conftest.py
-----------
Shared pytest fixtures for the Hospital Readmission test suite.

Fixtures defined here are automatically available to all test files
without needing to import them.

What is a pytest fixture?
    A fixture is a reusable setup function that provides test data or
    objects.  The @pytest.fixture decorator marks a function as a fixture.
    Tests receive fixtures by including the fixture name in their parameter list.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Add project root to Python path so src imports work during testing
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Minimal sample DataFrame that mimics the UCI Diabetes dataset structure
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    A minimal sample DataFrame that mimics the raw UCI Diabetes dataset.

    Contains:
    - 5 valid rows
    - 1 row with '?' missing-value codes
    - 1 duplicate row
    - 1 row with invalid gender
    - Various readmission classes

    This lets tests check cleaning and validation behaviour
    without needing the full 101k-row dataset.
    """
    data = {
        "encounter_id": [1001, 1002, 1003, 1004, 1002, 1005, 1006],
        "patient_nbr": [2001, 2002, 2003, 2004, 2002, 2005, 2006],
        "race": ["Caucasian", "AfricanAmerican", "?", "Asian", "AfricanAmerican", "Caucasian", "Hispanic"],
        "gender": ["Male", "Female", "Male", "Female", "Female", "Unknown/Invalid", "Male"],
        "age": ["[70-80)", "[60-70)", "[80-90)", "[50-60)", "[60-70)", "[40-50)", "[30-40)"],
        "weight": ["?", "?", "?", "?", "?", "?", "?"],
        "admission_type_id": [1, 1, 3, 2, 1, 1, 2],
        "discharge_disposition_id": [1, 1, 1, 1, 1, 1, 1],
        "admission_source_id": [7, 7, 7, 7, 7, 7, 7],
        "time_in_hospital": [3, 5, 7, 2, 5, 4, 6],
        "payer_code": ["MC", "MC", "?", "HM", "MC", "BC", "MC"],
        "medical_specialty": ["InternalMedicine", "?", "Cardiology", "?", "?", "Surgery", "?"],
        "num_lab_procedures": [41, 59, 33, 27, 59, 43, 31],
        "num_procedures": [0, 0, 1, 2, 0, 0, 3],
        "num_medications": [18, 13, 22, 8, 13, 16, 10],
        "number_outpatient": [0, 0, 0, 1, 0, 0, 0],
        "number_emergency": [0, 0, 0, 0, 0, 0, 1],
        "number_inpatient": [1, 2, 0, 0, 2, 1, 0],
        "diag_1": ["428", "250.01", "414", "?", "250.01", "401", "276"],
        "diag_2": ["250", "276", "250", "250", "276", "250", "250"],
        "diag_3": ["250", "58", "0", "0", "58", "0", "0"],
        "number_diagnoses": [9, 6, 7, 5, 6, 8, 4],
        "max_glu_serum": ["None", ">300", "None", "None", ">300", "None", "None"],
        "A1Cresult": [">8", "None", "None", "None", "None", ">8", "None"],
        "metformin": ["No", "No", "No", "Steady", "No", "No", "No"],
        "repaglinide": ["No"] * 7,
        "nateglinide": ["No"] * 7,
        "chlorpropamide": ["No"] * 7,
        "glimepiride": ["No"] * 7,
        "acetohexamide": ["No"] * 7,
        "glipizide": ["Steady", "No", "No", "No", "No", "No", "Up"],
        "glyburide": ["No"] * 7,
        "tolbutamide": ["No"] * 7,
        "pioglitazone": ["No"] * 7,
        "rosiglitazone": ["No"] * 7,
        "acarbose": ["No"] * 7,
        "miglitol": ["No"] * 7,
        "troglitazone": ["No"] * 7,
        "tolazamide": ["No"] * 7,
        "examide": ["No"] * 7,
        "citoglipton": ["No"] * 7,
        "insulin": ["Up", "No", "Steady", "No", "No", "Down", "No"],
        "glyburide-metformin": ["No"] * 7,
        "glipizide-metformin": ["No"] * 7,
        "glimepiride-pioglitazone": ["No"] * 7,
        "metformin-rosiglitazone": ["No"] * 7,
        "metformin-pioglitazone": ["No"] * 7,
        "change": ["Ch", "No", "Ch", "No", "No", "Ch", "No"],
        "diabetesMed": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
        "readmitted": ["<30", "NO", ">30", "NO", "NO", "<30", "NO"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_cleaned_df(sample_raw_df) -> pd.DataFrame:
    """
    A sample DataFrame that has been through the cleaning pipeline.
    Used by tests that validate post-cleaning state.
    """
    df = sample_raw_df.copy()
    # Apply basic transformations manually (mirrors what clean_data.py does)
    df = df.replace("?", pd.NA)
    df.columns = df.columns.str.lower().str.strip()
    # Remove duplicate (encounter_id=1002 appears twice)
    df = df.drop_duplicates(subset=["encounter_id"], keep="first")
    # Remove invalid gender
    df = df[df["gender"].isin(["Male", "Female"]) | df["gender"].isna()]
    return df.reset_index(drop=True)


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """
    Provide a temporary directory for file-based tests.
    pytest's tmp_path fixture creates a unique temp dir per test.
    """
    return tmp_path


@pytest.fixture
def sample_config() -> dict:
    """Minimal config dict for tests that need configuration."""
    return {
        "dataset": {
            "name": "Test Dataset",
            "primary_file": "diabetic_data.csv",
            "readmission_classes": {
                "positive": "<30",
                "negative_gt30": ">30",
                "negative_no": "NO",
            },
        },
        "paths": {
            "data_raw": "data/raw",
            "data_staging": "data/staging",
            "data_processed": "data/processed",
            "data_curated": "data/curated",
            "data_rejected": "data/rejected",
            "data_logs": "data/logs",
        },
        "etl": {
            "missing_value_codes": ["?", "Unknown/Invalid", "NULL", ""],
            "required_columns": [
                "encounter_id", "patient_nbr", "race", "gender", "age",
                "admission_type_id", "discharge_disposition_id",
                "admission_source_id", "time_in_hospital",
                "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient",
                "diag_1", "diag_2", "diag_3", "number_diagnoses", "readmitted",
            ],
            "numeric_ranges": {
                "time_in_hospital": {"min": 1, "max": 14},
                "num_medications": {"min": 1, "max": 81},
            },
        },
    }
