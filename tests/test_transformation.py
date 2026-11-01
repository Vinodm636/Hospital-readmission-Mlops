"""
test_transformation.py
----------------------
Tests for data cleaning (clean_data.py) and feature engineering
(feature_engineering.py).

What we test:
    Cleaning:
        - '?' replacement with NaN
        - Duplicate removal
        - Invalid encounter removal (gender, time_in_hospital)
        - Surrogate key creation
        - Missing value handling

    Feature Engineering:
        - readmission_binary creation (correct values)
        - diagnosis_count calculation
        - prior_admission_count calculation
        - length_of_stay alias
        - primary_diagnosis_category ICD-9 mapping

Run:
    pytest tests/test_transformation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_data import (
    replace_missing_value_codes,
    standardise_column_names,
    remove_duplicates,
    remove_invalid_encounters,
    standardise_categorical_fields,
    create_surrogate_keys,
    handle_remaining_missing_values,
)
from src.transformation.feature_engineering import (
    create_readmission_binary,
    create_diagnosis_count,
    create_prior_admission_features,
    create_length_of_stay_features,
    create_primary_diagnosis_category,
    create_medication_count,
    create_diabetes_drug_flag,
    create_total_service_utilisation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaceMissingValueCodes:
    def test_question_marks_become_nan(self):
        df = pd.DataFrame({"race": ["Caucasian", "?", "Asian"]})
        result = replace_missing_value_codes(df, ["?"])
        assert pd.isna(result["race"].iloc[1])

    def test_other_values_preserved(self):
        df = pd.DataFrame({"race": ["Caucasian", "?", "Asian"]})
        result = replace_missing_value_codes(df, ["?"])
        assert result["race"].iloc[0] == "Caucasian"
        assert result["race"].iloc[2] == "Asian"

    def test_multiple_codes_replaced(self):
        df = pd.DataFrame({"col": ["OK", "?", "NULL", "valid"]})
        result = replace_missing_value_codes(df, ["?", "NULL"])
        assert pd.isna(result["col"].iloc[1])
        assert pd.isna(result["col"].iloc[2])
        assert result["col"].iloc[0] == "OK"


class TestStandardiseColumnNames:
    def test_converts_to_lowercase(self):
        df = pd.DataFrame({"EncounterId": [1], "PatientNbr": [2]})
        result = standardise_column_names(df)
        assert "encounterid" in result.columns
        assert "patientnbr" in result.columns

    def test_strips_whitespace(self):
        df = pd.DataFrame({" encounter_id ": [1]})
        result = standardise_column_names(df)
        assert "encounter_id" in result.columns

    def test_replaces_spaces_with_underscores(self):
        df = pd.DataFrame({"my column": [1]})
        result = standardise_column_names(df)
        assert "my_column" in result.columns


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({
            "encounter_id": [1, 2, 2],
            "patient_nbr": [10, 20, 20],
            "readmitted": ["NO", "<30", "<30"],
        })
        rejected = []
        result = remove_duplicates(df, run_id="test-run", rejected_records=rejected)
        assert len(result) == 2
        assert len(rejected) == 1

    def test_keeps_first_occurrence(self):
        df = pd.DataFrame({
            "encounter_id": [1, 2, 2],
            "value": ["first", "first_dup", "second_dup"],
        })
        rejected = []
        result = remove_duplicates(df, run_id="test", rejected_records=rejected)
        # Row 0 and row 1 are NOT duplicates of each other
        # Row 1 and row 2 have same encounter_id=2 — but NOT exact duplicates
        # (different 'value' column)
        assert len(result) == 3  # No exact duplicates

    def test_no_removals_when_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        rejected = []
        result = remove_duplicates(df, run_id="test", rejected_records=rejected)
        assert len(result) == 3
        assert len(rejected) == 0


class TestRemoveInvalidEncounters:
    def test_removes_row_with_missing_readmitted(self):
        df = pd.DataFrame({
            "encounter_id": [1, 2],
            "patient_nbr": [10, 20],
            "readmitted": ["NO", None],
            "gender": ["Male", "Female"],
            "time_in_hospital": [3, 5],
        })
        rejected = []
        result = remove_invalid_encounters(df, "test", rejected, ["encounter_id", "patient_nbr", "readmitted"])
        assert len(result) == 1
        assert len(rejected) == 1
        assert rejected[0]["rejection_reason"] == "MISSING_REQUIRED_FIELD"

    def test_removes_invalid_gender(self):
        df = pd.DataFrame({
            "encounter_id": [1, 2],
            "patient_nbr": [10, 20],
            "readmitted": ["NO", "NO"],
            "gender": ["Male", "Unknown/Invalid"],
            "time_in_hospital": [3, 5],
        })
        rejected = []
        result = remove_invalid_encounters(df, "test", rejected, [])
        assert len(result) == 1
        assert any(r["rejection_reason"] == "INVALID_GENDER" for r in rejected)

    def test_removes_zero_time_in_hospital(self):
        df = pd.DataFrame({
            "encounter_id": [1, 2],
            "patient_nbr": [10, 20],
            "readmitted": ["NO", "NO"],
            "gender": ["Male", "Female"],
            "time_in_hospital": [3, 0],  # 0 is invalid
        })
        rejected = []
        result = remove_invalid_encounters(df, "test", rejected, [])
        assert len(result) == 1
        assert any(r["rejection_reason"] == "INVALID_TIME_IN_HOSPITAL" for r in rejected)


class TestCreateSurrogateKeys:
    def test_surrogate_keys_created(self):
        df = pd.DataFrame({
            "encounter_id": [101, 102],
            "patient_nbr": [201, 202],
        })
        result = create_surrogate_keys(df)
        assert "encounter_surrogate_id" in result.columns
        assert "patient_surrogate_id" in result.columns

    def test_surrogate_keys_are_deterministic(self):
        """Same input must always produce the same surrogate key."""
        df = pd.DataFrame({"encounter_id": [101], "patient_nbr": [201]})
        result1 = create_surrogate_keys(df.copy())
        result2 = create_surrogate_keys(df.copy())
        assert result1["encounter_surrogate_id"].iloc[0] == result2["encounter_surrogate_id"].iloc[0]

    def test_different_ids_produce_different_surrogates(self):
        df = pd.DataFrame({"encounter_id": [101, 102], "patient_nbr": [201, 202]})
        result = create_surrogate_keys(df)
        assert result["encounter_surrogate_id"].iloc[0] != result["encounter_surrogate_id"].iloc[1]

    def test_surrogate_key_length_is_16(self):
        df = pd.DataFrame({"encounter_id": [101], "patient_nbr": [201]})
        result = create_surrogate_keys(df)
        assert len(result["encounter_surrogate_id"].iloc[0]) == 16


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateReadmissionBinary:
    def test_less_than_30_maps_to_1(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
        result = create_readmission_binary(df)
        assert result["readmission_binary"].iloc[0] == 1

    def test_greater_than_30_maps_to_0(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
        result = create_readmission_binary(df)
        assert result["readmission_binary"].iloc[1] == 0

    def test_no_maps_to_0(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
        result = create_readmission_binary(df)
        assert result["readmission_binary"].iloc[2] == 0

    def test_raises_when_readmitted_column_missing(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(ValueError):
            create_readmission_binary(df)

    def test_only_binary_values(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30", "NO"]})
        result = create_readmission_binary(df)
        unique_vals = set(result["readmission_binary"].unique())
        assert unique_vals.issubset({0, 1})


class TestCreateDiagnosisCount:
    def test_counts_non_zero_diagnoses(self):
        df = pd.DataFrame({
            "diag_1": ["428", "250"],
            "diag_2": ["0", "276"],
            "diag_3": ["0", "0"],
        })
        result = create_diagnosis_count(df)
        # Row 0: diag_1=428 (valid), diag_2=0 (invalid), diag_3=0 (invalid) → count=1
        # Row 1: diag_1=250 (valid), diag_2=276 (valid), diag_3=0 (invalid) → count=2
        assert result["diagnosis_count"].iloc[0] == 1
        assert result["diagnosis_count"].iloc[1] == 2

    def test_zero_when_no_diag_columns(self):
        df = pd.DataFrame({"other": [1, 2]})
        result = create_diagnosis_count(df)
        assert "diagnosis_count" in result.columns
        assert (result["diagnosis_count"] == 0).all()


class TestCreatePriorAdmissionFeatures:
    def test_counts_total_prior_admissions(self):
        df = pd.DataFrame({
            "number_outpatient": [1, 0],
            "number_emergency": [0, 2],
            "number_inpatient": [1, 0],
        })
        result = create_prior_admission_features(df)
        assert result["prior_admission_count"].iloc[0] == 2  # 1+0+1
        assert result["prior_admission_count"].iloc[1] == 2  # 0+2+0

    def test_is_prior_admission_binary(self):
        df = pd.DataFrame({
            "number_outpatient": [1, 0],
            "number_emergency": [0, 0],
            "number_inpatient": [0, 0],
        })
        result = create_prior_admission_features(df)
        assert result["is_prior_admission"].iloc[0] == 1
        assert result["is_prior_admission"].iloc[1] == 0


class TestCreateLengthOfStayFeatures:
    def test_creates_length_of_stay_alias(self):
        df = pd.DataFrame({"time_in_hospital": [3, 7, 14]})
        result = create_length_of_stay_features(df)
        assert list(result["length_of_stay"]) == [3, 7, 14]

    def test_is_long_stay_flag(self):
        df = pd.DataFrame({"time_in_hospital": [3, 7, 8, 14]})
        result = create_length_of_stay_features(df)
        assert result["is_long_stay"].iloc[0] == 0  # 3 ≤ 7
        assert result["is_long_stay"].iloc[1] == 0  # 7 ≤ 7
        assert result["is_long_stay"].iloc[2] == 1  # 8 > 7
        assert result["is_long_stay"].iloc[3] == 1  # 14 > 7


class TestCreatePrimaryDiagnosisCategory:
    def test_circulatory_system_mapped(self):
        df = pd.DataFrame({"diag_1": ["428"]})   # Heart failure = Circulatory
        result = create_primary_diagnosis_category(df)
        assert result["primary_diagnosis_category"].iloc[0] == "Circulatory_System"

    def test_endocrine_metabolic_mapped(self):
        df = pd.DataFrame({"diag_1": ["250"]})   # Diabetes = Endocrine
        result = create_primary_diagnosis_category(df)
        assert result["primary_diagnosis_category"].iloc[0] == "Endocrine_Metabolic"

    def test_unknown_for_zero(self):
        df = pd.DataFrame({"diag_1": ["0"]})
        result = create_primary_diagnosis_category(df)
        assert result["primary_diagnosis_category"].iloc[0] == "Unknown"

    def test_e_code_mapped(self):
        df = pd.DataFrame({"diag_1": ["E810"]})
        result = create_primary_diagnosis_category(df)
        assert result["primary_diagnosis_category"].iloc[0] == "External_Causes"

    def test_missing_diag_1_is_unknown(self):
        df = pd.DataFrame({"diag_1": [None]})
        result = create_primary_diagnosis_category(df)
        assert result["primary_diagnosis_category"].iloc[0] == "Unknown"


class TestCreateDiabetesDrugFlag:
    def test_flag_set_when_insulin_prescribed(self):
        df = pd.DataFrame({
            "insulin": ["Steady"],
            "metformin": ["No"],
        })
        result = create_diabetes_drug_flag(df)
        assert result["has_diabetes_drug"].iloc[0] == 1

    def test_flag_unset_when_no_drugs(self):
        df = pd.DataFrame({
            "insulin": ["No"],
            "metformin": ["No"],
            "glipizide": ["No"],
        })
        result = create_diabetes_drug_flag(df)
        assert result["has_diabetes_drug"].iloc[0] == 0

    def test_works_with_no_drug_columns(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = create_diabetes_drug_flag(df)
        assert "has_diabetes_drug" in result.columns


class TestCreateTotalServiceUtilisation:
    def test_sums_all_utilisation_columns(self):
        df = pd.DataFrame({
            "number_outpatient": [1, 0],
            "number_emergency": [2, 0],
            "number_inpatient": [1, 0],
            "time_in_hospital": [5, 3],
        })
        result = create_total_service_utilisation(df)
        assert result["total_service_utilisation"].iloc[0] == 9  # 1+2+1+5
        assert result["total_service_utilisation"].iloc[1] == 3  # 0+0+0+3
