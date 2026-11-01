"""
test_validation.py
------------------
Tests for the data-quality validation module (src/validation/data_quality.py).

What we test:
    - Required column checks (pass/fail)
    - Row count checks
    - Null percentage checks
    - Duplicate row detection
    - Numeric range validation
    - Categorical value validation
    - Readmission distribution check
    - Surrogate key validation

Run:
    pytest tests/test_validation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.data_quality import (
    check_required_columns,
    check_row_count,
    check_null_percentages,
    check_duplicate_rows,
    check_numeric_ranges,
    check_categorical_values,
    check_readmission_distribution,
    check_surrogate_keys,
    make_check_result,
)


# ─────────────────────────────────────────────────────────────────────────────
# make_check_result tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeCheckResult:
    def test_result_has_required_keys(self):
        result = make_check_result("test_check", "PASS", "All good")
        required = ["check_name", "status", "description", "failed_count", "timestamp"]
        for key in required:
            assert key in result

    def test_default_failed_count_is_zero(self):
        result = make_check_result("test_check", "PASS", "All good")
        assert result["failed_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# check_required_columns tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckRequiredColumns:
    def test_pass_when_all_columns_present(self):
        df = pd.DataFrame({"encounter_id": [1], "patient_nbr": [2], "readmitted": ["NO"]})
        result = check_required_columns(df, ["encounter_id", "patient_nbr", "readmitted"])
        assert result["status"] == "PASS"

    def test_fail_when_column_missing(self):
        df = pd.DataFrame({"encounter_id": [1], "patient_nbr": [2]})
        result = check_required_columns(df, ["encounter_id", "readmitted"])
        assert result["status"] == "FAIL"
        assert result["failed_count"] == 1

    def test_fail_count_matches_missing_columns(self):
        df = pd.DataFrame({"col_a": [1]})
        result = check_required_columns(df, ["col_a", "col_b", "col_c"])
        assert result["failed_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# check_row_count tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckRowCount:
    def test_pass_when_above_minimum(self):
        df = pd.DataFrame({"x": range(5000)})
        result = check_row_count(df, min_rows=1000)
        assert result["status"] == "PASS"

    def test_fail_when_below_minimum(self):
        df = pd.DataFrame({"x": range(50)})
        result = check_row_count(df, min_rows=1000)
        assert result["status"] == "FAIL"

    def test_fail_on_empty_dataframe(self):
        df = pd.DataFrame()
        result = check_row_count(df, min_rows=1)
        assert result["status"] == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# check_null_percentages tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckNullPercentages:
    def test_pass_when_no_high_null_columns(self):
        df = pd.DataFrame({
            "col_a": [1, 2, 3, 4, 5],
            "col_b": ["a", "b", "c", None, "e"],  # 20% null — below 50%
        })
        result = check_null_percentages(df, threshold_pct=50.0)
        assert result["status"] == "PASS"

    def test_warn_when_column_exceeds_threshold(self):
        df = pd.DataFrame({
            "col_a": [1, 2, 3, 4, 5],
            "col_b": [None, None, None, None, "e"],  # 80% null — above 50%
        })
        result = check_null_percentages(df, threshold_pct=50.0)
        assert result["status"] == "WARN"
        assert result["failed_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# check_duplicate_rows tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDuplicateRows:
    def test_pass_when_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        result = check_duplicate_rows(df)
        assert result["status"] == "PASS"
        assert result["failed_count"] == 0

    def test_fail_when_duplicates_exist(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "a", "b"]})
        result = check_duplicate_rows(df)
        assert result["status"] == "FAIL"
        assert result["failed_count"] == 1

    def test_correct_duplicate_count(self):
        # Rows 0 and 1 are identical, rows 2 and 3 are identical → 2 duplicates
        df = pd.DataFrame({
            "id": [1, 1, 2, 2],
            "val": ["a", "a", "b", "b"]
        })
        result = check_duplicate_rows(df)
        assert result["failed_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# check_numeric_ranges tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckNumericRanges:
    def test_pass_when_values_in_range(self):
        df = pd.DataFrame({"time_in_hospital": [1, 3, 7, 14]})
        ranges = {"time_in_hospital": {"min": 1, "max": 14}}
        result = check_numeric_ranges(df, ranges)
        assert result["status"] == "PASS"

    def test_warn_when_value_out_of_range(self):
        df = pd.DataFrame({"time_in_hospital": [1, 3, 7, 99]})  # 99 > max 14
        ranges = {"time_in_hospital": {"min": 1, "max": 14}}
        result = check_numeric_ranges(df, ranges)
        assert result["status"] == "WARN"
        assert result["failed_count"] >= 1

    def test_pass_when_column_not_in_dataframe(self):
        """Missing column should not cause a failure (it is skipped)."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        ranges = {"time_in_hospital": {"min": 1, "max": 14}}
        result = check_numeric_ranges(df, ranges)
        assert result["status"] == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# check_categorical_values tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckCategoricalValues:
    def test_pass_with_valid_gender(self):
        df = pd.DataFrame({
            "gender": ["Male", "Female", "Male"],
            "readmitted": ["NO", "<30", ">30"],
            "age": ["[70-80)", "[60-70)", "[50-60)"],
        })
        result = check_categorical_values(df)
        assert result["status"] == "PASS"

    def test_fail_with_invalid_gender(self):
        df = pd.DataFrame({
            "gender": ["Male", "Unknown/Invalid", "Female"],
            "readmitted": ["NO", "NO", "<30"],
            "age": ["[70-80)", "[60-70)", "[50-60)"],
        })
        result = check_categorical_values(df)
        assert result["status"] == "FAIL"

    def test_fail_with_invalid_readmitted(self):
        df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "readmitted": ["YES", "NO"],   # 'YES' is invalid
            "age": ["[70-80)", "[60-70)"],
        })
        result = check_categorical_values(df)
        assert result["status"] == "FAIL"

    def test_ignores_nan_values(self):
        """NaN in categorical columns should not trigger failures."""
        df = pd.DataFrame({
            "gender": ["Male", None, "Female"],
            "readmitted": ["NO", None, "<30"],
            "age": ["[70-80)", None, "[50-60)"],
        })
        result = check_categorical_values(df)
        assert result["status"] == "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# check_readmission_distribution tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckReadmissionDistribution:
    def test_pass_with_all_three_classes(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30", "NO"]})
        result = check_readmission_distribution(df)
        assert result["status"] == "PASS"

    def test_warn_with_missing_class(self):
        df = pd.DataFrame({"readmitted": ["<30", "NO", "<30"]})  # '>30' missing
        result = check_readmission_distribution(df)
        assert result["status"] == "WARN"
        assert result["failed_count"] >= 1

    def test_fail_when_column_missing(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = check_readmission_distribution(df)
        assert result["status"] == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# check_surrogate_keys tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckSurrogateKeys:
    def test_pass_with_valid_surrogate_keys(self):
        df = pd.DataFrame({
            "encounter_surrogate_id": ["abc1", "abc2", "abc3"],
            "patient_surrogate_id": ["pat1", "pat2", "pat1"],
        })
        result = check_surrogate_keys(df)
        assert result["status"] == "PASS"

    def test_warn_when_encounter_surrogate_not_unique(self):
        df = pd.DataFrame({
            "encounter_surrogate_id": ["abc1", "abc1", "abc3"],  # duplicate
            "patient_surrogate_id": ["pat1", "pat2", "pat1"],
        })
        result = check_surrogate_keys(df)
        assert result["status"] == "WARN"

    def test_warn_when_column_missing(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = check_surrogate_keys(df)
        # Should warn, not crash
        assert result["status"] in ("PASS", "WARN")
