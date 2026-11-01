"""
test_ingestion.py
-----------------
Tests for the data ingestion module (src/ingestion/ingest.py).

What we test:
    - FileNotFoundError raised when dataset is missing
    - Successful CSV loading
    - Data profile generation
    - Metadata structure
    - Staging copy creation
    - Row/column counts in metadata match actual file

Run:
    pytest tests/test_ingestion.py -v
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.ingest import (
    build_metadata,
    load_csv,
    profile_dataframe,
    save_metadata_to_json,
    save_staging_copy,
    preserve_raw_copy,
)


# ─────────────────────────────────────────────────────────────────────────────
# load_csv tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    """Tests for the load_csv() function."""

    def test_raises_file_not_found_when_missing(self, temp_dir):
        """load_csv must raise FileNotFoundError for a non-existent file."""
        missing_path = temp_dir / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_csv(missing_path)
        # Error message should tell the user where to place the file
        assert "data/raw" in str(exc_info.value)

    def test_loads_valid_csv(self, temp_dir, sample_raw_df):
        """load_csv must return a DataFrame with correct shape."""
        csv_path = temp_dir / "test_data.csv"
        sample_raw_df.to_csv(csv_path, index=False)

        result = load_csv(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_raw_df)
        assert len(result.columns) == len(sample_raw_df.columns)

    def test_raises_on_malformed_csv(self, temp_dir):
        """load_csv must raise ValueError for an unparseable file."""
        bad_file = temp_dir / "bad.csv"
        bad_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")  # binary garbage

        with pytest.raises(Exception):
            load_csv(bad_file)


# ─────────────────────────────────────────────────────────────────────────────
# profile_dataframe tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileDataframe:
    """Tests for the profile_dataframe() function."""

    def test_profile_contains_required_keys(self, sample_raw_df):
        """Profile dict must contain all expected keys."""
        profile = profile_dataframe(sample_raw_df)

        required_keys = [
            "row_count", "column_count", "columns", "dtypes",
            "null_counts", "question_mark_counts", "duplicate_rows",
            "memory_usage_mb",
        ]
        for key in required_keys:
            assert key in profile, f"Missing key in profile: {key}"

    def test_row_count_matches_dataframe(self, sample_raw_df):
        """Profile row_count must match the actual DataFrame length."""
        profile = profile_dataframe(sample_raw_df)
        assert profile["row_count"] == len(sample_raw_df)

    def test_question_mark_counts_detected(self, sample_raw_df):
        """Profile must count '?' values in the raw dataset."""
        profile = profile_dataframe(sample_raw_df)
        # The sample data has '?' in 'race', 'weight', 'payer_code', etc.
        total_qmarks = sum(profile["question_mark_counts"].values())
        assert total_qmarks > 0, "Expected '?' values in sample data"

    def test_duplicate_count_detected(self, sample_raw_df):
        """Profile must detect the duplicate row in sample data."""
        profile = profile_dataframe(sample_raw_df)
        # sample_raw_df has encounter_id=1002 duplicated
        assert profile["duplicate_rows"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# build_metadata tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMetadata:
    """Tests for the build_metadata() function."""

    def test_metadata_contains_all_fields(self):
        """build_metadata must produce a dict with all required fields."""
        metadata = build_metadata(
            run_id="test-run-001",
            source_name="UCI Test",
            file_name="test.csv",
            status="SUCCESS",
            row_count=100,
            column_count=50,
        )
        required_fields = [
            "run_id", "source_name", "extraction_timestamp",
            "file_name", "status", "row_count", "column_count", "error_message",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_metadata_status_success(self):
        """build_metadata must record SUCCESS status correctly."""
        metadata = build_metadata(
            run_id="r1", source_name="Test", file_name="f.csv",
            status="SUCCESS", row_count=200, column_count=50,
        )
        assert metadata["status"] == "SUCCESS"
        assert metadata["error_message"] is None

    def test_metadata_status_failed(self):
        """build_metadata must record FAILED status with error message."""
        metadata = build_metadata(
            run_id="r2", source_name="Test", file_name="f.csv",
            status="FAILED", row_count=0, column_count=0,
            error_message="File not found",
        )
        assert metadata["status"] == "FAILED"
        assert metadata["error_message"] == "File not found"

    def test_row_count_stored_correctly(self):
        """build_metadata must store the exact row count passed."""
        metadata = build_metadata(
            run_id="r3", source_name="S", file_name="f.csv",
            status="SUCCESS", row_count=101766, column_count=50,
        )
        assert metadata["row_count"] == 101766


# ─────────────────────────────────────────────────────────────────────────────
# save_metadata_to_json tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveMetadataToJson:
    """Tests for the save_metadata_to_json() function."""

    def test_creates_json_file(self, temp_dir):
        """save_metadata_to_json must create a JSON file."""
        metadata = build_metadata(
            run_id="test-001", source_name="UCI", file_name="test.csv",
            status="SUCCESS", row_count=100, column_count=50,
        )
        log_file = save_metadata_to_json(metadata, log_dir=str(temp_dir))
        assert Path(log_file).exists()

    def test_json_is_valid(self, temp_dir):
        """The written JSON must be parseable."""
        metadata = build_metadata(
            run_id="test-002", source_name="UCI", file_name="test.csv",
            status="SUCCESS", row_count=100, column_count=50,
        )
        log_file = save_metadata_to_json(metadata, log_dir=str(temp_dir))
        with open(log_file, "r") as f:
            records = json.load(f)
        assert isinstance(records, list)
        assert len(records) >= 1

    def test_appends_multiple_runs(self, temp_dir):
        """Calling save_metadata_to_json twice must append, not overwrite."""
        for run_id in ["run-A", "run-B"]:
            metadata = build_metadata(
                run_id=run_id, source_name="UCI", file_name="test.csv",
                status="SUCCESS", row_count=100, column_count=50,
            )
            log_file = save_metadata_to_json(metadata, log_dir=str(temp_dir))

        with open(log_file, "r") as f:
            records = json.load(f)
        assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# save_staging_copy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveStagingCopy:
    """Tests for the save_staging_copy() function."""

    def test_creates_staging_csv(self, temp_dir, sample_raw_df):
        """save_staging_copy must write a CSV file to the staging directory."""
        staging_dir = temp_dir / "staging"
        staging_file = save_staging_copy(sample_raw_df, staging_dir=str(staging_dir))

        assert Path(staging_file).exists()

    def test_staging_csv_has_correct_row_count(self, temp_dir, sample_raw_df):
        """Staging CSV must contain the same number of rows as the input."""
        staging_dir = temp_dir / "staging"
        staging_file = save_staging_copy(sample_raw_df, staging_dir=str(staging_dir))

        loaded = pd.read_csv(staging_file)
        assert len(loaded) == len(sample_raw_df)

    def test_staging_csv_has_correct_columns(self, temp_dir, sample_raw_df):
        """Staging CSV must preserve all original column names."""
        staging_dir = temp_dir / "staging"
        staging_file = save_staging_copy(sample_raw_df, staging_dir=str(staging_dir))

        loaded = pd.read_csv(staging_file)
        assert set(loaded.columns) == set(sample_raw_df.columns)
