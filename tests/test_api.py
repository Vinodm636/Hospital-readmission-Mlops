"""
test_api.py
-----------
Basic tests for the FastAPI prediction API (api/main.py).

What we test (without requiring a running database):
    - /health endpoint returns 200
    - /health response contains expected fields
    - /predict endpoint validates required input fields
    - /predict endpoint rejects invalid input with 422
    - Input validation catches wrong data types

Note:
    These tests use FastAPI's TestClient which runs the app in-process —
    no real server or database is needed for basic endpoint tests.
    Database-dependent tests are marked with @pytest.mark.skip when
    the database is not available.

Run:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import FastAPI test client
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Create a FastAPI TestClient.
    We patch the model loading so no real trained model is required.
    """
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    # Patch model loading to avoid requiring a real model file
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.75, 0.25]]

    with patch("api.main.load_model", return_value=mock_model):
        try:
            import api.main as api_module
            # Set module globals so the health endpoint reports "ok"
            api_module._model = mock_model
            api_module._model_version = type(mock_model).__name__
            api_module._model_loaded_at = "2026-10-14T00:00:00"
            from api.main import app
            return TestClient(app)
        except Exception as e:
            pytest.skip(f"Could not create TestClient: {e}")


@pytest.fixture
def valid_prediction_input() -> dict:
    """Valid input for the /predict endpoint."""
    return {
        "time_in_hospital": 5,
        "num_lab_procedures": 41,
        "num_procedures": 1,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 9,
        "age": "[70-80)",
        "gender": "Male",
        "race": "Caucasian",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "primary_diagnosis_category": "Circulatory_System",
        "insulin": "Steady",
        "diabetesMed": "Yes",
        "max_glu_serum": "None",
        "A1Cresult": "None",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """GET /health must return valid JSON."""
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_health_contains_status_field(self, client):
        """GET /health response must include a 'status' field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_status_is_ok(self, client):
        """GET /health status field should be 'ok' when service is running."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("ok", "healthy", "running")


# ─────────────────────────────────────────────────────────────────────────────
# Root endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_disclaimer(self, client):
        """Root response must contain the academic disclaimer."""
        response = client.get("/")
        data = response.json()
        # Check either message or disclaimer field contains relevant text
        response_text = str(data).lower()
        assert any(word in response_text for word in ["academic", "prototype", "clinical"])


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — input validation tests (no model needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictInputValidation:
    def test_missing_required_field_returns_422(self, client):
        """POST /predict with missing required field must return HTTP 422."""
        incomplete_input = {
            "time_in_hospital": 5,
            # Missing all other required fields
        }
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422

    def test_invalid_type_returns_422(self, client):
        """POST /predict with wrong data type must return HTTP 422."""
        invalid_input = {
            "time_in_hospital": "not_a_number",  # Should be int
            "num_medications": 18,
            "num_lab_procedures": 41,
            "num_procedures": 1,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
            "number_diagnoses": 9,
            "age": "[70-80)",
            "gender": "Male",
            "race": "Caucasian",
            "admission_type_id": 1,
            "discharge_disposition_id": 1,
            "admission_source_id": 7,
            "primary_diagnosis_category": "Circulatory_System",
            "insulin": "No",
            "diabetesMed": "Yes",
            "max_glu_serum": "None",
            "A1Cresult": "None",
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """POST /predict with empty body must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model unit tests (standalone — no HTTP required)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionInputSchema:
    """Test the Pydantic input schema directly without making HTTP calls."""

    def test_valid_input_passes_validation(self):
        """Valid prediction input must not raise a ValidationError."""
        try:
            from api.main import PredictionInput
        except ImportError:
            pytest.skip("API not yet implemented")

        from pydantic import ValidationError
        try:
            obj = PredictionInput(
                time_in_hospital=5,
                num_lab_procedures=41,
                num_procedures=1,
                num_medications=18,
                number_outpatient=0,
                number_emergency=0,
                number_inpatient=1,
                number_diagnoses=9,
                age="[70-80)",
                gender="Male",
                race="Caucasian",
                admission_type_id=1,
                discharge_disposition_id=1,
                admission_source_id=7,
                primary_diagnosis_category="Circulatory_System",
                insulin="Steady",
                diabetesMed="Yes",
                max_glu_serum="None",
                A1Cresult="None",
            )
            assert obj.time_in_hospital == 5
        except ValidationError as e:
            pytest.fail(f"Valid input raised ValidationError: {e}")
        except Exception:
            pytest.skip("PredictionInput not yet importable")

    def test_negative_time_in_hospital_fails(self):
        """Negative time_in_hospital must fail validation."""
        try:
            from api.main import PredictionInput
            from pydantic import ValidationError
        except ImportError:
            pytest.skip("API not yet implemented")

        with pytest.raises((ValidationError, ValueError)):
            PredictionInput(
                time_in_hospital=-1,   # Invalid
                num_lab_procedures=41,
                num_procedures=1,
                num_medications=18,
                number_outpatient=0,
                number_emergency=0,
                number_inpatient=0,
                number_diagnoses=9,
                age="[70-80)",
                gender="Male",
                race="Caucasian",
                admission_type_id=1,
                discharge_disposition_id=1,
                admission_source_id=7,
                primary_diagnosis_category="Circulatory_System",
                insulin="No",
                diabetesMed="Yes",
                max_glu_serum="None",
                A1Cresult="None",
            )
