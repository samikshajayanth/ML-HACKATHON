"""Tests for the PathSnap FastAPI Backend API."""
from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PathSnap API"
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_trips_endpoint():
    response = client.get("/trips?split=student_test")
    assert response.status_code == 200
    data = response.json()
    assert data["split"] == "student_test"
    assert data["count"] == 10
    assert "00000000" in data["trips"]
    assert "00000052" in data["trips"]


def test_predict_endpoint_success():
    # Use trip 00000052 which is a compact trip (262 observations) for quick test execution
    response = client.post("/predict/00000052?split=student_test&emission_mode=hybrid")
    assert response.status_code == 200
    data = response.json()

    assert data["trip_id"] == "00000052"
    assert data["status"] == "success"
    assert data["num_observations"] > 0
    assert len(data["matched_arcs"]) == data["num_observations"]
    assert len(data["ml_top1_arcs"]) == data["num_observations"]

    # Metrics validation
    metrics = data["metrics"]
    assert "continuity_hmm" in metrics
    assert "continuity_ml" in metrics
    assert 0.0 <= metrics["continuity_hmm"] <= 1.0
    assert 0.0 <= metrics["continuity_ml"] <= 1.0

    # Coordinates validation for Folium
    assert len(data["gps_coordinates"]) == data["num_observations"]
    assert len(data["matched_coordinates"]) >= 2
    assert isinstance(data["inference_time_s"], float)


def test_predict_nonexistent_trip():
    response = client.post("/predict/invalid_trip_id_999999")
    assert response.status_code == 404
