"""Pydantic schemas for the PathSnap FastAPI backend."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "PathSnap API"
    version: str = "1.0.0"
    model_loaded: bool = True
    available_splits: list[str] = ["train", "student_test"]


class TripListResponse(BaseModel):
    split: str
    count: int
    trips: list[str]


class MapMatchRequest(BaseModel):
    trip_id: str = Field(..., description="Trip ID to reconstruct (e.g. '00000000')")
    split: str = Field(default="student_test", description="Dataset split: 'student_test' or 'train'")
    emission_mode: str = Field(default="hybrid", description="Emission mode: 'hybrid', 'ml', or 'gaussian'")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid weight for ML emission")
    sigma_z: float = Field(default=10.0, gt=0.0, description="GPS error standard deviation in meters")
    beta: float = Field(default=10.0, gt=0.0, description="Transition model beta parameter in meters")


class RouteEvaluationMetrics(BaseModel):
    continuity_hmm: float
    continuity_ml: float
    continuity_improvement_pct: float
    accuracy_hmm: Optional[float] = None
    accuracy_ml: Optional[float] = None
    accuracy_improvement_pct: Optional[float] = None
    candidate_recall: Optional[float] = None


class MapMatchResponse(BaseModel):
    trip_id: str
    split: str
    status: str = "success"
    num_observations: int
    num_candidates: int
    matched_arcs: list[int]
    ml_top1_arcs: list[int]
    log_likelihood: float
    restarts: int
    metrics: RouteEvaluationMetrics
    gps_coordinates: list[list[float]]  # [lat, lon]
    matched_coordinates: list[list[float]]  # [lat, lon]
    ml_coordinates: list[list[float]]  # [lat, lon]
    ground_truth_coordinates: Optional[list[list[float]]] = None  # [lat, lon]
    inference_time_s: float
