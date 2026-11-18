"""PathSnap FastAPI Application.

Provides RESTful endpoints for map matching, route reconstruction,
evaluation trip listing, and health checks.
"""
from __future__ import annotations

import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    HealthResponse,
    MapMatchRequest,
    MapMatchResponse,
    TripListResponse,
)
from .services import map_matching_service, prediction_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap API",
    description="Distributed GPS Trajectory Map Matching and Route Reconstruction API",
    version="1.0.0",
)

# CORS middleware for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Root")
def root():
    return {
        "service": "PathSnap API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    model_ready = prediction_service.is_ready()
    return HealthResponse(
        status="healthy" if model_ready else "degraded",
        model_loaded=model_ready,
    )


@app.get("/trips", response_model=TripListResponse, summary="List Evaluation Trips")
def list_trips(split: str = Query(default="student_test", description="Dataset split to list")):
    try:
        trips = map_matching_service.get_available_trips(split=split)
        return TripListResponse(split=split, count=len(trips), trips=trips)
    except FileNotFoundError as err:
        logger.error("Trips directory error: %s", err)
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        logger.error("Error retrieving trips: %s", err)
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/predict/{trip_id}", response_model=MapMatchResponse, summary="Match Trip by ID")
def predict_trip(
    trip_id: str,
    split: str = Query(default="student_test", description="Dataset split"),
    emission_mode: str = Query(default="hybrid", description="Emission mode: hybrid, ml, gaussian"),
    alpha: float = Query(default=0.5, ge=0.0, le=1.0, description="Hybrid ML weight"),
    sigma_z: float = Query(default=10.0, gt=0.0, description="GPS error std in meters"),
    beta: float = Query(default=10.0, gt=0.0, description="Transition beta in meters"),
):
    """Primary Hackathon evaluation endpoint:

    Receives a trip_id, processes GPS points through Candidate Generation -> Feature Extraction
    -> ML CandidateScorer -> HMM Viterbi Decoder, and computes server-side evaluation metrics.
    """
    logger.info("Received map-matching request for trip '%s' in split '%s'", trip_id, split)
    try:
        result = map_matching_service.process_trip(
            trip_id=trip_id,
            split=split,
            emission_mode=emission_mode,
            alpha=alpha,
            sigma_z=sigma_z,
            beta=beta,
        )
        logger.info(
            "Trip '%s' matched successfully in %.2fs. Continuity: %.2f%%, Accuracy: %s",
            trip_id,
            result["inference_time_s"],
            result["metrics"]["continuity_hmm"] * 100,
            (
                f"{result['metrics']['accuracy_hmm'] * 100:.2f}%"
                if result["metrics"]["accuracy_hmm"] is not None
                else "N/A"
            ),
        )
        return result
    except FileNotFoundError as err:
        logger.warning("Trip not found: %s", err)
        raise HTTPException(status_code=404, detail=f"Trip data not found: {err}")
    except ValueError as err:
        logger.warning("Validation error processing trip: %s", err)
        raise HTTPException(status_code=422, detail=str(err))
    except Exception as err:
        logger.exception("Internal error processing trip '%s': %s", trip_id, err)
        raise HTTPException(status_code=500, detail=f"Map matching failed: {err}")


@app.post("/match", response_model=MapMatchResponse, summary="Custom Map Matching Request")
def match_custom(req: MapMatchRequest):
    """Match trip with full parameter customization in JSON body."""
    return predict_trip(
        trip_id=req.trip_id,
        split=req.split,
        emission_mode=req.emission_mode,
        alpha=req.alpha,
        sigma_z=req.sigma_z,
        beta=req.beta,
    )
