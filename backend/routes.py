
from fastapi import APIRouter, HTTPException

from backend.pipeline import (
    process_trip,
    get_available_trips
)

router = APIRouter()


@router.get("/trips")
def get_trips():
    try:
        trips = get_available_trips("student_test")

        return {
            "split": "student_test",
            "count": len(trips),
            "trips": trips
        }

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.post("/predict/{trip_id}")
def predict_trip(trip_id: str):
    try:
        return process_trip(
            trip_id,
            split="student_test"
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline processing failed: {error}"
        )
