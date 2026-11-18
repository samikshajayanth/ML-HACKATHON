from fastapi import APIRouter, HTTPException
from backend.pipeline import process_trip

router = APIRouter()


@router.get("/trips")
def get_trips():
    # Replace with actual trip IDs from the permitted input data.
    return {"trips": []}


@router.post("/predict/{trip_id}")
def predict_trip(trip_id: str):
    try:
        result = process_trip(trip_id)
        return result
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error))