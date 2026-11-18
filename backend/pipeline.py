
import os
from pathlib import Path

from src.data.loader import load_trip
from src.features.candidates import CandidateGenerator
from src.features.feature_extraction import extract_features


def get_dataset_root():
    value = os.getenv("PATHSNAP_DATASET_ROOT")

    if not value:
        raise ValueError(
            "PATHSNAP_DATASET_ROOT is not configured."
        )

    root = Path(value).expanduser()

    if not root.is_dir():
        raise ValueError(
            f"Dataset directory does not exist: {root}"
        )

    return root


def get_available_trips(split="student_test"):
    if split not in ("train", "student_test"):
        raise ValueError("Invalid dataset split.")

    root = get_dataset_root()
    split_dir = root / split

    if not split_dir.is_dir():
        raise ValueError(
            f"Dataset split directory not found: {split_dir}"
        )

    return sorted(
        folder.name
        for folder in split_dir.iterdir()
        if folder.is_dir()
    )


def process_trip(trip_id, split="student_test"):
    if split not in ("train", "student_test"):
        raise ValueError("Invalid dataset split.")

    root = get_dataset_root()

    available = get_available_trips(split)

    if str(trip_id) not in available:
        raise ValueError(
            f"Trip {trip_id} not found in {split}."
        )

    trip = load_trip(
        root,
        split,
        str(trip_id)
    )

    # Generate candidate road arcs.
    generator = CandidateGenerator(
        radius_m=100,
        max_candidates=8
    )

    candidates = generator.generate(trip)

    # Extract features for every candidate.
    features = extract_features(
        trip,
        candidates
    )

    # GPS coordinates are longitude, latitude.
    gps_coordinates = [
        [float(row[0]), float(row[1])]
        for row in trip.track
    ]

    candidate_records = candidates.where(
        candidates.notna(), None
    ).to_dict(orient="records")

    return {
        "trip_id": trip.trip_id,
        "split": split,
        "status": "features_generated",
        "reference_route_loaded": trip.route is not None,
        "gps_observations": len(trip.track),
        "road_nodes": len(trip.nodes),
        "road_arcs": len(trip.arcs),
        "candidate_records": len(candidates),
        "feature_records": len(features),
        "feature_columns": list(features.columns),
        "gps_coordinates": gps_coordinates,
        "candidates": candidate_records,
        "metrics": {
            "observations_with_candidates":
                int(candidates["observation_id"].nunique()),
            "observations_without_candidates":
                int(
                    len(trip.track)
                    - candidates["observation_id"].nunique()
                ),
            "fallback_candidates":
                int(candidates["is_fallback"].sum()),
            "missing_feature_values":
                int(features.isna().sum().sum())
        }
    }
