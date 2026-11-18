
from pathlib import Path

from src.data.loader import load_trip
from src.features.candidates import CandidateGenerator


DATASET_ROOT = Path(
    "/Users/rujuljain/Downloads/Student folder"
)

trip = load_trip(
    DATASET_ROOT,
    "train",
    "00000001"
)

generator = CandidateGenerator(
    radius_m=100,
    max_candidates=8
)

candidates = generator.generate(trip)

print("\n===== CANDIDATE GENERATION =====")
print("Trip:", trip.trip_id)
print("GPS observations:", len(trip.track))
print("Candidate records:", len(candidates))
print(
    "Observations with candidates:",
    candidates["observation_id"].nunique()
)
print(
    "Observations without candidates:",
    len(trip.track) - candidates["observation_id"].nunique()
)
print(
    "Average candidates per observation:",
    round(
        len(candidates) / len(trip.track), 2
    )
)
print(
    "Fallback candidates:",
    candidates["is_fallback"].sum()
)
print(
    "Maximum candidate distance:",
    round(candidates["distance_m"].max(), 2),
    "metres"
)

print("\n===== FIRST 15 CANDIDATES =====")
print(candidates.head(15).to_string(index=False))

output_dir = Path("outputs/candidates")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "trip_00000001_candidates.csv"
candidates.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Projected CRS:", candidates.attrs["projected_epsg"])
