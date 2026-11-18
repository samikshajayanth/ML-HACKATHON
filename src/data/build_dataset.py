import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_trip
from src.features.candidates import CandidateGenerator
from src.features.feature_extraction import extract_features


def main():
    parser = argparse.ArgumentParser(
        description="Build candidate and feature datasets for training trips."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset folder containing train/ and student_test/."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=100
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    train_dir = dataset_root / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    output_root = Path("outputs/training")
    candidate_dir = output_root / "candidates"
    feature_dir = output_root / "features"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    generator = CandidateGenerator(
        radius_m=args.radius,
        max_candidates=args.max_candidates
    )

    all_candidates = []
    all_features = []
    report = []

    trip_folders = sorted(
        folder for folder in train_dir.iterdir()
        if folder.is_dir()
    )

    print(f"Found {len(trip_folders)} training trips.")

    for index, folder in enumerate(trip_folders, start=1):
        trip_id = folder.name

        try:
            trip = load_trip(dataset_root, "train", trip_id)

            candidates = generator.generate(trip)
            features = extract_features(trip, candidates)

            candidates.to_csv(
                candidate_dir / f"trip_{trip_id}_candidates.csv",
                index=False
            )
            features.to_csv(
                feature_dir / f"trip_{trip_id}_features.csv",
                index=False
            )

            all_candidates.append(candidates)
            all_features.append(features)

            observations = len(trip.track)
            covered = candidates["observation_id"].nunique()

            report.append({
                "trip_id": trip_id,
                "status": "success",
                "gps_observations": observations,
                "candidate_records": len(candidates),
                "feature_records": len(features),
                "observations_with_candidates": covered,
                "observations_without_candidates": observations - covered,
                "fallback_candidates": int(candidates["is_fallback"].sum()),
                "missing_feature_values": int(features.isna().sum().sum()),
                "error": ""
            })

            print(
                f"[{index}/{len(trip_folders)}] {trip_id}: "
                f"{len(candidates)} candidates, "
                f"{len(features)} features"
            )

        except Exception as error:
            report.append({
                "trip_id": trip_id,
                "status": "failed",
                "gps_observations": 0,
                "candidate_records": 0,
                "feature_records": 0,
                "observations_with_candidates": 0,
                "observations_without_candidates": 0,
                "fallback_candidates": 0,
                "missing_feature_values": 0,
                "error": str(error)
            })

            print(f"[{index}/{len(trip_folders)}] {trip_id}: FAILED - {error}")

    if all_candidates:
        pd.concat(all_candidates, ignore_index=True).to_csv(
            output_root / "all_training_candidates.csv",
            index=False
        )

    if all_features:
        pd.concat(all_features, ignore_index=True).to_csv(
            output_root / "all_training_features.csv",
            index=False
        )

    report_df = pd.DataFrame(report)
    report_df.to_csv(
        output_root / "processing_report.csv",
        index=False
    )

    print("\n========== BATCH PROCESSING SUMMARY ==========")
    print("Trips found:", len(trip_folders))
    print("Successful:", (report_df["status"] == "success").sum())
    print("Failed:", (report_df["status"] == "failed").sum())
    print("Total candidate records:", report_df["candidate_records"].sum())
    print("Total feature records:", report_df["feature_records"].sum())
    print("Total uncovered observations:",
          report_df["observations_without_candidates"].sum())

    print("\nOutputs saved in:", output_root.resolve())


if __name__ == "__main__":
    main()
