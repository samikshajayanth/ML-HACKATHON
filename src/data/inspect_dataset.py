import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine(lon1, lat1, lon2, lat2):
    """Calculate geographic distance in metres."""
    radius = 6_371_000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def inspect_trip(trip_dir):
    trip_id = trip_dir.name

    try:
        track = np.atleast_2d(
            np.loadtxt(trip_dir / f"{trip_id}.track")
        )
        nodes = np.atleast_2d(
            np.loadtxt(trip_dir / f"{trip_id}.nodes")
        )
        arcs = np.atleast_2d(
            np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=int)
        )
        route = np.atleast_1d(
            np.loadtxt(trip_dir / f"{trip_id}.route", dtype=int)
        )

        diffs = np.diff(track[:, 2])
        route_arcs = arcs[route]

        disconnected = np.sum(
            route_arcs[:-1, 1] != route_arcs[1:, 0]
        )

        invalid_arc_nodes = np.sum(
            (arcs < 0) | (arcs >= len(nodes))
        )

        invalid_route_ids = np.sum(
            (route < 0) | (route >= len(arcs))
        )

        start_node = nodes[route_arcs[0, 0]]
        end_node = nodes[route_arcs[-1, 1]]

        start_distance = haversine(
            track[0, 0], track[0, 1],
            start_node[0], start_node[1]
        )

        end_distance = haversine(
            track[-1, 0], track[-1, 1],
            end_node[0], end_node[1]
        )

        return {
            "trip_id": trip_id,
            "track_rows": len(track),
            "node_rows": len(nodes),
            "arc_rows": len(arcs),
            "route_rows": len(route),
            "timestamp_gaps": int(np.sum(diffs > 1)),
            "max_timestamp_gap": float(diffs.max()),
            "disconnected_transitions": int(disconnected),
            "invalid_arc_node_refs": int(invalid_arc_nodes),
            "invalid_route_arc_ids": int(invalid_route_ids),
            "start_distance_m": round(float(start_distance), 2),
            "end_distance_m": round(float(end_distance), 2),
            "status": "OK",
        }

    except Exception as exc:
        return {
            "trip_id": trip_id,
            "status": f"ERROR: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PathSnap training dataset."
    )
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    train_dir = args.dataset_root / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {train_dir}")

    trip_dirs = sorted(
        path for path in train_dir.iterdir()
        if path.is_dir()
    )

    results = [inspect_trip(path) for path in trip_dirs]
    df = pd.DataFrame(results)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "training_dataset_inspection.csv"
    df.to_csv(output_file, index=False)

    successful = df[df["status"] == "OK"]

    print("\n===== TRAINING DATASET INSPECTION =====")
    print("Trips found:", len(trip_dirs))
    print("Successful trips:", len(successful))
    print("Failed trips:", len(df) - len(successful))
    print("Report saved:", output_file)

    if len(successful):
        print("\n===== SUMMARY =====")
        columns = [
            "track_rows", "node_rows", "arc_rows",
            "route_rows", "timestamp_gaps",
            "start_distance_m", "end_distance_m"
        ]
        print(successful[columns].describe().round(2))

        print("\n===== INTEGRITY CHECKS =====")
        for column in [
            "disconnected_transitions",
            "invalid_arc_node_refs",
            "invalid_route_arc_ids",
        ]:
            print(f"{column}: {successful[column].sum()}")

        print("\n===== FIRST 10 TRIPS =====")
        print(successful.head(10).to_string(index=False))

    if len(df) > len(successful):
        print("\n===== ERRORS =====")
        print(df[df["status"] != "OK"].to_string(index=False))


if __name__ == "__main__":
    main()
