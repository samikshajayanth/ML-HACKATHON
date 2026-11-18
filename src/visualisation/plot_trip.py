
from pathlib import Path
import argparse

import folium
import numpy as np


def create_map(dataset_root, trip_id, output_dir):
    folder = dataset_root / "train" / trip_id

    track = np.loadtxt(folder / f"{trip_id}.track")
    nodes = np.loadtxt(folder / f"{trip_id}.nodes")
    arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=int)
    route = np.atleast_1d(
        np.loadtxt(folder / f"{trip_id}.route", dtype=int)
    ).astype(int)

    route_arcs = arcs[route]
    route_coords = nodes[
        np.concatenate([route_arcs[:1, 0], route_arcs[:, 1]])
    ]

    center = track[:, :2].mean(axis=0)

    m = folium.Map(
        location=[center[1], center[0]],
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    # GPS observations
    gps_coords = [
        [lat, lon] for lon, lat in track[:, :2]
    ]

    folium.PolyLine(
        gps_coords,
        color="blue",
        weight=3,
        opacity=0.75,
        tooltip="Noisy GPS track"
    ).add_to(m)

    # Draw route as separate continuous segments.
    # Do not draw artificial lines across disconnected transitions.
    segment = [route_coords[0]]

    for i in range(len(route_arcs) - 1):
        current_end = route_arcs[i, 1]
        next_start = route_arcs[i + 1, 0]

        if current_end == next_start:
            segment.append(route_coords[i + 2])
        else:
            if len(segment) >= 2:
                folium.PolyLine(
                    [[p[1], p[0]] for p in segment],
                    color="red",
                    weight=4,
                    opacity=0.8,
                    tooltip="Ground-truth route"
                ).add_to(m)

            segment = [
                nodes[next_start],
                nodes[route_arcs[i + 1, 1]]
            ]

    if len(segment) >= 2:
        folium.PolyLine(
            [[p[1], p[0]] for p in segment],
            color="red",
            weight=4,
            opacity=0.8,
            tooltip="Ground-truth route"
        ).add_to(m)

    # Start and end markers
    folium.Marker(
        gps_coords[0],
        tooltip="GPS start",
        icon=folium.Icon(color="green")
    ).add_to(m)

    folium.Marker(
        gps_coords[-1],
        tooltip="GPS end",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"trip_{trip_id}_map.html"
    m.save(str(output_file))

    print("Map saved:", output_file.resolve())
    print("GPS observations:", len(track))
    print("Ground-truth route arcs:", len(route))
    print(
        "Disconnected route transitions:",
        int(np.sum(route_arcs[:-1, 1] != route_arcs[1:, 0]))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--trip", default="00000001")
    args = parser.parse_args()

    create_map(
        args.dataset_root,
        args.trip,
        Path("outputs/maps")
    )
