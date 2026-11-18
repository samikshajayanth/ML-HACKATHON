
import numpy as np
import pandas as pd

from src.features.candidates import CandidateGenerator


def calculate_heading(dx, dy):
    """Bearing clockwise from projected grid north."""
    return np.degrees(np.arctan2(dx, dy)) % 360


def calculate_heading_difference(a, b):
    """Smallest angular difference in degrees."""
    return np.abs((a - b + 180) % 360 - 180)


def calculate_gps_motion(track, projected_gps):
    """
    Estimate local GPS speed and heading.

    Uses forward differences at the first point,
    backward differences at the last point,
    and central differences for interior points.
    """
    n = len(track)

    heading = np.full(n, np.nan)
    speed = np.full(n, np.nan)

    for i in range(n):
        if n < 2:
            continue

        if i == 0:
            left, right = 0, 1
        elif i == n - 1:
            left, right = n - 2, n - 1
        else:
            left, right = i - 1, i + 1

        dt = track[right, 2] - track[left, 2]

        if dt <= 0:
            continue

        dx = projected_gps[right, 0] - projected_gps[left, 0]
        dy = projected_gps[right, 1] - projected_gps[left, 1]

        distance = np.hypot(dx, dy)

        speed[i] = distance / dt

        if distance > 0:
            heading[i] = calculate_heading(dx, dy)

    return heading, speed


def extract_features(trip, candidates):
    """
    Add GPS movement and road geometry features.

    Ground-truth route labels are never accessed.
    """

    result = candidates.copy()

    projected_nodes, projected_gps, epsg = (
        CandidateGenerator._project_coordinates(
            trip.track, trip.nodes
        )
    )

    gps_heading, speed = calculate_gps_motion(
        trip.track, projected_gps
    )

    # Directed road geometry
    arc_vectors = (
        projected_nodes[trip.arcs[:, 1]]
        - projected_nodes[trip.arcs[:, 0]]
    )

    arc_lengths = np.linalg.norm(
        arc_vectors, axis=1
    )

    arc_headings = calculate_heading(
        arc_vectors[:, 0],
        arc_vectors[:, 1]
    )

    observation_ids = result["observation_id"].to_numpy(
        dtype=int
    )
    arc_ids = result["candidate_arc_id"].to_numpy(
        dtype=int
    )

    result["gps_heading_deg"] = gps_heading[observation_ids]
    result["speed_mps"] = speed[observation_ids]

    result["arc_heading_deg"] = arc_headings[arc_ids]

    result["heading_diff_deg"] = (
        calculate_heading_difference(
            result["gps_heading_deg"].to_numpy(),
            result["arc_heading_deg"].to_numpy()
        )
    )

    result["arc_length_m"] = arc_lengths[arc_ids]

    # Time since previous observation.
    timestamp_gaps = np.full(len(trip.track), np.nan)

    if len(trip.track) > 1:
        timestamp_gaps[1:] = np.diff(trip.track[:, 2])

    result["timestamp_gap_s"] = timestamp_gaps[
        observation_ids
    ]

    result["projected_epsg"] = epsg

    return result
