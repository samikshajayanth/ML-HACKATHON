
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from shapely import LineString, Point, STRtree


class CandidateGenerator:
    """
    Generate nearby directed road-arc candidates for GPS observations.

    Ground-truth route labels are deliberately not used here.
    """

    def __init__(self, radius_m=100, max_candidates=8):
        self.radius_m = radius_m
        self.max_candidates = max_candidates

    @staticmethod
    def _project_coordinates(track, nodes):
        """Project coordinates into the local UTM zone."""

        longitude = float(np.mean(nodes[:, 0]))
        latitude = float(np.mean(nodes[:, 1]))

        zone = int((longitude + 180) // 6) + 1

        if latitude >= 0:
            epsg = 32600 + zone
        else:
            epsg = 32700 + zone

        transformer = Transformer.from_crs(
            "EPSG:4326",
            CRS.from_epsg(epsg),
            always_xy=True
        )

        node_x, node_y = transformer.transform(
            nodes[:, 0], nodes[:, 1]
        )

        gps_x, gps_y = transformer.transform(
            track[:, 0], track[:, 1]
        )

        projected_nodes = np.column_stack(
            [node_x, node_y]
        )

        projected_gps = np.column_stack(
            [gps_x, gps_y]
        )

        return projected_nodes, projected_gps, epsg

    def generate(self, trip):
        """
        Parameters
        ----------
        trip : TripData
            Loaded trip object from src.data.loader.

        Returns
        -------
        pandas.DataFrame
            Candidate records, one row per observation-candidate pair.
        """

        track = trip.track
        nodes = trip.nodes
        arcs = trip.arcs

        projected_nodes, projected_gps, epsg = (
            self._project_coordinates(track, nodes)
        )

        # Convert every directed arc into a road LineString.
        arc_coordinates = projected_nodes[arcs]

        geometries = [
            LineString(coords)
            for coords in arc_coordinates
        ]

        # Build spatial index once per trip.
        tree = STRtree(geometries)

        records = []

        for observation_id, (x, y) in enumerate(projected_gps):
            point = Point(x, y)

            # Find all road arcs within the search radius.
            indices = tree.query(
                point,
                predicate="dwithin",
                distance=self.radius_m
            )

            fallback = False

            # If none are nearby, retain the nearest road arc.
            if len(indices) == 0:
                nearest_index = int(tree.nearest(point))
                indices = np.array([nearest_index])
                fallback = True

            # Calculate exact point-to-segment distances.
            candidates = []

            for arc_index in indices:
                arc_index = int(arc_index)
                line = geometries[arc_index]

                distance = float(line.distance(point))

                projected_position = line.project(point)

                fraction = (
                    projected_position / line.length
                    if line.length > 0 else 0.0
                )

                candidates.append({
                    "arc_id": arc_index,
                    "distance_m": distance,
                    "fraction_along_arc": fraction,
                    "fallback": fallback
                })

            # Keep the nearest candidates.
            candidates.sort(
                key=lambda item: item["distance_m"]
            )

            candidates = candidates[:self.max_candidates]

            for candidate_rank, candidate in enumerate(candidates):
                arc_id = candidate["arc_id"]

                records.append({
                    "trip_id": trip.trip_id,
                    "observation_id": observation_id,
                    "timestamp": float(track[observation_id, 2]),
                    "candidate_rank": candidate_rank,
                    "candidate_arc_id": arc_id,
                    "from_node_id": int(arcs[arc_id, 0]),
                    "to_node_id": int(arcs[arc_id, 1]),
                    "distance_m": candidate["distance_m"],
                    "fraction_along_arc":
                        candidate["fraction_along_arc"],
                    "is_fallback": candidate["fallback"]
                })

        result = pd.DataFrame(records)

        result.attrs["projected_epsg"] = epsg
        result.attrs["search_radius_m"] = self.radius_m

        return result
