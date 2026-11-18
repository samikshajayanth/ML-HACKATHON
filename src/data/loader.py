
from pathlib import Path
import numpy as np


class TripData:
    """Load one trip without exposing reference routes by default."""

    def __init__(self, trip_dir, include_route=False):
        self.trip_dir = Path(trip_dir)
        self.trip_id = self.trip_dir.name

        self.track = np.atleast_2d(
            self._load("track", dtype=float)
        )

        self.nodes = np.atleast_2d(
            self._load("nodes", dtype=float)
        )

        self.arcs = np.atleast_2d(
            self._load("arcs", dtype=int)
        )

        # Reference route is loaded only when explicitly requested.
        self.route = None

        if include_route:
            self.route = np.atleast_1d(
                self._load("route", dtype=int)
            )

    def _load(self, extension, dtype):
        path = self.trip_dir / f"{self.trip_id}.{extension}"

        if not path.is_file():
            raise FileNotFoundError(
                f"Required trip file not found: {path}"
            )

        return np.loadtxt(path, dtype=dtype)

    @property
    def is_training(self):
        return self.route is not None

    def get_route_nodes(self):
        if self.route is None:
            raise ValueError(
                "Reference route is not loaded for this trip."
            )

        if len(self.route) == 0:
            raise ValueError("Reference route is empty.")

        route_arcs = self.arcs[self.route]

        return np.concatenate([
            route_arcs[:1, 0],
            route_arcs[:, 1]
        ])

    def validate(self):
        if self.track.ndim != 2 or self.track.shape[1] != 3:
            raise ValueError("Track must contain 3 columns.")

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("Nodes must contain 2 columns.")

        if self.arcs.ndim != 2 or self.arcs.shape[1] != 2:
            raise ValueError("Arcs must contain 2 columns.")

        if len(self.nodes) == 0 or len(self.arcs) == 0:
            raise ValueError("Road network cannot be empty.")

        if not np.isfinite(self.track).all():
            raise ValueError("Track contains non-finite values.")

        if not np.isfinite(self.nodes).all():
            raise ValueError("Nodes contain non-finite values.")

        if np.any(self.arcs < 0):
            raise ValueError("Arc node IDs cannot be negative.")

        if np.any(self.arcs >= len(self.nodes)):
            raise ValueError("Arc references an invalid node ID.")

        if self.route is not None:
            if len(self.route) == 0:
                raise ValueError("Reference route is empty.")

            if np.any(self.route < 0):
                raise ValueError("Route contains negative arc IDs.")

            if np.any(self.route >= len(self.arcs)):
                raise ValueError("Route references an invalid arc ID.")

        return True


def load_trip(dataset_root, split, trip_id):
    """
    Load a trip from train or student_test.

    Training trips load their reference routes.
    Student-test trips never load reference routes.
    """

    if split not in ("train", "student_test"):
        raise ValueError(
            "split must be 'train' or 'student_test'."
        )

    trip_dir = Path(dataset_root) / split / str(trip_id)

    trip = TripData(
        trip_dir,
        include_route=(split == "train")
    )

    trip.validate()

    return trip
