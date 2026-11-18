
from pathlib import Path
import numpy as np


class TripData:
    """Load and expose all files belonging to one trip."""

    def __init__(self, trip_dir):
        self.trip_dir = Path(trip_dir)
        self.trip_id = self.trip_dir.name

        self.track = self._load(
            "track", dtype=float
        )
        self.nodes = self._load(
            "nodes", dtype=float
        )
        self.arcs = self._load(
            "arcs", dtype=int
        )
        self.route = self._load(
            "route", dtype=int
        )

        self.track = np.atleast_2d(self.track)
        self.nodes = np.atleast_2d(self.nodes)
        self.arcs = np.atleast_2d(self.arcs)
        self.route = np.atleast_1d(self.route)

    def _load(self, extension, dtype):
        path = self.trip_dir / (
            f"{self.trip_id}.{extension}"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        return np.loadtxt(path, dtype=dtype)

    @property
    def is_training(self):
        return self.route is not None

    def get_route_nodes(self):
        """Return ordered node IDs from supplied route arcs."""
        route_arcs = self.arcs[self.route]

        return np.concatenate([
            route_arcs[:1, 0],
            route_arcs[:, 1]
        ])

    def validate(self):
        """Check basic structure and index validity."""
        assert self.track.shape[1] == 3
        assert self.nodes.shape[1] == 2
        assert self.arcs.shape[1] == 2

        assert np.all(self.arcs >= 0)
        assert np.all(self.arcs < len(self.nodes))

        assert np.all(self.route >= 0)
        assert np.all(self.route < len(self.arcs))

        return True


def load_trip(dataset_root, split, trip_id):
    """
    Load one trip.

    split: train or student_test
    """
    trip_dir = (
        Path(dataset_root) / split / str(trip_id)
    )

    trip = TripData(trip_dir)
    trip.validate()

    return trip
