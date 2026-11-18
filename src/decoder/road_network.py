
from pathlib import Path

import networkx as nx
import numpy as np


def load_road_network(trip_dir: str | Path):
    """
    Load nodes and directed arcs for one trip.

    Assumption to verify:
    - Node IDs correspond to row indices in the .nodes file.
    - Arc IDs correspond to row indices in the .arcs file.
    """
    trip_dir = Path(trip_dir)

    nodes_file = next(trip_dir.glob("*.nodes"))
    arcs_file = next(trip_dir.glob("*.arcs"))

    nodes = np.loadtxt(nodes_file)
    arcs = np.loadtxt(arcs_file, dtype=int)

    graph = nx.MultiDiGraph()

    for node_id, (lon, lat) in enumerate(nodes):
        graph.add_node(
            node_id,
            lon=float(lon),
            lat=float(lat),
        )

    for arc_id, (source, target) in enumerate(arcs):
        graph.add_edge(
            int(source),
            int(target),
            key=arc_id,
            arc_id=arc_id,
        )

    return graph


if __name__ == "__main__":
    graph = load_road_network("Student folder/train/00000001")

    print("Nodes:", graph.number_of_nodes())
    print("Directed arcs:", graph.number_of_edges())
    print("Is directed:", graph.is_directed())
