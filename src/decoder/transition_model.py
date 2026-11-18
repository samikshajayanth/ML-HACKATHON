import math
import networkx as nx


class TransitionModel:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.arc_endpoints = {}

        for source, target, data in graph.edges(data=True):
            self.arc_endpoints[data["arc_id"]] = (source, target)

        self._distance_cache = {}

    def score(self, previous_arc: int, current_arc: int, dt: float = 1.0) -> float:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a positive finite number")

        if previous_arc == current_arc:
            return 0.0

        if previous_arc not in self.arc_endpoints:
            return -12.0

        if current_arc not in self.arc_endpoints:
            return -12.0

        _, previous_end = self.arc_endpoints[previous_arc]
        current_start, _ = self.arc_endpoints[current_arc]

        if previous_end == current_start:
            return -0.5

        key = (previous_end, current_start)

        if key not in self._distance_cache:
            try:
                distance = nx.shortest_path_length(
                    self.graph,
                    source=previous_end,
                    target=current_start,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance = math.inf

            self._distance_cache[key] = distance

        distance = self._distance_cache[key]

        if not math.isfinite(distance):
            return -12.0

        return -0.5 - (distance / 100.0)