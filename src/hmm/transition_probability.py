"""Transition probability model for HMM map matching.

Implements:
1. RoadNetworkGraph: Directed road network with node-to-node shortest path queries,
   Dijkstra caching, and arc length computation.
2. TransitionModel: Computes transition log-probabilities based on the difference
   between road network shortest path distance and Euclidean GPS displacement
   (Newson & Krumm 2009 formulation).
"""
from __future__ import annotations

import math
from typing import Any
import networkx as nx
import numpy as np


class RoadNetworkGraph:
    """Directed graph representation of a road network for shortest path calculations."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray, projected_nodes: np.ndarray | None = None):
        """
        Parameters
        ----------
        nodes : np.ndarray
            (N, 2) array of (lon, lat)
        arcs : np.ndarray
            (M, 2) array of (from_node_id, to_node_id)
        projected_nodes : np.ndarray | None
            (N, 2) array of UTM projected (x, y) coordinates. If None, Euclidean
            distances on projected coords should be provided or computed.
        """
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)
        self.arcs = np.asarray(arcs, dtype=int)

        if projected_nodes is not None:
            self.projected_nodes = np.asarray(projected_nodes, dtype=float)
            arc_v = self.projected_nodes[self.arcs[:, 1]] - self.projected_nodes[self.arcs[:, 0]]
            self.arc_lengths = np.linalg.norm(arc_v, axis=1)
        else:
            self.projected_nodes = None
            # Fallback approximate Euclidean on degrees * 111,000m if projected not given
            dlat = (nodes[self.arcs[:, 1], 1] - nodes[self.arcs[:, 0], 1]) * 111139.0
            dlon = (nodes[self.arcs[:, 1], 0] - nodes[self.arcs[:, 0], 0]) * (
                111139.0 * np.cos(np.radians(np.mean(nodes[:, 1])))
            )
            self.arc_lengths = np.hypot(dlon, dlat)

        self.graph = nx.DiGraph()
        for arc_id in range(self.num_arcs):
            u = int(self.arcs[arc_id, 0])
            v = int(self.arcs[arc_id, 1])
            w = float(self.arc_lengths[arc_id])
            self.graph.add_edge(u, v, weight=w, arc_id=arc_id)

        self._shortest_cache: dict[int, dict[int, float]] = {}

    def get_arc_length(self, arc_id: int) -> float:
        if 0 <= arc_id < len(self.arc_lengths):
            return float(self.arc_lengths[arc_id])
        return 0.0

    def shortest_path_distance(self, source_node: int, target_node: int, cutoff: float = 2500.0) -> float:
        """Find network shortest path distance between source and target nodes."""
        if source_node == target_node:
            return 0.0

        if source_node not in self.graph or target_node not in self.graph:
            return float("inf")

        # Check local cache for source_node
        if source_node not in self._shortest_cache:
            if len(self._shortest_cache) > 2000:
                self._shortest_cache.clear()
            try:
                dists = nx.single_source_dijkstra_path_length(
                    self.graph, source_node, cutoff=cutoff, weight="weight"
                )
                self._shortest_cache[source_node] = dists
            except Exception:
                self._shortest_cache[source_node] = {}

        return self._shortest_cache[source_node].get(target_node, float("inf"))


class TransitionModel:
    """Transition probability model: p(c_t | c_{t-1}) = (1 / beta) * exp(-|d_net - d_gps| / beta).

    In log-space: log p = -log(beta) - (|d_net - d_gps| / beta).
    """

    def __init__(
        self,
        road_network: RoadNetworkGraph,
        beta: float = 10.0,
        unreachable_penalty: float = 50.0,
        max_cutoff_m: float = 3000.0,
        backwards_penalty_m: float = 200.0,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.road_network = road_network
        self.beta = float(beta)
        self.unreachable_penalty = float(unreachable_penalty)
        self.max_cutoff_m = float(max_cutoff_m)
        self.backwards_penalty_m = float(backwards_penalty_m)
        self._log_beta = math.log(self.beta)

    def compute_network_distance(
        self,
        from_arc_id: int,
        from_fraction: float,
        to_arc_id: int,
        to_fraction: float,
        d_gps: float = 0.0,
    ) -> float:
        """Calculate road network distance between two projected candidate points."""
        from_arc_id = int(from_arc_id)
        to_arc_id = int(to_arc_id)
        from_len = self.road_network.get_arc_length(from_arc_id)
        to_len = self.road_network.get_arc_length(to_arc_id)

        from_frac = float(np.clip(from_fraction, 0.0, 1.0))
        to_frac = float(np.clip(to_fraction, 0.0, 1.0))

        # Case 1: Same arc
        if from_arc_id == to_arc_id:
            if to_frac >= from_frac:
                return (to_frac - from_frac) * from_len
            else:
                # Vehicle moving backwards along one-way arc
                # Try finding loop in graph from v to u
                u = int(self.road_network.arcs[from_arc_id, 0])
                v = int(self.road_network.arcs[from_arc_id, 1])
                loop_dist = self.road_network.shortest_path_distance(v, u, cutoff=self.max_cutoff_m)
                if math.isfinite(loop_dist):
                    return (1.0 - from_frac) * from_len + loop_dist + to_frac * to_len
                return (from_frac - to_frac) * from_len + self.backwards_penalty_m

        # Case 2: Different arcs
        u1 = int(self.road_network.arcs[from_arc_id, 0])
        v1 = int(self.road_network.arcs[from_arc_id, 1])
        u2 = int(self.road_network.arcs[to_arc_id, 0])
        v2 = int(self.road_network.arcs[to_arc_id, 1])

        # Subcase 2A: Direct topological connection (v1 == u2)
        rem_from = (1.0 - from_frac) * from_len
        enter_to = to_frac * to_len
        if v1 == u2:
            return rem_from + enter_to

        # Subcase 2B: Intermediate path through graph
        cutoff = max(self.max_cutoff_m, d_gps * 3.0 + 500.0)
        graph_dist = self.road_network.shortest_path_distance(v1, u2, cutoff=cutoff)
        if math.isfinite(graph_dist):
            return rem_from + graph_dist + enter_to

        return float("inf")

    def log_transition_probability(
        self,
        from_candidate: dict[str, Any],
        to_candidate: dict[str, Any],
        d_gps: float,
    ) -> float:
        """Compute log p(to_candidate | from_candidate)."""
        from_arc = from_candidate.get("candidate_arc_id", from_candidate.get("candidate_id", 0))
        from_frac = from_candidate.get("fraction_along_arc", from_candidate.get("proj_frac", 0.0))
        to_arc = to_candidate.get("candidate_arc_id", to_candidate.get("candidate_id", 0))
        to_frac = to_candidate.get("fraction_along_arc", to_candidate.get("proj_frac", 0.0))

        d_net = self.compute_network_distance(from_arc, from_frac, to_arc, to_frac, d_gps=d_gps)

        if not math.isfinite(d_net):
            return -self.unreachable_penalty

        delta = abs(d_net - d_gps)
        log_prob = -self._log_beta - (delta / self.beta)
        return max(log_prob, -self.unreachable_penalty)

    def transition_matrix(
        self,
        prev_candidates: list[dict[str, Any]],
        curr_candidates: list[dict[str, Any]],
        d_gps: float,
    ) -> np.ndarray:
        """Compute log transition matrix of shape (len(prev_candidates), len(curr_candidates))."""
        n_prev = len(prev_candidates)
        n_curr = len(curr_candidates)
        mat = np.empty((n_prev, n_curr), dtype=float)

        for i in range(n_prev):
            for j in range(n_curr):
                mat[i, j] = self.log_transition_probability(
                    prev_candidates[i], curr_candidates[j], d_gps
                )
        return mat
