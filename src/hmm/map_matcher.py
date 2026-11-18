"""End-to-end Hidden Markov Model Map Matcher for PathSnap.

Orchestrates candidate grouping, emission probabilities, transition probabilities,
Viterbi decoding, sequence reconstruction, and route metrics comparison.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from .emission_probability import (
    EmissionModel,
    GaussianEmissionModel,
    HybridEmissionModel,
    MLEmissionModel,
)
from .transition_probability import RoadNetworkGraph, TransitionModel
from .viterbi import ViterbiDecoder
from ..features.candidates import CandidateGenerator


class HMMMapMatcher:
    """Full HMM Map-Matching pipeline integrating GPS observations, candidates,

    machine learning predictions, and road-network topology.
    """

    def __init__(
        self,
        sigma_z: float = 10.0,
        beta: float = 10.0,
        emission_mode: str = "hybrid",
        alpha: float = 0.5,
        unreachable_penalty: float = 50.0,
    ):
        """
        Parameters
        ----------
        sigma_z : float
            GPS observation measurement uncertainty standard deviation in meters.
        beta : float
            Exponential transition parameter in meters.
        emission_mode : str
            One of 'hybrid', 'ml', 'gaussian'.
        alpha : float
            Weight for ML in hybrid mode: alpha * log(ML) + (1 - alpha) * log(Gaussian).
        unreachable_penalty : float
            Log penalty for topologically unreachable candidate transitions.
        """
        self.sigma_z = float(sigma_z)
        self.beta = float(beta)
        self.emission_mode = emission_mode
        self.alpha = float(alpha)
        self.unreachable_penalty = float(unreachable_penalty)

        if emission_mode == "gaussian":
            self.emission_model: EmissionModel = GaussianEmissionModel(sigma_z=sigma_z)
        elif emission_mode == "ml":
            self.emission_model = MLEmissionModel()
        elif emission_mode == "hybrid":
            self.emission_model = HybridEmissionModel(sigma_z=sigma_z, alpha=alpha)
        else:
            raise ValueError(f"Unknown emission_mode: {emission_mode}. Choose 'hybrid', 'ml', or 'gaussian'")

        self.viterbi = ViterbiDecoder(unreachable_penalty=unreachable_penalty)

    def match(
        self,
        trip,
        candidates_df: pd.DataFrame,
        scored_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Match a trip's GPS track to the road network.

        Parameters
        ----------
        trip : TripData
            Loaded trip object with track, nodes, arcs, and optional route.
        candidates_df : pd.DataFrame
            Generated candidate road segments.
        scored_df : pd.DataFrame | None
            ML model scored candidates (with 'score', 'prob_norm', 'rank').
            If provided, merges scores into candidate layers.

        Returns
        -------
        dict with:
            matched_arcs: list[int]
            ml_top1_arcs: list[int]
            matched_coordinates: list[list[float]]  ([lat, lon] for Folium)
            ml_coordinates: list[list[float]]
            continuity_hmm: float
            continuity_ml: float
            accuracy_hmm: float | None
            accuracy_ml: float | None
            log_likelihood: float
            num_observations: int
            evaluation_summary: dict
        """
        # Ensure projected coordinates
        projected_nodes, projected_gps, epsg = CandidateGenerator._project_coordinates(
            trip.track, trip.nodes
        )

        # Build road network graph
        road_network = RoadNetworkGraph(
            trip.nodes, trip.arcs, projected_nodes=projected_nodes
        )
        transition_model = TransitionModel(
            road_network=road_network,
            beta=self.beta,
            unreachable_penalty=self.unreachable_penalty,
        )

        # Prepare candidate layers per observation
        merged_cands = candidates_df.copy()
        if scored_df is not None:
            # Merge score, prob_norm, and rank
            score_cols = [c for c in ["score", "prob_norm", "rank"] if c in scored_df.columns]
            join_keys = ["obs_id", "candidate_id"]
            cand_id_col = "candidate_arc_id" if "candidate_arc_id" in merged_cands.columns else "candidate_id"
            obs_id_col = "observation_id" if "observation_id" in merged_cands.columns else "obs_id"

            merged_cands["_join_obs"] = merged_cands[obs_id_col]
            merged_cands["_join_cand"] = merged_cands[cand_id_col]

            scored_temp = scored_df.copy()
            s_cand_col = "candidate_arc_id" if "candidate_arc_id" in scored_temp.columns else "candidate_id"
            s_obs_col = "observation_id" if "observation_id" in scored_temp.columns else "obs_id"
            scored_temp["_join_obs"] = scored_temp[s_obs_col]
            scored_temp["_join_cand"] = scored_temp[s_cand_col]

            to_merge = scored_temp[["_join_obs", "_join_cand"] + score_cols].drop_duplicates(
                subset=["_join_obs", "_join_cand"]
            )
            merged_cands = pd.merge(
                merged_cands, to_merge, on=["_join_obs", "_join_cand"], how="left"
            ).drop(columns=["_join_obs", "_join_cand"])

        obs_col = "observation_id" if "observation_id" in merged_cands.columns else "obs_id"
        candidate_layers: list[list[dict[str, Any]]] = []
        emission_log_probs: list[np.ndarray] = []
        ml_top1_arcs: list[int] = []

        # Group by observation index
        grouped = merged_cands.groupby(obs_col, sort=True)
        obs_indices = sorted(grouped.groups.keys())

        for obs_idx in obs_indices:
            grp = grouped.get_group(obs_idx)
            layer = grp.to_dict(orient="records")
            candidate_layers.append(layer)

            # Compute emissions for this layer
            obs_pt = projected_gps[obs_idx] if obs_idx < len(projected_gps) else np.zeros(2)
            log_emit = self.emission_model.log_emission_probabilities(obs_pt, grp)
            emission_log_probs.append(np.asarray(log_emit, dtype=float))

            # Record ML / nearest top-1 arc
            if "score" in grp.columns:
                best_cand_row = grp.sort_values(by=["score", "distance_m"], ascending=[False, True]).iloc[0]
            else:
                dist_col = "dist_m" if "dist_m" in grp.columns else "distance_m"
                best_cand_row = grp.sort_values(by=dist_col, ascending=True).iloc[0]
            arc_key = "candidate_arc_id" if "candidate_arc_id" in best_cand_row else "candidate_id"
            ml_top1_arcs.append(int(best_cand_row[arc_key]))

        # Define transition function for Viterbi
        def transition_fn(t: int, prev_cands: list[dict], curr_cands: list[dict]) -> np.ndarray:
            t_prev = obs_indices[t - 1]
            t_curr = obs_indices[t]
            if t_curr < len(projected_gps) and t_prev < len(projected_gps):
                d_gps = float(np.linalg.norm(projected_gps[t_curr] - projected_gps[t_prev]))
            else:
                d_gps = 10.0
            return transition_model.transition_matrix(prev_cands, curr_cands, d_gps=d_gps)

        # Run Viterbi
        viterbi_res = self.viterbi.decode(
            candidate_layers=candidate_layers,
            emission_probs=emission_log_probs,
            transition_fn=transition_fn,
        )

        matched_arcs = viterbi_res["matched_arcs"]

        # Compute route continuity
        continuity_hmm = self.compute_continuity(matched_arcs, trip.arcs)
        continuity_ml = self.compute_continuity(ml_top1_arcs, trip.arcs)

        # Compute accuracy if route labels are available
        accuracy_hmm, accuracy_ml = None, None
        if hasattr(trip, "route") and trip.route is not None and len(trip.route) > 0:
            accuracy_hmm = self.compute_route_accuracy(matched_arcs, trip.route)
            accuracy_ml = self.compute_route_accuracy(ml_top1_arcs, trip.route)

        # Extract coordinates for Folium ([lat, lon])
        matched_coords = self.arcs_to_coordinates(matched_arcs, trip.nodes, trip.arcs)
        ml_coords = self.arcs_to_coordinates(ml_top1_arcs, trip.nodes, trip.arcs)
        gps_coords = [[float(row[1]), float(row[0])] for row in trip.track]

        return {
            "trip_id": trip.trip_id,
            "num_observations": len(trip.track),
            "num_matched_points": len(matched_arcs),
            "matched_arcs": matched_arcs,
            "ml_top1_arcs": ml_top1_arcs,
            "log_likelihood": viterbi_res["log_likelihood"],
            "restarts": viterbi_res["num_fallback_restarts"],
            "continuity_hmm": continuity_hmm,
            "continuity_ml": continuity_ml,
            "accuracy_hmm": accuracy_hmm,
            "accuracy_ml": accuracy_ml,
            "matched_coordinates": matched_coords,
            "ml_coordinates": ml_coords,
            "gps_coordinates": gps_coords,
            "evaluation_summary": {
                "continuity_improvement_percent": round((continuity_hmm - continuity_ml) * 100, 2),
                "accuracy_improvement_percent": (
                    round((accuracy_hmm - accuracy_ml) * 100, 2)
                    if accuracy_hmm is not None and accuracy_ml is not None
                    else None
                ),
            },
        }

    @staticmethod
    def compute_continuity(arc_sequence: list[int], arcs: np.ndarray) -> float:
        """Calculate percentage of consecutive transitions that are topologically continuous.

        Two consecutive arcs a1 and a2 are continuous if:
        1. a1 == a2 (vehicle remains on same segment), or
        2. to_node(a1) == from_node(a2) (vehicle moves directly to connected segment).
        """
        if len(arc_sequence) <= 1:
            return 1.0

        continuous_steps = 0
        total_transitions = len(arc_sequence) - 1

        for i in range(total_transitions):
            a1 = arc_sequence[i]
            a2 = arc_sequence[i + 1]
            if a1 == a2:
                continuous_steps += 1
            else:
                end_node_a1 = arcs[a1, 1]
                start_node_a2 = arcs[a2, 0]
                if end_node_a1 == start_node_a2:
                    continuous_steps += 1

        return float(continuous_steps / total_transitions)

    @staticmethod
    def compute_route_accuracy(predicted_arcs: list[int], ground_truth_route: np.ndarray) -> float:
        """Percentage of predicted arcs that appear in the ground truth route sequence."""
        if not len(predicted_arcs):
            return 0.0
        gt_set = set(int(a) for a in ground_truth_route)
        hits = sum(1 for a in predicted_arcs if a in gt_set)
        return float(hits / len(predicted_arcs))

    @staticmethod
    def arcs_to_coordinates(
        arc_sequence: list[int],
        nodes: np.ndarray,
        arcs: np.ndarray,
    ) -> list[list[float]]:
        """Convert a sequence of arcs to ordered [latitude, longitude] coordinates."""
        if not arc_sequence:
            return []

        coords: list[list[float]] = []
        for i, arc_id in enumerate(arc_sequence):
            u = int(arcs[arc_id, 0])
            v = int(arcs[arc_id, 1])
            u_pt = [float(nodes[u, 1]), float(nodes[u, 0])]  # lat, lon
            v_pt = [float(nodes[v, 1]), float(nodes[v, 0])]  # lat, lon

            if not coords:
                coords.extend([u_pt, v_pt])
            else:
                last_pt = coords[-1]
                # Avoid exact duplicates
                if last_pt != u_pt:
                    coords.append(u_pt)
                if coords[-1] != v_pt:
                    coords.append(v_pt)
        return coords
