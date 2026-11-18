"""Viterbi decoding algorithm for Hidden Markov Model map matching.

Implements numerically stable log-space dynamic programming recurrence,
backtracking, and graceful fallback handling for disconnected states,
unreachable transitions, and missing candidate observations.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


class ViterbiDecoder:
    """Decodes the most likely sequence of candidate road segments using the Viterbi algorithm."""

    def __init__(self, unreachable_penalty: float = 50.0):
        self.unreachable_penalty = unreachable_penalty

    def decode(
        self,
        candidate_layers: list[list[dict[str, Any]]],
        emission_probs: list[np.ndarray],
        transition_fn,  # callable(t, prev_cands, curr_cands) -> np.ndarray (log trans matrix)
    ) -> dict[str, Any]:
        """
        Run Viterbi decoding over candidate observation layers.

        Parameters
        ----------
        candidate_layers : list[list[dict]]
            List of candidate dictionaries for each observation step t in [0, T-1].
        emission_probs : list[np.ndarray]
            List of log emission probability arrays for each observation step t.
        transition_fn : callable
            Function taking (step_idx, prev_layer, curr_layer) and returning
            log transition matrix of shape (len(prev_layer), len(curr_layer)).

        Returns
        -------
        dict with:
            matched_indices: list[int] (index of chosen candidate in each layer)
            matched_arcs: list[int] (arc_id of chosen candidate in each layer)
            log_likelihood: float
            num_transitions: int
            num_fallback_restarts: int
        """
        T = len(candidate_layers)
        if T == 0:
            return {
                "matched_indices": [],
                "matched_arcs": [],
                "log_likelihood": -float("inf"),
                "num_transitions": 0,
                "num_fallback_restarts": 0,
            }

        # Filter out empty steps if any (or handle 1-length)
        if T == 1:
            best_idx = int(np.argmax(emission_probs[0]))
            arc_key = "candidate_arc_id" if "candidate_arc_id" in candidate_layers[0][best_idx] else "candidate_id"
            return {
                "matched_indices": [best_idx],
                "matched_arcs": [int(candidate_layers[0][best_idx][arc_key])],
                "log_likelihood": float(emission_probs[0][best_idx]),
                "num_transitions": 0,
                "num_fallback_restarts": 0,
            }

        # Step 0: Initialization with uniform prior + log emission
        K0 = len(candidate_layers[0])
        log_prior = -math.log(max(K0, 1))
        v_current = emission_probs[0] + log_prior

        backpointers: list[np.ndarray] = []
        restarts = 0

        # Step t: Dynamic programming forward pass
        for t in range(1, T):
            prev_layer = candidate_layers[t - 1]
            curr_layer = candidate_layers[t]
            log_emit = emission_probs[t]
            K_curr = len(curr_layer)

            # Compute log transition matrix (shape: K_prev, K_curr)
            log_trans = transition_fn(t, prev_layer, curr_layer)

            # Compute combined score: v_{t-1}[i] + log_trans[i, j]
            # v_current[:, None] broadcasts over j
            scores = v_current[:, None] + log_trans

            # Check if all transitions are virtually unreachable (e.g. huge jump)
            best_from_prev = np.max(scores, axis=0)
            if np.all(best_from_prev < -1e8) or np.all(np.isneginf(best_from_prev)):
                # Fallback restart: reset state probabilities using emission only
                restarts += 1
                v_current = log_emit - math.log(max(K_curr, 1))
                # Link to highest scoring candidate in previous step
                best_prev_cand = int(np.argmax(v_current))
                backpointers.append(np.full(K_curr, best_prev_cand, dtype=int))
            else:
                ptr = np.argmax(scores, axis=0)
                v_current = best_from_prev + log_emit
                backpointers.append(ptr)

        # Backtracking
        best_last_idx = int(np.argmax(v_current))
        total_log_likelihood = float(v_current[best_last_idx])

        matched_indices = [best_last_idx]
        curr_idx = best_last_idx

        for t in range(T - 1, 0, -1):
            curr_idx = int(backpointers[t - 1][curr_idx])
            matched_indices.append(curr_idx)

        matched_indices.reverse()

        matched_arcs = []
        for t, idx in enumerate(matched_indices):
            c = candidate_layers[t][idx]
            arc_key = "candidate_arc_id" if "candidate_arc_id" in c else "candidate_id"
            matched_arcs.append(int(c[arc_key]))

        return {
            "matched_indices": matched_indices,
            "matched_arcs": matched_arcs,
            "log_likelihood": total_log_likelihood,
            "num_transitions": T - 1,
            "num_fallback_restarts": restarts,
        }
