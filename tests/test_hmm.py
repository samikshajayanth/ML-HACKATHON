"""Tests for the Hidden Markov Model (HMM) and Viterbi Map-Matching components."""
import math
import numpy as np
import pandas as pd
import pytest

from src.hmm.emission_probability import (
    GaussianEmissionModel,
    MLEmissionModel,
    HybridEmissionModel,
)
from src.hmm.transition_probability import RoadNetworkGraph, TransitionModel
from src.hmm.viterbi import ViterbiDecoder
from src.hmm.map_matcher import HMMMapMatcher


class TestEmissionProbability:
    def test_gaussian_emission_calculation(self):
        sigma = 10.0
        model = GaussianEmissionModel(sigma_z=sigma)

        # At distance 0, probability is 1 / (sqrt(2*pi)*sigma)
        expected_p0 = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        assert np.isclose(model.emission_probability(0.0), expected_p0)
        assert np.isclose(model.log_p(0.0), math.log(expected_p0))

        # Farther distance should have lower probability
        log_p10 = model.log_p(10.0)
        log_p20 = model.log_p(20.0)
        assert log_p10 > log_p20

    def test_gaussian_emission_from_dataframe(self):
        model = GaussianEmissionModel(sigma_z=10.0)
        df = pd.DataFrame({"dist_m": [0.0, 5.0, 20.0]})
        log_probs = model.log_emission_probabilities((0, 0), df)
        assert len(log_probs) == 3
        assert log_probs[0] > log_probs[1] > log_probs[2]

    def test_ml_emission(self):
        model = MLEmissionModel()
        df = pd.DataFrame({"prob_norm": [0.8, 0.15, 0.05]})
        log_probs = model.log_emission_probabilities((0, 0), df)
        assert log_probs[0] > log_probs[1] > log_probs[2]
        assert np.isclose(log_probs[0], math.log(0.8))

    def test_hybrid_emission(self):
        model = HybridEmissionModel(sigma_z=10.0, alpha=0.5)
        df = pd.DataFrame({"dist_m": [5.0, 25.0], "prob_norm": [0.9, 0.1]})
        log_probs = model.log_emission_probabilities((0, 0), df)
        assert len(log_probs) == 2
        assert log_probs[0] > log_probs[1]


class TestTransitionProbability:
    @pytest.fixture
    def sample_network(self):
        # 3 nodes in a line: (0,0) -> (0, 100) -> (0, 200)
        nodes = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
        # 2 directed arcs: 0 -> 1, 1 -> 2
        arcs = np.array([[0, 1], [1, 2]])
        projected = np.array([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
        return RoadNetworkGraph(nodes, arcs, projected_nodes=projected)

    def test_road_network_graph(self, sample_network):
        assert sample_network.get_arc_length(0) == pytest.approx(100.0)
        assert sample_network.get_arc_length(1) == pytest.approx(100.0)
        # Shortest path from node 0 to node 2
        d = sample_network.shortest_path_distance(0, 2)
        assert d == pytest.approx(200.0)

    def test_transition_same_arc_forward(self, sample_network):
        model = TransitionModel(sample_network, beta=10.0)
        # Same arc 0, moving from frac 0.1 to 0.6 (delta = 50m)
        d_net = model.compute_network_distance(
            from_arc_id=0, from_fraction=0.1, to_arc_id=0, to_fraction=0.6
        )
        assert d_net == pytest.approx(50.0)

        # If GPS distance is also 50m, log prob is maximum (-log(beta))
        c1 = {"candidate_arc_id": 0, "fraction_along_arc": 0.1}
        c2 = {"candidate_arc_id": 0, "fraction_along_arc": 0.6}
        log_p = model.log_transition_probability(c1, c2, d_gps=50.0)
        assert log_p == pytest.approx(-math.log(10.0))

    def test_transition_connected_arcs(self, sample_network):
        model = TransitionModel(sample_network, beta=10.0)
        # From arc 0 (frac 0.8, 20m remaining) to arc 1 (frac 0.3, 30m entered)
        # Total network distance = 20 + 30 = 50m
        d_net = model.compute_network_distance(
            from_arc_id=0, from_fraction=0.8, to_arc_id=1, to_fraction=0.3
        )
        assert d_net == pytest.approx(50.0)

    def test_transition_unreachable(self, sample_network):
        model = TransitionModel(sample_network, beta=10.0, unreachable_penalty=50.0)
        # Backwards from arc 1 to arc 0 cannot be reached directly without reverse edge
        d_net = model.compute_network_distance(
            from_arc_id=1, from_fraction=0.5, to_arc_id=0, to_fraction=0.5
        )
        assert d_net == float("inf")
        c1 = {"candidate_arc_id": 1, "fraction_along_arc": 0.5}
        c2 = {"candidate_arc_id": 0, "fraction_along_arc": 0.5}
        log_p = model.log_transition_probability(c1, c2, d_gps=50.0)
        assert log_p == -50.0


class TestViterbiDecoder:
    def test_viterbi_simple_chain(self):
        decoder = ViterbiDecoder()

        # 3 time steps, 2 candidates per step
        candidate_layers = [
            [{"candidate_arc_id": 10}, {"candidate_arc_id": 20}],
            [{"candidate_arc_id": 11}, {"candidate_arc_id": 21}],
            [{"candidate_arc_id": 12}, {"candidate_arc_id": 22}],
        ]

        # Arc sequence 10 -> 11 -> 12 is favored by transitions
        emission_probs = [
            np.array([math.log(0.6), math.log(0.4)]),
            np.array([math.log(0.5), math.log(0.5)]),
            np.array([math.log(0.7), math.log(0.3)]),
        ]

        def transition_fn(t, prev_cands, curr_cands):
            # 10->11, 11->12 has high log prob, others have -10 penalty
            mat = np.full((2, 2), -10.0)
            mat[0, 0] = -0.1  # favored path (0 -> 0)
            mat[1, 1] = -1.0
            return mat

        result = decoder.decode(candidate_layers, emission_probs, transition_fn)
        assert result["matched_arcs"] == [10, 11, 12]
        assert result["matched_indices"] == [0, 0, 0]
        assert result["num_transitions"] == 2

    def test_viterbi_handles_single_step(self):
        decoder = ViterbiDecoder()
        candidate_layers = [[{"candidate_arc_id": 42}, {"candidate_arc_id": 99}]]
        emission_probs = [np.array([-2.0, -0.5])]

        result = decoder.decode(candidate_layers, emission_probs, lambda *args: None)
        assert result["matched_arcs"] == [99]
        assert result["matched_indices"] == [1]

    def test_viterbi_handles_empty(self):
        decoder = ViterbiDecoder()
        result = decoder.decode([], [], lambda *args: None)
        assert result["matched_arcs"] == []


class TestHMMMapMatcher:
    def test_continuity_calculation(self):
        arcs = np.array([
            [0, 1],  # arc 0
            [1, 2],  # arc 1
            [2, 3],  # arc 2
            [10, 11] # arc 3 (disconnected)
        ])

        # 0 -> 1 -> 2 is 100% continuous
        assert HMMMapMatcher.compute_continuity([0, 1, 2], arcs) == 1.0

        # 0 -> 0 -> 1 is 100% continuous (staying on same arc is valid)
        assert HMMMapMatcher.compute_continuity([0, 0, 1], arcs) == 1.0

        # 0 -> 3 (jump to 10->11) is discontinuous: 0/1 continuous = 0.0
        assert HMMMapMatcher.compute_continuity([0, 3], arcs) == 0.0

        # 0 -> 1 -> 3 has 1 continuous step out of 2 = 50%
        assert HMMMapMatcher.compute_continuity([0, 1, 3], arcs) == 0.5
