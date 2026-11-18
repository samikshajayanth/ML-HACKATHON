"""End-to-end integration test running a full sample trip through the complete pipeline."""
import numpy as np
import pytest

from src.data.loader import load_trip
from src.features.candidates import CandidateGenerator
from src.features.feature_extraction import extract_features
from src.candidate_data import apply_aliases
from src.predict import CandidateScorer
from src.hmm import HMMMapMatcher


def test_full_pipeline_end_to_end():
    # 1. Load Trip
    trip = load_trip("data/raw", "student_test", "00000052")
    assert trip.track.shape[1] == 3
    assert trip.nodes.shape[1] == 2
    assert trip.arcs.shape[1] == 2
    n_obs = len(trip.track)

    # 2. Candidate Generation
    gen = CandidateGenerator(radius_m=100.0, max_candidates=8)
    candidates = gen.generate(trip)
    assert len(candidates) > 0
    assert candidates["observation_id"].nunique() == n_obs

    # 3. Feature Extraction
    features = apply_aliases(extract_features(trip, candidates))
    assert len(features) == len(candidates)
    assert "dist_m" in features.columns
    assert "speed_mps" in features.columns
    assert "heading_diff_deg" in features.columns

    # 4. ML Candidate Scoring
    scorer = CandidateScorer.load()
    scored = scorer.predict(features)
    assert len(scored) == len(candidates)
    assert "score" in scored.columns
    assert "prob_norm" in scored.columns
    assert "rank" in scored.columns

    # 5. HMM Map Matching with Viterbi
    matcher = HMMMapMatcher(emission_mode="hybrid", alpha=0.5, sigma_z=10.0, beta=10.0)
    result = matcher.match(trip, candidates, scored_df=scored)

    # Validate output
    assert result["trip_id"] == "00000052"
    assert result["num_observations"] == n_obs
    assert len(result["matched_arcs"]) == n_obs
    assert len(result["ml_top1_arcs"]) == n_obs
    assert 0.0 <= result["continuity_hmm"] <= 1.0
    assert 0.0 <= result["continuity_ml"] <= 1.0

    # Ensure HMM provides high continuity
    assert result["continuity_hmm"] >= 0.80

    # Check coordinate formatting for Folium map
    gps_coords = result["gps_coordinates"]
    matched_coords = result["matched_coordinates"]
    assert len(gps_coords) == n_obs
    assert len(matched_coords) >= 2
    for pt in matched_coords[:5]:
        assert len(pt) == 2
        # Check latitude and longitude ranges
        assert -90.0 <= pt[0] <= 90.0
        assert -180.0 <= pt[1] <= 180.0


def test_edge_case_short_track():
    # Test handling of very short tracks (2 points)
    trip = load_trip("data/raw", "student_test", "00000052")
    # Subsample trip to 2 observations
    trip.track = trip.track[:2]

    gen = CandidateGenerator(radius_m=100.0, max_candidates=8)
    candidates = gen.generate(trip)
    features = apply_aliases(extract_features(trip, candidates))

    scorer = CandidateScorer.load()
    scored = scorer.predict(features)

    matcher = HMMMapMatcher()
    result = matcher.match(trip, candidates, scored_df=scored)

    assert result["num_observations"] == 2
    assert len(result["matched_arcs"]) == 2
