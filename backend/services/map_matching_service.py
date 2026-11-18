"""Map matching service orchestrating trip loading, candidate generation,

feature extraction, ML scoring, HMM Viterbi decoding, and server-side metrics.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.data.loader import load_trip
from src.features.candidates import CandidateGenerator
from src.features.feature_extraction import extract_features
from src.candidate_data import apply_aliases
from src.hmm import HMMMapMatcher
from .prediction_service import prediction_service


def get_dataset_root() -> Path:
    """Resolve dataset root path with priority:

    1. PATHSNAP_DATASET_ROOT env var
    2. Local workspace data/raw
    3. ~/Downloads/Student folder
    """
    env_path = os.getenv("PATHSNAP_DATASET_ROOT")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir():
            return p

    local_raw = Path("data/raw")
    if local_raw.is_dir():
        return local_raw.resolve()

    download_dir = Path.home() / "Downloads" / "Student folder"
    if download_dir.is_dir():
        return download_dir.resolve()

    raise FileNotFoundError(
        "Dataset root not found. Please place dataset in data/raw or set PATHSNAP_DATASET_ROOT."
    )


class MapMatchingService:
    """End-to-end map-matching service for API endpoints."""

    def __init__(self, dataset_root: Path | None = None):
        self._dataset_root = dataset_root

    @property
    def dataset_root(self) -> Path:
        if self._dataset_root is None:
            self._dataset_root = get_dataset_root()
        return self._dataset_root

    def get_available_trips(self, split: str = "student_test") -> list[str]:
        split_dir = self.dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory {split_dir} does not exist")
        return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    def process_trip(
        self,
        trip_id: str,
        split: str = "student_test",
        emission_mode: str = "hybrid",
        alpha: float = 0.5,
        sigma_z: float = 10.0,
        beta: float = 10.0,
    ) -> dict[str, Any]:
        """Execute full map matching pipeline on a trip."""
        t0 = time.perf_counter()

        # 1. Load Trip
        trip = load_trip(self.dataset_root, split, str(trip_id))

        # 2. Candidate Generation
        gen = CandidateGenerator(radius_m=100.0, max_candidates=8)
        candidates = gen.generate(trip)

        # 3. Feature Extraction
        features = apply_aliases(extract_features(trip, candidates))

        # 4. Machine Learning Scoring
        scored = prediction_service.score(features)

        # 5. HMM Map Matching with Viterbi
        matcher = HMMMapMatcher(
            sigma_z=sigma_z,
            beta=beta,
            emission_mode=emission_mode,
            alpha=alpha,
        )
        match_result = matcher.match(trip, candidates, scored_df=scored)

        # 6. Server-side reference route metrics & coordinates (without leaking to ML)
        gt_coords = None
        cand_recall = None
        if hasattr(trip, "route") and trip.route is not None and len(trip.route) > 0:
            gt_arcs = trip.arcs[trip.route]
            gt_coords = [
                [float(trip.nodes[node_id, 1]), float(trip.nodes[node_id, 0])]
                for node_id in np.concatenate([gt_arcs[:1, 0], gt_arcs[:, 1]])
            ]
            gt_set = set(int(a) for a in trip.route)
            # Candidate recall: fraction of observations having at least one true arc among candidates
            cand_arc_col = "candidate_arc_id" if "candidate_arc_id" in candidates.columns else "candidate_id"
            obs_col = "observation_id" if "observation_id" in candidates.columns else "obs_id"
            obs_has_gt = candidates.groupby(obs_col)[cand_arc_col].apply(
                lambda s: any(int(a) in gt_set for a in s)
            )
            cand_recall = float(obs_has_gt.mean())

        elapsed = time.perf_counter() - t0

        c_hmm = float(match_result["continuity_hmm"])
        c_ml = float(match_result["continuity_ml"])
        a_hmm = float(match_result["accuracy_hmm"]) if match_result["accuracy_hmm"] is not None else None
        a_ml = float(match_result["accuracy_ml"]) if match_result["accuracy_ml"] is not None else None

        return {
            "trip_id": str(trip_id),
            "split": split,
            "status": "success",
            "num_observations": match_result["num_observations"],
            "num_candidates": len(candidates),
            "matched_arcs": match_result["matched_arcs"],
            "ml_top1_arcs": match_result["ml_top1_arcs"],
            "log_likelihood": float(match_result["log_likelihood"]),
            "restarts": int(match_result["restarts"]),
            "metrics": {
                "continuity_hmm": c_hmm,
                "continuity_ml": c_ml,
                "continuity_improvement_pct": round((c_hmm - c_ml) * 100, 2),
                "accuracy_hmm": a_hmm,
                "accuracy_ml": a_ml,
                "accuracy_improvement_pct": (
                    round((a_hmm - a_ml) * 100, 2)
                    if a_hmm is not None and a_ml is not None
                    else None
                ),
                "candidate_recall": cand_recall,
            },
            "gps_coordinates": match_result["gps_coordinates"],
            "matched_coordinates": match_result["matched_coordinates"],
            "ml_coordinates": match_result["ml_coordinates"],
            "ground_truth_coordinates": gt_coords,
            "inference_time_s": round(elapsed, 3),
        }


map_matching_service = MapMatchingService()
