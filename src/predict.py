"""Reusable prediction interface.

Member 3 (HMM emissions):   scored = CandidateScorer.load(path).predict(candidates_df)
Member 4 (FastAPI):         scorer = load_scorer();  scorer.predict_records(list_of_dicts)

The input must contain ALL candidates of each observation together (context features such
as dist_rank are computed within the observation) and the same raw feature columns used
in training. Output rows are in the SAME ORDER as the input:

    trip_id, obs_id, candidate_id, score, prob_norm, rank

score      = P(candidate is the correct segment), model output, in [0, 1]
prob_norm  = score rescaled to sum to 1 within the observation (use for HMM emissions;
             take logs with a small floor, e.g. log(max(prob_norm, 1e-12)))
rank       = 1 for the best candidate in the observation
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import config as C
from .candidate_data import add_context_features, apply_aliases, feature_matrix, validate_schema
from .evaluation.metrics import normalise_within_obs, rank_within_obs


def save_bundle(path, model, feature_columns, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "model": model, "feature_columns": list(feature_columns),
                 "meta": meta}, path, compress=3)


class CandidateScorer:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.meta: dict = bundle.get("meta", {})
        self._pos = list(self.model.classes_).index(1)

    @classmethod
    def load(cls, path=None) -> "CandidateScorer":
        path = Path(path) if path else C.DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return cls(joblib.load(path))

    @property
    def required_columns(self) -> list[str]:
        """Raw columns the caller must send (derived context features are recomputed)."""
        derived = {"n_candidates", "dist_minus_min", "dist_ratio_to_min", "dist_rank",
                   "heading_minus_min", "heading_rank"}
        raw = [c for c in self.feature_columns if c not in derived]
        return C.ID_COLS + raw

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = apply_aliases(df.reset_index(drop=True))
        validate_schema(df, source="predict input", need_label=False)
        d = add_context_features(df)
        missing = [c for c in self.feature_columns if c not in d.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        p = self.model.predict_proba(feature_matrix(d, self.feature_columns))[:, self._pos]
        out = d[C.ID_COLS].copy()
        out["score"] = p
        out["prob_norm"] = normalise_within_obs(d, p)
        tb = d["dist_rank"].to_numpy() if "dist_rank" in d else None
        out["rank"] = rank_within_obs(d, p, tb)
        return out

    def predict_records(self, records: list[dict]) -> list[dict]:
        """JSON-friendly wrapper for FastAPI."""
        return self.predict(pd.DataFrame.from_records(records)).to_dict(orient="records")

    @staticmethod
    def best_per_observation(scored: pd.DataFrame) -> pd.DataFrame:
        return scored[scored["rank"] == 1].reset_index(drop=True)


_CACHE: dict[str, CandidateScorer] = {}


def load_scorer(path=None) -> CandidateScorer:
    """Cached loader (load once at FastAPI startup)."""
    key = str(path or C.DEFAULT_MODEL_PATH)
    if key not in _CACHE:
        _CACHE[key] = CandidateScorer.load(path)
    return _CACHE[key]


def predict_candidates(df: pd.DataFrame, model_path=None) -> pd.DataFrame:
    return load_scorer(model_path).predict(df)
