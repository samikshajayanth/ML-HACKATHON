"""Prediction service encapsulating ML CandidateScorer loading and scoring."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.predict import CandidateScorer, load_scorer
from src import config as C


class PredictionService:
    """Manages the trained candidate classifier and inference."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = model_path or C.DEFAULT_MODEL_PATH
        self._scorer: CandidateScorer | None = None

    @property
    def scorer(self) -> CandidateScorer:
        if self._scorer is None:
            self._scorer = load_scorer(self.model_path)
        return self._scorer

    def score(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Predict P(candidate is correct road segment) for all candidates of each observation."""
        return self.scorer.predict(candidates_df)

    def is_ready(self) -> bool:
        try:
            return self.scorer is not None
        except Exception:
            return False


# Singleton instance
prediction_service = PredictionService()
