"""Emission probability models for HMM map matching.

Implements:
1. GaussianEmissionModel: Evaluates spatial likelihood using perpendicular distance
   between GPS observations and candidate road segments based on GPS measurement uncertainty.
2. MLEmissionModel: Evaluates candidate likelihood using calibrated machine learning scores.
3. HybridEmissionModel: Combines spatial geometry with learned ML representations.
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import pandas as pd


class EmissionModel:
    """Base interface for emission probability calculations."""

    def log_emission_probabilities(
        self,
        observation_coords: tuple[float, float] | np.ndarray,
        candidates: pd.DataFrame | list[dict],
    ) -> np.ndarray:
        """Compute log emission probabilities for all candidates of one GPS observation."""
        raise NotImplementedError


class GaussianEmissionModel(EmissionModel):
    """Gaussian emission probability based on perpendicular distance.

    p(z_t | c_{t,i}) = (1 / (sqrt(2*pi)*sigma)) * exp(-d_i^2 / (2 * sigma^2))
    log p(z_t | c_{t,i}) = -0.5 * log(2*pi*sigma^2) - (d_i^2 / (2 * sigma^2))
    """

    def __init__(self, sigma_z: float = 10.0):
        if sigma_z <= 0:
            raise ValueError("sigma_z must be positive")
        self.sigma_z = float(sigma_z)
        self._log_const = -0.5 * math.log(2.0 * math.pi * (self.sigma_z**2))
        self._two_sigma_sq = 2.0 * (self.sigma_z**2)

    def log_p(self, distance_m: float | np.ndarray) -> float | np.ndarray:
        dist = np.asarray(distance_m, dtype=float)
        return self._log_const - (dist**2) / self._two_sigma_sq

    def emission_probability(self, distance_m: float | np.ndarray) -> float | np.ndarray:
        return np.exp(self.log_p(distance_m))

    def log_emission_probabilities(
        self,
        observation_coords: tuple[float, float] | np.ndarray,
        candidates: pd.DataFrame | list[dict],
    ) -> np.ndarray:
        if isinstance(candidates, pd.DataFrame):
            col = "dist_m" if "dist_m" in candidates.columns else "distance_m"
            distances = candidates[col].to_numpy(dtype=float)
        else:
            distances = np.array([
                float(c.get("dist_m", c.get("distance_m", 0.0)))
                for c in candidates
            ], dtype=float)
        return self.log_p(distances)


class MLEmissionModel(EmissionModel):
    """Emission probability derived from trained ML classifier probabilities.

    Uses normalized within-observation candidate probabilities (prob_norm)
    or raw confidence scores.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def log_p(self, scores: float | np.ndarray) -> float | np.ndarray:
        s = np.asarray(scores, dtype=float)
        s_safe = np.clip(s, self.eps, 1.0)
        return np.log(s_safe)

    def log_emission_probabilities(
        self,
        observation_coords: tuple[float, float] | np.ndarray,
        candidates: pd.DataFrame | list[dict],
    ) -> np.ndarray:
        if isinstance(candidates, pd.DataFrame):
            if "prob_norm" in candidates.columns:
                scores = candidates["prob_norm"].to_numpy(dtype=float)
            elif "score" in candidates.columns:
                scores = candidates["score"].to_numpy(dtype=float)
                tot = np.sum(scores)
                if tot > 0:
                    scores = scores / tot
            else:
                raise ValueError("Candidates DataFrame must have 'prob_norm' or 'score'")
        else:
            scores = np.array([
                float(c.get("prob_norm", c.get("score", 1.0 / max(len(candidates), 1))))
                for c in candidates
            ], dtype=float)
            tot = np.sum(scores)
            if tot > 0:
                scores = scores / tot
        return self.log_p(scores)


class HybridEmissionModel(EmissionModel):
    """Blended emission model: alpha * log(ML) + (1 - alpha) * log(Gaussian)."""

    def __init__(self, sigma_z: float = 10.0, alpha: float = 0.5, eps: float = 1e-12):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0.0, 1.0]")
        self.gaussian_model = GaussianEmissionModel(sigma_z=sigma_z)
        self.ml_model = MLEmissionModel(eps=eps)
        self.alpha = float(alpha)

    def log_emission_probabilities(
        self,
        observation_coords: tuple[float, float] | np.ndarray,
        candidates: pd.DataFrame | list[dict],
    ) -> np.ndarray:
        log_g = self.gaussian_model.log_emission_probabilities(observation_coords, candidates)
        if self.alpha == 0.0:
            return log_g
        log_m = self.ml_model.log_emission_probabilities(observation_coords, candidates)
        if self.alpha == 1.0:
            return log_m
        return self.alpha * log_m + (1.0 - self.alpha) * log_g
