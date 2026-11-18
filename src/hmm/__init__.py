"""PathSnap Hidden Markov Model (HMM) and Viterbi Map-Matching package.

Modules:
- emission_probability: Gaussian, ML, and Hybrid emission models
- transition_probability: Road network graph and distance-based transition likelihood
- viterbi: Log-space Viterbi dynamic programming recurrence and backtracking
- map_matcher: High-level map matcher integrating candidate generation, ML scoring, and HMM decoding
"""

from .emission_probability import (
    EmissionModel,
    GaussianEmissionModel,
    HybridEmissionModel,
    MLEmissionModel,
)
from .map_matcher import HMMMapMatcher
from .transition_probability import RoadNetworkGraph, TransitionModel
from .viterbi import ViterbiDecoder

__all__ = [
    "EmissionModel",
    "GaussianEmissionModel",
    "MLEmissionModel",
    "HybridEmissionModel",
    "RoadNetworkGraph",
    "TransitionModel",
    "ViterbiDecoder",
    "HMMMapMatcher",
]
