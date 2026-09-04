"""Uncertainty-Aware LeWorldModel (UA-LeWM).

This package adds epistemic-uncertainty estimation and uncertainty-aware
rollout truncation to the LeWorldModel (LeWM) latent planner, following the
adaptive-truncation idea of Wu et al. (UA-MBRL, arXiv:2104.07246) and the
project proposal (488proposal.pdf).

Public API
----------
- UncertaintyAwareJEPA : wraps a pretrained ``jepa.JEPA`` and exposes the same
  ``get_cost`` interface used by ``stable_worldmodel`` solvers, but evaluates
  the planning cost at an *uncertainty-truncated* rollout step.
- MCDropoutEstimator / EnsembleEstimator : produce stochastic rollouts whose
  per-step disagreement is used as the epistemic uncertainty signal sigma^2.
- TruncationConfig / compute_truncation_index : the k* = -omega * sigma^2 + k_base rule,
  plus ``gated`` mode (full horizon when sigma^2 is in-distribution, adaptive otherwise).
"""

from .truncation import TruncationConfig, compute_truncation_index
from .uncertainty import (
    EnsembleEstimator,
    MCDropoutEstimator,
    UncertaintyEstimator,
)
from .ua_jepa import UncertaintyAwareJEPA

__all__ = [
    "TruncationConfig",
    "compute_truncation_index",
    "UncertaintyEstimator",
    "MCDropoutEstimator",
    "EnsembleEstimator",
    "UncertaintyAwareJEPA",
]
