"""Uncertainty-aware planning on top of a frozen LeWorldModel (LeWM).

All planners expose the ``get_cost(info_dict, action_candidates)`` interface
that the ``stable_worldmodel`` CEM solver calls, so they are drop-in
replacements for the plain LeWM cost model:

- ``UncertaintyAwareJEPA`` — samples ``N`` MC-dropout rollouts, scores the mean
  trajectory (MC aggregation) and optionally truncates the scored horizon when
  the step-0 epistemic uncertainty exceeds a calibrated threshold (``gated``).
- ``GoalCVaRJEPA`` — same rollouts, plus an elite-calibrated soft penalty on the
  worst dropout samples (Goal-CVaR).
- ``MCDropoutEstimator`` / ``EnsembleEstimator`` — produce the stochastic
  rollouts whose disagreement is the uncertainty signal sigma^2.
- ``TruncationConfig`` / ``compute_truncation_index`` — the horizon-truncation
  rules.
"""

from .goal_cvar import GoalCVaRConfig, GoalCVaRJEPA, goal_cvar_cost
from .truncation import TruncationConfig, compute_truncation_index
from .ua_jepa import UncertaintyAwareJEPA
from .uncertainty import EnsembleEstimator, MCDropoutEstimator, UncertaintyEstimator

__all__ = [
    "TruncationConfig",
    "compute_truncation_index",
    "UncertaintyEstimator",
    "MCDropoutEstimator",
    "EnsembleEstimator",
    "UncertaintyAwareJEPA",
    "GoalCVaRConfig",
    "GoalCVaRJEPA",
    "goal_cvar_cost",
]
