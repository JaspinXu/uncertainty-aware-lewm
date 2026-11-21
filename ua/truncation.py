"""Uncertainty-aware rollout truncation.

Implements the adaptive truncation rule from Wu et al. (UA-MBRL):

    k* = clamp( round(-omega * sigma^2_M(z_t) + k_base), k_min, k_max )

where ``sigma^2_M`` is the epistemic uncertainty (ensemble / MC-dropout
disagreement) of the latent prediction. The planner then evaluates its cost at
the truncated step k* instead of the fixed final horizon H, so it cannot
exploit "hallucinated" deep-rollout predictions.

Several modes are provided for ablations (fixed horizon vs. adaptive,
different penalty weights):

- ``"none"``            : no truncation, score the final step (standard LeWM MPC).
- ``"fixed"``           : always truncate at a constant length ``k_base`` (ablation).
- ``"adaptive_length"`` : single k* per candidate from the start-state sigma^2
                          (faithful to UA-MBRL Eq. 18).
- ``"adaptive_perstep"``: walk the rollout and cut at the first step whose
                          sigma^2 exceeds a calibrated ``threshold``.
- ``"gated"``             : if the start-state sigma^2 is below a calibrated
                          ``threshold``, keep the full horizon (k* = H, like baseline);
                          otherwise apply the adaptive_length rule. This preserves
                          baseline behavior on familiar states while truncating only
                          high-uncertainty candidates.
- ``"gated_tc"``          : gate every predicted step against a per-step ``threshold``
                          (a quantile of planning-time sigma^2 at that depth, so the
                          rule does not depend on the raw sigma^2 scale) and keep only
                          the steps before the first uncertain one. Meant to be used with
                          ``terminal_weight`` > 0 (see ``terminal_charge``): a truncated
                          candidate is charged for the part of its plan that was cut,
                          instead of having it scored as free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


@dataclass
class TruncationConfig:
    """Configuration for the uncertainty-aware truncation rule."""

    mode: str = "adaptive_length"
    omega: float = 1.0          # slope on sigma^2 (UA-MBRL Eq. 18)
    k_base: float = 5.0         # baseline rollout length
    k_min: int = 1              # never truncate below this many steps
    k_max: Optional[int] = None  # upper bound (defaults to full horizon)
    # Scalar for gated, or scalar/per-step sequence for adaptive_perstep.
    threshold: Optional[Union[float, Sequence[float]]] = None
    penalty_weight: float = 0.0  # optional extra cost = weight * sigma^2 at k*
                                 # (the uncertainty-as-penalty variant)
    terminal_weight: float = 0.0  # HALT-style terminal charge on the truncated part (see terminal_charge)

    def __post_init__(self):
        valid = {"none", "fixed", "adaptive_length", "adaptive_perstep", "gated", "gated_tc"}
        if self.mode not in valid:
            raise ValueError(f"Unknown truncation mode {self.mode!r}; expected one of {valid}")


def _adaptive_length_index(
    sigma2_ref: torch.Tensor,
    cfg: TruncationConfig,
    P: int,
    k_min: int,
    k_max: int,
) -> torch.Tensor:
    """Return per-candidate index from the UA-MBRL Eq. 18 rule."""
    k = torch.round(-cfg.omega * sigma2_ref + cfg.k_base)
    k = k.clamp(min=k_min, max=k_max)
    return (k.long() - 1).clamp(min=0, max=P - 1)


def _perstep_threshold(
    threshold: Union[float, Sequence[float]],
    sigma2_steps: torch.Tensor,
) -> torch.Tensor:
    """Return a threshold broadcastable over ``(..., P)`` uncertainty."""
    values = torch.as_tensor(
        threshold,
        dtype=sigma2_steps.dtype,
        device=sigma2_steps.device,
    )
    if values.ndim == 0:
        return values
    if values.ndim != 1 or values.numel() != sigma2_steps.shape[-1]:
        raise ValueError(
            "adaptive_perstep threshold sequence must contain exactly one "
            f"value per predicted step ({sigma2_steps.shape[-1]}), got "
            f"shape {tuple(values.shape)}"
        )
    return values.view(*([1] * (sigma2_steps.ndim - 1)), -1)


def compute_truncation_index(
    sigma2_steps: torch.Tensor,
    cfg: TruncationConfig,
) -> torch.Tensor:
    """Return the per-candidate rollout index at which to evaluate the cost.

    Parameters
    ----------
    sigma2_steps : (..., P) tensor
        Epistemic uncertainty sigma^2 at each of the ``P`` predicted rollout
        steps, for every action candidate (leading dims are arbitrary, e.g.
        ``(B, S)``).
    cfg : TruncationConfig

    Returns
    -------
    idx : (...) long tensor
        Index in ``[0, P-1]`` selecting the truncated step for each candidate.
    """
    *lead, P = sigma2_steps.shape
    k_max = cfg.k_max if cfg.k_max is not None else P
    k_max = int(min(k_max, P))
    k_min = int(max(1, min(cfg.k_min, k_max)))

    if cfg.mode == "none":
        idx = sigma2_steps.new_full(lead, P - 1, dtype=torch.long)

    elif cfg.mode == "fixed":
        k = int(min(max(round(cfg.k_base), k_min), k_max))
        idx = sigma2_steps.new_full(lead, k - 1, dtype=torch.long)

    elif cfg.mode == "adaptive_length":
        # k* from the uncertainty of the *first* predicted step (branch start).
        sigma2_ref = sigma2_steps[..., 0]
        idx = _adaptive_length_index(sigma2_ref, cfg, P, k_min, k_max)

    elif cfg.mode == "gated":
        if cfg.threshold is None:
            raise ValueError(
                "gated requires cfg.threshold (calibrate it on validation rollouts, "
                "see ua/calibrate.py and scripts/calibrate_gated_threshold.py)"
            )
        sigma2_ref = sigma2_steps[..., 0]
        threshold = torch.as_tensor(
            cfg.threshold,
            dtype=sigma2_steps.dtype,
            device=sigma2_steps.device,
        )
        if threshold.numel() != 1:
            raise ValueError("gated requires a scalar threshold")
        idx_adaptive = _adaptive_length_index(sigma2_ref, cfg, P, k_min, k_max)
        idx_full = sigma2_steps.new_full(lead, P - 1, dtype=torch.long)
        idx = torch.where(sigma2_ref < threshold.reshape(()), idx_full, idx_adaptive)

    elif cfg.mode == "gated_tc":
        if cfg.threshold is None:
            raise ValueError(
                "gated_tc requires per-step thresholds (quantiles of planning-time sigma^2 per "
                "depth, see scripts/cube_compare.py +planner=calibrate)"
            )
        exceeds = sigma2_steps > _perstep_threshold(cfg.threshold, sigma2_steps)  # (..., P)
        first_exceed = torch.argmax(exceeds.int(), dim=-1)
        # trusted horizon: the steps before the first uncertain one (at least k_min)
        idx = torch.where(
            exceeds.any(dim=-1),
            first_exceed - 1,
            torch.full_like(first_exceed, k_max - 1),
        )
        idx = idx.clamp(min=k_min - 1, max=k_max - 1)

    else:  # adaptive_perstep
        if cfg.threshold is None:
            raise ValueError(
                "adaptive_perstep requires cfg.threshold (calibrate it on "
                "validation rollouts, see ua/calibrate.py)"
            )
        # first step whose uncertainty exceeds the threshold -> cut there.
        # A per-step threshold is important because rollout uncertainty normally
        # grows with depth. Reusing a step-0 scalar at every step causes nearly
        # every candidate to be cut immediately.
        threshold = _perstep_threshold(cfg.threshold, sigma2_steps)
        exceeds = sigma2_steps > threshold  # (..., P)
        # ignore the first (k_min) steps so we always keep a minimal rollout.
        if k_min > 1:
            exceeds[..., : k_min - 1] = False
        any_exceed = exceeds.any(dim=-1)
        first_exceed = torch.argmax(exceeds.int(), dim=-1)
        # if nothing exceeds, keep up to k_max.
        idx = torch.where(
            any_exceed,
            first_exceed,
            torch.full_like(first_exceed, k_max - 1),
        )
        idx = idx.clamp(min=k_min - 1, max=P - 1)

    return idx


def terminal_charge(dist_steps: torch.Tensor, idx: torch.Tensor, weight: float) -> torch.Tensor:
    """HALT-style terminal cost for the part of each candidate's horizon that was cut.

    Truncation scores a candidate at an earlier step, so the cut part of its plan is never
    scored, yet the whole plan is executed (receding horizon = horizon). As in MOReL's
    pessimistic MDP, a candidate that enters an uncertain region is instead charged:

        charge = weight * (q90 - q10 of the population's full-horizon costs) * (P - 1 - idx) / (P - 1)

    i.e. zero when nothing is cut, and for a fully cut candidate a whole 10-90% spread of the
    current CEM population, enough to rank it behind most confident candidates. The scale comes
    from the population being ranked, so no calibration is needed.

    ``dist_steps`` (..., S, P) per-step costs, ``idx`` (..., S) truncation indices -> (..., S).
    """
    P = dist_steps.shape[-1]
    if P < 2 or not weight:
        return torch.zeros(idx.shape, dtype=dist_steps.dtype, device=dist_steps.device)
    final = dist_steps[..., P - 1].sort(dim=-1).values  # (..., S) full-horizon costs, sorted
    S = final.shape[-1]
    spread = (final[..., round(0.9 * (S - 1))] - final[..., round(0.1 * (S - 1))]).clamp(min=0)
    cut = (P - 1 - idx).to(dist_steps.dtype) / (P - 1)
    return weight * spread.unsqueeze(-1) * cut


def gather_at_index(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``values[..., t]`` where ``t = idx[...]``.

    ``values`` is (..., P); ``idx`` is (...). Returns (...).
    """
    return torch.gather(values, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
