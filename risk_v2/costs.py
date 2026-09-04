"""Pure tensor implementation of Soft Penalty v2."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ua.truncation import gather_at_index

from .config import RiskConfig


def terminal_goal_quantities(
    trajectories: torch.Tensor,
    mean_trajectory: torch.Tensor,
    goal: torch.Tensor,
    tail_count: int,
    idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return point distance, per-rollout losses, and excess upper-tail loss.

    When ``idx`` is provided with shape ``(B, S)``, quantities are evaluated at
    the truncated step ``k* = idx + 1`` instead of the final horizon step.
    """
    if trajectories.ndim != 5:
        raise ValueError("trajectories must have shape (N, B, S, P, D)")
    if mean_trajectory.shape != trajectories.shape[1:]:
        raise ValueError("mean_trajectory must have shape (B, S, P, D)")
    if goal.shape != (trajectories.shape[1], trajectories.shape[-1]):
        raise ValueError("goal must have shape (B, D)")
    if not 1 <= tail_count <= trajectories.shape[0]:
        raise ValueError("tail_count must be in [1, N]")

    target = goal[:, None, None, :]
    d_steps = ((mean_trajectory - target) ** 2).sum(dim=-1)
    loss_steps = ((trajectories - target.unsqueeze(0)) ** 2).sum(dim=-1)
    if idx is None:
        d = d_steps[..., -1]
        losses = loss_steps[..., -1]
    else:
        if idx.shape != d_steps.shape[:-1]:
            raise ValueError("idx must have shape (B, S)")
        d = gather_at_index(d_steps, idx)
        idx_n = idx.unsqueeze(0).expand(loss_steps.shape[0], -1, -1)
        losses = torch.gather(loss_steps, dim=-1, index=idx_n.unsqueeze(-1)).squeeze(-1)
    cvar = losses.topk(tail_count, dim=0, largest=True, sorted=False).values.mean(dim=0)
    return d, losses, cvar - d


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(values, q, dim=-1)


def _robust_floor(values: torch.Tensor, eps: float) -> torch.Tensor:
    """A positive, unit-aware numerical floor for each batch row."""
    magnitude = _quantile(values.abs(), 0.5).clamp_min(1.0)
    return magnitude * eps


def _elite_overlap(d: torch.Tensor, objective: torch.Tensor, topk: int) -> torch.Tensor:
    d_indices = d.topk(topk, dim=-1, largest=False, sorted=False).indices
    j_indices = objective.topk(topk, dim=-1, largest=False, sorted=False).indices
    d_mask = torch.zeros_like(d, dtype=torch.bool).scatter(-1, d_indices, True)
    j_mask = torch.zeros_like(d, dtype=torch.bool).scatter(-1, j_indices, True)
    return (d_mask & j_mask).sum(dim=-1).to(d.dtype) / float(topk)


def soft_penalty_v2(
    d: torch.Tensor,
    risk: torch.Tensor,
    config: RiskConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply per-call elite calibration to candidate risks."""
    if d.ndim != 2 or risk.shape != d.shape:
        raise ValueError("d and risk must share shape (B, S)")
    if d.shape[-1] != config.num_samples:
        raise ValueError(
            f"received S={d.shape[-1]} candidates, configured num_samples="
            f"{config.num_samples}"
        )

    elite_count = min(2 * config.topk, config.num_samples)
    elite_indices = d.topk(elite_count, dim=-1, largest=False, sorted=False).indices
    elite_risk = risk.gather(-1, elite_indices)
    tau = _quantile(elite_risk, 0.5)
    iqr = _quantile(elite_risk, 0.75) - _quantile(elite_risk, 0.25)
    scale = torch.maximum(iqr, _robust_floor(elite_risk, config.eps))

    q = config.topk / config.num_samples
    distance_gap = _quantile(d, min(1.0, 2.0 * q)) - _quantile(d, q)
    delta_d = torch.maximum(distance_gap, _robust_floor(d, config.eps))

    if config.alpha == 0.0:
        # Avoid even a rounding-level change to the control objective.
        penalty = torch.zeros_like(d)
        objective = d
    else:
        excess = torch.relu((risk - tau.unsqueeze(-1)) / scale.unsqueeze(-1))
        penalty = config.alpha * delta_d.unsqueeze(-1) * excess.clamp(max=config.cap)
        objective = d + penalty

    diagnostics = {
        "terminal_distance": d,
        "risk": risk,
        "penalty": penalty,
        "penalized_fraction": (penalty > 0).to(d.dtype).mean(dim=-1),
        "penalty_to_goal": penalty / d.abs().clamp_min(config.eps),
        "topk_elite_overlap": _elite_overlap(d, objective, config.topk),
        "tau": tau,
        "scale": scale,
        "delta_d": delta_d,
        "cost": objective,
    }
    return objective, diagnostics


def compute_risk_cost(
    trajectories: torch.Tensor,
    mean_trajectory: torch.Tensor,
    sigma2: torch.Tensor,
    goal: torch.Tensor,
    config: RiskConfig,
    idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a configured objective and diagnostics.

    ``idx`` selects the scored rollout step (adaptive truncation). When omitted,
    behavior matches the original full-horizon Soft Penalty v2 objectives.
    """
    tail_count = config.resolved_tail_count(trajectories.shape[0])
    d, _, cvar_excess = terminal_goal_quantities(
        trajectories, mean_trajectory, goal, tail_count, idx=idx
    )
    if sigma2.shape != mean_trajectory.shape[:-1]:
        raise ValueError("sigma2 must have shape (B, S, P)")
    if idx is not None and idx.shape != sigma2.shape[:-1]:
        raise ValueError("idx must have shape (B, S)")

    sigma2_scored = (
        gather_at_index(sigma2, idx) if idx is not None else sigma2[..., -1]
    )

    if config.mode == "mc_terminal":
        risk = torch.zeros_like(d)
        objective = d
        penalty = torch.zeros_like(d)
        diagnostics = {
            "terminal_distance": d,
            "risk": risk,
            "penalty": penalty,
            "penalized_fraction": penalty.mean(dim=-1),
            "penalty_to_goal": penalty,
            "topk_elite_overlap": torch.ones(
                d.shape[0], device=d.device, dtype=d.dtype
            ),
            "cost": objective,
        }
        if idx is not None:
            diagnostics["truncation_length"] = idx.to(dtype=d.dtype) + 1
        return objective, diagnostics

    if config.mode == "soft_v1_control":
        # Keep the path-averaged sigma^2 control; truncation only moves the
        # goal-distance reference used by the soft scaffold.
        risk = (sigma2 / float(mean_trajectory.shape[-1])).mean(dim=-1)
        if config.alpha == 0.0:
            penalty = torch.zeros_like(d)
            objective = d
        else:
            penalty = config.resolved_lambda * risk
            objective = d + penalty
        diagnostics = {
            "terminal_distance": d,
            "risk": risk,
            "penalty": penalty,
            "penalized_fraction": (penalty > 0).to(d.dtype).mean(dim=-1),
            "penalty_to_goal": penalty / d.abs().clamp_min(config.eps),
            "topk_elite_overlap": _elite_overlap(d, objective, config.topk),
            "cost": objective,
        }
        if idx is not None:
            diagnostics["truncation_length"] = idx.to(dtype=d.dtype) + 1
        return objective, diagnostics

    if config.mode == "goal_cvar_eot":
        risk = cvar_excess
    else:
        # Scalar control: scored-step per-dimension latent variance.
        risk = (sigma2_scored / float(mean_trajectory.shape[-1])).clamp_min(0)
    objective, diagnostics = soft_penalty_v2(d, risk, config)
    if idx is not None:
        diagnostics["truncation_length"] = idx.to(dtype=d.dtype) + 1
    return objective, diagnostics
