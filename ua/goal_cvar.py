"""Goal-CVaR soft-penalty planner (the "Soft Penalty" method).

Every CEM candidate is scored with

    J = d + alpha * delta_d * clip( relu((r - tau) / scale), 0, cap )

where ``d`` is the MC-mean terminal latent distance to the goal, ``r`` is the
excess upper-tail loss (mean of the ``tail_count`` largest per-rollout terminal
losses minus ``d``, i.e. a CVaR-style measure of how bad the worst dropout
samples are), and ``tau`` / ``scale`` / ``delta_d`` are robust statistics
re-estimated on the elite set of every planning call so the penalty is
unit-free. ``alpha = 0`` reproduces the plain MC-mean objective exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .truncation import TruncationConfig, compute_truncation_index, gather_at_index
from .ua_jepa import UncertaintyAwareJEPA
from .uncertainty import UncertaintyEstimator, rollout_statistics


@dataclass(frozen=True)
class GoalCVaRConfig:
    """Objective parameters plus the CEM sizes the elite calibration needs."""

    alpha: float = 0.1
    num_samples: int = 300  # CEM candidates per planning call
    topk: int = 30          # CEM elites per planning call
    tail_fraction: float = 0.4  # N=5 rollouts -> the two largest losses
    tail_count: Optional[int] = None
    cap: float = 2.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if not 1 <= self.topk <= self.num_samples:
            raise ValueError("topk must be in [1, num_samples]")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        if self.tail_count is not None and self.tail_count < 1:
            raise ValueError("tail_count must be positive when specified")
        if self.cap < 0:
            raise ValueError("cap must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def resolved_tail_count(self, rollout_samples: int) -> int:
        """Resolve a valid upper-tail count (N=5 defaults to the top two)."""
        if rollout_samples < 1:
            raise ValueError("rollout_samples must be positive")
        count = (
            self.tail_count
            if self.tail_count is not None
            else math.ceil(self.tail_fraction * rollout_samples)
        )
        return min(rollout_samples, max(1, int(count)))


# --------------------------------------------------------------------------- #
# pure tensor math
# --------------------------------------------------------------------------- #
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


def soft_penalty(
    d: torch.Tensor,
    risk: torch.Tensor,
    config: GoalCVaRConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Add the elite-calibrated soft penalty to the candidate distances."""
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


def goal_cvar_cost(
    trajectories: torch.Tensor,
    mean_trajectory: torch.Tensor,
    sigma2: torch.Tensor,
    goal: torch.Tensor,
    config: GoalCVaRConfig,
    idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Goal-CVaR objective and diagnostics for one planning call.

    ``idx`` selects the scored rollout step (optional truncation); when omitted
    the final horizon step is scored.
    """
    if sigma2.shape != mean_trajectory.shape[:-1]:
        raise ValueError("sigma2 must have shape (B, S, P)")
    if idx is not None and idx.shape != sigma2.shape[:-1]:
        raise ValueError("idx must have shape (B, S)")
    tail_count = config.resolved_tail_count(trajectories.shape[0])
    d, _, cvar_excess = terminal_goal_quantities(
        trajectories, mean_trajectory, goal, tail_count, idx=idx
    )
    objective, diagnostics = soft_penalty(d, cvar_excess, config)
    if idx is not None:
        diagnostics["truncation_length"] = idx.to(dtype=d.dtype) + 1
    return objective, diagnostics


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
class GoalCVaRJEPA(UncertaintyAwareJEPA):
    """UncertaintyAwareJEPA whose cost is the Goal-CVaR soft-penalty objective."""

    def __init__(
        self,
        base: torch.nn.Module,
        estimator: UncertaintyEstimator,
        cvar_cfg: GoalCVaRConfig,
        trunc_cfg: Optional[TruncationConfig] = None,
        history_size: int = 3,
    ):
        super().__init__(
            base=base,
            estimator=estimator,
            trunc_cfg=trunc_cfg or TruncationConfig(mode="none"),
            history_size=history_size,
        )
        self.cvar_cfg = cvar_cfg
        self.risk_diagnostics: list[Dict[str, float]] = []

    def reset_diagnostics(self) -> None:
        super().reset_diagnostics()
        self.risk_diagnostics.clear()

    def diagnostics_summary(self) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        if self.risk_diagnostics:
            keys = self.risk_diagnostics[0].keys()
            summary.update(
                {
                    key: sum(row[key] for row in self.risk_diagnostics)
                    / len(self.risk_diagnostics)
                    for key in keys
                }
            )
        if self.truncation_lengths:
            summary["mean_truncation_length"] = sum(self.truncation_lengths) / len(
                self.truncation_lengths
            )
        return summary

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        assert "goal" in info_dict, "goal not in info_dict"
        if action_candidates.shape[1] != self.cvar_cfg.num_samples:
            raise ValueError(
                f"solver supplied {action_candidates.shape[1]} candidates, but the "
                f"Goal-CVaR config expects {self.cvar_cfg.num_samples}"
            )

        device = self.device
        for key in list(info_dict):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)
        action_candidates = action_candidates.to(device)
        goal = self._encode_goal(info_dict)

        self.estimator.prepare(self)
        try:
            trajectories = torch.stack(
                [
                    self.estimator.trajectory(
                        self, sample_idx, info_dict, action_candidates
                    )
                    for sample_idx in range(self.estimator.n)
                ],
                dim=0,
            )
        finally:
            self.estimator.finish(self)

        mean_trajectory, sigma2 = rollout_statistics(trajectories)
        idx = compute_truncation_index(sigma2, self.trunc_cfg)
        cost, diagnostics = goal_cvar_cost(
            trajectories, mean_trajectory, sigma2, goal, self.cvar_cfg, idx=idx
        )
        if cost.shape != action_candidates.shape[:2]:
            raise RuntimeError(
                f"Goal-CVaR objective returned {tuple(cost.shape)}, expected "
                f"{tuple(action_candidates.shape[:2])}"
            )

        self.risk_diagnostics.append(
            {
                key: value.detach().float().mean().cpu().item()
                for key, value in diagnostics.items()
            }
        )
        self.mean_sigma2.append(sigma2.detach().float().mean().cpu().item())
        self.truncation_lengths.append((idx.float() + 1).mean().item())
        self.sigma2_refs.append(sigma2[..., 0].detach().float().mean().cpu().item())
        return cost
