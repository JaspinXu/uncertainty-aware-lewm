"""Planner wrapper for the standalone Soft Penalty v2 objective."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ua.truncation import TruncationConfig, compute_truncation_index
from ua.ua_jepa import UncertaintyAwareJEPA
from ua.uncertainty import UncertaintyEstimator, rollout_statistics

from .config import RiskConfig
from .costs import compute_risk_cost


class RiskAwareJEPA(UncertaintyAwareJEPA):
    def __init__(
        self,
        base: torch.nn.Module,
        estimator: UncertaintyEstimator,
        risk_cfg: RiskConfig,
        trunc_cfg: Optional[TruncationConfig] = None,
        history_size: int = 3,
    ):
        super().__init__(
            base=base,
            estimator=estimator,
            trunc_cfg=trunc_cfg or TruncationConfig(mode="none"),
            history_size=history_size,
        )
        self.risk_cfg = risk_cfg
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
        if action_candidates.shape[1] != self.risk_cfg.num_samples:
            raise ValueError(
                f"solver supplied {action_candidates.shape[1]} candidates, but risk "
                f"config expects {self.risk_cfg.num_samples}"
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
        cost, diagnostics = compute_risk_cost(
            trajectories,
            mean_trajectory,
            sigma2,
            goal,
            self.risk_cfg,
            idx=idx,
        )
        if cost.shape != action_candidates.shape[:2]:
            raise RuntimeError(
                f"risk objective returned {tuple(cost.shape)}, expected "
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
