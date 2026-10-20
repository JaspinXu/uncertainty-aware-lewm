"""Build the planner selected by ``cfg.method`` (config/eval/method/*.yaml).

All planners wrap the same frozen LeWM checkpoint; they differ only in how a
candidate action sequence is scored:

    baseline  -> the LeWM model itself (one deterministic rollout)
    ua        -> UncertaintyAwareJEPA (MC-dropout aggregation, optional gated
                 horizon truncation)
    goal_cvar -> GoalCVaRJEPA (MC-dropout aggregation + CVaR soft penalty)
"""

from __future__ import annotations

import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from ua import (
    GoalCVaRConfig,
    GoalCVaRJEPA,
    MCDropoutEstimator,
    TruncationConfig,
    UncertaintyAwareJEPA,
)


def load_base(policy: str, device: str) -> torch.nn.Module:
    """Load the pretrained LeWM cost model, frozen, on ``device``."""
    base = swm.wm.utils.load_pretrained(policy).to(device).eval()
    base.requires_grad_(False)
    base.interpolate_pos_encoding = True
    return base


def truncation_config(trunc: DictConfig) -> TruncationConfig:
    threshold = trunc.get("threshold")
    if threshold is not None:  # a scalar, or one value per predicted step (adaptive_perstep / gated_tc)
        threshold = [float(t) for t in threshold] if OmegaConf.is_list(threshold) else float(threshold)
    return TruncationConfig(
        mode=str(trunc.mode),
        omega=float(trunc.omega),
        k_base=float(trunc.k_base),
        k_min=int(trunc.k_min),
        k_max=None if trunc.get("k_max") is None else int(trunc.k_max),
        threshold=threshold,
        terminal_weight=float(trunc.get("terminal_weight", 0.0)),
    )


def build_planner(cfg: DictConfig, device: str) -> torch.nn.Module:
    method = cfg.method
    base = load_base(cfg.policy, device)
    planner = str(method.planner)

    if planner == "baseline":
        return base

    estimator = MCDropoutEstimator(n_samples=int(method.n_samples))
    trunc = truncation_config(method.truncation)
    history_size = int(method.get("history_size", 3))

    if planner == "ua":
        model = UncertaintyAwareJEPA(
            base=base, estimator=estimator, trunc_cfg=trunc, history_size=history_size
        )
    elif planner == "goal_cvar":
        cvar = GoalCVaRConfig(
            alpha=float(method.alpha),
            num_samples=int(cfg.solver.num_samples),
            topk=int(cfg.solver.topk),
            tail_fraction=float(method.tail_fraction),
            tail_count=None if method.get("tail_count") is None else int(method.tail_count),
            cap=float(method.cap),
            eps=float(method.eps),
        )
        model = GoalCVaRJEPA(
            base=base,
            estimator=estimator,
            cvar_cfg=cvar,
            trunc_cfg=trunc,
            history_size=history_size,
        )
    else:
        raise ValueError(f"Unknown method.planner {planner!r}; use baseline | ua | goal_cvar")
    return model.to(device)
