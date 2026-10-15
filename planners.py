"""Build the planner selected by ``cfg.method`` (config/eval/method/*.yaml).

All planners wrap the same frozen LeWM checkpoint; they differ only in how a
candidate action sequence is scored:

    baseline  -> the LeWM model itself (one deterministic rollout)
    ua        -> UncertaintyAwareJEPA (MC-dropout aggregation, optional gated
                 horizon truncation or sigma^2 penalty)
    goal_cvar -> GoalCVaRJEPA (MC-dropout aggregation + CVaR soft penalty)
    risk      -> RiskAwareJEPA (LeWM cost corrected by a learned risk head, ua/risk.py)
    ens       -> EnsemblePenaltyJEPA (LeWM cost + lambda * latent-ensemble sigma^2, ua/ensemble.py)
"""

from __future__ import annotations

import stable_worldmodel as swm
import torch
from omegaconf import DictConfig

from ua import (
    GoalCVaRConfig,
    GoalCVaRJEPA,
    MCDropoutEstimator,
    TruncationConfig,
    UncertaintyAwareJEPA,
)


def load_base(policy: str, device: str, overrides: dict | None = None) -> torch.nn.Module:
    """Load the pretrained LeWM cost model, frozen, on ``device``.

    ``overrides`` sets constructor arguments of the saved config (dotted keys),
    e.g. ``{"w_prop": 0.1}`` for the block-weighted cost of cube/variants.py.
    """
    base = swm.wm.utils.load_pretrained(policy, extra_args=dict(overrides) if overrides else None)
    base = base.to(device).eval()
    base.requires_grad_(False)
    base.interpolate_pos_encoding = True
    return base


def truncation_config(trunc: DictConfig) -> TruncationConfig:
    threshold = trunc.get("threshold")
    return TruncationConfig(
        mode=str(trunc.mode),
        omega=float(trunc.omega),
        k_base=float(trunc.k_base),
        k_min=int(trunc.k_min),
        k_max=None if trunc.get("k_max") is None else int(trunc.k_max),
        threshold=None if threshold is None else float(threshold),
        penalty_weight=float(trunc.get("penalty_weight", 0.0) or 0.0),
    )


def build_planner(cfg: DictConfig, device: str) -> torch.nn.Module:
    method = cfg.method
    overrides = cfg.get("model_overrides")
    if overrides is not None:
        from omegaconf import OmegaConf

        overrides = OmegaConf.to_container(overrides, resolve=True)
    base = load_base(cfg.policy, device, overrides)
    planner = str(method.planner)

    if planner == "baseline":
        return base

    if planner == "ens":
        from pathlib import Path

        from ua.ensemble import EnsemblePenaltyJEPA, load_ensemble

        ens_path = Path(str(method.ensemble_path)).expanduser()
        if not ens_path.is_absolute():
            import os

            ens_path = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm")) / ens_path
        ensemble, _ = load_ensemble(ens_path, device)
        return EnsemblePenaltyJEPA(
            base=base, ensemble=ensemble, lam=float(method.lam), history_size=int(method.get("history_size", 3))
        ).to(device)

    if planner == "risk":
        from pathlib import Path

        from ua.risk import RiskAwareJEPA, load_head

        head_path = Path(str(method.head_path))
        if not head_path.is_absolute():
            head_path = Path(__file__).resolve().parent / head_path
        head, meta = load_head(head_path, device)
        return RiskAwareJEPA(
            base=base,
            head=head,
            use_sigma=bool(meta["use_sigma"]),
            mode=str(method.mode),
            beta=float(method.beta),
            n_mc=int(method.get("n_mc", 8)),
            history_size=int(method.get("history_size", 3)),
        ).to(device)

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
        raise ValueError(f"Unknown method.planner {planner!r}; use baseline | ua | goal_cvar | risk | ens")
    return model.to(device)
