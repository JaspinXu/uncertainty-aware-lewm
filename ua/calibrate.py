"""Calibrate the truncation threshold / k_base on validation rollouts.

For ``adaptive_perstep`` and ``gated`` modes you need a sigma^2 cutoff that
matches the *planning-time* uncertainty distribution. Prefer:

  1. run CEM on held-out PushT states (same action-candidate distribution
     the planner sees);
  2. collect per-candidate sigma^2 at the first predicted step (sigma^2_0);
  3. set ``threshold`` to a percentile of that distribution (p50–p90).

Expert-action rollouts understate planning-time sigma^2 (CEM candidates are
much more uncertain), so they are only kept as an ablation (``source=expert``).

Use ``scripts/calibrate_gated_threshold.py`` (default ``source=planning``).
"""

from __future__ import annotations

import numpy as np
import torch

from .uncertainty import rollout_statistics


def threshold_from_sigma2(sigma2_values, percentile: float = 90.0) -> float:
    """Return the ``percentile``-th percentile of observed in-distribution sigma^2."""
    if torch.is_tensor(sigma2_values):
        sigma2_values = sigma2_values.detach().cpu().numpy()
    sigma2_values = np.asarray(sigma2_values).reshape(-1)
    sigma2_values = sigma2_values[np.isfinite(sigma2_values)]
    return float(np.percentile(sigma2_values, percentile))


def flatten_sigma2_refs(ua_model) -> np.ndarray:
    """Concatenate all per-candidate sigma^2_0 values recorded by ``get_cost``."""
    if not getattr(ua_model, "sigma2_refs", None):
        return np.array([], dtype=np.float64)
    return torch.cat(list(ua_model.sigma2_refs), dim=0).numpy().astype(np.float64)


def flatten_sigma2_perstep(ua_model) -> np.ndarray:
    """Stack planning-time sigma^2 as ``(num_candidates, rollout_steps)``."""
    if not getattr(ua_model, "sigma2_perstep", None):
        return np.empty((0, 0), dtype=np.float64)
    return (
        torch.cat(list(ua_model.sigma2_perstep), dim=0)
        .numpy()
        .astype(np.float64)
    )


def perstep_thresholds(
    sigma2_steps,
    percentile: float = 90.0,
) -> np.ndarray:
    """Calibrate one uncertainty cutoff per rollout depth."""
    if torch.is_tensor(sigma2_steps):
        sigma2_steps = sigma2_steps.detach().cpu().numpy()
    values = np.asarray(sigma2_steps)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            "sigma2_steps must be a non-empty (num_candidates, rollout_steps) array"
        )
    values = np.where(np.isfinite(values), values, np.nan)
    return np.nanpercentile(values, percentile, axis=0)


@torch.no_grad()
def collect_sigma2_ref(ua_model, info: dict, actions: torch.Tensor, step: int = 0) -> torch.Tensor:
    """Run stochastic rollouts and return sigma^2 at a reference step.

    Parameters
    ----------
    ua_model : UncertaintyAwareJEPA
    info : dict
        Planning info dict (pixels, goal, ...).
    actions : (B, S, T, A) tensor
        Action candidates to roll out.
    step : int
        Rollout step whose sigma^2 is returned (0 = first predicted step).

    Returns
    -------
    sigma2_ref : (B, S) tensor
    """
    ua_model.estimator.prepare(ua_model)
    trajs = [
        ua_model.estimator.trajectory(ua_model, i, info, actions)
        for i in range(ua_model.estimator.n)
    ]
    ua_model.estimator.finish(ua_model)
    trajs = torch.stack(trajs, dim=0)
    _, sigma2_steps = rollout_statistics(trajs)
    return sigma2_steps[..., step]


def summarize(sigma2_values) -> dict:
    """Quick descriptive stats to help pick ``omega`` / ``k_base`` / ``threshold``."""
    if torch.is_tensor(sigma2_values):
        sigma2_values = sigma2_values.detach().cpu().numpy()
    s = np.asarray(sigma2_values).reshape(-1)
    s = s[np.isfinite(s)]
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p50": float(np.percentile(s, 50)),
        "p90": float(np.percentile(s, 90)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "max": float(s.max()),
    }
