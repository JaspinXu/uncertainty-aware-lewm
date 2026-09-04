"""Epistemic-uncertainty estimators for latent rollouts.

The planner needs a per-step uncertainty signal sigma^2_M(z_t) over an imagined
latent rollout. We expose two estimators that both reduce to *sampling several
rollout trajectories and measuring their disagreement*:

- ``MCDropoutEstimator`` : a single pretrained LeWM, run ``n`` times with the
  predictor's dropout layers kept active at inference. Needs no extra training
  and no dataset -> ideal for the laptop "Tier 1" setup. (The shipped LeWM
  predictor uses dropout=0.1, so this produces a genuine signal.)

- ``EnsembleEstimator`` : ``n`` independently-seeded predictors, matching
  UA-MBRL Eq. (15)-(18). Use this for the faithful "Tier 2" results (freeze the
  pretrained encoder, train N small ``ARPredictor`` heads on the dataset).

Both implement the same ``trajectory(ua_model, idx, info, actions)`` interface,
so ``UncertaintyAwareJEPA`` is agnostic to which one is plugged in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


def _set_dropout_train(module: nn.Module, enabled: bool) -> None:
    """Toggle dropout-active mode while keeping BatchNorm statistics frozen.

    This is what makes MC-dropout valid at inference: we want stochastic dropout
    masks but deterministic normalization statistics.

    We set the *entire* module to train/eval mode (so ``self.training`` checks in
    forward methods return True), then force BatchNorm layers back to eval mode.
    This covers both ``nn.Dropout`` layers and functional dropout (e.g. in
    ``F.scaled_dot_product_attention``) that checks the module's training flag.
    """
    if enabled:
        module.train()  # makes self.training=True everywhere
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                              nn.InstanceNorm2d, nn.InstanceNorm3d)):
                m.eval()  # freeze normalization statistics
    else:
        module.eval()


class UncertaintyEstimator(ABC):
    """Produces ``n`` latent rollout trajectories whose variance is the signal."""

    #: number of trajectories sampled per cost evaluation
    n: int

    def prepare(self, ua_model) -> None:
        """Hook called once before sampling (e.g. enable dropout)."""

    def finish(self, ua_model) -> None:
        """Hook called once after sampling (e.g. restore eval mode)."""

    @abstractmethod
    def trajectory(self, ua_model, idx: int, info: dict, actions: torch.Tensor) -> torch.Tensor:
        """Return one rollout's predicted steps, shape ``(B, S, P, D)``."""


class MCDropoutEstimator(UncertaintyEstimator):
    """Monte-Carlo dropout on a single pretrained model (no retraining/data)."""

    def __init__(self, n_samples: int = 5):
        # n_samples=1 is allowed as a stochastic single-sample control (dropout
        # on, no aggregation). Epistemic variance is then identically zero.
        if n_samples < 1:
            raise ValueError("MC-dropout needs n_samples >= 1")
        self.n = n_samples

    def prepare(self, ua_model) -> None:
        _set_dropout_train(ua_model.base, True)

    def finish(self, ua_model) -> None:
        _set_dropout_train(ua_model.base, False)

    def trajectory(self, ua_model, idx: int, info: dict, actions: torch.Tensor) -> torch.Tensor:
        # same model every time; stochasticity comes from the active dropout masks
        return ua_model.single_rollout(ua_model.base, info, actions)


class EnsembleEstimator(UncertaintyEstimator):
    """Disagreement across ``n`` independent predictors (UA-MBRL style)."""

    def __init__(self, members: List[nn.Module]):
        if len(members) < 2:
            raise ValueError("EnsembleEstimator needs >= 2 members")
        self.members = members
        self.n = len(members)

    def prepare(self, ua_model) -> None:
        for m in self.members:
            m.eval()

    def trajectory(self, ua_model, idx: int, info: dict, actions: torch.Tensor) -> torch.Tensor:
        return ua_model.single_rollout(self.members[idx], info, actions)


def rollout_statistics(trajectories: torch.Tensor):
    """Compute the ensemble mean and per-step epistemic variance.

    Parameters
    ----------
    trajectories : (N, B, S, P, D) tensor
        ``N`` sampled rollouts of ``P`` predicted steps in latent dim ``D``.

    Returns
    -------
    mean_traj : (B, S, P, D)
        Mean latent trajectory (used as the point prediction for the cost).
    sigma2_steps : (B, S, P)
        Epistemic uncertainty per step = variance across the N rollouts,
        summed over the latent dimension (matches UA-MBRL's scalar sigma^2_M).
    """
    mean_traj = trajectories.mean(dim=0)
    if trajectories.shape[0] < 2:
        # Stochastic N=1 control: no disagreement estimate is available.
        sigma2_steps = torch.zeros(
            mean_traj.shape[:-1], dtype=mean_traj.dtype, device=mean_traj.device
        )
        return mean_traj, sigma2_steps
    # Bessel correction keeps the expected variance scale comparable across
    # N=2/3/5. With unbiased=False, small ensembles systematically report a
    # smaller sigma^2 and therefore receive weaker truncation at the same
    # omega/threshold.
    var_per_dim = trajectories.var(dim=0, unbiased=True)  # (B, S, P, D)
    sigma2_steps = var_per_dim.sum(dim=-1)  # (B, S, P)
    return mean_traj, sigma2_steps
