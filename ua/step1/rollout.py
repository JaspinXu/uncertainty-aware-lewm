"""Latent rollouts from cached embeddings and the two sigma^2 estimators.

``rollout_from_emb`` is ``UncertaintyAwareJEPA.single_rollout`` without the
encoder call: the encoder is frozen and dropout-free (HF ViT, p=0), so the
start embedding can be computed once and reused for any number of
deterministic, MC-dropout or ensemble rollouts. The action encoder is
per-timestep (kernel-1 conv + MLP), so embedding the whole sequence at once
equals the incremental embedding of the original code.
"""

from __future__ import annotations

from typing import Sequence

import torch
from einops import rearrange
from torch import nn

from ua.uncertainty import _set_dropout_train


class PredictorHead(nn.Module):
    """Predictor + action encoder + pred_proj: everything a rollout needs."""

    def __init__(self, predictor: nn.Module, action_encoder: nn.Module, pred_proj: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.pred_proj = pred_proj

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))


def rollout_from_emb(model, emb0: torch.Tensor, actions: torch.Tensor, history_size: int = 3):
    """Autoregressive rollout.

    emb0: (N, H, D) start embeddings, actions: (N, T, A) normalized action
    blocks (T >= H). Returns the T - H + 1 predicted embeddings (N, P, D);
    the last one is the state after all T blocks (the one the cost uses).
    """
    H, T = emb0.shape[1], actions.shape[1]
    act_emb = model.action_encoder(actions)
    emb = emb0
    preds = []
    for t in range(T - H + 1):
        ctx = emb[:, -history_size:]
        act = act_emb[:, : H + t][:, -history_size:]
        pred = model.predict(ctx, act)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        preds.append(pred)
    return torch.cat(preds, dim=1)


@torch.no_grad()
def deterministic_rollouts(model, emb0, actions, chunk: int = 4096) -> torch.Tensor:
    """(N, P, D) eval-mode predictions, computed in chunks of candidates."""
    was_training = model.training
    model.eval()
    out = [
        rollout_from_emb(model, emb0[i : i + chunk], actions[i : i + chunk])
        for i in range(0, actions.shape[0], chunk)
    ]
    model.train(was_training)
    return torch.cat(out, dim=0)


@torch.no_grad()
def mc_dropout_rollouts(model, emb0, actions, n: int, seed: int, chunk: int = 4096) -> torch.Tensor:
    """(n, N, P, D) rollouts with predictor dropout on and norms frozen."""
    torch.manual_seed(seed)
    _set_dropout_train(model, True)
    try:
        out = torch.stack(
            [
                torch.cat(
                    [
                        rollout_from_emb(model, emb0[i : i + chunk], actions[i : i + chunk])
                        for i in range(0, actions.shape[0], chunk)
                    ],
                    dim=0,
                )
                for _ in range(n)
            ],
            dim=0,
        )
    finally:
        _set_dropout_train(model, False)
    return out


@torch.no_grad()
def ensemble_rollouts(members: Sequence[nn.Module], emb0, actions, chunk: int = 4096) -> torch.Tensor:
    """(M, N, P, D) eval-mode rollouts, one per member."""
    return torch.stack([deterministic_rollouts(m, emb0, actions, chunk) for m in members], dim=0)


def sigma2(trajs: torch.Tensor) -> torch.Tensor:
    """(n, N, P, D) -> (N, P): unbiased variance across samples, summed over D
    (same estimator as ``ua.uncertainty.rollout_statistics``)."""
    return trajs.var(dim=0, unbiased=True).sum(dim=-1)
