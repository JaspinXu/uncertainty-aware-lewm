"""Latent rollouts of a frozen LeWM, deterministic or with MC-dropout.

``rollout_from_latents`` reproduces ``stable_worldmodel.wm.lewm.LeWM.rollout``
step for step (same history window, same action-embedding slices) but works on
an explicit history latent and returns only the predicted steps, so the same
code scores arbitrary candidate pools after planning.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from ua.uncertainty import _set_dropout_train


@torch.no_grad()
def encode_pixels(model: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    """Encode normalised frames ``(B, T, C, H, W)`` into latents ``(B, T, D)``."""
    was_training = model.training
    model.eval()
    try:
        return model.encode({"pixels": pixels})["emb"]
    finally:
        model.train(was_training)


def rollout_from_latents(
    model: nn.Module,
    z_hist: torch.Tensor,
    actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    """Autoregressive latent rollout.

    Parameters
    ----------
    z_hist : (B, T0, D) latents of the observed history (T0 = history_len).
    actions : (B, S, H, A) candidate action blocks; the first T0 blocks pair
        with the history frames exactly as in ``LeWM.rollout``.

    Returns
    -------
    (B, S, H - T0 + 1, D) predicted latents; the last entry is the latent the
    planning cost compares with the goal.
    """
    B, S, H, A = actions.shape
    T0, D = z_hist.shape[1], z_hist.shape[2]
    if H < T0:
        raise ValueError(f"need at least {T0} action blocks, got {H}")
    emb = z_hist[:, None].expand(B, S, T0, D).reshape(B * S, T0, D)
    act_emb = model.action_encoder(actions.reshape(B * S, H, A))
    embs = list(emb.unbind(dim=1))
    preds = []
    for t in range(H - T0 + 1):
        lo = max(0, T0 + t - history_size)
        e = torch.stack(embs[lo:], dim=1)
        a = act_emb[:, lo : T0 + t]
        nxt = model.predict(e, a)[:, -1]
        embs.append(nxt)
        preds.append(nxt)
    return torch.stack(preds, dim=1).reshape(B, S, len(preds), D)


@contextlib.contextmanager
def mc_dropout(model: nn.Module):
    """Dropout active, normalisation statistics frozen (as in ua.MCDropoutEstimator)."""
    was_training = model.training
    _set_dropout_train(model, True)
    try:
        yield model
    finally:
        _set_dropout_train(model, False)
        model.train(was_training)


@torch.no_grad()
def deterministic_rollout(model, z_hist, actions, history_size=3, chunk=4096):
    model.eval()
    return _chunked(lambda a: rollout_from_latents(model, z_hist, a, history_size), actions, chunk)


@torch.no_grad()
def mc_rollouts(model, z_hist, actions, n_samples=8, history_size=3, chunk=4096):
    """Stack of ``n_samples`` MC-dropout rollouts, shape (N, B, S, P, D)."""
    outs = []
    with mc_dropout(model):
        for _ in range(int(n_samples)):
            outs.append(
                _chunked(lambda a: rollout_from_latents(model, z_hist, a, history_size), actions, chunk)
            )
    return torch.stack(outs, dim=0)


def _chunked(fn, actions: torch.Tensor, chunk: int) -> torch.Tensor:
    """Apply ``fn`` over chunks of the candidate dimension S to bound memory."""
    S = actions.shape[1]
    if S <= chunk:
        return fn(actions)
    return torch.cat([fn(actions[:, i : i + chunk]) for i in range(0, S, chunk)], dim=1)


def goal_cost(pred_last: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Planning cost ||z_hat_H - z_goal||^2 (sum over latent dims), as LeWM.criterion."""
    return ((pred_last - z_goal) ** 2).sum(dim=-1)


def per_step_variance(samples: torch.Tensor) -> torch.Tensor:
    """Across-sample variance summed over latent dims (Bessel), as ua.rollout_statistics.

    samples: (N, ..., D) -> (...)
    """
    if samples.shape[0] < 2:
        return torch.zeros(samples.shape[1:-1], dtype=samples.dtype, device=samples.device)
    return samples.var(dim=0, unbiased=True).sum(dim=-1)
