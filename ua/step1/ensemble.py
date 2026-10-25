"""Predictor heads trained on a frozen encoder (the ensemble sigma^2 of Step 1).

1. ``cache_latents``: encode whole dataset episodes once with the frozen
   encoder + projector, using the planner's image transform.
2. ``new_head``: a predictor / action encoder / pred_proj with the released
   architecture and a fresh random init.
3. ``train_head``: the LeWM 1-step teacher-forcing loss on cached latents
   (history 3 frames, frameskip 5), episode-bootstrapped per member.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filter)
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from .common import encode_frames
from .rollout import PredictorHead

FRAMESKIP = 5
HISTORY = 3


@torch.no_grad()
def cache_latents(h5_path, episodes, model, transform, device, batch: int = 256, log_every: int = 200):
    """Encode every frame of ``episodes`` (same preprocessing as the planner).

    Returns ``{"emb": (F, D) float16, "action": (F, A) float32 raw,
    "ep": (F,) int, "step": (F,) int, "offsets": (E,), "lengths": (E,)}``
    with frames stored episode by episode in the order of ``episodes``.
    """
    model.eval()
    f = h5py.File(h5_path, "r")
    ep_off, ep_len = f["ep_offset"][:], f["ep_len"][:]
    embs, acts, eps, steps, offsets, lengths = [], [], [], [], [], []
    total = 0
    t0 = time.time()
    for n, ep in enumerate(episodes):
        s, L = int(ep_off[ep]), int(ep_len[ep])
        pix = f["pixels"][s : s + L]  # (L, 224, 224, 3) uint8
        act = f["action"][s : s + L].astype(np.float32)
        embs.append(encode_frames(model, pix, transform, device, batch).half().numpy())
        acts.append(act)
        eps.append(np.full(L, ep))
        steps.append(np.arange(L))
        offsets.append(total)
        lengths.append(L)
        total += L
        if (n + 1) % log_every == 0:
            rate = total / (time.time() - t0)
            print(f"[cache] {n + 1}/{len(episodes)} episodes, {rate:.0f} frames/s", flush=True)
    f.close()
    return {
        "emb": np.concatenate(embs),
        "action": np.concatenate(acts),
        "ep": np.concatenate(eps),
        "step": np.concatenate(steps),
        "offsets": np.asarray(offsets),
        "lengths": np.asarray(lengths),
    }


def new_head(config_json: str | Path, seed: int) -> PredictorHead:
    """Fresh head with the checkpoint's predictor / action_encoder / pred_proj config."""
    cfg = json.loads(Path(config_json).read_text())
    torch.manual_seed(seed)
    return PredictorHead(
        hydra.utils.instantiate(cfg["predictor"]),
        hydra.utils.instantiate(cfg["action_encoder"]),
        hydra.utils.instantiate(cfg["pred_proj"]),
    )


def load_members(ensemble_dir: str | Path, config_json: str | Path, device) -> list[PredictorHead]:
    """The trained members (member_<m>.pt, saved on whatever device they trained on), eval mode."""
    members = []
    for path in sorted(Path(ensemble_dir).glob("member_*.pt")):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        head = new_head(config_json, seed=ck["seed"])
        head.load_state_dict(ck["state_dict"])
        members.append(head.to(device).eval())
    assert members, f"no member_*.pt in {ensemble_dir}"
    return members


def base_head(base) -> PredictorHead:
    """The released predictor viewed as a head (shares parameters with ``base``)."""
    return PredictorHead(base.predictor, base.action_encoder, base.pred_proj)


class ClipSampler:
    """Random LeWM clips (4 frames, 3 action blocks) from cached latents."""

    def __init__(self, cache: dict, episode_rows, action_mean, action_std, device, seed: int):
        self.device = device
        self.emb = torch.as_tensor(cache["emb"], device=device).float()
        # the last step of every episode has a NaN action; like LeWM training
        # (train.py: nan_to_num), zero it so 0 * NaN never reaches the gradients
        act = np.nan_to_num((cache["action"] - action_mean) / action_std, nan=0.0)
        self.act = torch.as_tensor(act, device=device).float()
        span = (HISTORY + 1 - 1) * FRAMESKIP + FRAMESKIP  # frames 0,5,10,15 + last block's actions
        # every (episode row, start) whose clip fits in the episode
        starts = []
        for r in episode_rows:
            off, L = int(cache["offsets"][r]), int(cache["lengths"][r])
            starts.append(off + np.arange(L - span + 1))
        self.starts = torch.as_tensor(np.concatenate(starts), device=device)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.frame_off = torch.arange(HISTORY + 1, device=device) * FRAMESKIP  # (4,)
        self.act_off = torch.arange((HISTORY + 1) * FRAMESKIP, device=device)  # (20,)

    def __len__(self):
        return len(self.starts)

    def batch(self, size: int, idx: torch.Tensor | None = None):
        if idx is None:
            idx = torch.randint(len(self.starts), (size,), generator=self.g).to(self.device)
        s = self.starts[idx]
        emb = self.emb[s[:, None] + self.frame_off]  # (B, 4, D)
        act = self.act[s[:, None] + self.act_off]  # (B, 20, A)
        act = act.reshape(len(s), HISTORY + 1, FRAMESKIP * act.shape[-1])  # (B, 4, 25)
        return emb, act


def lewm_loss(head: PredictorHead, emb: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """LeWM pred_loss: predict frames 1..3 from frames 0..2 (causal predictor)."""
    act_emb = head.action_encoder(act)
    pred = head.predict(emb[:, :HISTORY], act_emb[:, :HISTORY])
    return F.mse_loss(pred, emb[:, 1:])


@torch.no_grad()
def eval_loss(head: PredictorHead, sampler: ClipSampler, n: int = 20000, batch: int = 2048) -> float:
    was = head.training
    head.eval()
    idx = torch.linspace(0, len(sampler) - 1, min(n, len(sampler))).long().to(sampler.device)
    tot, cnt = 0.0, 0
    for i in range(0, len(idx), batch):
        emb, act = sampler.batch(0, idx[i : i + batch])
        tot += lewm_loss(head, emb, act).item() * len(emb)
        cnt += len(emb)
    head.train(was)
    return tot / cnt


def train_head(head, sampler, val_sampler, steps, batch, lr, weight_decay=1e-3, warmup=500, log_every=500):
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(i):
        if i < warmup:
            return lr * (i + 1) / warmup
        p = (i - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    log = []
    t0 = time.time()
    for i in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(i)
        emb, act = sampler.batch(batch)
        loss = lewm_loss(head, emb, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (i + 1) % log_every == 0 or i + 1 == steps:
            val = eval_loss(head, val_sampler, n=8000)
            log.append({"step": i + 1, "train": loss.item(), "val": val, "sec": time.time() - t0})
            print(f"  step {i + 1}/{steps} train {loss.item():.4f} val {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
    head.eval()
    return log
