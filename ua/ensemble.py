"""Latent-space dynamics ensemble on the frozen LeWM encoder (deep-ensemble sigma^2).

MC-dropout reuses one network; a deep ensemble trains K independent dynamics
models, which is the uncertainty estimator of UA-MBRL. Training K full world
models is expensive, but the planner only needs the *dynamics* to disagree:
we freeze the pretrained encoder, pre-compute its latents for every dataset
frame once (scripts/cube/encode_dataset.py) and train K predictor heads
(LeWM predictor + action embedder + projection, random init, bootstrap
episodes) on those latents (train_ensemble.py). Disagreement between members
on a candidate action sequence is sigma^2_ens.

``EnsemblePenaltyJEPA`` keeps the pretrained LeWM cost and adds
lambda * sigma^2_ens at the final predicted block.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from cube.rollout import goal_cost, per_step_variance, rollout_from_latents

_MLP = {
    "_target_": "stable_worldmodel.wm.lewm.module.MLP",
    "input_dim": 192,
    "output_dim": 192,
    "hidden_dim": 2048,
    "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
}


def member_config(dim: int = 192, depth: int = 6, action_dim: int = 25) -> dict:
    """Same predictor architecture as the pretrained LeWM-Cube."""
    return {
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": 3,
            "input_dim": dim,
            "hidden_dim": dim,
            "output_dim": dim,
            "depth": depth,
            "heads": 16,
            "mlp_dim": 2048,
            "dim_head": 64,
            "dropout": 0.1,
            "emb_dropout": 0.0,
        },
        "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder", "input_dim": action_dim, "emb_dim": dim},
        "pred_proj": {**copy.deepcopy(_MLP), "input_dim": dim, "output_dim": dim},
    }


class LatentDynamics(nn.Module):
    """One ensemble member: (z_<=t, a_<=t) -> z_{t+1} in the frozen latent space."""

    def __init__(self, predictor: nn.Module, action_encoder: nn.Module, pred_proj: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.pred_proj = pred_proj

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        b = emb.size(0)
        return self.pred_proj(preds.flatten(0, 1)).reshape(b, -1, preds.size(-1))


def build_member(cfg: dict) -> LatentDynamics:
    from hydra.utils import instantiate

    parts = {k: instantiate(copy.deepcopy(v)) for k, v in cfg.items()}
    return LatentDynamics(**parts)


class LatentEnsemble(nn.Module):
    def __init__(self, members: list[LatentDynamics]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def __len__(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def rollouts(self, z_hist: torch.Tensor, actions: torch.Tensor, history_size: int = 3, chunk: int = 4096) -> torch.Tensor:
        """(K, B, S, P, D) predicted latents of every member (eval mode)."""
        outs = []
        S = actions.shape[1]
        for m in self.members:
            m.eval()
            parts = [rollout_from_latents(m, z_hist, actions[:, i : i + chunk], history_size) for i in range(0, S, chunk)]
            outs.append(torch.cat(parts, dim=1))
        return torch.stack(outs)


def save_ensemble(ens: LatentEnsemble, cfg: dict, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "state_dicts": [m.state_dict() for m in ens.members], "meta": meta}, path)


def load_ensemble(path: str | Path, device="cpu") -> tuple[LatentEnsemble, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    members = []
    for sd in ckpt["state_dicts"]:
        m = build_member(ckpt["config"])
        m.load_state_dict(sd)
        members.append(m.eval())
    return LatentEnsemble(members).to(device).eval(), ckpt["meta"]


class EnsemblePenaltyJEPA(nn.Module):
    """J = ||z_hat_H - z_goal||^2 (pretrained LeWM) + lam * sigma^2_ens(H)."""

    def __init__(self, base: nn.Module, ensemble: LatentEnsemble, lam: float, history_size: int = 3):
        super().__init__()
        self.base = base
        self.ensemble = ensemble
        self.lam = float(lam)
        self.history_size = int(history_size)
        self.sigma_log: list[float] = []
        self.penalty_log: list[float] = []

    def diagnostics_summary(self) -> dict:
        if not self.sigma_log:
            return {}
        return {
            "mean_sigma2_ens_H": float(sum(self.sigma_log) / len(self.sigma_log)),
            "mean_penalty": float(sum(self.penalty_log) / len(self.penalty_log)),
        }

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        dev = next(self.base.parameters()).device
        dtype = next(self.base.parameters()).dtype
        cache = info_dict.setdefault("_ens_cache", {})
        if "z_hist" not in cache:
            self.base.eval()
            cache["z_hist"] = self.base.encode({"pixels": info_dict["pixels"][:, 0].to(dev, dtype)})["emb"]
            cache["z_goal"] = self.base.encode({"pixels": info_dict["goal"][:, 0].to(dev, dtype)})["emb"][:, -1]
        z_hist, z_goal = cache["z_hist"], cache["z_goal"]
        acts = action_candidates.to(dev, dtype)
        self.base.eval()
        det = rollout_from_latents(self.base, z_hist, acts, self.history_size)
        cost = goal_cost(det[..., -1, :], z_goal[:, None]).float()
        sig2 = per_step_variance(self.ensemble.rollouts(z_hist, acts, self.history_size))[..., -1].float()
        self.sigma_log.append(float(sig2.mean()))
        self.penalty_log.append(float((self.lam * sig2).mean()))
        return cost + self.lam * sig2
