"""Learned risk head + risk-aware planner (L3).

The risk head is a small MLP that, from quantities the planner already has for
every CEM candidate (current/goal latents, the predicted terminal latent, the
MC-dropout sigma^2 profile and the planning cost), predicts what the simulator
oracle measured for replayed candidates:

    out[..., 0]  log1p of the true terminal prediction error ||z_hat_H - z_H||^2
    out[..., 1]  log1p of the true terminal cost            ||z_H - z_goal||^2
    out[..., 2]  logit of segment success

``RiskAwareJEPA`` exposes the solver's ``get_cost`` interface and scores

    mode="err":        J = d_hat + beta * err_hat      (learned error penalty)
    mode="true_cost":  J = c_true_hat                  (learned cost correction)

where d_hat is the frozen LeWM's own cost.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cube.rollout import goal_cost, mc_rollouts, per_step_variance, rollout_from_latents


def risk_features(
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    pred: torch.Tensor,
    det_cost: torch.Tensor,
    sig2: torch.Tensor | None = None,
    mc_cost_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feature vector per candidate.

    z0, z_goal: (..., D); pred: (..., P, D); det_cost: (...); sig2: (..., P); mc_cost_std: (...)
    """
    zH = pred[..., -1, :]
    feats = [zH - z_goal, zH - z0, z0, torch.log1p(det_cost.clamp_min(0))[..., None]]
    if sig2 is not None:
        feats.append(torch.log(sig2.clamp_min(0) + 1e-6))
        feats.append(torch.log1p(mc_cost_std.clamp_min(0))[..., None])
    return torch.cat(feats, dim=-1)


class RiskHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, n_out: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats - self.feat_mean) / self.feat_std
        return self.net(x)


def save_head(head: RiskHead, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "meta": meta}, path)


def load_head(path: str | Path, device="cpu") -> tuple[RiskHead, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    head = RiskHead(meta["in_dim"], meta.get("hidden", 256), meta.get("dropout", 0.1))
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), meta


class RiskAwareJEPA(nn.Module):
    """Planner cost = frozen LeWM cost corrected by the learned risk head."""

    def __init__(
        self,
        base: nn.Module,
        head: RiskHead,
        use_sigma: bool,
        mode: str = "err",
        beta: float = 1.0,
        n_mc: int = 8,
        history_size: int = 3,
    ):
        super().__init__()
        if mode not in ("err", "true_cost"):
            raise ValueError(f"unknown risk mode {mode!r}")
        self.base = base
        self.head = head
        self.use_sigma = bool(use_sigma)
        self.mode = mode
        self.beta = float(beta)
        self.n_mc = int(n_mc)
        self.history_size = int(history_size)
        self.penalty_log: list[float] = []

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def diagnostics_summary(self) -> dict:
        if not self.penalty_log:
            return {}
        return {"mean_risk_term": float(sum(self.penalty_log) / len(self.penalty_log))}

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        dev = self.device
        dtype = next(self.base.parameters()).dtype
        pixels = info_dict["pixels"][:, 0].to(dev, dtype)  # (B, T0, C, H, W)
        goal = info_dict["goal"][:, 0].to(dev, dtype)
        self.base.eval()
        z_hist = self.base.encode({"pixels": pixels})["emb"]  # (B, T0, D)
        z_goal = self.base.encode({"pixels": goal})["emb"][:, -1]  # (B, D)
        acts = action_candidates.to(dev, dtype)
        det = rollout_from_latents(self.base, z_hist, acts, self.history_size)  # (B,S,P,D)
        det_cost = goal_cost(det[..., -1, :], z_goal[:, None])  # (B, S)
        sig2 = mc_std = None
        if self.use_sigma:
            mc = mc_rollouts(self.base, z_hist, acts, self.n_mc, self.history_size)
            sig2 = per_step_variance(mc)
            mc_std = goal_cost(mc[..., -1, :], z_goal[None, :, None]).std(dim=0, unbiased=True)
        B, S = det_cost.shape
        z0 = z_hist[:, -1][:, None].expand(B, S, -1)
        zg = z_goal[:, None].expand(B, S, -1)
        feats = risk_features(z0, zg, det, det_cost, sig2, mc_std).float()
        out = self.head(feats)
        if self.mode == "err":
            risk = torch.expm1(out[..., 0]).clamp_min(0)
            cost = det_cost.float() + self.beta * risk
            self.penalty_log.append(float((self.beta * risk).mean()))
        else:
            cost = torch.expm1(out[..., 1]).clamp_min(0)
            self.penalty_log.append(float((cost - det_cost.float()).mean()))
        return cost
