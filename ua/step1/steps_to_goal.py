"""Steps-to-goal estimate k̂(z, g) on frozen LeWM latents.

A hindsight temporal distance: for latents z_t and z_{t+k} of the same expert
episode, regress k. With MSE, k̂(z, g) is the expected number of expert steps
from z to the goal latent g; beyond ~25-50 steps that number is not
identifiable from (z, g) (the expert's detours are random), so the MSE estimate
collapses towards the mean. A low expectile (``expectile`` = 0.1, as in
IQL/HIQL) instead estimates the shortest observed time, an optimistic distance
that hindsight pairs only bound from above. Applied to the world model's
predicted latents it gives a cost-to-go for choosing among a time-conditioned
prior's proposals (prior.PriorSelectSolver), in place of the myopic latent
distance. Trained on the same cached latents and episodes as the priors
(prior.PriorData); see scripts/cube_train.py.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from ua.step1.prior import PriorData

SCALE = 100.0  # steps; the network predicts k / SCALE


class StepsToGoal(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """(B, D), (B, D) -> (B,) estimated expert steps from z to g."""
        return self.net(torch.cat([z, g, g - z], dim=-1)).squeeze(-1) * SCALE


def objective(pred: torch.Tensor, k: torch.Tensor, expectile: float | None) -> torch.Tensor:
    """MSE (expectile None) or expectile loss on k / SCALE."""
    u = k.float() / SCALE - pred / SCALE
    if expectile is None:
        return (u ** 2).mean()
    return (torch.where(u < 0, 1 - expectile, expectile) * u ** 2).mean()


@torch.no_grad()
def evaluate(model: StepsToGoal, data: PriorData, batch: int = 4096, expectile: float | None = None,
             buckets=((0, 10), (10, 25), (25, 50), (50, 100), (100, 151))) -> dict:
    """On the data's fixed evaluation pairs: the training objective, MAE overall and by true k, Spearman rho."""
    model.eval()
    pred, true = [], []
    for i in range(0, len(data.eval_s), batch):
        k = data.eval_k[i : i + batch]
        z, g, _ = data.batch(data.eval_s[i : i + batch], k)
        pred.append(model(z, g).cpu())
        true.append(k.cpu())
    obj = float(objective(torch.cat(pred), torch.cat(true), expectile))
    pred, true = torch.cat(pred).numpy(), torch.cat(true).numpy().astype(np.float64)
    err = np.abs(pred - true)
    return {
        "objective": obj, "mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": float(spearmanr(pred, true).statistic),
        "mae_by_k": {f"{lo}-{hi - 1}": float(err[(true >= lo) & (true < hi)].mean()) for lo, hi in buckets},
        "mean_pred_by_k": {f"{lo}-{hi - 1}": float(pred[(true >= lo) & (true < hi)].mean()) for lo, hi in buckets},
    }


def train_steps_to_goal(train: PriorData, val: PriorData, steps: int = 20_000, batch: int = 1024,
                        lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0, log_every: int = 1000,
                        expectile: float | None = None) -> tuple[StepsToGoal, list]:
    """MSE or expectile loss on k / SCALE; early stopping on the validation objective."""
    torch.manual_seed(seed)
    model = StepsToGoal().to(train.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    log, t0, best = [], time.time(), (float("inf"), None, 0)
    for i in range(steps):
        model.train()
        for gr in opt.param_groups:  # cosine decay after 500 warm-up steps
            gr["lr"] = lr * min((i + 1) / 500, 0.5 * (1 + math.cos(math.pi * i / steps)))
        s, k = train.draw(batch, gen)
        z, g, _ = train.batch(s, k)
        loss = objective(model(z, g), k, expectile)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % log_every == 0 or i + 1 == steps:
            ev = evaluate(model, val, expectile=expectile)
            v = ev["objective"]
            log.append({"step": i + 1, "train": loss.item(), "val": v, "val_mae": ev["mae"], "sec": time.time() - t0})
            if v < best[0]:
                best = (v, {n: t.detach().clone() for n, t in model.state_dict().items()}, i + 1)
            print(f"  step {i + 1}/{steps} train {loss.item():.4f} val {v:.4f} (MAE {ev['mae']:.1f} steps, "
                  f"Spearman {ev['spearman']:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best[1])
    model.eval()
    print(f"  kept step {best[2]} (val objective {best[0]:.4f})", flush=True)
    return model, log


def load_steps_to_goal(path, device) -> StepsToGoal:
    model = StepsToGoal().to(device)
    model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu")["state_dict"])
    return model.eval()
