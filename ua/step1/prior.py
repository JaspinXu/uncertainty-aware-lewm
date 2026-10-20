"""Goal-conditioned behaviour-cloning priors on frozen LeWM latents, and the planners built on them.

pi(z_t, z_goal [, k]) predicts the expert's next 25 actions as the planner's 5
normalized action blocks (5 x 25), from the frozen-encoder latents of the
current frame and of a goal frame k steps ahead. Trained on cached latents
(ensemble.cache_latents) with hindsight relabelling: k is drawn uniformly from
``offsets`` (exactly 25 = the evaluation's goal offset by default).

Prior variants:
- point prior (MSE): the mean action chunk;
- time-conditioned prior (``k_max`` > 0): the goal offset k is an input;
- Gaussian prior (``gaussian``): a mean and a per-dimension std, trained by
  Gaussian NLL. The std is the prior's own uncertainty about the expert's
  action, which ``PriorInitCEMSolver`` uses as the width of the CEM search.

Planners:
- ``PriorSolver``: execute the prior's mean (behaviour cloning, no world model).
- ``PriorInitCEMSolver``: CEM from N(mu_p, sigma^2) at every planning call,
  with sigma = 1 (CEM's default width), a constant, or the prior's sigma_p.
- ``PriorSelectSolver``: a time-conditioned prior proposes one chunk per goal
  offset in ``ks``; the world model rolls each out and a steps-to-goal estimate
  k̂ (ua/step1/steps_to_goal.py) scores each one's time to the goal.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from stable_worldmodel.solver import CEMSolver
from torch import nn

from ua.step1.rollout import deterministic_rollouts

HORIZON, BLOCK, ACT = 5, 5, 5   # planner blocks, env steps per block, action dim
GOAL_OFFSET = HORIZON * BLOCK    # 25 env steps
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # clamp of the Gaussian prior's log std (normalized action units)


class BCPrior(nn.Module):
    """MLP on (z, g, g - z) [+ a learned embedding of the goal offset k in 1..k_max].

    Outputs the mean action chunk, plus a per-dimension log std when ``gaussian``.
    """

    def __init__(self, dim: int = 192, hidden: int = 1024, dropout: float = 0.1, k_max: int = 0,
                 gaussian: bool = False):
        super().__init__()
        self.k_max, self.gaussian = int(k_max), bool(gaussian)
        k_dim = 32 if self.k_max else 0
        self.k_emb = nn.Embedding(self.k_max + 1, k_dim) if self.k_max else None
        out = HORIZON * BLOCK * ACT
        self.net = nn.Sequential(
            nn.Linear(3 * dim + k_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out * (2 if self.gaussian else 1)),
        )

    def dist(self, z: torch.Tensor, g: torch.Tensor, k: torch.Tensor | None = None):
        """(B, D), (B, D)[, (B,) goal offsets] -> mean and std, each (B, HORIZON, BLOCK * ACT).

        The std is None for a point prior. k is required by a time-conditioned prior and ignored otherwise."""
        x = torch.cat([z, g, g - z], dim=-1)
        if self.k_max:
            x = torch.cat([x, self.k_emb(k.long().clamp(1, self.k_max))], dim=-1)
        y = self.net(x)
        if not self.gaussian:
            return y.view(-1, HORIZON, BLOCK * ACT), None
        mu, log_std = y.chunk(2, dim=-1)
        std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return mu.reshape(-1, HORIZON, BLOCK * ACT), std.reshape(-1, HORIZON, BLOCK * ACT)

    def forward(self, z: torch.Tensor, g: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """The mean action chunk (B, HORIZON, BLOCK * ACT) in normalized action units."""
        return self.dist(z, g, k)[0]


def gaussian_nll(mu: torch.Tensor, std: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Mean per-dimension Gaussian negative log-likelihood (up to the constant 0.5 log 2 pi)."""
    return (0.5 * ((a - mu) / std) ** 2 + std.log()).mean()


class PriorData:
    """(z_t, z_{t+k}, normalized actions t..t+24) triples from cached latents.

    k is drawn uniformly from ``offsets`` = (lo, hi); t uniformly where both the
    goal frame and the 25 actions lie inside the episode. A fixed draw of
    ``n_eval`` pairs serves as the evaluation set.
    """

    def __init__(self, cache: dict, episode_rows, action_mean, action_scale, device,
                 offsets=(GOAL_OFFSET, GOAL_OFFSET), n_eval: int = 50_000):
        self.device = device
        self.offsets = (int(offsets[0]), int(offsets[1]))
        self.emb = torch.as_tensor(cache["emb"], device=device).float()
        act = np.nan_to_num((cache["action"] - action_mean) / action_scale, nan=0.0).astype(np.float32)
        self.act = torch.as_tensor(act, device=device)
        self.ep_off = torch.as_tensor([int(cache["offsets"][r]) for r in episode_rows])
        self.ep_len = torch.as_tensor([int(cache["lengths"][r]) for r in episode_rows])
        assert int(self.ep_len.min()) > max(self.offsets[1], GOAL_OFFSET), "episodes too short for the offsets"
        self.steps = torch.arange(GOAL_OFFSET, device=device)
        self.eval_s, self.eval_k = self.draw(n_eval, torch.Generator().manual_seed(12345))

    def draw(self, n: int, gen: torch.Generator, k: int | None = None):
        """n (start row, offset) pairs, k uniform in ``offsets`` unless given."""
        lo, hi = self.offsets
        k = torch.randint(lo, hi + 1, (n,), generator=gen) if k is None else torch.full((n,), int(k))
        e = torch.randint(len(self.ep_off), (n,), generator=gen)
        room = self.ep_len[e] - torch.clamp(k, min=GOAL_OFFSET)  # t + k and t + 24 stay inside the episode
        t = (torch.rand(n, generator=gen) * room).long()
        return (self.ep_off[e] + t).to(self.device), k.to(self.device)

    def batch(self, s: torch.Tensor, k: torch.Tensor):
        a = self.act[s[:, None] + self.steps].reshape(len(s), HORIZON, BLOCK * ACT)
        return self.emb[s], self.emb[s + k], a


@torch.no_grad()
def eval_mse(prior: BCPrior, data: PriorData, batch: int = 4096) -> dict:
    """On the data's fixed evaluation pairs: MSE of the mean, overall and per action dimension
    (normalized units); for a Gaussian prior also the NLL and the mean predicted std."""
    prior.eval()
    sq = torch.zeros(ACT, device=data.device)
    nll, std_sum = 0.0, 0.0
    for i in range(0, len(data.eval_s), batch):
        k = data.eval_k[i : i + batch]
        z, g, a = data.batch(data.eval_s[i : i + batch], k)
        mu, std = prior.dist(z, g, k)
        sq += ((mu - a).view(len(z), -1, ACT) ** 2).sum(dim=(0, 1))
        if std is not None:
            nll += float(gaussian_nll(mu, std, a)) * len(z)
            std_sum += float(std.mean()) * len(z)
    n = len(data.eval_s)
    per_dim = (sq / (n * GOAL_OFFSET)).cpu().numpy()
    out = {"mse": float(per_dim.mean()), "mse_per_dim": per_dim.tolist()}
    if prior.gaussian:
        out.update(nll=nll / n, sigma_mean=std_sum / n)
    return out


def train_prior(train: PriorData, val: PriorData, steps: int = 8_000, batch: int = 1024, lr: float = 1e-3,
                weight_decay: float = 1e-4, seed: int = 0, log_every: int = 500,
                k_max: int = 0, gaussian: bool = False) -> tuple[BCPrior, list]:
    """MSE (point prior) or Gaussian NLL (``gaussian``), with early stopping: the returned weights
    are those with the lowest validation objective (held-out episodes; evaluation episodes are never used)."""
    torch.manual_seed(seed)
    prior = BCPrior(k_max=k_max, gaussian=gaussian).to(train.device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    key = "nll" if gaussian else "mse"
    log, t0, best = [], time.time(), (float("inf"), None, 0)
    for i in range(steps):
        prior.train()
        for gr in opt.param_groups:  # cosine decay after 500 warm-up steps
            gr["lr"] = lr * min((i + 1) / 500, 0.5 * (1 + math.cos(math.pi * i / steps)))
        s, k = train.draw(batch, gen)
        z, g, a = train.batch(s, k)
        mu, std = prior.dist(z, g, k)
        loss = gaussian_nll(mu, std, a) if gaussian else F.mse_loss(mu, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % log_every == 0 or i + 1 == steps:
            ev = eval_mse(prior, val)
            log.append({"step": i + 1, "train": loss.item(), "val": ev[key], "val_mse": ev["mse"],
                        "sec": time.time() - t0})
            if ev[key] < best[0]:
                best = (ev[key], {n: t.detach().clone() for n, t in prior.state_dict().items()}, i + 1)
            print(f"  step {i + 1}/{steps} train {loss.item():.4f} val {key} {ev[key]:.4f} "
                  f"(mse {ev['mse']:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    prior.load_state_dict(best[1])
    prior.eval()
    print(f"  kept step {best[2]} (val {key} {best[0]:.4f})", flush=True)
    return prior, log


def load_prior(path, device) -> BCPrior:
    state = torch.load(path, weights_only=False, map_location="cpu")
    prior = BCPrior(k_max=int(state.get("k_max", 0)), gaussian=bool(state.get("gaussian", False))).to(device)
    prior.load_state_dict(state["state_dict"])
    return prior.eval()


# --------------------------------------------------------------------------- #
# planning with the prior
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_history_goal(model, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents of the observation history (B, H, D) and of the goal (B, D) from the prepared planning info."""
    dev, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    z = model.encode({"pixels": info_dict["pixels"].to(dev, dtype)})["emb"]
    g = model.encode({"pixels": info_dict["goal"].to(dev, dtype)})["emb"][:, -1]
    return z.float(), g.float()


@torch.no_grad()
def encode_obs_goal(model, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents (B, D) of the current frame and of the goal from the prepared planning info."""
    z, g = encode_history_goal(model, info_dict)
    return z[:, -1], g


@torch.no_grad()
def propose(prior: BCPrior, model, info_dict: dict):
    """The prior's mean and std (std None for a point prior), each (B, HORIZON, 25), for the planning info."""
    assert not prior.k_max, "a time-conditioned prior needs k: use PriorSelectSolver"
    return prior.dist(*encode_obs_goal(model, info_dict))


class PriorInitCEMSolver(CEMSolver):
    """CEM whose initial distribution at every planning call is N(mu_p, sigma^2) (scoring unchanged).

    ``init_std``: None -> sigma = ``var_scale`` (CEM's default width, 1); "prior" -> the Gaussian
    prior's per-dimension sigma_p; a number -> that constant. CEM refits mean and std to its
    elites after the first iteration, as usual.
    """

    def __init__(self, *args, prior: BCPrior, init_std=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert init_std != "prior" or prior.gaussian, "init_std='prior' needs a Gaussian prior"
        self.prior, self.init_std = prior, init_std
        self._std = None

    def solve(self, info_dict: dict, init_action=None) -> dict:
        mu, std = propose(self.prior, self.model, info_dict)
        if self.init_std == "prior":
            self._std = std.cpu().to(self.dtype)
        elif self.init_std is not None:
            self._std = torch.full_like(mu, float(self.init_std)).cpu().to(self.dtype)
        try:
            return super().solve(info_dict, init_action=mu.cpu())
        finally:
            self._std = None

    def init_action_distrib(self, n_envs: int, actions: torch.Tensor | None = None):
        mean, var = super().init_action_distrib(n_envs, actions)
        return mean, (var if self._std is None else self._std.to(var.device))


class PriorSolver:
    """The prior as a planner: returns its mean proposal, no world-model scoring."""

    def __init__(self, prior: BCPrior, model):
        self.prior, self.model = prior, model

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._n_envs, self._config = n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def solve(self, info_dict: dict, init_action=None) -> dict:
        return {"actions": propose(self.prior, self.model, info_dict)[0].cpu()}

    def __call__(self, *args, **kwargs) -> dict:
        return self.solve(*args, **kwargs)


class PriorSelectSolver(PriorSolver):
    """Proposals of a time-conditioned prior for goal offsets ``ks``; the world model picks one per env.

    Every proposal is rolled out by the world model; its cost is its estimated time to the goal,
    min over its blocks j of (steps up to block j + k̂(predicted latent after block j, goal)),
    a learned cost-to-go instead of the myopic latent distance. ``chosen_k`` records the selected
    offset of every planning call.
    """

    def __init__(self, prior: BCPrior, model, khat, ks=(5, 10, 15, 25, 35, 50, 75, 100)):
        assert prior.k_max, "PriorSelectSolver needs a time-conditioned prior"
        super().__init__(prior, model)
        self.khat, self.ks = khat, [int(k) for k in ks]
        self.chosen_k: list[int] = []

    @torch.no_grad()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        zh, g = encode_history_goal(self.model, info_dict)
        n, S, H = len(zh), len(self.ks), zh.shape[1]
        k = torch.as_tensor(self.ks, device=zh.device).repeat(n)
        cand = self.prior(zh[:, -1].repeat_interleave(S, 0), g.repeat_interleave(S, 0), k).view(n, S, HORIZON, -1)
        pred = deterministic_rollouts(self.model, zh.repeat_interleave(S, 0), cand.flatten(0, 1))  # (n S, P, D)
        P = pred.shape[1]  # the states after blocks H .. HORIZON
        togo = self.khat(pred.flatten(0, 1), g.repeat_interleave(S * P, 0)).clamp(min=0).view(n, S, P)
        best = (togo + BLOCK * torch.arange(H, H + P, device=togo.device)).min(dim=-1).values.argmin(dim=1)
        self.chosen_k += [self.ks[j] for j in best.tolist()]
        return {"actions": cand[torch.arange(n, device=cand.device), best].cpu()}
