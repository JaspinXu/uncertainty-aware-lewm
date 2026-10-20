"""Uncertainty-aware wrapper around a pretrained LeWM ``jepa.JEPA``.

``UncertaintyAwareJEPA`` exposes the exact ``get_cost(info_dict, action_candidates)``
interface that ``stable_worldmodel`` solvers (CEM / gradient) call, so it is a
drop-in replacement for the plain LeWM cost model in the planning loop.

Difference vs. plain LeWM:
  1. it samples ``N`` latent rollouts (MC-dropout or ensemble) instead of one;
  2. it derives a per-step epistemic uncertainty sigma^2_M(z_t);
  3. it evaluates the planning cost at the *uncertainty-truncated* step k*
     (and optionally adds an uncertainty penalty), rather than at the fixed
     final horizon H.

The latent rollout math mirrors ``jepa.JEPA.rollout`` exactly; we only refactor
it to (a) run against an arbitrary member model and (b) return every predicted
step so we can measure disagreement and truncate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .truncation import TruncationConfig, compute_truncation_index, gather_at_index, terminal_charge
from .uncertainty import EnsembleEstimator, UncertaintyEstimator, rollout_statistics


class UncertaintyAwareJEPA(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        estimator: UncertaintyEstimator,
        trunc_cfg: TruncationConfig,
        history_size: int = 3,
    ):
        super().__init__()
        self.base = base
        self.estimator = estimator
        self.trunc_cfg = trunc_cfg
        self.history_size = history_size

        # register ensemble members so .to(device)/.eval() reach them
        if isinstance(estimator, EnsembleEstimator):
            self._members = nn.ModuleList(
                [m for m in estimator.members if m is not base]
            )

        # diagnostics collected across planning iterations (for logging/plots)
        self.truncation_lengths: list = []
        self.truncated_frac: list = []  # fraction of candidates scored before the full horizon
        self.mean_sigma2: list = []
        # per-candidate sigma^2 at the first predicted step (for gated calibration)
        self.sigma2_refs: list = []
        # full per-step sigma^2 rows for depth-aware threshold calibration
        self.sigma2_perstep: list = []

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_diagnostics(self) -> None:
        self.truncation_lengths.clear()
        self.truncated_frac.clear()
        self.mean_sigma2.clear()
        self.sigma2_refs.clear()
        self.sigma2_perstep.clear()

    def single_rollout(
        self,
        model: nn.Module,
        info: dict,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """One autoregressive latent rollout, returning *only* the predicted steps.

        Mirrors ``jepa.JEPA.rollout`` but (a) uses ``model`` for encode/predict so
        ensemble members can be swapped in, and (b) stacks the predicted
        embeddings into ``(B, S, P, D)`` with ``P = n_steps + 1``. Does NOT mutate
        ``info`` so it can be called repeatedly for sampling.
        """
        HS = self.history_size
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        n_steps = T - H

        # Expand raw 2D actions to match action_encoder's expected input_dim
        # (training uses frameskip-expanded actions, e.g. 5×2=10D)
        act_in_dim = model.action_encoder.patch_embed.in_channels
        raw_act_dim = act_0.shape[-1]
        if raw_act_dim != act_in_dim:
            repeat = act_in_dim // raw_act_dim
            act_0 = act_0.repeat_interleave(repeat, dim=-1)
            act_future = act_future.repeat_interleave(repeat, dim=-1)

        # Set info["action"] to act_0 so model.encode() uses it
        info["action"] = act_0

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = model.encode(_init)
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        preds = []
        for t in range(n_steps):
            act_emb = model.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)
            preds.append(pred_emb)
            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # final step (matches base rollout's trailing prediction)
        act_emb = model.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
        preds.append(pred_emb)

        preds = torch.cat(preds, dim=1)  # (BS, P, D)
        preds = rearrange(preds, "(b s) p d -> b s p d", b=B, s=S)
        return preds

    def _encode_goal(self, info_dict: dict) -> torch.Tensor:
        """Encode the goal image into its latent embedding (last frame)."""
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(goal.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.base.encode(goal)
        return goal["emb"][..., -1, :]  # (B, D)

    # ------------------------------------------------------------------ #
    # planning interface (called by the solver)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        assert "goal" in info_dict, "goal not in info_dict"

        device = self.device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        action_candidates = action_candidates.to(device)

        goal_last = self._encode_goal(info_dict)  # (B, D)

        # ---- sample N rollouts and measure disagreement ----
        self.estimator.prepare(self)
        trajs = [
            self.estimator.trajectory(self, i, info_dict, action_candidates)
            for i in range(self.estimator.n)
        ]
        self.estimator.finish(self)
        trajs = torch.stack(trajs, dim=0)  # (N, B, S, P, D)

        mean_traj, sigma2_steps = rollout_statistics(trajs)  # (B,S,P,D), (B,S,P)

        # ---- per-step distance to goal (sum of squared latent error) ----
        B, S, P, D = mean_traj.shape
        goal_b = goal_last.view(B, 1, 1, D)
        dist_steps = ((mean_traj - goal_b) ** 2).sum(dim=-1)  # (B, S, P)

        # ---- uncertainty-aware truncation ----
        idx = compute_truncation_index(sigma2_steps, self.trunc_cfg)  # (B, S)
        cost = gather_at_index(dist_steps, idx)  # (B, S)

        if self.trunc_cfg.penalty_weight:
            cost = cost + self.trunc_cfg.penalty_weight * gather_at_index(sigma2_steps, idx)
        if self.trunc_cfg.terminal_weight:
            cost = cost + terminal_charge(dist_steps, idx, self.trunc_cfg.terminal_weight)

        # ---- diagnostics ----
        self.truncation_lengths.append((idx.float() + 1).mean().item())
        self.truncated_frac.append((idx < P - 1).float().mean().item())
        self.mean_sigma2.append(sigma2_steps.mean().item())
        # store step-0 sigma^2 for every candidate (used by gated threshold calibration)
        self.sigma2_refs.append(sigma2_steps[..., 0].detach().cpu().reshape(-1))
        self.sigma2_perstep.append(
            sigma2_steps.detach().cpu().reshape(-1, sigma2_steps.shape[-1])
        )

        return cost

    # plain LeWM cost (no truncation) — handy for A/B comparison in code
    @torch.no_grad()
    def get_cost_baseline(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        return self.base.get_cost(info_dict, action_candidates)
