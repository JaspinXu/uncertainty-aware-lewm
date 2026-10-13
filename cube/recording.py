"""Log every CEM planning call of an evaluation run (L1 signal test).

Recording never changes what the planner does: the solver and policy below
only *observe* the final CEM population and re-score it afterwards.

For each planning call and each re-planning env the recorder stores

* the simulator state at planning time (qpos/qvel) and the goal target, so the
  oracle can replay any candidate from exactly that state;
* the executed plan (final CEM mean) and the final CEM population with the
  solver's own costs;
* for the executed plan + the whole population (``pool``): deterministic cost,
  MC-dropout mean cost, MC cost spread and per-step sigma^2 (N MC masks);
* a candidate subset for oracle replay -- the executed plan, the ``n_elite``
  best candidates of the final population and ``n_random`` others -- with their
  action sequences and deterministic predicted latents.

Pool index 0 is always the executed plan; pool index ``1 + i`` is final-population
candidate ``i``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.callbacks import Callback

from .rollout import (
    deterministic_rollout,
    encode_pixels,
    goal_cost,
    mc_rollouts,
    per_step_variance,
)

STATE_KEYS = ("qpos", "qvel")
GOAL_PREFIX = "goal_privileged"


class FinalPopulation(Callback):
    """Keep the candidates/costs of the last CEM iteration (per solver batch)."""

    name = "final_population"

    def __init__(self, n_steps: int):
        super().__init__(reduction="none")
        self.n_steps = int(n_steps)

    def compute(self, **state: Any):
        if int(state["step"]) != self.n_steps - 1:
            return None
        return {
            "candidates": state["candidates"].detach().float().cpu(),
            "costs": state["costs"].detach().float().cpu(),
        }


class RecordingCEMSolver(CEMSolver):
    """``CEMSolver`` that hands its final population to a ``PlanRecorder``."""

    def __init__(self, *args, recorder: "PlanRecorder | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self._final = FinalPopulation(self.n_steps)
        self.callbacks.append(self._final)

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        outputs = super().solve(info_dict, init_action=init_action)
        if self.recorder is not None:
            batches = outputs["callbacks"][self._final.output_key]
            recs = [b[-1] for b in batches if b]
            candidates = torch.cat([r["candidates"] for r in recs], dim=0)
            costs = torch.cat([r["costs"] for r in recs], dim=0)
            self.recorder.on_solve(info_dict, outputs["actions"], candidates, costs)
        return outputs


class RecordingPolicy(WorldModelPolicy):
    """``WorldModelPolicy`` that tells the recorder which envs re-plan and when."""

    def __init__(self, *args, recorder: "PlanRecorder | None" = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = recorder
        self._t = 0

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self.recorder is not None:
            n_envs = self.env.num_envs
            term = info_dict.get("terminated")
            dead = (
                np.asarray(term, dtype=bool).reshape(n_envs, -1)[:, 0]
                if term is not None
                else np.zeros(n_envs, dtype=bool)
            )
            replan = [
                i for i in range(n_envs) if len(self._action_buffer[i]) == 0 and not dead[i]
            ]
            if replan:
                self.recorder.begin_plan(replan, self._t, info_dict)
        action = super().get_action(info_dict, **kwargs)
        self._t += 1
        return action


class PlanRecorder:
    """Re-scores each final CEM population and accumulates per-plan records."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        history_size: int = 3,
        n_mc: int = 8,
        n_elite: int = 30,
        n_random: int = 20,
        seed: int = 0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.history_size = int(history_size)
        self.n_mc = int(n_mc)
        self.n_elite = int(n_elite)
        self.n_random = int(n_random)
        self.rng = np.random.default_rng(seed)
        self.plans: list[dict] = []
        self._pending: dict | None = None
        self._plan_count: dict[int, int] = {}
        self.max_cost_mismatch = 0.0

    # ------------------------------------------------------------------ #
    def begin_plan(self, env_ids: list[int], t: int, raw_info: dict) -> None:
        state = {}
        for key in raw_info:
            if key in STATE_KEYS or key.startswith(GOAL_PREFIX):
                arr = np.asarray(raw_info[key])
                state[key] = arr[env_ids].reshape(len(env_ids), *arr.shape[2:]).copy()
        self._pending = {"env_ids": list(env_ids), "t": int(t), "state": state}

    @torch.no_grad()
    def on_solve(
        self,
        info_dict: dict,
        executed: torch.Tensor,
        final_candidates: torch.Tensor,
        final_costs: torch.Tensor,
    ) -> None:
        pend = self._pending
        if pend is None:
            raise RuntimeError("PlanRecorder.on_solve called without begin_plan")
        self._pending = None
        ids = pend["env_ids"]
        n = len(ids)
        if executed.shape[0] != n or final_candidates.shape[0] != n:
            raise RuntimeError(
                f"recorder expected {n} envs, solver returned {executed.shape[0]}"
            )
        dev = self.device
        dtype = next(self.model.parameters()).dtype
        pixels = info_dict["pixels"].to(dev, dtype)
        goal = info_dict["goal"].to(dev, dtype)
        z_hist = encode_pixels(self.model, pixels)  # (n, T0, D)
        z_goal = encode_pixels(self.model, goal)[:, -1]  # (n, D)

        pool = torch.cat([executed[:, None].float(), final_candidates.float()], dim=1).to(dev, dtype)
        det = deterministic_rollout(self.model, z_hist, pool, self.history_size)  # (n,1+S,P,D)
        det_cost = goal_cost(det[..., -1, :], z_goal[:, None])  # (n, 1+S)
        mc = mc_rollouts(self.model, z_hist, pool, self.n_mc, self.history_size)  # (N,n,1+S,P,D)
        sigma2 = per_step_variance(mc)  # (n, 1+S, P)
        mc_cost = goal_cost(mc.mean(0)[..., -1, :], z_goal[:, None])
        mc_each = goal_cost(mc[..., -1, :], z_goal[None, :, None])  # (N, n, 1+S)
        mc_cost_std = mc_each.std(dim=0, unbiased=True) if self.n_mc > 1 else torch.zeros_like(mc_cost)

        mismatch = (det_cost[:, 1:].float().cpu() - final_costs).abs().max().item()
        self.max_cost_mismatch = max(self.max_cost_mismatch, mismatch)

        S = final_candidates.shape[1]
        for j, env in enumerate(ids):
            order = torch.argsort(final_costs[j]).numpy()
            elite = order[: self.n_elite]
            rest = order[self.n_elite :]
            k = min(self.n_random, len(rest))
            rnd = self.rng.choice(rest, size=k, replace=False) if k else np.array([], int)
            subset = np.concatenate([[0], elite + 1, np.sort(rnd) + 1]).astype(np.int64)
            rank = np.full(1 + S, -1, dtype=np.int64)
            rank[order + 1] = np.arange(S)
            plan_idx = self._plan_count.get(env, 0)
            self._plan_count[env] = plan_idx + 1
            rec = {
                "env": int(env),
                "t": pend["t"],
                "plan_idx": plan_idx,
                "state": {k_: v[j] for k_, v in pend["state"].items()},
                "solver_cost": final_costs[j].numpy().astype(np.float32),
                "solver_rank": rank,  # rank in the solver's final ranking (-1 = executed)
                "det_cost": det_cost[j].float().cpu().numpy(),
                "mc_cost": mc_cost[j].float().cpu().numpy(),
                "mc_cost_std": mc_cost_std[j].float().cpu().numpy(),
                "sigma2": sigma2[j].float().cpu().numpy(),  # (1+S, P)
                "subset": subset,
                "subset_actions": pool[j, subset].float().cpu().numpy(),  # (K, H, A)
                "subset_pred": det[j, subset].half().cpu().numpy(),  # (K, P, D)
                "z_hist": z_hist[j].float().cpu().numpy(),
                "z_goal": z_goal[j].float().cpu().numpy(),
            }
            self.plans.append(rec)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, meta: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(meta)
        meta["max_cost_mismatch"] = float(self.max_cost_mismatch)
        meta["n_mc"] = self.n_mc
        meta["n_elite"] = self.n_elite
        meta["n_random"] = self.n_random
        torch.save({"meta": meta, "plans": self.plans}, path)
        return path
