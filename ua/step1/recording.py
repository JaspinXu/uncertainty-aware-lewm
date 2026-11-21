"""Planning-time logging for the Step-1 diagnostics.

``LoggingPolicy`` (a ``WorldModelPolicy``) and ``LoggingCEMSolver`` (a
``CEMSolver``) leave planning untouched and record, at every replan and for
every replanning env:

- the full MuJoCo state (so any action sequence can be replayed exactly),
- the cube and goal positions,
- frozen-encoder embeddings of the observation and the goal,
- the CEM populations of the selected iterations with their costs,
- the executed plan (the final elite mean).

Per-step cube positions are read straight from the simulator, because the
World's ``infos`` hold stale privileged keys at t=0 (``set_state`` runs after
the reset that filled them).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.callbacks import Callback

SNAPSHOT_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat")


# --------------------------------------------------------------------------- #
# simulator state
# --------------------------------------------------------------------------- #
def snapshot(env) -> dict:
    """Copy everything ``mj_step`` reads from ``env._data`` (unwrapped env)."""
    data = env._data
    snap = {k: np.array(getattr(data, k), copy=True) for k in SNAPSHOT_FIELDS}
    snap["time"] = float(data.time)
    return snap


def restore(env, snap: dict) -> None:
    data = env._data
    for k in SNAPSHOT_FIELDS:
        getattr(data, k)[:] = snap[k]
    data.time = snap["time"]
    mujoco.mj_forward(env._model, data)


def cube_pos(env) -> np.ndarray:
    return np.array(env._data.joint("object_joint_0").qpos[:3], copy=True)


def goal_pos(env) -> np.ndarray:
    return np.array(env._data.mocap_pos[env._cube_target_mocap_ids[0]], copy=True)


# --------------------------------------------------------------------------- #
# CEM
# --------------------------------------------------------------------------- #
class PopulationRecorder(Callback):
    """Keep the CEM candidates and costs of the selected iterations."""

    def __init__(self, steps):
        super().__init__(reduction="none")
        self.steps = set(int(s) for s in steps)

    def compute(self, **state):
        if state["step"] not in self.steps:
            return None
        return {
            "step": int(state["step"]),
            "candidates": state["candidates"].detach().float().cpu().clone(),
            "costs": state["costs"].detach().float().cpu().clone(),
        }


class LoggingCEMSolver(CEMSolver):
    """CEM that stores one record per (replan, env).

    ``LoggingPolicy`` sets ``pending`` (replan time, env indices, snapshots)
    right before the policy calls the solver; ``solve`` pairs it with the
    populations of that call. Requires ``batch_size=1`` so that recorder
    batches map one-to-one to envs.
    """

    def __init__(self, *args, record_steps=(0,), **kwargs):
        self.recorder = PopulationRecorder(record_steps)
        callbacks = list(kwargs.pop("callbacks", None) or []) + [self.recorder]
        super().__init__(*args, callbacks=callbacks, **kwargs)
        if self.batch_size != 1:
            raise ValueError("LoggingCEMSolver needs batch_size=1")
        self.pending: dict | None = None
        self.records: list[dict] = []

    @torch.inference_mode()
    def _encode(self, info_dict: dict):
        dtype = next(self.model.parameters()).dtype
        pix = info_dict["pixels"].to(self.device, dtype)
        goal = info_dict["goal"].to(self.device, dtype)
        emb = self.model.encode({"pixels": pix})["emb"]  # (B, H, D)
        goal_emb = self.model.encode({"pixels": goal})["emb"][:, -1]  # (B, D)
        return emb.float().cpu(), goal_emb.float().cpu()

    def solve(self, info_dict: dict, init_action=None) -> dict:
        pending, self.pending = self.pending, None
        if pending is None:
            raise RuntimeError("LoggingCEMSolver.solve called without LoggingPolicy")
        emb, goal_emb = self._encode(info_dict)
        out = super().solve(info_dict, init_action=init_action)
        history = self.recorder.history
        actions = out["actions"]
        n = len(pending["env_idx"])
        if not (n == actions.shape[0] == len(history) == emb.shape[0]):
            raise RuntimeError(
                f"replan bookkeeping mismatch: envs={n} actions={actions.shape[0]} "
                f"batches={len(history)} emb={emb.shape[0]}"
            )
        for row, env_i in enumerate(pending["env_idx"]):
            rec = {
                "t": pending["t"],
                "env": int(env_i),
                "snapshot": pending["snapshots"][row],
                "cube_pos": pending["cube_pos"][row],
                "goal_pos": pending["goal_pos"][row],
                "emb": emb[row],
                "goal_emb": goal_emb[row],
                "plan": actions[row].float().clone(),
                "elite_cost": float(out["costs"][row]),
            }
            for item in history[row]:
                rec[f"cand_{item['step']}"] = item["candidates"][0]
                rec[f"cost_{item['step']}"] = item["costs"][0]
            self.records.append(rec)
        return out


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class LoggingPolicy(WorldModelPolicy):
    """``WorldModelPolicy`` that snapshots replanning envs and tracks the cube.

    Use one instance per evaluation: action buffers are not cleared between
    ``World.evaluate`` calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.solver, LoggingCEMSolver):
            raise TypeError("LoggingPolicy needs a LoggingCEMSolver")
        self.t = 0
        self.cube_traj: list[np.ndarray] = []  # one (n_envs, 3) array per step
        self.dead_traj: list[np.ndarray] = []

    def _unwrapped(self):
        return [self.env.envs[i].unwrapped for i in range(self.env.num_envs)]

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        n = self.env.num_envs
        envs = self._unwrapped()
        term = info_dict.get("terminated")
        dead = (
            np.zeros(n, dtype=bool)
            if term is None
            else np.asarray(term, dtype=bool).reshape(n, -1).any(axis=1)
        )
        self.cube_traj.append(np.stack([cube_pos(e) for e in envs]))
        self.dead_traj.append(dead.copy())

        # same rule as WorldModelPolicy.get_action
        pending = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if pending:
            self.solver.pending = {
                "t": self.t,
                "env_idx": pending,
                "snapshots": [snapshot(envs[i]) for i in pending],
                "cube_pos": [cube_pos(envs[i]) for i in pending],
                "goal_pos": [goal_pos(envs[i]) for i in pending],
            }
        action = super().get_action(info_dict, **kwargs)
        self.t += 1
        return action

    def finish(self) -> dict:
        """Close the log after ``World.evaluate`` returned."""
        envs = self._unwrapped()
        self.cube_traj.append(np.stack([cube_pos(e) for e in envs]))
        return {
            "cube_traj": np.stack(self.cube_traj, axis=1),  # (n_envs, T+1, 3)
            "dead_traj": np.stack(self.dead_traj, axis=1),  # (n_envs, T)
            "goal_pos": np.stack([goal_pos(e) for e in envs]),
            "records": self.solver.records,
        }
