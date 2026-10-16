"""Log the actions a policy actually executed, for exact offline replay.

    evaluate.py ... ++output.save_rollout=true   ->  <output.dir>/rollouts/<run_id>.npz

The evaluation starts every episode from a dataset state; with these
actions an episode can be replayed exactly in a fresh simulator (same state
setup as cube/oracle.replay), e.g. to render it at high quality
(scripts/cube/viz_export.py). Saved per episode (N episodes, T = eval budget):

  episodes, start_steps   dataset episode ids and start steps
  expert_qpos/qvel        (N, G+1, nq/nv) dataset states from start to goal
  goal_pos, goal_quat     (N, 3), (N, 4) cube goal (dataset state G steps ahead)
  actions                 (N, T, A) env actions the policy returned (NaN once done)
  cube_pos                (N, T, 3) cube position the policy saw at each step
  success                 (N,) episode success as reported by the evaluation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class RolloutLogger:
    def __init__(self, policy):
        self.actions: list[np.ndarray] = []
        self.cube: list[np.ndarray] = []
        self._orig = policy.get_action
        policy.get_action = self._get_action  # instance attribute shadows the method

    def _get_action(self, info_dict, **kwargs):
        cube = _cube_pos(info_dict)
        action = self._orig(info_dict, **kwargs)
        self.actions.append(np.array(action, dtype=np.float32, copy=True))
        if cube is not None:
            self.cube.append(cube)
        return action

    def save(self, path: Path, dataset, episodes, start_steps, goal_offset: int, successes) -> Path:
        n = len(episodes)
        acts = np.stack(self.actions, axis=1).reshape(n, len(self.actions), -1)
        data = dataset.load_chunk(np.asarray(episodes), np.asarray(start_steps),
                                  np.asarray(start_steps) + int(goal_offset) + 1)
        col = lambda ep, name: np.asarray(ep[name].numpy() if hasattr(ep[name], "numpy") else ep[name])  # noqa: E731
        out = {
            "episodes": np.asarray(episodes, dtype=np.int64),
            "start_steps": np.asarray(start_steps, dtype=np.int64),
            "expert_qpos": np.stack([col(ep, "qpos") for ep in data]).astype(np.float64),
            "expert_qvel": np.stack([col(ep, "qvel") for ep in data]).astype(np.float64),
            "goal_pos": np.stack([col(ep, "privileged_block_0_pos")[-1] for ep in data]).astype(np.float64),
            "goal_quat": np.stack([col(ep, "privileged_block_0_quat")[-1] for ep in data]).astype(np.float64),
            "actions": acts,
            "success": np.asarray(successes, dtype=bool),
        }
        if len(self.cube) == len(self.actions):
            out["cube_pos"] = np.stack(self.cube, axis=1).reshape(n, len(self.cube), 3)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        return path


def _cube_pos(info_dict) -> np.ndarray | None:
    """Current cube position from the world's info dict (last history entry)."""
    for key, value in info_dict.items():
        if key.endswith("block_0_pos") and not key.startswith("goal"):
            v = np.array(value.cpu() if hasattr(value, "cpu") else value, dtype=np.float64)  # copy: the
            return v.reshape(v.shape[0], -1, 3)[:, -1]  # world updates its info buffers in place
    return None
