"""Execute logged action sequences from saved MuJoCo states.

A ``ReplayEnv`` is a one-env ``World`` built from the evaluation's ``world``
config, so physics, cameras and the pixel pipeline (``AddPixelsWrapper``) are
those of the closed-loop run. Replays step the unwrapped env (no per-step
render) and only render at the requested steps.

``replay_many`` fans jobs out to worker processes (one ReplayEnv each).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Iterable, Iterator, Sequence

import numpy as np

from .recording import cube_pos, restore


class ReplayEnv:
    def __init__(self, world_cfg: dict):
        import stable_worldmodel as swm
        from stable_worldmodel.wrapper.default import AddPixelsWrapper

        cfg = dict(world_cfg)
        cfg["num_envs"] = 1
        cfg.setdefault("max_episode_steps", 100)
        self.world = swm.World(**cfg, image_shape=(224, 224))
        self.world.reset(seed=0)
        env = self.world.envs.envs[0]
        self.u = env.unwrapped
        self._pixels = None
        e = env
        while hasattr(e, "env"):
            if isinstance(e, AddPixelsWrapper):
                self._pixels = e
                break
            e = e.env
        if self._pixels is None:
            raise RuntimeError("AddPixelsWrapper not found in the env chain")

    def render(self) -> np.ndarray:
        pixels, _ = self._pixels._get_pixels()
        return np.asarray(pixels["pixels"], dtype=np.uint8)

    def run(self, snap: dict, actions: np.ndarray, render_steps: Sequence[int]):
        """Replay ``actions`` (T, action_dim) in env units from ``snap``.

        Returns cube positions after every step ``(T, 3)`` and frames rendered
        after the steps in ``render_steps`` (1-based), ``(len, 224, 224, 3)``.
        """
        restore(self.u, snap)
        render_steps = set(int(s) for s in render_steps)
        pos = np.empty((len(actions), 3))
        frames = []
        for k, a in enumerate(actions, start=1):
            self.u.step(np.asarray(a))
            pos[k - 1] = cube_pos(self.u)
            if k in render_steps:
                frames.append(self.render())
        return pos, np.stack(frames) if frames else np.empty((0, 224, 224, 3), np.uint8)


# --------------------------------------------------------------------------- #
# worker pool
# --------------------------------------------------------------------------- #
_ENV: ReplayEnv | None = None


def _init_worker(world_cfg: dict, mujoco_gl: str) -> None:
    os.environ["MUJOCO_GL"] = mujoco_gl
    global _ENV
    _ENV = ReplayEnv(world_cfg)


def _run_job(job: dict) -> dict:
    """job: {key, snapshot, actions (K, T, A), render_steps}."""
    actions = job["actions"]
    K, T = actions.shape[:2]
    pos = np.empty((K, T, 3))
    frames = np.empty((K, len(job["render_steps"]), 224, 224, 3), np.uint8)
    for i in range(K):
        pos[i], frames[i] = _ENV.run(job["snapshot"], actions[i], job["render_steps"])
    return {"key": job["key"], "pos": pos, "frames": frames}


def replay_many(
    jobs: Iterable[dict], world_cfg: dict, workers: int
) -> Iterator[dict]:
    """Yield ``_run_job`` results (unordered) for every job."""
    mujoco_gl = os.environ.get("MUJOCO_GL", "egl")
    if workers <= 1:
        _init_worker(world_cfg, mujoco_gl)
        for job in jobs:
            yield _run_job(job)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(world_cfg, mujoco_gl)) as pool:
        yield from pool.imap_unordered(_run_job, jobs)
