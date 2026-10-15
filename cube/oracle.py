"""Ground-truth replay of planned action sequences in the Cube simulator.

For a recorded planning call we restore the exact simulator state (qpos/qvel,
solver warm start and end-effector pose -- see cube/simstate.py -- plus the goal
target) and execute each candidate action sequence open loop, exactly as
the evaluation would have executed it (normalised CEM actions -> per-step env
actions through the dataset action scaler). We log the cube-to-target distance
after every env step, the first step at which the task is solved, and render
the frame at the end of every action block, which the LeWM encoder turns into
the *true* latent the model tried to predict.

Replays are independent, so plans are distributed over worker processes,
each with its own MuJoCo env (EGL rendering).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

CUBE_ENV_ID = "swm/OGBCube-v0"
DEFAULT_ENV_KWARGS = dict(
    env_type="single",
    ob_type="states",
    multiview=False,
    width=224,
    height=224,
    visualize_info=False,
    terminate_at_goal=True,
)
SUCCESS_RADIUS = 0.04  # CubeEnv._compute_successes


def make_cube_env(env_kwargs: dict | None = None, seed: int = 0):
    """Bare CubeEnv (no World wrappers) reset once so that buffers exist."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/* envs)

    kwargs = dict(DEFAULT_ENV_KWARGS)
    kwargs.update(env_kwargs or {})
    kwargs.setdefault("render_mode", "rgb_array")
    env = gym.make(CUBE_ENV_ID, max_episode_steps=10_000, **kwargs).unwrapped
    env.reset(seed=seed)
    return env


def cube_qpos_slice(env) -> slice:
    adr = int(env._model.joint("object_joint_0").qposadr[0])
    return slice(adr, adr + 3)


def render_like_world(env, image_shape=(224, 224)) -> np.ndarray:
    """Same pixels as World/AddPixelsWrapper: render_multiview + PIL bilinear resize."""
    from PIL import Image

    img = env.render_multiview()
    if isinstance(img, dict):  # multiview disabled for Cube eval; keep first view
        img = next(iter(img.values()))
    h, w = image_shape
    pil = Image.fromarray(np.asarray(img)).resize((w, h), Image.BILINEAR)
    return np.asarray(pil)


def normalized_to_env_actions(actions: np.ndarray, action_block: int, mean, scale) -> np.ndarray:
    """(K, H, block*A) normalised CEM actions -> (K, H*block, A) env actions.

    Bit-identical to WorldModelPolicy, which hands the env float32 actions from
    ``StandardScaler.inverse_transform`` of the float32 plan: a float64
    re-implementation differs by ~1e-7, which the contact dynamics can amplify
    over 25 steps.
    """
    from sklearn.preprocessing import StandardScaler

    actions = np.asarray(actions, dtype=np.float32)
    K, H, BA = actions.shape
    A = BA // int(action_block)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = A
    scaler.n_samples_seen_ = 1
    per_step = scaler.inverse_transform(actions.reshape(K * H * int(action_block), A))
    return per_step.reshape(K, H * int(action_block), A)


@dataclass
class ReplayResult:
    cube_dist: np.ndarray  # (K, T) cube-target distance after each env step
    success_step: np.ndarray  # (K,) first env step (1-based) with success, 0 = never
    final_qpos: np.ndarray  # (K, nq)
    frames: np.ndarray | None = None  # (K, n_blocks, H, W, 3) uint8
    extra: dict = field(default_factory=dict)


def replay(
    env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray | None,
    env_actions: np.ndarray,
    render_every: int = 5,
    image_shape=(224, 224),
    stop_at_success: bool = False,
    sim: dict | None = None,
    mid_episode: bool = False,
) -> ReplayResult:
    """Execute every candidate (open loop) from the same start state.

    ``sim`` holds the extra state recorded with qpos/qvel (cube/simstate.py);
    ``mid_episode`` says whether the plan was made after the first env step.
    """
    from .simstate import restore_sim_state

    K, T, _ = env_actions.shape
    sl = cube_qpos_slice(env)
    goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    n_render = T // render_every if render_every else 0
    dist = np.full((K, T), np.nan)
    succ = np.zeros(K, dtype=np.int64)
    final_qpos = np.zeros((K, env._model.nq))
    frames = np.zeros((K, n_render, *image_shape, 3), dtype=np.uint8) if n_render else None
    for k in range(K):
        env._reset_next_step = False
        env.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
        env.set_target_pos(0, goal_pos, None if goal_quat is None else np.asarray(goal_quat, np.float64))
        env.pre_step()
        env.post_step()
        if sim:
            restore_sim_state(env, sim, mid_episode)
        for t in range(T):
            env.step(env_actions[k, t])
            cube = env._data.qpos[sl]
            dist[k, t] = float(np.linalg.norm(cube - goal_pos))
            if succ[k] == 0 and bool(env._success):
                succ[k] = t + 1
            if n_render and (t + 1) % render_every == 0:
                frames[k, (t + 1) // render_every - 1] = render_like_world(env, image_shape)
            if stop_at_success and succ[k]:
                break
        final_qpos[k] = env._data.qpos.copy()
    return ReplayResult(cube_dist=dist, success_step=succ, final_qpos=final_qpos, frames=frames)


# ---------------------------------------------------------------------- #
# multiprocessing
# ---------------------------------------------------------------------- #
_ENV = None


def _init_worker(env_kwargs):
    global _ENV
    _ENV = make_cube_env(env_kwargs)


def _work(job: dict) -> dict:
    res = replay(
        _ENV,
        job["qpos"],
        job["qvel"],
        job["goal_pos"],
        job.get("goal_quat"),
        job["env_actions"],
        render_every=job.get("render_every", 5),
        image_shape=tuple(job.get("image_shape", (224, 224))),
        sim=job.get("sim"),
        mid_episode=bool(job.get("mid_episode", False)),
    )
    sl = cube_qpos_slice(_ENV)
    return {
        "key": job["key"],
        "cube_dist": res.cube_dist.astype(np.float32),
        "success_step": res.success_step,
        "final_qpos": res.final_qpos,
        "frames": res.frames,
        "start_cube_dist": float(np.linalg.norm(np.asarray(job["qpos"])[sl] - np.asarray(job["goal_pos"]).reshape(3))),
    }


def run_jobs(jobs: Iterable[dict], workers: int, env_kwargs: dict | None = None):
    """Yield results as they finish (unordered); ``workers<=1`` runs inline."""
    jobs = list(jobs)
    if workers <= 1:
        _init_worker(env_kwargs)
        for job in jobs:
            yield _work(job)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(env_kwargs,)) as pool:
        for out in pool.imap_unordered(_work, jobs, chunksize=1):
            yield out
