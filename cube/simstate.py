"""Simulator state that qpos/qvel alone do not capture (exact oracle replays).

Restoring qpos/qvel reproduces the evaluation exactly only at the start of an
episode. In the middle of one, two more pieces of MuJoCo state decide what the
next env step does:

* ``qacc_warmstart`` -- the constraint solver starts from the previous
  accelerations; with the gripper and the cube in contact a different warm
  start gives a slightly different solution, which grows over 25 steps.
* the end-effector ("pinch" site) pose. ``ManipSpaceEnv.set_control`` turns the
  relative action into a target pose from ``data.site_xpos`` / ``site_xmat``,
  and after ``mj_step`` those are one physics substep old (MuJoCo computes them
  before integrating). ``set_state`` calls ``mj_forward`` and refreshes them,
  so a replay would start from a slightly different target.

Measured on expert-like trajectories (12 episodes, replay of 25 steps from
step 25 or 50): restoring qpos/qvel only gives a median max-|qpos| error of
0.15 (up to 1.3); restoring both pieces below makes every replay bit-exact.

``SimStateInfo`` adds both to every info dict of the evaluation env (the
recorder stores them with qpos/qvel); ``restore_sim_state`` puts them back
before an oracle replay.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

SIM_KEYS = ("sim_qacc_warmstart", "sim_pinch_xpos", "sim_pinch_xmat")


def sim_state(env) -> dict:
    """Current extra state of an (unwrapped) ManipSpace env; {} for other envs."""
    data = getattr(env, "_data", None)
    if data is None:
        return {}
    out = {"sim_qacc_warmstart": np.array(data.qacc_warmstart, dtype=np.float64)}
    site = getattr(env, "_pinch_site_id", None)
    if site is not None:
        out["sim_pinch_xpos"] = np.array(data.site_xpos[site], dtype=np.float64)
        out["sim_pinch_xmat"] = np.array(data.site_xmat[site], dtype=np.float64)
    return out


class SimStateInfo(gym.Wrapper):
    """Adds :func:`sim_state` of the raw env to the info of every reset/step."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.update(sim_state(self.env.unwrapped))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(sim_state(self.env.unwrapped))
        return obs, reward, terminated, truncated, info


def restore_sim_state(env, state: dict, mid_episode: bool) -> None:
    """Call after ``env.set_state(qpos, qvel)`` and ``env.set_target_pos(...)``.

    The warm start is always restored. The stale end-effector pose only applies
    in the middle of an episode: at an episode start the evaluation itself had
    just called ``set_state`` (fresh kinematics), exactly like the replay.
    """
    data = env._data
    ws = state.get("sim_qacc_warmstart")
    if ws is not None:
        data.qacc_warmstart[:] = np.asarray(ws, dtype=np.float64).reshape(-1)
    if mid_episode and state.get("sim_pinch_xpos") is not None:
        site = env._pinch_site_id
        data.site_xpos[site] = np.asarray(state["sim_pinch_xpos"], dtype=np.float64).reshape(3)
        data.site_xmat[site] = np.asarray(state["sim_pinch_xmat"], dtype=np.float64).reshape(9)
