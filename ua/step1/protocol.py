"""Non-trivial Cube start states.

With the goal 25 expert steps ahead, ~38% of dataset start states already have
the cube within the 4 cm success radius of the goal, so every planner "solves"
them at t=0. Step 1 only samples start states that need the cube to move.
"""

from __future__ import annotations

import numpy as np

SUCCESS_RADIUS = 0.04  # m, cube_env._compute_successes
CUBE_POS_COL = "privileged_block_0_pos"


def episode_lengths(ep_idx: np.ndarray, step_idx: np.ndarray) -> np.ndarray:
    """Length of every episode id (``max(step_idx) + 1``), indexed by id."""
    lengths = np.zeros(int(ep_idx.max()) + 1, dtype=np.int64)
    np.maximum.at(lengths, ep_idx, step_idx + 1)
    return lengths


def nontrivial_rows(dataset, goal_offset: int, min_init_dist: float = SUCCESS_RADIUS, exclude_episodes=None):
    """Rows that leave room for the goal and start > ``min_init_dist`` from it.

    Returns ``(rows, init_dist)``: dataset row indices and the cube-goal
    distance (m) at each of them. Rows of ``exclude_episodes`` (episode ids,
    e.g. those a learned prior was trained on) are left out.
    """
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_idx = np.asarray(dataset.get_col_data(col)).astype(np.int64)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).astype(np.int64)
    lengths = episode_lengths(ep_idx, step_idx)
    valid = np.nonzero(step_idx <= lengths[ep_idx] - goal_offset - 1)[0]
    if exclude_episodes is not None:
        valid = valid[~np.isin(ep_idx[valid], np.asarray(list(exclude_episodes), dtype=np.int64))]

    # rows are stored episode-major and in step order: row + offset is the goal
    goal = valid + goal_offset
    assert np.all(ep_idx[goal] == ep_idx[valid])
    assert np.all(step_idx[goal] == step_idx[valid] + goal_offset)

    pos = np.asarray(dataset.get_col_data(CUBE_POS_COL), dtype=np.float64)
    dist = np.linalg.norm(pos[goal] - pos[valid], axis=1)
    keep = dist > min_init_dist
    return valid[keep], dist[keep]


def sample_nontrivial_start_states(
    dataset,
    num_eval: int,
    goal_offset: int,
    seed: int,
    min_init_dist: float = SUCCESS_RADIUS,
    exclude_episodes=None,
):
    """Sample ``num_eval`` non-trivial (episode, start step) pairs.

    Returns ``(episodes, start_steps, rows, init_dist)``, sorted by row so the
    HDF5 reads stay monotone (same convention as ``evaluate.sample_start_states``).
    """
    rows, dist = nontrivial_rows(dataset, goal_offset, min_init_dist, exclude_episodes)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(rows), size=num_eval, replace=False))
    picked_rows = rows[pick]
    data = dataset.get_row_data(picked_rows)
    col = "episode_idx" if "episode_idx" in data else "ep_idx"
    return (
        np.asarray(data[col]).astype(int).tolist(),
        np.asarray(data["step_idx"]).astype(int).tolist(),
        picked_rows,
        dist[pick],
    )
