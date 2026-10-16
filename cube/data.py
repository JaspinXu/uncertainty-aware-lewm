"""Dataset helpers shared by evaluate.py and the Cube experiment scripts.

The official LeWM HDF5 files store one column per info key with ``/``
replaced by ``_`` (``privileged/block_0_pos`` -> ``privileged_block_0_pos``),
plus ``ep_idx`` (or ``episode_idx``) and ``step_idx`` columns that index each
row inside its episode. Everything here only relies on those conventions.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

EPISODE_COLUMNS = ("episode_idx", "ep_idx")


def episode_column(dataset) -> str:
    """Name of the per-row episode index column."""
    names = list(dataset.column_names)
    for name in EPISODE_COLUMNS:
        if name in names:
            return name
    raise KeyError(f"dataset has none of {EPISODE_COLUMNS}; columns: {sorted(names)}")


def first_present(dataset, candidates: Iterable[str]) -> str:
    """Return the first candidate column that exists (informative error otherwise)."""
    names = set(dataset.column_names)
    candidates = list(candidates)
    for name in candidates:
        if name in names:
            return name
    raise KeyError(f"none of {candidates} in dataset columns {sorted(names)}")


def episode_max_step(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(episode_ids, max_step_idx)`` for every episode id in the file.

    Vectorised replacement for the per-episode ``np.max(step_idx[ep == e])``
    loop of the upstream evaluation script (identical result, O(rows)).
    """
    ep = np.asarray(dataset.get_col_data(episode_column(dataset))).reshape(-1)
    step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ids, inverse = np.unique(ep, return_inverse=True)
    max_step = np.full(len(ids), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_step, inverse, step.astype(np.int64))
    return ids, max_step


def episodes_length(dataset, episodes: Sequence[int]) -> np.ndarray:
    """``max(step_idx) + 1`` for each requested episode id (upstream semantics)."""
    ids, max_step = episode_max_step(dataset)
    pos = np.searchsorted(ids, np.asarray(episodes))
    if np.any(pos >= len(ids)) or np.any(ids[np.minimum(pos, len(ids) - 1)] != np.asarray(episodes)):
        raise KeyError("unknown episode id requested")
    return max_step[pos] + 1


def valid_start_rows(dataset, goal_offset_steps: int) -> np.ndarray:
    """Row indices whose step leaves room for a goal ``goal_offset_steps`` later.

    Same rule as the upstream protocol: a row is valid when
    ``step_idx <= episode_length - goal_offset_steps - 1``.
    """
    ep = np.asarray(dataset.get_col_data(episode_column(dataset))).reshape(-1)
    step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ids, inverse = np.unique(ep, return_inverse=True)
    max_step = np.full(len(ids), np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_step, inverse, step.astype(np.int64))
    max_start = (max_step + 1) - int(goal_offset_steps) - 1
    return np.nonzero(step <= max_start[inverse])[0]


def sample_start_rows(dataset, num_eval: int, seed: int, goal_offset_steps: int) -> np.ndarray:
    """Sorted dataset rows used as evaluation start states.

    Reproduces the upstream sampler exactly (``default_rng(seed).choice`` over
    ``len(valid) - 1`` positions without replacement), so a given seed selects
    the same start states as ``le-wm/eval.py``.
    """
    valid = valid_start_rows(dataset, goal_offset_steps)
    if len(valid) < num_eval:
        raise ValueError(
            f"Not enough valid starting points for evaluation ({len(valid)} < {num_eval})"
        )
    g = np.random.default_rng(int(seed))
    picked = g.choice(len(valid) - 1, size=int(num_eval), replace=False)
    return np.sort(valid[picked])


def rows_to_episode_starts(dataset, rows: np.ndarray) -> tuple[list[int], list[int]]:
    """Map dataset rows to ``(episode_ids, start_steps)`` lists."""
    ep = np.asarray(dataset.get_col_data(episode_column(dataset))).reshape(-1)
    step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    return ep[rows].astype(int).tolist(), step[rows].astype(int).tolist()


def fit_scaler(values: np.ndarray):
    """StandardScaler fitted on the rows without NaNs (as in evaluate.build_world)."""
    from sklearn import preprocessing

    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    scaler = preprocessing.StandardScaler()
    scaler.fit(values[~np.isnan(values).any(axis=1)])
    return scaler
