"""Sanity-check the Cube dataset layout and summarise the evaluation protocol.

    python scripts/cube/inspect_dataset.py [--seeds 42 43 ...]

Prints every column with shape/dtype, checks the columns the evaluation and
training code rely on, and measures how hard the protocol's start/goal pairs
are: the cube-to-goal distance at the start state (an episode whose cube is
already within the 4 cm success radius is solved by any action).
Writes results/cube/dataset_info.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from cube import data as cube_data  # noqa: E402

REQUIRED = ["pixels", "action", "qpos", "qvel", "step_idx"]
GOAL_POS = ["privileged_block_0_pos", "privileged/block_0_pos"]
GOAL_QUAT = ["privileged_block_0_quat", "privileged/block_0_quat"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055])
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/dataset_info.json")
    args = ap.parse_args()

    import stable_worldmodel as swm

    ds = swm.data.HDF5Dataset(args.dataset, keys_to_cache=["action"], cache_dir=swm.data.utils.get_cache_dir())
    info: dict = {"dataset": args.dataset, "path": str(ds.h5_path)}
    with ds._open_h5() as f:
        cols = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in f.keys() if hasattr(f[k], "shape")}
    info["columns"] = cols
    print(f"dataset: {ds.h5_path}")
    for k, v in sorted(cols.items()):
        print(f"  {k:34s} {tuple(v['shape'])} {v['dtype']}")

    missing = [c for c in REQUIRED if c not in cols]
    ep_col = cube_data.episode_column(ds)
    pos_col = cube_data.first_present(ds, GOAL_POS)
    quat_col = cube_data.first_present(ds, GOAL_QUAT)
    info.update({"episode_column": ep_col, "goal_pos_column": pos_col, "goal_quat_column": quat_col, "missing": missing})
    proprio = [c for c in cols if c.startswith("proprio")]
    info["proprio_columns"] = proprio
    info["proprio_dim"] = int(sum(int(np.prod(cols[c]["shape"][1:] or [1])) for c in proprio))
    print(f"episode column={ep_col} goal pos={pos_col} quat={quat_col} missing={missing}")
    print(f"proprio columns ({info['proprio_dim']} dims): {proprio}")

    lengths = np.asarray(ds.lengths)
    ids, max_step = cube_data.episode_max_step(ds)
    step = np.asarray(ds.get_col_data("step_idx")).reshape(-1)
    info["episodes"] = int(len(lengths))
    info["rows"] = int(lengths.sum())
    info["episode_length"] = {"min": int(lengths.min()), "median": float(np.median(lengths)), "max": int(lengths.max())}
    info["step_idx_starts_at"] = int(step[np.asarray(ds.offsets)].min())
    info["step_idx_consistent"] = bool(np.all(max_step + 1 == lengths[: len(max_step)]))
    ep = np.asarray(ds.get_col_data(ep_col)).reshape(-1)
    info["ep_idx_positional"] = bool(np.all(ep[np.asarray(ds.offsets)] == np.arange(len(lengths))))
    print(f"{info['episodes']} episodes, {info['rows']} rows, lengths {info['episode_length']}, "
          f"step_idx starts at {info['step_idx_starts_at']}, consistent={info['step_idx_consistent']}, "
          f"ep_idx positional={info['ep_idx_positional']}")

    # difficulty of the protocol's start/goal pairs
    pos = np.asarray(ds.get_col_data(pos_col)).reshape(-1, 3)
    per_seed = {}
    for seed in args.seeds:
        rows = cube_data.sample_start_rows(ds, args.num_eval, seed, args.goal_offset)
        d = np.linalg.norm(pos[rows] - pos[rows + args.goal_offset], axis=1)
        per_seed[str(seed)] = {
            "trivial_frac": float((d <= 0.04).mean()),
            "median_start_dist": float(np.median(d)),
            "start_dist": d.round(4).tolist(),
        }
        print(f"  seed {seed}: trivial (start within 4 cm) {100 * (d <= 0.04).mean():.0f}%  "
              f"median start distance {np.median(d) * 100:.1f} cm")
    info["protocol"] = {"num_eval": args.num_eval, "goal_offset": args.goal_offset, "per_seed": per_seed}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=1))
    print(f"wrote {args.out}")
    if missing:
        sys.exit(f"missing required columns: {missing}")


if __name__ == "__main__":
    main()
    os._exit(0)
