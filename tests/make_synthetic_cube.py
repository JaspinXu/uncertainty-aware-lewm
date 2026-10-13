"""Build a tiny Cube dataset in the official LeWM HDF5 layout (tests only).

The official ``ogbench/cube_single_expert.h5`` is 46 GB; the test-suite and the
sandbox integration runs use a few short expert episodes collected with the same
environment, expert policy and column naming (``/`` in info keys -> ``_``):

    python tests/make_synthetic_cube.py --out /tmp/stablewm --episodes 6 --steps 80

writes ``<out>/datasets/ogbench/cube_single_expert.h5``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


class _RenamingWriter:
    """In-memory writer for ``World.collect``: renames keys, adds ep_idx."""

    def __init__(self):
        self.episodes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_episodes(self, episodes):
        for ep in episodes:
            out = {}
            for k, v in ep.items():
                arr = np.asarray(v)
                if arr.dtype.kind in "OUS":
                    continue
                out[k.replace("/", "_")] = arr
            n = len(out["step_idx"])
            # World.collect starts recording after the first step; the official
            # files index steps from 0 within each episode, so re-index.
            out["step_idx"] = np.arange(n, dtype=np.int64)
            out["ep_idx"] = np.full(n, len(self.episodes), dtype=np.int64)
            self.episodes.append(out)


def make_dataset(out_root: Path, episodes: int = 6, steps: int = 80, seed: int = 0) -> Path:
    import stable_worldmodel as swm
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=2,
        image_shape=(224, 224),
        max_episode_steps=steps,
        env_type="single",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )
    world.set_policy(ExpertPolicy(seed=seed))
    writer = _RenamingWriter()
    world.collect(writer=writer, episodes=episodes, seed=seed, progress=False)

    path = Path(out_root) / "datasets" / "ogbench" / "cube_single_expert.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with HDF5Writer(path, mode="overwrite") as w:
        for ep in writer.episodes:
            w.write_episode({k: list(v) for k, v in ep.items()})
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = make_dataset(Path(args.out), args.episodes, args.steps, args.seed)
    print(path)
    os._exit(0)  # skip noisy EGL teardown


if __name__ == "__main__":
    main()
