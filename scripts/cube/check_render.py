"""Check headless MuJoCo rendering + Cube env speed on this node.

    python scripts/cube/check_render.py [--out results/cube/smoke/render_check.png]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/smoke/render_check.png")
    args = ap.parse_args()
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    try:
        from cube.oracle import make_cube_env, render_like_world

        env = make_cube_env()
        t0 = time.time()
        for _ in range(50):
            env.step(np.random.uniform(-1, 1, 5))
        step_ms = (time.time() - t0) / 50 * 1e3
        t0 = time.time()
        for _ in range(20):
            img = render_like_world(env)
        render_ms = (time.time() - t0) / 20 * 1e3
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"RENDER CHECK FAILED: {type(e).__name__}: {e}")
        print("Try: export MUJOCO_GL=osmesa  (needs `conda install -c conda-forge mesalib`), "
              "or run this on a GPU node; see cluster/README.md")
        os._exit(1)
    from PIL import Image

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(args.out)
    print(f"render OK: {img.shape} {img.dtype}, mean pixel {img.mean():.1f}; "
          f"env step {step_ms:.2f} ms, render {render_ms:.2f} ms -> {args.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
