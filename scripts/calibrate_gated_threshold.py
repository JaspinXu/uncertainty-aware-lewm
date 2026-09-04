"""Calibrate the sigma^2 gate threshold used by config/eval/method/gated.yaml.

Runs a short CEM-planning evaluation with the MC-dropout planner (truncation
disabled) and records, for every candidate the solver scored, the epistemic
uncertainty sigma^2 at the first predicted step. The recommended gate
threshold is the median of those values: candidates above it are truncated.

    python scripts/calibrate_gated_threshold.py method=mc5 seed=910 +num_eval=8 \\
        +output_filename=gated_lewm_strong_rh5.json

Writes results/calibration/<output_filename> (+ the raw sigma^2 arrays as .npy).
Episodes are sampled with seed+1000 so calibration never reuses evaluation
start states.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import (  # noqa: E402
    build_policy,
    build_world,
    evaluate_policy,
    pick_device,
    sample_start_states,
    seed_everything,
)
from planners import build_planner  # noqa: E402
from ua.calibrate import (  # noqa: E402
    flatten_sigma2_perstep,
    flatten_sigma2_refs,
    perstep_thresholds,
    summarize,
    threshold_from_sigma2,
)


@hydra.main(version_base=None, config_path="../config/eval", config_name="pusht")
def main(cfg: DictConfig) -> None:
    if str(cfg.method.planner) != "ua":
        raise ValueError("calibration needs an MC-dropout planner, e.g. method=mc5")
    num_eval = int(cfg.get("num_eval", 8))
    output_filename = str(cfg.get("output_filename", "gated_threshold.json"))
    if Path(output_filename).name != output_filename or not output_filename.endswith(".json"):
        raise ValueError("+output_filename must be a .json basename without directories")

    seed_everything(int(cfg.seed))
    device = pick_device()
    OmegaConf.set_struct(cfg, False)
    cfg.method.truncation.mode = "none"  # collect sigma^2 only; do not truncate
    cfg.eval.num_eval = num_eval
    cal_seed = int(cfg.seed) + 1000
    print(f"[calibrate] device={device} num_eval={num_eval} cal_seed={cal_seed}")

    world, dataset, transform, process = build_world(cfg)
    model = build_planner(cfg, device)
    model.reset_diagnostics()
    policy = build_policy(cfg, model, device, process, transform)
    episodes, start_steps = sample_start_states(cfg, dataset, num_eval, cal_seed)

    t0 = time.time()
    metrics = evaluate_policy(cfg, world, dataset, policy, episodes, start_steps)
    elapsed = time.time() - t0
    print(f"[calibrate] planning done in {elapsed:.1f}s, metrics={metrics}")

    sigma2_vals = flatten_sigma2_refs(model)
    sigma2_steps = flatten_sigma2_perstep(model)
    if len(sigma2_vals) == 0:
        raise RuntimeError("No sigma^2 values collected")
    print(f"[calibrate] {len(sigma2_vals)} candidate sigma^2_0 values from {len(model.sigma2_refs)} get_cost calls")

    stats = summarize(sigma2_vals)
    p50 = threshold_from_sigma2(sigma2_vals, 50.0)
    p90 = threshold_from_sigma2(sigma2_vals, 90.0)
    p95 = threshold_from_sigma2(sigma2_vals, 95.0)
    print("\n=== sigma^2_0 (first predicted step) ===")
    for k, v in stats.items():
        print(f"  {k:>4s}: {v:.6f}")
    print(f"\nRecommended gate threshold (p50): {p50:.6f}   (p90 {p90:.6f}, p95 {p95:.6f})")
    print("Put it in config/eval/method/gated.yaml -> truncation.threshold")

    out_dir = ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_filename).stem
    payload = {
        "policy": cfg.policy,
        "estimator": "mcdropout",
        "n_samples": int(cfg.method.n_samples),
        "source": "planning",
        "stats": stats,
        "threshold_p50": p50,
        "threshold_p90": p90,
        "threshold_p95": p95,
        "recommended_threshold": p50,
        "perstep_threshold_p50": perstep_thresholds(sigma2_steps, 50.0).tolist(),
        "perstep_threshold_p90": perstep_thresholds(sigma2_steps, 90.0).tolist(),
        "perstep_threshold_p95": perstep_thresholds(sigma2_steps, 95.0).tolist(),
        "meta": {
            "source": "planning",
            "num_eval": num_eval,
            "cal_seed": cal_seed,
            "n_get_cost": len(model.sigma2_refs),
            "n_sigma2_refs": int(len(sigma2_vals)),
            "eval_time": elapsed,
            "success_rate": float(metrics.get("success_rate", float("nan"))),
        },
    }
    out_path = out_dir / output_filename
    out_path.write_text(json.dumps(payload, indent=2))
    np.save(out_dir / f"{stem}_sigma2_refs_planning.npy", sigma2_vals)
    np.save(out_dir / f"{stem}_sigma2_perstep_planning.npy", sigma2_steps)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
