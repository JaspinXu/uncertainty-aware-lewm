"""Calibrate the sigma^2 gate threshold for ``gated`` truncation.

By default collects reference epistemic uncertainty (sigma^2 at the first
predicted step) from **CEM planning** on PushT episodes — the same candidate
action distribution the planner sees at eval time.

Usage:
    export STABLEWM_HOME=$PWD
    export PYTHONPATH=$PWD/stable-worldmodel:$PWD:$PYTHONPATH

    # recommended: CEM planning-time calibration (8 parallel episodes)
    python scripts/calibrate_gated_threshold.py policy=pusht/lewm \
        +source=planning +num_eval=8

    # ablation: expert-action rollouts (underestimates planning-time sigma^2)
    python scripts/calibrate_gated_threshold.py policy=pusht/lewm \
        +source=expert +num_samples=64

    # synthetic smoke test
    python scripts/calibrate_gated_threshold.py policy=pusht/lewm \
        +source=synthetic +num_samples=32

Then run gated eval with the printed planning-time p50/p90 threshold.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_worldmodel as swm

from ua.calibrate import (
    collect_sigma2_ref,
    flatten_sigma2_perstep,
    flatten_sigma2_refs,
    perstep_thresholds,
    summarize,
    threshold_from_sigma2,
)
from ua_eval import (
    build_ua_model,
    get_dataset,
    get_episodes_length,
    img_transform,
    pick_device,
)

os.environ.setdefault("MUJOCO_GL", "egl")


def get_dataset_with_pixels(cfg):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        cfg.eval.dataset_name,
        keys_to_load=["pixels", "action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state", "episode_idx", "step_idx"],
        cache_dir=dataset_path,
    )


def normalize_action(action: np.ndarray, processor) -> np.ndarray:
    flat = action.reshape(-1, action.shape[-1])
    return processor.transform(flat).reshape(action.shape)


def build_info_from_chunk(chunk, goal_chunk, transform, device, history_size):
    pix = chunk["pixels"][-history_size:]
    goal_pix = goal_chunk["pixels"][-history_size:]

    def _to_tensor(frames):
        t = torch.from_numpy(np.asarray(frames)).float()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.shape[-1] in (1, 3, 4):
            t = t.permute(0, 3, 1, 2)
        out = torch.stack([transform(f) for f in t], dim=0)
        return out.unsqueeze(0).unsqueeze(0)

    return {
        "pixels": _to_tensor(pix).to(device),
        "goal": _to_tensor(goal_pix).to(device),
    }


@torch.no_grad()
def collect_from_expert(cfg, ua_model, device, num_samples: int, goal_offset: int):
    dataset = get_dataset_with_pixels(cfg)
    transform = img_transform(cfg)

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_idx_col = dataset.get_col_data(col_name)
    step_idx_col = dataset.get_col_data("step_idx")
    episode_len = np.array(
        [np.max(step_idx_col[episode_idx_col == ep_id]) + 1 for ep_id in ep_indices]
    )

    processor = preprocessing.StandardScaler()
    action_col = dataset.get_col_data("action")
    action_col = action_col[~np.isnan(action_col).any(axis=1)]
    processor.fit(action_col)

    history_size = int(cfg.ua.history_size)
    T = history_size + int(cfg.plan_config.horizon)
    max_start = episode_len - goal_offset - T - 1
    valid_eps = ep_indices[max_start > 0]
    if len(valid_eps) == 0:
        raise ValueError("No episodes long enough for calibration windows")

    rng = np.random.default_rng(int(cfg.seed))
    sigma2_vals = []
    for _ in range(num_samples):
        ep = int(rng.choice(valid_eps))
        ep_local = int(np.where(ep_indices == ep)[0][0])
        start = int(rng.integers(history_size - 1, max_start[ep_indices == ep][0]))
        goal_start = start + goal_offset

        hist = dataset.load_chunk(
            np.array([ep_local]),
            np.array([start - history_size + 1]),
            np.array([start + 1]),
        )[0]
        goal = dataset.load_chunk(
            np.array([ep_local]),
            np.array([goal_start - history_size + 1]),
            np.array([goal_start + 1]),
        )[0]
        act_chunk = dataset.load_chunk(
            np.array([ep_local]),
            np.array([start - history_size + 1]),
            np.array([start - history_size + 1 + T]),
        )[0]

        info = build_info_from_chunk(hist, goal, transform, device, history_size)
        actions = normalize_action(act_chunk["action"], processor).astype(np.float32)
        actions = torch.from_numpy(actions).view(1, 1, T, -1).to(device)
        sigma2_vals.append(collect_sigma2_ref(ua_model, info, actions, step=0).item())

    return np.array(sigma2_vals)


@torch.no_grad()
def collect_synthetic(ua_model, device, num_samples: int, history_size: int, T: int):
    sigma2_vals = []
    for _ in range(num_samples):
        info = {
            "pixels": torch.rand(1, 1, history_size, 3, 224, 224, device=device),
            "goal": torch.rand(1, 1, history_size, 3, 224, 224, device=device),
        }
        actions = torch.randn(1, 1, T, 2, device=device)
        sigma2_vals.append(collect_sigma2_ref(ua_model, info, actions, step=0).item())
    return np.array(sigma2_vals)


def collect_from_planning(cfg, device, num_eval: int):
    """Run a short PushT eval so CEM calls get_cost; harvest sigma2_refs."""
    # force no truncation while collecting — we only need the sigma^2 signal
    OmegaConf.set_struct(cfg, False)
    cfg.ua.truncation.mode = "none"
    cfg.eval.num_eval = num_eval
    cfg.world.num_envs = num_eval

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be <= eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = build_ua_model(cfg, device)
    model.reset_diagnostics()
    OmegaConf.set_struct(cfg, False)
    cfg.solver.device = device
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(cfg.seed)
    # use a different seed offset so calibration episodes != formal eval seeds
    cal_seed = int(cfg.seed) + 1000
    g = np.random.default_rng(cal_seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)
    print(
        f"[calibrate] running CEM planning on {num_eval} episodes "
        f"(cal_seed={cal_seed}) to collect sigma^2_0 ..."
    )
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=None,
    )
    elapsed = time.time() - t0
    print(f"[calibrate] planning done in {elapsed:.1f}s, metrics={metrics}")

    sigma2_vals = flatten_sigma2_refs(model)
    sigma2_steps = flatten_sigma2_perstep(model)
    print(
        f"[calibrate] collected {len(sigma2_vals)} per-candidate sigma^2_0 "
        f"from {len(model.sigma2_refs)} get_cost calls"
    )
    return sigma2_vals, sigma2_steps, {
        "num_eval": num_eval,
        "cal_seed": cal_seed,
        "n_get_cost": len(model.sigma2_refs),
        "n_sigma2_refs": int(len(sigma2_vals)),
        "eval_time": elapsed,
        "success_rate": float(metrics.get("success_rate", float("nan"))),
    }


@hydra.main(version_base=None, config_path="../config/eval", config_name="ua_pusht")
def main(cfg: DictConfig):
    source = str(cfg.get("source", "planning"))
    num_samples = int(cfg.get("num_samples", 64))
    num_eval = int(cfg.get("num_eval", 8))
    device = pick_device()
    print(f"[calibrate] device={device}, source={source}")

    if cfg.policy == "random":
        raise ValueError("Set policy=pusht/lewm (or another checkpoint) for calibration")

    meta = {"source": source}
    if source == "planning":
        sigma2_vals, sigma2_steps, plan_meta = collect_from_planning(
            cfg, device, num_eval
        )
        meta.update(plan_meta)
    elif source == "expert":
        sigma2_steps = None
        ua_model = build_ua_model(cfg, device)
        history_size = int(cfg.ua.history_size)
        T = history_size + int(cfg.plan_config.horizon)
        sigma2_vals = collect_from_expert(
            cfg, ua_model, device, num_samples, int(cfg.eval.goal_offset_steps)
        )
        meta["num_rollouts"] = num_samples
    elif source == "synthetic":
        sigma2_steps = None
        ua_model = build_ua_model(cfg, device)
        history_size = int(cfg.ua.history_size)
        T = history_size + int(cfg.plan_config.horizon)
        sigma2_vals = collect_synthetic(ua_model, device, num_samples, history_size, T)
        meta["num_rollouts"] = num_samples
    else:
        raise ValueError(f"Unknown source={source!r}; use planning | expert | synthetic")

    if len(sigma2_vals) == 0:
        raise RuntimeError("No sigma^2 values collected — check planning diagnostics")

    stats = summarize(sigma2_vals)
    p50 = threshold_from_sigma2(sigma2_vals, 50.0)
    p90 = threshold_from_sigma2(sigma2_vals, 90.0)
    p95 = threshold_from_sigma2(sigma2_vals, 95.0)

    print("\n=== sigma^2_0 (first predicted step) ===")
    for k, v in stats.items():
        print(f"  {k:>4s}: {v:.6f}")
    print(f"\nRecommended gated thresholds ({source}):")
    print(f"  p50 (aggressive gate — truncate top half): {p50:.6f}")
    print(f"  p90 (moderate gate — truncate top 10%):    {p90:.6f}")
    print(f"  p95 (conservative gate):                   {p95:.6f}")
    # For planning-time calibration, p50 is usually the useful operating point:
    # expert p90 (~0.15) was far below planning mean (~1.0), so the gate never closed.
    recommend = p50 if source == "planning" else p90
    print(
        "\nExample eval command:\n"
        f"  python ua_eval.py --config-name=ua_pusht policy={cfg.policy} "
        f"ua.truncation.mode=gated ua.truncation.threshold={recommend:.4f} "
        f"ua.truncation.omega={cfg.ua.truncation.omega}"
    )

    out_dir = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_filename = str(cfg.get("output_filename", "gated_threshold.json"))
    if Path(output_filename).name != output_filename or not output_filename.endswith(".json"):
        raise ValueError("+output_filename must be a .json basename without directories")
    out_path = out_dir / output_filename
    output_stem = Path(output_filename).stem
    perstep = {}
    if sigma2_steps is not None and sigma2_steps.size:
        perstep = {
            "perstep_threshold_p50": perstep_thresholds(
                sigma2_steps, 50.0
            ).tolist(),
            "perstep_threshold_p90": perstep_thresholds(
                sigma2_steps, 90.0
            ).tolist(),
            "perstep_threshold_p95": perstep_thresholds(
                sigma2_steps, 95.0
            ).tolist(),
        }
        print("\nRecommended adaptive_perstep threshold (p90 by depth):")
        print("  " + json.dumps(perstep["perstep_threshold_p90"]))
    payload = {
        "policy": cfg.policy,
        "estimator": cfg.ua.estimator,
        "n_samples": cfg.ua.n_samples,
        "source": source,
        "stats": stats,
        "threshold_p50": p50,
        "threshold_p90": p90,
        "threshold_p95": p95,
        "recommended_threshold": recommend,
        **perstep,
        "meta": meta,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    # also dump raw values for plotting
    refs_path = out_dir / f"{output_stem}_sigma2_refs_{source}.npy"
    np.save(refs_path, sigma2_vals)
    if sigma2_steps is not None:
        perstep_path = out_dir / f"{output_stem}_sigma2_perstep_{source}.npy"
        np.save(perstep_path, sigma2_steps)
    print(f"\nWrote {out_path}")
    print(f"Wrote {refs_path}")


if __name__ == "__main__":
    main()
