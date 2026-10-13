"""Planning evaluation for every method in config/eval/method/ (PushT and Cube).

    python evaluate.py method=baseline seed=42                      # PushT (default)
    python evaluate.py --config-name cube method=baseline seed=42   # OGBench-Cube
    python evaluate.py --config-name cube seed=42 record.enabled=true  # + L1 records

The protocol (dataset, start-state sampling, CEM budget, replanning) lives in
config/eval/<env>.yaml and is shared by all methods; ``method=`` only changes
how candidate action sequences are scored (see planners.py).

Writes ``<output.dir>/<run_id>.txt`` (resolved config + metrics) and
``<output.dir>/<run_id>.json`` (per-episode outcomes), where ``run_id``
defaults to ``<method.name>_s<seed>``. With ``record.enabled=true`` every
planning call is also logged to ``<record.dir>/<run_id>.pt`` (cube/recording.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from cube import data as cube_data
from planners import build_planner

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# shared pieces (also used by scripts/calibrate_gated_threshold.py)
# --------------------------------------------------------------------------- #
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed episode sampling, CEM sampling and MC-dropout masks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def episode_column(dataset) -> str:
    return cube_data.episode_column(dataset)


def get_episodes_length(dataset, episodes):
    """``max(step_idx) + 1`` per episode (vectorised; same values as before)."""
    return cube_data.episodes_length(dataset, episodes)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path
    )


def build_world(cfg):
    """Environment, image transforms, dataset and per-column normalizers."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    extra_wrappers = []
    if cfg.get("map_proprio_keys", False):
        # env info keys 'proprio/x' -> dataset column names 'proprio_x', so that the
        # start state's proprio is taken from the dataset (as qpos/qvel are) and
        # proprio-conditioned models (cube/variants.py) see one naming everywhere.
        from functools import partial

        from stable_worldmodel.wrapper import MapKeysWrapper

        key_map = {
            "proprio/" + c[len("proprio_"):]: c
            for c in dataset.column_names
            if c.startswith("proprio_")
        }
        extra_wrappers.append(partial(MapKeysWrapper, key_map=key_map))
    world = swm.World(**cfg.world, image_shape=(224, 224), extra_wrappers=extra_wrappers or None)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        values = dataset.get_col_data(col)
        scaler.fit(values[~np.isnan(values).any(axis=1)])
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return world, dataset, transform, process


def build_policy(cfg, model, device, process, transform, recorder=None):
    """CEM + MPC policy. With a ``recorder`` the solver/policy also log every
    planning call (cube/recording.py); the chosen actions are unchanged."""
    OmegaConf.set_struct(cfg, False)
    cfg.solver.device = device
    plan_config = swm.PlanConfig(**cfg.plan_config)
    if recorder is None:
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        return swm.policy.WorldModelPolicy(
            solver=solver, config=plan_config, process=process, transform=transform
        )
    from cube.recording import RecordingPolicy

    solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_cfg["_target_"] = "cube.recording.RecordingCEMSolver"
    solver = hydra.utils.instantiate(solver_cfg, model=model, recorder=recorder)
    return RecordingPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
        recorder=recorder,
    )


def sample_start_states(cfg, dataset, num_eval: int, sample_seed: int):
    """Pick ``num_eval`` (episode, start step) pairs that leave room for the goal.

    Same rule and RNG stream as the upstream LeWM protocol (see cube/data.py).
    """
    valid = cube_data.valid_start_rows(dataset, int(cfg.eval.goal_offset_steps))
    print(len(valid), "valid starting points found for evaluation.")
    rows = cube_data.sample_start_rows(
        dataset, num_eval, sample_seed, int(cfg.eval.goal_offset_steps)
    )
    return cube_data.rows_to_episode_starts(dataset, rows)


def evaluate_policy(cfg, world, dataset, policy, episodes, start_steps, video_dir=None):
    world.set_policy(policy)
    return world.evaluate(
        dataset=dataset,
        start_steps=start_steps,
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=video_dir,
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def make_recorder(cfg, model, device):
    """PlanRecorder on the frozen base LeWM (``model.base`` for MC planners)."""
    from cube.recording import PlanRecorder

    rec = cfg.record
    base = getattr(model, "base", model)
    return PlanRecorder(
        base,
        device=device,
        history_size=int(rec.get("history_size", 3)),
        n_mc=int(rec.n_mc),
        n_elite=int(rec.n_elite),
        n_random=int(rec.n_random),
        seed=int(cfg.seed),
    )


def _plain(node):
    """DictConfig/ListConfig/None -> plain python (for JSON)."""
    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def write_outputs(cfg, run_id, results_dir, device, metrics, model, elapsed, episodes, start_steps):
    """``<run_id>.txt`` (human readable, as before) + ``<run_id>.json`` (per episode)."""
    lines = [f"device: {device}\n", f"metrics: {metrics}\n"]
    extra = {}
    if getattr(model, "truncation_lengths", None):
        lines.append(f"mean_truncation_length: {np.mean(model.truncation_lengths)}\n")
        lines.append(f"mean_sigma2: {np.mean(model.mean_sigma2)}\n")
        print(f"[evaluate] mean truncated rollout length: {np.mean(model.truncation_lengths):.2f}")
        extra["mean_truncation_length"] = float(np.mean(model.truncation_lengths))
        extra["mean_sigma2"] = float(np.mean(model.mean_sigma2))
    if hasattr(model, "diagnostics_summary"):
        diagnostics = model.diagnostics_summary()
        print("[evaluate] diagnostics:", diagnostics)
        lines.append("risk_diagnostics: " + json.dumps(diagnostics, sort_keys=True) + "\n")
        extra["risk_diagnostics"] = diagnostics
    lines.append(f"evaluation_time: {elapsed} seconds\n")

    results_path = results_dir / f"{run_id}.txt"
    with results_path.open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        f.write("\n==== RESULTS ====\n")
        f.writelines(lines)

    successes = np.asarray(metrics["episode_successes"]).astype(bool)
    payload = {
        "run_id": run_id,
        "env": str(cfg.world.env_name),
        "method": str(cfg.method.name),
        "seed": int(cfg.seed),
        "policy": str(cfg.policy),
        "num_eval": int(cfg.eval.num_eval),
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": successes.astype(int).tolist(),
        "episodes": [int(e) for e in episodes],
        "start_steps": [int(t) for t in start_steps],
        "evaluation_time": float(elapsed),
        "device": str(device),
        "model_overrides": _plain(cfg.get("model_overrides")),
        **extra,
    }
    (results_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=1))
    print(f"[evaluate] wrote {results_path}")
    return payload


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = pick_device()
    run_id = cfg.output.run_id or f"{cfg.method.name}_s{cfg.seed}"
    print(f"[evaluate] method={cfg.method.name} seed={cfg.seed} device={device} run_id={run_id}")

    world, dataset, transform, process = build_world(cfg)
    model = build_planner(cfg, device)
    recording = bool(cfg.get("record") is not None and cfg.record.get("enabled", False))
    recorder = make_recorder(cfg, model, device) if recording else None
    policy = build_policy(cfg, model, device, process, transform, recorder=recorder)
    episodes, start_steps = sample_start_states(cfg, dataset, int(cfg.eval.num_eval), int(cfg.seed))

    results_dir = ROOT / cfg.output.dir
    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir = results_dir / "videos" / run_id if cfg.output.video else None

    start_time = time.time()
    metrics = evaluate_policy(cfg, world, dataset, policy, episodes, start_steps, video_dir)
    elapsed = time.time() - start_time
    print(metrics)

    write_outputs(cfg, run_id, results_dir, device, metrics, model, elapsed, episodes, start_steps)

    if recorder is not None:
        scaler = process["action"]
        meta = {
            "run_id": run_id,
            "method": str(cfg.method.name),
            "seed": int(cfg.seed),
            "policy": str(cfg.policy),
            "dataset": str(cfg.eval.dataset_name),
            "episodes": [int(e) for e in episodes],
            "start_steps": [int(t) for t in start_steps],
            "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
            "goal_offset_steps": int(cfg.eval.goal_offset_steps),
            "eval_budget": int(cfg.eval.eval_budget),
            "plan_config": OmegaConf.to_container(cfg.plan_config, resolve=True),
            "solver": {k: cfg.solver[k] for k in ("num_samples", "n_steps", "topk")},
            "world": OmegaConf.to_container(cfg.world, resolve=True),
            "action_mean": scaler.mean_.tolist(),
            "action_scale": scaler.scale_.tolist(),
            "evaluation_time": float(elapsed),
        }
        rec_dir = ROOT / str(cfg.record.dir)
        path = recorder.save(rec_dir / f"{run_id}.pt", meta)
        print(f"[evaluate] recorded {len(recorder.plans)} plans -> {path} "
              f"(max |det-solver| cost mismatch {recorder.max_cost_mismatch:.3g})")


if __name__ == "__main__":
    run()
