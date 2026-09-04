"""PushT planning evaluation for every method in config/eval/method/.

    python evaluate.py method=baseline seed=42
    python evaluate.py method=gated seed=1011 eval.num_eval=50

The protocol (dataset, start-state sampling, CEM budget, replanning) lives in
config/eval/pusht.yaml and is shared by all methods; ``method=`` only changes
how candidate action sequences are scored (see planners.py).

Writes ``results/<run_id>.txt`` (resolved config + metrics), where ``run_id``
defaults to ``<method.name>_s<seed>``.
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
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def get_episodes_length(dataset, episodes):
    episode_idx = dataset.get_col_data(episode_column(dataset))
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


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
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

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


def build_policy(cfg, model, device, process, transform):
    OmegaConf.set_struct(cfg, False)
    cfg.solver.device = device
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**cfg.plan_config),
        process=process,
        transform=transform,
    )


def sample_start_states(cfg, dataset, num_eval: int, sample_seed: int):
    """Pick ``num_eval`` (episode, start step) pairs that leave room for the goal."""
    col_name = episode_column(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_by_episode = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_by_episode[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_indices = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]
    print(len(valid_indices), "valid starting points found for evaluation.")
    if len(valid_indices) < num_eval:
        raise ValueError("Not enough valid starting points for evaluation")

    g = np.random.default_rng(sample_seed)
    picked = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    picked = np.sort(valid_indices[picked])  # sorted for HDF5 indexing
    rows = dataset.get_row_data(picked)
    return rows[col_name].tolist(), rows["step_idx"].tolist()


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
@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = pick_device()
    run_id = cfg.output.run_id or f"{cfg.method.name}_s{cfg.seed}"
    print(f"[evaluate] method={cfg.method.name} seed={cfg.seed} device={device} run_id={run_id}")

    world, dataset, transform, process = build_world(cfg)
    model = build_planner(cfg, device)
    policy = build_policy(cfg, model, device, process, transform)
    episodes, start_steps = sample_start_states(cfg, dataset, int(cfg.eval.num_eval), int(cfg.seed))

    results_dir = ROOT / cfg.output.dir
    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir = results_dir / "videos" / run_id if cfg.output.video else None

    start_time = time.time()
    metrics = evaluate_policy(cfg, world, dataset, policy, episodes, start_steps, video_dir)
    elapsed = time.time() - start_time
    print(metrics)

    lines = [f"device: {device}\n", f"metrics: {metrics}\n"]
    if getattr(model, "truncation_lengths", None):
        lines.append(f"mean_truncation_length: {np.mean(model.truncation_lengths)}\n")
        lines.append(f"mean_sigma2: {np.mean(model.mean_sigma2)}\n")
        print(f"[evaluate] mean truncated rollout length: {np.mean(model.truncation_lengths):.2f}")
    if hasattr(model, "diagnostics_summary"):
        diagnostics = model.diagnostics_summary()
        print("[evaluate] diagnostics:", diagnostics)
        lines.append("risk_diagnostics: " + json.dumps(diagnostics, sort_keys=True) + "\n")
    lines.append(f"evaluation_time: {elapsed} seconds\n")

    results_path = results_dir / f"{run_id}.txt"
    with results_path.open("w") as f:
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
        f.write("\n==== RESULTS ====\n")
        f.writelines(lines)
    print(f"[evaluate] wrote {results_path}")


if __name__ == "__main__":
    run()
