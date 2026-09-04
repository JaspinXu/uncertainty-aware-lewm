"""Evaluation entry point for Uncertainty-Aware LeWM (UA-LeWM).

This mirrors ``eval.py`` (same dataset sampling / world.evaluate flow) but:
  * picks the device automatically (CUDA -> MPS -> CPU) instead of hardcoding CUDA,
    so it runs on an Apple-Silicon laptop;
  * wraps the pretrained LeWM cost model in ``ua.UncertaintyAwareJEPA`` so the
    CEM/gradient solver plans with uncertainty-aware rollout truncation.

Run, e.g.:
    python ua_eval.py --config-name=ua_pusht policy=pusht/lewm \
        ua.estimator=mcdropout ua.truncation.mode=adaptive_length
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import copy
import random
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

from ua import (
    EnsembleEstimator,
    MCDropoutEstimator,
    TruncationConfig,
    UncertaintyAwareJEPA,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed episode sampling, CEM sampling, dropout, and perturbed ensembles."""
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


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path
    )


def build_ua_model(cfg, device):
    """Load the pretrained LeWM cost model and wrap it for uncertainty-aware planning."""
    base = swm.wm.utils.load_pretrained(cfg.policy)
    base = base.to(device).eval()
    base.requires_grad_(False)
    base.interpolate_pos_encoding = True

    ua_cfg = cfg.ua
    est_name = ua_cfg.estimator

    if est_name == "mcdropout":
        estimator = MCDropoutEstimator(n_samples=int(ua_cfg.n_samples))

    elif est_name == "ensemble":
        members = [base]
        for ckpt in list(ua_cfg.get("ensemble_ckpts", []) or []):
            m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
            m.requires_grad_(False)
            m.interpolate_pos_encoding = True
            members.append(m)
        if len(members) < 2:
            raise ValueError(
                "ua.estimator=ensemble needs ua.ensemble_ckpts with >=1 extra checkpoint"
            )
        estimator = EnsembleEstimator(members)

    elif est_name == "ensemble_perturb":
        # cheap stand-in ensemble: clone base and jitter predictor weights.
        # (no extra data/training; use for quick disagreement experiments)
        members = [base]
        std = float(ua_cfg.get("perturb_std", 0.01))
        for _ in range(int(ua_cfg.n_samples) - 1):
            m = copy.deepcopy(base)
            with torch.no_grad():
                for p in m.predictor.parameters():
                    p.add_(torch.randn_like(p) * std)
            m = m.to(device).eval()
            m.requires_grad_(False)
            m.interpolate_pos_encoding = True
            members.append(m)
        estimator = EnsembleEstimator(members)

    else:
        raise ValueError(f"Unknown ua.estimator {est_name!r}")

    threshold_cfg = ua_cfg.truncation.get("threshold")
    if threshold_cfg is None:
        threshold = None
    elif isinstance(threshold_cfg, (list, tuple)):
        threshold = [float(value) for value in threshold_cfg]
    else:
        # OmegaConf ListConfig is iterable but is not a built-in list.
        try:
            threshold = (
                [float(value) for value in threshold_cfg]
                if not isinstance(threshold_cfg, (str, bytes))
                else float(threshold_cfg)
            )
        except TypeError:
            threshold = float(threshold_cfg)

    trunc = TruncationConfig(
        mode=ua_cfg.truncation.mode,
        omega=float(ua_cfg.truncation.omega),
        k_base=float(ua_cfg.truncation.k_base),
        k_min=int(ua_cfg.truncation.k_min),
        k_max=(None if ua_cfg.truncation.get("k_max") is None else int(ua_cfg.truncation.k_max)),
        threshold=threshold,
        penalty_weight=float(ua_cfg.truncation.get("penalty_weight", 0.0)),
    )

    model = UncertaintyAwareJEPA(
        base=base,
        estimator=estimator,
        trunc_cfg=trunc,
        history_size=int(ua_cfg.get("history_size", 3)),
    ).to(device)
    return model


@hydra.main(version_base=None, config_path="./config/eval", config_name="ua_pusht")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    seed_everything(int(cfg.seed))
    device = pick_device()
    print(f"[ua_eval] using device: {device}")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy = cfg.get("policy", "random")
    if policy != "random":
        model = build_ua_model(cfg, device)
        # solver device must match the model device (overrides the cuda default)
        OmegaConf.set_struct(cfg, False)
        cfg.solver.device = device
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        results_root = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    else:
        policy = swm.policy.RandomPolicy()
        results_root = Path(__file__).parent

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_root.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_root,
    )
    end_time = time.time()
    print(metrics)

    if cfg.policy != "random" and hasattr(model, "truncation_lengths") and model.truncation_lengths:
        print(
            f"[ua_eval] mean truncated rollout length: "
            f"{np.mean(model.truncation_lengths):.2f} steps "
            f"(over {len(model.truncation_lengths)} planning iterations)"
        )

    results_path = results_root / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"device: {device}\n")
        f.write(f"metrics: {metrics}\n")
        if cfg.policy != "random" and getattr(model, "truncation_lengths", None):
            f.write(f"mean_truncation_length: {np.mean(model.truncation_lengths)}\n")
            f.write(f"mean_sigma2: {np.mean(model.mean_sigma2)}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
