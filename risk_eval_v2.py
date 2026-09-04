"""Standalone evaluation entry point for Soft Penalty v2."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

from risk_v2 import RiskAwareJEPA, RiskConfig
from ua import EnsembleEstimator, MCDropoutEstimator
from ua.truncation import TruncationConfig
from ua_eval import get_dataset, get_episodes_length, img_transform, pick_device, seed_everything


def _build_truncation_config(risk_cfg: DictConfig) -> TruncationConfig:
    trunc_cfg = risk_cfg.get("truncation")
    if trunc_cfg is None:
        return TruncationConfig(mode="none")

    threshold_cfg = trunc_cfg.get("threshold")
    if threshold_cfg is None:
        threshold = None
    elif isinstance(threshold_cfg, (list, tuple)):
        threshold = [float(value) for value in threshold_cfg]
    else:
        try:
            threshold = (
                [float(value) for value in threshold_cfg]
                if not isinstance(threshold_cfg, (str, bytes))
                else float(threshold_cfg)
            )
        except TypeError:
            threshold = float(threshold_cfg)

    return TruncationConfig(
        mode=str(trunc_cfg.get("mode", "none")),
        omega=float(trunc_cfg.get("omega", 0.3)),
        k_base=float(trunc_cfg.get("k_base", 5)),
        k_min=int(trunc_cfg.get("k_min", 1)),
        k_max=(None if trunc_cfg.get("k_max") is None else int(trunc_cfg.k_max)),
        threshold=threshold,
        penalty_weight=float(trunc_cfg.get("penalty_weight", 0.0)),
    )


def build_risk_model(cfg: DictConfig, device: str) -> RiskAwareJEPA:
    base = swm.wm.utils.load_pretrained(cfg.policy).to(device).eval()
    base.requires_grad_(False)
    base.interpolate_pos_encoding = True

    risk_cfg = cfg.risk
    estimator_name = str(risk_cfg.estimator)
    if estimator_name == "mcdropout":
        estimator = MCDropoutEstimator(n_samples=int(risk_cfg.n_samples))
    elif estimator_name in {"ensemble", "ensemble_perturb"}:
        members = [base]
        if estimator_name == "ensemble":
            for checkpoint in list(risk_cfg.get("ensemble_ckpts", []) or []):
                member = swm.wm.utils.load_pretrained(checkpoint).to(device).eval()
                member.requires_grad_(False)
                member.interpolate_pos_encoding = True
                members.append(member)
            if len(members) < 2:
                raise ValueError(
                    "risk.estimator=ensemble needs at least one ensemble checkpoint"
                )
        else:
            perturb_std = float(risk_cfg.get("perturb_std", 0.01))
            for _ in range(int(risk_cfg.n_samples) - 1):
                member = copy.deepcopy(base)
                with torch.no_grad():
                    for parameter in member.predictor.parameters():
                        parameter.add_(torch.randn_like(parameter) * perturb_std)
                member = member.to(device).eval()
                member.requires_grad_(False)
                member.interpolate_pos_encoding = True
                members.append(member)
        estimator = EnsembleEstimator(members)
    else:
        raise ValueError(f"Unknown risk.estimator {estimator_name!r}")

    objective = OmegaConf.to_container(risk_cfg.objective, resolve=True)
    objective["num_samples"] = int(cfg.solver.num_samples)
    if cfg.solver.get("topk") is None:
        raise ValueError("Soft Penalty v2 requires a solver with topk")
    objective["topk"] = int(cfg.solver.topk)
    resolved = RiskConfig.from_mapping(objective)
    trunc = _build_truncation_config(risk_cfg)

    OmegaConf.set_struct(cfg, False)
    cfg.risk.objective = OmegaConf.create(asdict(resolved))
    return RiskAwareJEPA(
        base=base,
        estimator=estimator,
        risk_cfg=resolved,
        trunc_cfg=trunc,
        history_size=int(risk_cfg.get("history_size", 3)),
    ).to(device)


@hydra.main(version_base=None, config_path="./config/eval", config_name="risk_v2_pusht")
def run(cfg: DictConfig) -> None:
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    seed_everything(int(cfg.seed))
    device = pick_device()
    print(f"[risk_eval_v2] using device: {device}")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(column)
        processor.fit(values[~np.isnan(values).any(axis=1)])
        process[column] = processor
        if column != "action":
            process[f"goal_{column}"] = processor

    selected_policy = cfg.get("policy", "random")
    model = None
    if selected_policy != "random":
        model = build_risk_model(cfg, device)
        OmegaConf.set_struct(cfg, False)
        cfg.solver.device = device
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        selected_policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(**cfg.plan_config),
            process=process,
            transform=transform,
        )
        results_root = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    else:
        selected_policy = swm.policy.RandomPolicy()
        results_root = Path(__file__).parent

    episode_lengths = get_episodes_length(dataset, episode_ids)
    max_starts = episode_lengths - cfg.eval.goal_offset_steps - 1
    max_start_by_episode = {
        episode_id: max_starts[index] for index, episode_id in enumerate(episode_ids)
    }
    row_max_starts = np.array(
        [max_start_by_episode[value] for value in dataset.get_col_data(col_name)]
    )
    valid_indices = np.nonzero(
        dataset.get_col_data("step_idx") <= row_max_starts
    )[0]
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError("Not enough valid starting points for evaluation")

    generator = np.random.default_rng(cfg.seed)
    sampled_positions = generator.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    sampled_indices = np.sort(valid_indices[sampled_positions])
    sampled_rows = dataset.get_row_data(sampled_indices)

    world.set_policy(selected_policy)
    results_root.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=sampled_rows["step_idx"].tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=sampled_rows[col_name].tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_root,
    )
    elapsed = time.time() - start_time
    diagnostics = model.diagnostics_summary() if model is not None else {}
    print(metrics)
    if diagnostics:
        print("[risk_eval_v2] diagnostics:", diagnostics)

    results_path = results_root / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as stream:
        stream.write("\n==== RESOLVED CONFIG ====\n")
        stream.write(OmegaConf.to_yaml(cfg, resolve=True))
        stream.write("\n==== RESULTS ====\n")
        stream.write(f"device: {device}\n")
        stream.write(f"metrics: {metrics}\n")
        stream.write(
            "risk_diagnostics: "
            + json.dumps(diagnostics, sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.write(f"evaluation_time: {elapsed} seconds\n")


if __name__ == "__main__":
    run()
