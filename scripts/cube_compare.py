"""Multi-seed closed-loop comparison of planners on OGBench Cube with a frozen LeWM (config/eval/compare.yaml).

    python scripts/cube_compare.py +planner=calibrate             # gate thresholds, on compare.cal_seeds
    python scripts/cube_compare.py +planner=lewm                  # every seed in compare.seeds
    python scripts/cube_compare.py +planner=gated_tc "+seeds=[0,1]"
    python scripts/cube_compare.py +planner=gated_tc +quantile=0.8 +terminal_weight=2   # -> run gated_tc-q0.8-w2
    python scripts/cube_compare.py +planner=prior_cem_sp compare.offset=50

Every planner sees the same non-trivial start states for a seed (episodes outside the
latent cache of scripts/cube_train.py, so no learned component has seen them); success =
the cube within 4 cm of the goal's cube position, which ends the episode. CEM is
300 samples / 10 iterations / top-30 over 5 blocks of 5 steps, replanning every 5 blocks.

Planners:
  lewm              original LeWM: CEM on the deterministic terminal latent distance
  mc5               mean of 5 MC-dropout rollouts (config/eval/method/mc5.yaml)
  gated             MC-5 + gated horizon truncation (method/gated.yaml); threshold re-calibrated
                    on Cube with its own recipe: p50 of planning-time step-0 sigma^2
  gated_tc          MC-5 + per-step quantile gate + HALT-style terminal charge on the cut part
                    of the horizon (method/gated_tc.yaml, ua/truncation.py)
  goal_cvar         MC-5 + Goal-CVaR soft penalty (method/goal_cvar.yaml)
  bc_only           the Gaussian prior's mean, executed without the world model (reference)
  prior_cem_s1      CEM from N(mu_p, 1)          sigma = CEM's default width (as bc_init)
  prior_cem_sconst  CEM from N(mu_p, c^2)        c = the prior's mean sigma_p on held-out data
  prior_cem_sp      CEM from N(mu_p, sigma_p^2)  the prior's own per-dimension uncertainty
                    (these four use the prior for goals exactly compare.offset steps ahead;
  bc_only_var,      the same four with the prior that is not told how far the goal is,
  prior_cem_*_var   trained on goals 1-100 steps ahead: models/sigma_prior_var.pt)
  bc_select_khat    time-conditioned prior proposes a chunk per goal offset (5..100); LeWM rolls
                    each out; k̂ picks the shortest estimated time to the goal
  lewm_khat,        lewm and prior_cem_*_var with every CEM candidate scored by that k̂ time to
  prior_cem_*_var_khat  the goal instead of the terminal latent distance (ua/step1/prior.KhatCost)
  bc_select_khat_m1         bc_select_khat on the Gaussian time-conditioned prior (models/
                            sigma_prior_tc.pt): its mean per k, the M = 1 reference
  bc_select_khat_sp_m4      + 3 draws per k from N(mu_k, sigma_k^2): 32 proposals, still a weak selection
  bc_select_khat_sconst_m4  the same with sigma = the prior's mean sigma_k (a constant)
  bc_select_khat_sp_m16     + 15 draws per k (128 proposals): does a stronger selection over-exploit k̂?
  bc_select_khat_m1_ens,    m1 / sp_m4 with lambda x the Step-1 ensemble's latent sigma^2 added to every
  bc_select_khat_sp_m4_ens  proposal's time to goal; lambda = 1 unit of models/ens_scale.json (outcome-free:
  *_ens3                    median T / median sigma^2 on held-out cached states), or 3 units for _ens3
  Tuning (seeds 100-104, out_dir results/cube/compare_tune; see scripts/submit_cube_tune.sh):
  prior_cem_sp_var_a<alpha>     CEM from N(mu_p, (alpha sigma_p)^2), alpha 0.1 / 0.25 / 0.5 (a01 / a025 / a05)
  prior_cem_sconst_var_a01      its control at the chosen alpha = 0.1: the constant width (mean sigma_p) x 0.1
  bc_select_khat_sp_m4_ens03    sp_m4 with lambda = 0.3 units
  calibrate         MC-5 planning on compare.cal_seeds, recording planning-time sigma^2 per
                    step -> the thresholds of gated (p50, step 0) and gated_tc (quantile, per step)

Writes <compare.out_dir>/k<offset>/<run>/seed_<s>.json (existing seeds are skipped) and
k<offset>/calibration/; summarize with scripts/cube_compare_summary.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import build_world, evaluate_policy, pick_device, seed_everything  # noqa: E402
from planners import build_planner, load_base  # noqa: E402
from ua.calibrate import flatten_sigma2_perstep  # noqa: E402
from ua.step1.ensemble import load_members  # noqa: E402
from ua.step1.prior import KhatCost, PriorInitCEMSolver, PriorSelectSolver, PriorSolver, load_prior  # noqa: E402
from ua.step1.protocol import sample_nontrivial_start_states  # noqa: E402
from ua.step1.steps_to_goal import load_steps_to_goal  # noqa: E402

UA_METHODS = ("mc5", "gated", "gated_tc", "goal_cvar")  # config/eval/method/<name>.yaml on the MC-5 planner
PRIOR_CEM = {"prior_cem_s1": None, "prior_cem_sconst": "const", "prior_cem_sp": "prior"}  # -> init std
PRIOR_PLANNERS = ("bc_only", *PRIOR_CEM)  # on the Gaussian prior; "<name>_var" uses sigma_prior_var
KHAT_COST = ("lewm_khat", *(f"{p}_var_khat" for p in PRIOR_CEM))  # CEM scored by k̂ (KhatCost)
SELECT_SAMPLED = {  # bc_select_khat on sigma_prior_tc: (width of the draws, proposals per k, ensemble penalty units)
    "bc_select_khat_m1": (None, 1, 0), "bc_select_khat_sp_m4": ("prior", 4, 0),
    "bc_select_khat_sconst_m4": ("const", 4, 0), "bc_select_khat_sp_m16": ("prior", 16, 0),
    "bc_select_khat_m1_ens": (None, 1, 1), "bc_select_khat_sp_m4_ens": ("prior", 4, 1),
    "bc_select_khat_m1_ens3": (None, 1, 3), "bc_select_khat_sp_m4_ens3": ("prior", 4, 3),
    "bc_select_khat_sp_m4_ens03": ("prior", 4, 0.3),
}
SCALED_SP = {  # run -> (planner whose CEM width is scaled, alpha)
    "prior_cem_sp_var_a01": ("prior_cem_sp_var", 0.1), "prior_cem_sp_var_a025": ("prior_cem_sp_var", 0.25),
    "prior_cem_sp_var_a05": ("prior_cem_sp_var", 0.5), "prior_cem_sconst_var_a01": ("prior_cem_sconst_var", 0.1),
}
PLANNERS = ("lewm", *UA_METHODS, *PRIOR_PLANNERS, *(f"{p}_var" for p in PRIOR_PLANNERS), "bc_select_khat",
            *KHAT_COST, *SELECT_SAMPLED, *SCALED_SP, "calibrate")
GATED_PERCENTILE = 50.0  # the original Gated recipe: threshold = p50 of planning-time step-0 sigma^2
TYPES = ("carry", "pick", "place", "move")  # cube held at the start / at the goal: yes-yes, no-yes, yes-no, no-no
CONTACT_COL = "proprio_gripper_contact"


def task_types(contact: np.ndarray, rows: np.ndarray, offset: int) -> list[str]:
    names = {(True, True): "carry", (False, True): "pick", (True, False): "place", (False, False): "move"}
    return [names[(bool(c0), bool(cg))] for c0, cg in zip(contact[rows] > 0.5, contact[rows + offset] > 0.5)]


def cached_episodes(cfg) -> list[int]:
    """Every episode a learned component could have been trained on (the latent cache)."""
    split = ROOT / cfg.compare.models_dir / "split.json"
    if split.exists():
        s = json.loads(split.read_text())
        return [int(e) for e in s["train_episodes"] + s["val_episodes"]]
    path = ROOT / cfg.compare.latents
    if not path.exists():
        raise FileNotFoundError(f"neither {split} nor {path}: run scripts/cube_train.py first")
    return [int(e) for e in torch.load(path, weights_only=False, map_location="cpu")["episodes"]]


def model_file(cfg, name: str) -> Path:
    """Checkpoint of a learned component (the same rule as scripts/cube_train.py)."""
    return ROOT / cfg.compare.models_dir / str(cfg.compare.train[name].get("file", f"{name}.pt"))


def method_cfg(name: str) -> DictConfig:
    return OmegaConf.load(ROOT / "config" / "eval" / "method" / f"{name}.yaml")


def load_calibration(kdir: Path) -> np.ndarray:
    path = kdir / "calibration" / "sigma2_perstep.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/cube_compare.py +planner=calibrate first")
    return np.load(path)


def build(planner: str, cfg, device: str, kdir: Path):
    """Cost model, run name, solver factory (seed -> solver) and the settings to record."""
    cem_args = dict(batch_size=1, num_samples=int(cfg.solver.num_samples), var_scale=float(cfg.solver.var_scale),
                    n_steps=int(cfg.solver.n_steps), topk=int(cfg.solver.topk), device=device)
    settings = {"cem": {k: v for k, v in cem_args.items() if k != "device"}}
    run = str(cfg.get("run_name", planner))

    if planner == "lewm":
        model = load_base(cfg.policy, device)
        return model, run, lambda seed: swm.solver.CEMSolver(model=model, seed=seed, **cem_args), settings

    if planner in UA_METHODS or planner == "calibrate":
        mcfg = method_cfg("mc5" if planner == "calibrate" else planner)
        if planner == "gated":
            sig = load_calibration(kdir)
            mcfg.truncation.threshold = float(np.percentile(sig[:, 0], GATED_PERCENTILE))
        elif planner == "gated_tc":
            q = float(cfg.get("quantile", mcfg.quantile))
            w = float(cfg.get("terminal_weight", mcfg.truncation.terminal_weight))
            if "run_name" not in cfg and (q, w) != (float(mcfg.quantile), float(mcfg.truncation.terminal_weight)):
                run = f"gated_tc-q{q:g}-w{w:g}"
            sig = load_calibration(kdir)
            mcfg.truncation.threshold = np.nanpercentile(sig, 100 * q, axis=0).tolist()
            mcfg.truncation.terminal_weight = w
            settings["quantile"] = q
        cfg.method = mcfg
        model = build_planner(cfg, device)
        settings["method"] = OmegaConf.to_container(mcfg, resolve=True)
        return model, run, lambda seed: swm.solver.CEMSolver(model=model, seed=seed, **cem_args), settings

    base = load_base(cfg.policy, device)
    if planner == "bc_select_khat":
        prior = load_prior(model_file(cfg, "prior_tc"), device)
        khat = load_steps_to_goal(model_file(cfg, "khat"), device)
        return base, run, lambda seed: PriorSelectSolver(prior, base, khat), settings
    if planner in SELECT_SAMPLED:
        init_std, m, units = SELECT_SAMPLED[planner]
        ckpt = model_file(cfg, "sigma_prior_tc")
        prior, khat = load_prior(ckpt, device), load_steps_to_goal(model_file(cfg, "khat"), device)
        if init_std == "const":
            init_std = float(torch.load(ckpt, weights_only=False, map_location="cpu")["sigma_mean"])
        members, penalty = None, 0.0
        if units:
            scale = json.loads((ROOT / cfg.compare.models_dir / "ens_scale.json").read_text())
            penalty = units * scale["unit"]
            members = load_members(ROOT / cfg.compare.ensemble_dir, ROOT / "checkpoints" / cfg.policy / "config.json",
                                   device)
        settings.update(prior=ckpt.name, samples=m, init_std=init_std, penalty=penalty)
        return base, run, (lambda seed: PriorSelectSolver(prior, base, khat, samples=m, init_std=init_std, seed=seed,
                                                          members=members, penalty=penalty)), settings
    if planner in KHAT_COST:  # the same planner with k̂'s time to the goal as the CEM cost
        base = KhatCost(base, load_steps_to_goal(model_file(cfg, "khat"), device))
        planner = planner[: -len("_khat")]
        settings["cost"] = "khat time to goal"
        if planner == "lewm":
            return base, run, lambda seed: swm.solver.CEMSolver(model=base, seed=seed, **cem_args), settings
    std_scale = 1.0
    if planner in SCALED_SP:  # a prior_cem_* planner whose CEM width is scaled by alpha
        planner, std_scale = SCALED_SP[planner]
        settings["std_scale"] = std_scale
    name, component = (planner[:-4], "sigma_prior_var") if planner.endswith("_var") else (planner, "sigma_prior")
    ckpt = model_file(cfg, component)
    prior = load_prior(ckpt, device)
    settings["prior"] = ckpt.name
    if name == "bc_only":
        return base, run, lambda seed: PriorSolver(prior, base), settings
    init_std = PRIOR_CEM[name]
    if init_std == "const":
        init_std = float(torch.load(ckpt, weights_only=False, map_location="cpu")["sigma_mean"])
    settings["init_std"] = "cem default (var_scale)" if init_std is None else init_std
    return base, run, (lambda seed: PriorInitCEMSolver(model=base, seed=seed, prior=prior, init_std=init_std,
                                                       std_scale=std_scale, **cem_args)), settings


@hydra.main(version_base=None, config_path="../config/eval", config_name="compare")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    planner = str(cfg.planner)
    assert planner in PLANNERS, f"unknown planner {planner!r}; choose from {PLANNERS}"
    calibrating = planner == "calibrate"
    offset = int(cfg.compare.offset)
    cfg.eval.goal_offset_steps = offset
    cfg.eval.eval_budget = 2 * offset
    kdir = ROOT / cfg.compare.out_dir / f"k{offset}"
    seeds = [int(s) for s in cfg.get("seeds", cfg.compare.cal_seeds if calibrating else cfg.compare.seeds)]
    device = pick_device()

    model, run, make_solver, settings = build(planner, cfg, device, kdir)
    out = kdir / ("calibration" if calibrating else run)
    if calibrating and (out / "sigma2_perstep.npy").exists():
        print(f"[compare] calibration exists: {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    todo = seeds if calibrating else [s for s in seeds if not (out / f"seed_{s}.json").exists()]
    if not todo:
        print(f"[compare k={offset} {run}] all {len(seeds)} seeds done", flush=True)
        return

    world, dataset, transform, process = build_world(cfg)
    cached = cached_episodes(cfg)
    contact = np.asarray(dataset.get_col_data(CONTACT_COL)).reshape(-1)
    for seed in todo:
        seed_everything(seed)
        eps, starts, rows, init_dist = sample_nontrivial_start_states(
            dataset, int(cfg.eval.num_eval), offset, seed, float(cfg.compare.min_init_dist),
            exclude_episodes=cached)
        if hasattr(model, "reset_diagnostics") and not calibrating:
            model.reset_diagnostics()
        solver = make_solver(seed)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                             process=process, transform=transform)
        t0 = time.time()
        succ = np.asarray(evaluate_policy(cfg, world, dataset, policy, eps, starts)["episode_successes"], bool)
        print(f"[compare k={offset} {run}] seed {seed}: {succ.mean():.0%} ({time.time() - t0:.0f}s)", flush=True)
        if calibrating:
            continue
        diag = {}
        for key, values in (("mean_scored_blocks", getattr(model, "truncation_lengths", None)),
                            ("truncated_frac", getattr(model, "truncated_frac", None)),
                            ("mean_sigma2", getattr(model, "mean_sigma2", None))):
            if values:
                diag[key] = float(np.mean(values))
        if hasattr(model, "diagnostics_summary"):  # Goal-CVaR risk terms
            diag.update(model.diagnostics_summary())
        if getattr(solver, "chosen_k", None):
            diag["chosen_k"] = solver.chosen_k
        if getattr(solver, "samples", 1) > 1:  # which proposal of the chosen k won (0 = the mean)
            diag["chosen_sample"] = solver.chosen_sample
        if getattr(solver, "penalty", 0):  # how often the ensemble penalty changed the choice
            diag["penalty_changed"] = float(np.mean(solver.penalty_changed))
        (out / f"seed_{seed}.json").write_text(json.dumps({
            "seed": seed, "offset": offset, "budget": 2 * offset, "planner": planner, "run": run,
            "episodes": eps, "start_steps": starts, "rows": np.asarray(rows).tolist(),
            "init_dist": np.asarray(init_dist).tolist(), "types": task_types(contact, np.asarray(rows), offset),
            "episode_successes": succ.astype(int).tolist(), "seconds": time.time() - t0,
            "settings": settings, "diagnostics": diag,
        }))

    if calibrating:
        sig = flatten_sigma2_perstep(model)
        np.save(out / "sigma2_perstep.npy", sig)
        pct = (50, 80, 90, 95)
        (out / "thresholds.json").write_text(json.dumps({
            "seeds": seeds, "n_candidates": int(len(sig)),
            "gated_step0_p50": float(np.percentile(sig[:, 0], GATED_PERCENTILE)),
            "perstep_percentiles": {f"p{p}": np.nanpercentile(sig, p, axis=0).tolist() for p in pct},
        }, indent=1))
        print(f"[compare] calibration: {len(sig)} candidates -> {out}", flush=True)


if __name__ == "__main__":
    main()
