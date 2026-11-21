"""Step 1 / long-goal protocol: is the BC prior still enough when the goal is far away? (exploratory)

    python scripts/step1_longgoal.py +offset=50 +planner=cem
    python scripts/step1_longgoal.py +offset=100 +planner=bc_only +prior=prior_var
    python scripts/step1_longgoal.py +offset=50 +planner=bc_init "+seeds=[61,62]"

The Step-1 evaluation (non-trivial start states; success = the cube within
4 cm of the goal's cube position, which ends the episode; CEM 300/10/30,
horizon 5) with the goal ``offset`` expert steps ahead instead of 25 and a
budget of 2 x offset steps. Start states come only from dataset episodes
outside the latent cache (results/cube/step1/ensemble/latents.pt), which no
prior has been trained on. For a seed every planner sees the same start
states, so runs are paired.

Planners (receding horizon in 5-step blocks):
  cem          the LeWM planner (Step-1 baseline), replans every 5 blocks
  bc_only      the prior's 25-step chunk, queried again every 5 blocks
  bc_only_rh1  the prior's first block, queried again after every block
  bc_init      CEM started at the prior's proposal at every replan

The full method comparison at any goal offset (including bc_select_khat) is
scripts/cube_compare.py.

Writes results/cube/step1/longgoal/k<offset>/<run>/seed_<s>.json (run = cem,
or <planner>-<prior>) and longgoal/overview.md: every run overall and by task
type, paired with cem on the same start states.
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
from planners import load_base  # noqa: E402
from ua.step1.common import out_dir  # noqa: E402
from ua.step1.prior import PriorInitCEMSolver, PriorSolver, load_prior  # noqa: E402
from ua.step1.protocol import sample_nontrivial_start_states  # noqa: E402
from ua.step1.stats import mcnemar  # noqa: E402

PLANNERS = {"cem": 5, "bc_only": 5, "bc_only_rh1": 1, "bc_init": 5}  # planner -> receding horizon (blocks)
TYPES = ("carry", "pick", "place", "move")  # cube held at the start / at the goal: yes-yes, no-yes, yes-no, no-no
CONTACT_COL = "proprio_gripper_contact"


def task_types(contact: np.ndarray, rows: np.ndarray, offset: int) -> list[str]:
    names = {(True, True): "carry", (False, True): "pick", (True, False): "place", (False, False): "move"}
    return [names[(bool(c0), bool(cg))] for c0, cg in zip(contact[rows] > 0.5, contact[rows + offset] > 0.5)]


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    offset, planner = int(cfg.offset), str(cfg.planner)
    prior_name = str(cfg.get("prior", "prior_var"))
    cfg.eval.goal_offset_steps = offset
    cfg.eval.eval_budget = 2 * offset
    cfg.plan_config.receding_horizon = PLANNERS[planner]
    seeds = [int(s) for s in cfg.get("seeds", list(cfg.step1.seeds)[:4])]
    root = out_dir(cfg, ROOT) / "longgoal"
    out = root / f"k{offset}" / (planner if planner == "cem" else f"{planner}-{prior_name}")
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    world, dataset, transform, process = build_world(cfg)
    base = load_base(cfg.policy, device)
    prior = None if planner == "cem" else load_prior(out_dir(cfg, ROOT) / "prior" / f"{prior_name}.pt", device)
    cache = torch.load(out_dir(cfg, ROOT) / "ensemble" / "latents.pt", weights_only=False)
    cached = [int(e) for e in cache["episodes"]]  # every episode any prior could have been trained on
    del cache
    contact = np.asarray(dataset.get_col_data(CONTACT_COL)).reshape(-1)
    for seed in seeds:
        path = out / f"seed_{seed}.json"
        if path.exists():
            continue
        seed_everything(seed)
        eps, starts, rows, init_dist = sample_nontrivial_start_states(
            dataset, int(cfg.eval.num_eval), offset, seed, float(cfg.step1.min_init_dist), exclude_episodes=cached)
        cem = dict(model=base, batch_size=1, num_samples=int(cfg.solver.num_samples),
                   var_scale=float(cfg.solver.var_scale), n_steps=int(cfg.solver.n_steps),
                   topk=int(cfg.solver.topk), device=device, seed=seed)
        if planner == "cem":
            solver = swm.solver.CEMSolver(**cem)
        elif planner == "bc_init":
            solver = PriorInitCEMSolver(**cem, prior=prior)
        else:
            solver = PriorSolver(prior, base)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                             process=process, transform=transform)
        t0 = time.time()
        metrics = evaluate_policy(cfg, world, dataset, policy, eps, starts)
        succ = np.asarray(metrics["episode_successes"], dtype=bool)
        path.write_text(json.dumps({
            "seed": seed, "offset": offset, "budget": 2 * offset, "planner": planner,
            "prior": None if planner == "cem" else prior_name,
            "episodes": eps, "start_steps": starts, "rows": np.asarray(rows).tolist(),
            "init_dist": np.asarray(init_dist).tolist(), "types": task_types(contact, np.asarray(rows), offset),
            "episode_successes": succ.astype(int).tolist(), "seconds": time.time() - t0,
        }))
        print(f"[longgoal k={offset} {out.name}] seed {seed}: {succ.mean():.0%} ({time.time() - t0:.0f}s)", flush=True)

    md = overview(root)
    (root / "overview.md").write_text(md)
    print(md, flush=True)


def load_runs(run_dir: Path) -> dict[int, dict]:
    return {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in run_dir.glob("seed_*.json")}


def overview(root: Path) -> str:
    """Every run of every offset, paired with cem on the seeds both have."""
    md = ("# Long-goal protocol (exploratory)\n\n"
          "Goal k expert steps ahead, budget 2k steps; non-trivial start states from episodes no prior was "
          "trained on. Task type = cube held at the start / at the goal (carry yes/yes, pick no/yes, "
          "place yes/no, move no/no). Δ, gained / lost and exact McNemar p are against `cem` on the same "
          "start states.\n")
    for kdir in sorted(root.glob("k*"), key=lambda p: int(p.name[1:])):
        runs = {d.name: r for d in sorted(kdir.iterdir()) if d.is_dir() and (r := load_runs(d))}
        if not runs:
            continue
        k, ref = int(kdir.name[1:]), runs.get("cem", {})
        md += (f"\n## Goal {k} steps ahead, budget {2 * k} steps\n\n| run | seeds | n | success | "
               + " | ".join(TYPES) + " | Δ vs cem (pp) | gained / lost | McNemar p |\n"
               + "|---|---|---:|---:|" + "---:|" * len(TYPES) + "---:|---:|---:|\n")
        for name in sorted(runs, key=lambda n: (n != "cem", n)):
            r = runs[name]
            seeds = sorted(set(r) & set(ref)) if name != "cem" and set(r) & set(ref) else sorted(r)
            succ = np.concatenate([np.asarray(r[s]["episode_successes"], bool) for s in seeds])
            types = np.concatenate([np.asarray(r[s]["types"]) for s in seeds])
            cells = [f"{succ[types == t].mean():.0%}" if (types == t).any() else "–" for t in TYPES]
            paired = "| |"
            if name != "cem" and set(seeds) <= set(ref):
                for s in seeds:
                    assert r[s]["episodes"] == ref[s]["episodes"] and r[s]["start_steps"] == ref[s]["start_steps"]
                a = np.concatenate([np.asarray(ref[s]["episode_successes"], bool) for s in seeds])
                g, l_, p = mcnemar(a, succ)
                paired = f"{100 * (succ.mean() - a.mean()):+.1f} | {g} / {l_} | {p:.2g}"
            md += (f"| `{name}` | {','.join(map(str, seeds))} | {len(succ)} | {succ.mean():.1%} | "
                   + " | ".join(cells) + f" | {paired} |\n")
        r = next(iter(runs.values()))
        types = np.concatenate([np.asarray(v["types"]) for v in r.values()])
        md += f"\nTask types over the {len(r)} seed(s) of `{next(iter(runs))}`: " + ", ".join(
            f"{t} {int((types == t).sum())}" for t in TYPES) + ".\n"
    return md


if __name__ == "__main__":
    main()
