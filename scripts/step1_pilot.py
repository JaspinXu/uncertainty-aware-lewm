"""Step 1 / closed-loop pilots of fixes for the failure modes (exploratory).

    python scripts/step1_pilot.py +pilot=hold_goal         # keep squeezing when the goal shows the cube held
    python scripts/step1_pilot.py +pilot=rh1               # replan after every action block
    python scripts/step1_pilot.py +pilot=hold_goal_rh1     # both
    python scripts/step1_pilot.py +pilot=hold_naive        # keep squeezing always (breaks place tasks)
    python scripts/step1_pilot.py +pilot=bc_init           # CEM started at the BC prior's proposal
    python scripts/step1_pilot.py +pilot=bc_only           # the BC prior alone (no world-model scoring)
    python scripts/step1_pilot.py +pilot=bc_only_rh1       # ... re-queried after every action block
    python scripts/step1_pilot.py +pilot=bc_only_rh1 +prior=prior_var   # another prior checkpoint

Same non-trivial start states and CEM budget as the Step-1 baseline (seeds 61-68),
so every episode is paired with its baseline outcome.

hold: OGBench gripper actions are relative (target closure = current closure +
command). A held cube slips out unless the command keeps squeezing, which the
planner's noisy commands do not (scripts/step1_failures.py). While the gripper
closure says a cube is held (proprioception, read from the simulator), the
executed gripper command is raised to at least +0.4 (the expert's median while
holding is +0.43). The planner itself is unchanged. The expert releases with a
command near 0, so squeezing always also stops every deliberate release
(hold_naive); hold_goal only squeezes when the goal image shows the cube held,
decided by a logistic probe on the frozen goal latent (results/cube/step1/
probe_held.npz, 97.5% held-out accuracy).

bc_*: the goal-conditioned behaviour-cloning prior of scripts/step1_prior.py
(results/cube/step1/prior/<prior>.pt, default prior.pt, trained without the
Step-1 episodes); runs with another prior go to pilots/<pilot>-<prior>.
bc_init keeps the world-model scoring and only changes where the search starts,
at every replan; bc_only executes the prior's proposal directly. The full
method comparison (including bc_select_khat) is scripts/cube_compare.py.

Writes results/cube/step1/pilots/<pilot>/seed_<s>.json and summary.md, and
pilots/overview.md (every pilot by task type, paired with the baseline).
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

PILOTS = {  # name -> (hold rule: None | "always" | "goal", receding horizon in blocks)
    "hold_naive": ("always", 5),
    "hold_goal": ("goal", 5),
    "rh1": (None, 1),
    "hold_goal_rh1": ("goal", 1),
    # diagnostic (privileged): the first CEM call starts from the expert's own 25
    # actions, as a perfect behaviour-cloning prior would; later calls are unchanged
    "expert_init": (None, 5),
    # behaviour-cloning prior (learned, no privileged information at test time)
    "bc_init": (None, 5),
    "bc_only": (None, 5),
    "bc_only_rh1": (None, 1),
}
TYPES = ("carry", "pick", "place")  # task type of each start state (scripts/step1_failures.py)


class ExpertInitPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy whose first CEM call is initialised at the expert's actions."""

    def __init__(self, *args, expert_init=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_init, self.calls = expert_init, 0

    def get_action(self, info_dict, **kwargs):
        if self.calls == 0:
            self._next_init = self.expert_init.clone()
        action = super().get_action(info_dict, **kwargs)
        if self.calls == 0:
            self._next_init = None  # later replans start from scratch, as in the baseline
        self.calls += 1
        return action


HOLD_CLOSURE = 0.53   # gripper closure above this = a cube is held (dataset: 0.545-0.562 when holding)
HOLD_COMMAND = 0.4    # minimum gripper command while holding
GRIP = 4              # gripper dimension of the 5-d OGBench action


class HoldPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that keeps squeezing a held cube (always, or when the goal shows it held)."""

    def __init__(self, *args, mode="always", probe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode, self.probe, self.keep = mode, probe, None

    def _goal_shows_held(self, info_dict) -> np.ndarray:
        goal = self._prepare_info({"goal": info_dict["goal"]})["goal"]  # (n, 1, 3, H, W)
        model = self.solver.model
        dev = next(model.parameters()).device
        with torch.no_grad():
            z = model.encode({"pixels": goal.to(dev)})["emb"][:, -1].float().cpu().numpy()
        return z @ self.probe["w"] + self.probe["b"] > 0

    def get_action(self, info_dict, **kwargs):
        if self.keep is None:
            n = self.env.num_envs
            self.keep = np.ones(n, bool) if self.mode == "always" else self._goal_shows_held(info_dict)
        action = super().get_action(info_dict, **kwargs)
        for i, env in enumerate(self.env.envs):
            u = env.unwrapped
            closure = float(np.clip(u._data.qpos[u._gripper_opening_joint_id] / 0.8, 0, 1))
            if self.keep[i] and closure > HOLD_CLOSURE and np.isfinite(action[i, GRIP]):
                action[i, GRIP] = max(action[i, GRIP], HOLD_COMMAND)
        return action


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    name = str(cfg.pilot)
    hold, rh = PILOTS[name]
    cfg.plan_config.receding_horizon = rh
    device = pick_device()
    prior_name = str(cfg.get("prior", "prior"))
    run = f"{name}-{prior_name}" if name.startswith("bc_") and prior_name != "prior" else name
    out = out_dir(cfg, ROOT) / "pilots" / run
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg.step1.seeds]
    world, dataset, transform, process = build_world(cfg)
    base = load_base(cfg.policy, device)
    prior = load_prior(out_dir(cfg, ROOT) / "prior" / f"{prior_name}.pt", device) if name.startswith("bc_") else None
    for seed in seeds:
        path = out / f"seed_{seed}.json"
        if path.exists():
            continue
        seed_everything(seed)
        eps, starts, rows, _ = sample_nontrivial_start_states(
            dataset, int(cfg.eval.num_eval), int(cfg.eval.goal_offset_steps), seed, float(cfg.step1.min_init_dist))
        cem = dict(model=base, batch_size=1, num_samples=int(cfg.solver.num_samples),
                   var_scale=float(cfg.solver.var_scale), n_steps=int(cfg.solver.n_steps),
                   topk=int(cfg.solver.topk), device=device, seed=seed)
        if name == "bc_init":
            solver = PriorInitCEMSolver(**cem, prior=prior)
        elif name.startswith("bc_only"):
            solver = PriorSolver(prior, base)
        else:
            solver = swm.solver.CEMSolver(**cem)
        common = dict(solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform)
        if name == "expert_init":
            scaler = process["action"]
            steps = int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block)
            raw = np.stack([dataset.get_row_data(np.arange(r, r + steps))["action"] for r in rows])  # (n, 25, 5)
            norm = scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(len(rows), int(cfg.plan_config.horizon), -1)
            policy = ExpertInitPolicy(**common, expert_init=torch.as_tensor(norm, dtype=torch.float32))
        elif hold is None:
            policy = swm.policy.WorldModelPolicy(**common)
        else:
            probe = dict(np.load(out_dir(cfg, ROOT) / "probe_held.npz")) if hold == "goal" else None
            policy = HoldPolicy(**common, mode=hold, probe=probe)
        t0 = time.time()
        metrics = evaluate_policy(cfg, world, dataset, policy, eps, starts)
        succ = np.asarray(metrics["episode_successes"], dtype=bool)
        path.write_text(json.dumps({"seed": seed, "pilot": name, "episodes": eps, "start_steps": starts,
                                    "episode_successes": succ.astype(int).tolist(), "seconds": time.time() - t0}))
        print(f"[pilot {name}] seed {seed}: {succ.mean():.0%} ({time.time() - t0:.0f}s)", flush=True)

    failures = json.loads((out_dir(cfg, ROOT) / "failures.json").read_text())["episodes"]
    types = {(e["seed"], e["env"]): e["type"] for e in failures}
    res = paired(out, seeds, out_dir(cfg, ROOT) / "closed_loop", types)
    if res is None:
        return
    a, b, t = res
    md = (f"# Pilot `{out.name}` (hold rule: {hold}, replan every {rh} block(s))\n\n"
          f"{len(a)} non-trivial episodes, paired with the Step-1 baseline on the same start states.\n\n"
          + table([(out.name, a, b, t)]))
    (out / "summary.md").write_text(md)
    print(md, flush=True)

    # every pilot with all seeds done
    rows = []
    for d in sorted(p for p in out.parent.iterdir() if p.is_dir()):
        r = paired(d, seeds, out_dir(cfg, ROOT) / "closed_loop", types, complete=True)
        if r is not None:
            rows.append((d.name, *r))
    if rows:
        (out.parent / "overview.md").write_text(
            f"# Step-1 pilots, paired with the baseline (seeds {seeds[0]}-{seeds[-1]}, complete runs only)\n\n"
            "Success rate overall and by task type; gained / lost episodes and exact McNemar p against "
            "the baseline on the same start states.\n\n" + table(rows))


def paired(pilot_dir: Path, seeds, baseline_dir: Path, types: dict, complete: bool = False):
    """Baseline and pilot successes on the same start states, with each episode's task type."""
    a_all, b_all, t_all = [], [], []
    for seed in seeds:
        p = pilot_dir / f"seed_{seed}.json"
        if not p.exists():
            if complete:
                return None
            continue
        run = json.loads(p.read_text())
        base_log = torch.load(baseline_dir / f"seed_{seed}.pt", weights_only=False)
        assert list(base_log["episodes"]) == run["episodes"], "start states differ from the baseline"
        a_all.append(np.asarray(base_log["success"], bool))
        b_all.append(np.asarray(run["episode_successes"], bool))
        t_all.append(np.array([types[(seed, i)] for i in range(len(run["episodes"]))]))
    if not a_all:
        return None
    return np.concatenate(a_all), np.concatenate(b_all), np.concatenate(t_all)


def table(rows) -> str:
    """Markdown table, baseline first, then one row per (name, baseline, pilot, types)."""

    def line(label, x, t):
        return f"| {label} | {len(x)} | {x.mean():.1%} | " + " | ".join(f"{x[t == k].mean():.0%}" for k in TYPES)

    _, a, _, t = rows[0]
    out = "| | n | success | " + " | ".join(TYPES) + " | Δ (pp) | gained / lost | McNemar p |\n"
    out += "|---|---:|---:|" + "---:|" * len(TYPES) + "---:|---:|---:|\n"
    out += line("baseline (CEM, replan every 5 blocks)", a, t) + " | | | |\n"
    for name, a, b, t in rows:
        g, l_, p = mcnemar(a, b)
        out += line(f"`{name}`", b, t) + f" | {100 * (b.mean() - a.mean()):+.1f} | {g} / {l_} | {p:.2g} |\n"
    counts = ", ".join(f"{k} {int((t == k).sum())}" for k in TYPES)
    return out + f"\nTask types (n = {len(t)}): {counts}.\n"



if __name__ == "__main__":
    main()
