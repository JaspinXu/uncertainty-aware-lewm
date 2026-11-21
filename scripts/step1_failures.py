"""Step 1 / why do the baseline's plans fail? (exploratory, not pre-registered)

    python scripts/step1_failures.py

For the 400 non-trivial Step-1 episodes (seeds 61-68):

1. Exact replays of the executed plans, logging the cube, end-effector,
   gripper closure and gripper contact after every env step.
2. The expert's own 25 steps from the dataset, which reach the goal by
   construction: what kind of task each episode is (carry, pick, place, move).
3. What the model predicted: ridge probes fitted on frozen-encoder latents
   decode cube and end-effector position from each plan's predicted final
   latent, to compare with where they really ended.

Writes results/cube/step1/failures.json and prints the tables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
from collections import Counter  # noqa: E402

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from evaluate import pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.common import out_dir, to_env_actions  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402
from ua.step1.recording import cube_pos, restore  # noqa: E402
from ua.step1.replay import ReplayEnv  # noqa: E402
from ua.step1.rollout import deterministic_rollouts  # noqa: E402

REST_Z = 0.020      # m, cube resting on the table (dataset: 25th-50th percentile of z)
LIFT_Z = 0.030      # above this the cube is off the table
MOVED = 0.02        # m, the cube counts as moved
GOAL_EE = 0.05      # m, end-effector "at the goal" (within this of the goal cube position)


class _Scaler:
    def __init__(self, mean, scale):
        self.mean_, self.scale_ = np.asarray(mean), np.asarray(scale)

    def inverse_transform(self, x):
        x = np.array(x, dtype=np.float32, copy=True)
        x *= self.scale_
        x += self.mean_
        return x


def replay_episode(env, recs, scaler):
    """Replay the executed plans of one episode in order; per step: cube, ee, closure, contact."""
    u = env.u
    out = {"cube": [], "ee": [], "closure": [], "contact": []}

    def log():
        ob = u.compute_ob_info()
        out["cube"].append(cube_pos(u))
        out["ee"].append(np.asarray(ob["proprio/effector_pos"], dtype=np.float64).copy())
        out["closure"].append(float(np.asarray(ob["proprio/gripper_opening"]).ravel()[0]))
        out["contact"].append(float(np.asarray(ob["proprio/gripper_contact"]).ravel()[0]))

    for k, r in enumerate(sorted(recs, key=lambda r: r["t"])):
        restore(u, r["snapshot"])  # identical to the state the previous plan reached (exact replays)
        if k == 0:
            log()
        for a in to_env_actions(r["plan"][None], scaler)[0]:
            u.step(np.asarray(a))
            log()
    return {k: np.asarray(v) for k, v in out.items()}


def task_type(c0: bool, cg: bool) -> str:
    return {(True, True): "carry", (False, True): "pick", (True, False): "place", (False, False): "move"}[(c0, cg)]


def failure_mode(tr: dict, goal: np.ndarray, start_in_hand: bool) -> str:
    """Primary physical reason a failed episode failed."""
    contact = tr["contact"] > 0.5
    disp = np.linalg.norm(tr["cube"] - tr["cube"][0], axis=1).max()
    held_ever = start_in_hand or contact.any()
    if contact[-1]:
        return "held, wrong place"
    if held_ever:
        return "dropped / released"
    if disp < MOVED:
        return "never touched the cube"
    return "pushed, never grasped"


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    seeds = [int(s) for s in cfg.step1.seeds]
    logs = {s: torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in seeds}
    base = load_base(cfg.policy, device)

    h5 = h5py.File(ROOT / "datasets" / f"{cfg.eval.dataset_name}.h5", "r")
    cube_col = h5["privileged_block_0_pos"][:]
    ee_col = h5["proprio_effector_pos"][:]
    contact_col = h5["proprio_gripper_contact"][:, 0] > 0.5
    ep_off = h5["ep_offset"][:]

    # ---- probes: frozen latent -> cube / end-effector position ----
    cache = torch.load(out / "ensemble" / "latents.pt", weights_only=False)
    rows = ep_off[cache["ep"]] + cache["step"]
    Z = cache["emb"].astype(np.float32)
    rng = np.random.default_rng(0)
    n_val = int(cfg.step1.ensemble.val_episodes)
    val = np.isin(cache["ep"], cache["episodes"][-n_val:])
    tr_idx = rng.choice(np.nonzero(~val)[0], 200_000, replace=False)
    probes, probe_err = {}, {}
    for name, col in (("cube", cube_col), ("ee", ee_col)):
        probes[name] = Ridge(alpha=1.0).fit(Z[tr_idx], col[rows[tr_idx]])
        err = np.linalg.norm(probes[name].predict(Z[val]) - col[rows[val]], axis=1)
        probe_err[name] = float(100 * np.median(err))
    print(f"[failures] probe median error on held-out frames: cube {probe_err['cube']:.1f} cm, "
          f"end-effector {probe_err['ee']:.1f} cm")

    env = ReplayEnv(OmegaConf.to_container(cfg.world, resolve=True) | {"max_episode_steps": 100})
    episodes, plans = [], []
    for s in seeds:
        log = logs[s]
        scaler = _Scaler(log["action_mean"], log["action_scale"])
        by_env = {}
        for r in log["records"]:
            by_env.setdefault(r["env"], []).append(r)
        for e in range(len(log["success"])):
            recs = sorted(by_env[e], key=lambda r: r["t"])
            tr = replay_episode(env, recs, scaler)
            goal = log["goal_pos"][e]
            dist = np.linalg.norm(tr["cube"] - goal, axis=1)
            succ_step = int(np.argmax(dist <= SUCCESS_RADIUS)) if (dist <= SUCCESS_RADIUS).any() else None
            assert (succ_step is not None) == bool(log["success"][e]), "replay disagrees with the closed loop"
            if succ_step is not None:  # the closed loop stopped there
                tr = {k: v[: succ_step + 1] for k, v in tr.items()}
            row = int(log["rows"][e])
            c0, cg = bool(contact_col[row]), bool(contact_col[row + 25])
            ep = {
                "seed": s, "env": e, "success": bool(log["success"][e]),
                "type": task_type(c0, cg), "start_in_hand": c0,
                "goal_in_air": bool(cube_col[row + 25, 2] > LIFT_Z),
                "init_dist": float(log["init_dist"][e]),
                "final_dist": float(dist[: len(tr["cube"])][-1]), "min_dist": float(dist[: len(tr["cube"])].min()),
                "cube_moved": float(np.linalg.norm(tr["cube"] - tr["cube"][0], axis=1).max()),
                "max_lift": float(tr["cube"][:, 2].max() - REST_Z),
                "contact_any": bool((tr["contact"] > 0.5).any()),
                "ee_final_to_goal": float(np.linalg.norm(tr["ee"][-1] - goal)),
                "expert_grasps": bool(contact_col[row : row + 26].any()),
            }
            if not ep["success"]:
                ep["mode"] = failure_mode(tr, goal, c0)
                ep["arm_at_goal_without_cube"] = bool(ep["ee_final_to_goal"] < GOAL_EE and ep["final_dist"] > SUCCESS_RADIUS)
            episodes.append(ep)
            # model's view of every plan of the episode: decoded final cube / ee vs the real ones
            t_index = 0
            for r in recs:
                end = min(t_index + 25, len(tr["cube"]) - 1) if succ_step is None else min(t_index + 25, succ_step)
                plans.append({"seed": s, "env": e, "t": r["t"], "emb": r["emb"], "plan": r["plan"],
                              "goal_emb": r["goal_emb"], "goal": goal,
                              "real_cube": tr["cube"][end] if end < len(tr["cube"]) else tr["cube"][-1],
                              "real_ee": tr["ee"][end] if end < len(tr["ee"]) else tr["ee"][-1],
                              "episode_success": ep["success"]})
                t_index += 25

    # ---- decode the predicted final latents ----
    emb0 = torch.stack([p["emb"] for p in plans]).to(device)
    acts = torch.stack([p["plan"] for p in plans]).to(device)
    zhat = deterministic_rollouts(base, emb0, acts)[:, -1].cpu().numpy()
    pred_cube = probes["cube"].predict(zhat)
    pred_ee = probes["ee"].predict(zhat)
    goal_z = torch.stack([p["goal_emb"] for p in plans]).numpy()
    goal_cube_dec = probes["cube"].predict(goal_z)
    for p, pc, pe, gc in zip(plans, pred_cube, pred_ee, goal_cube_dec):
        p["pred_cube_to_goal"] = float(np.linalg.norm(pc - p["goal"]))
        p["real_cube_to_goal"] = float(np.linalg.norm(p["real_cube"] - p["goal"]))
        p["pred_cube_err"] = float(np.linalg.norm(pc - p["real_cube"]))
        p["pred_ee_err"] = float(np.linalg.norm(pe - p["real_ee"]))
        p["goal_decode_err"] = float(np.linalg.norm(gc - p["goal"]))

    # ---- tables ----
    res = {"probe_median_err_cm": probe_err}
    n = len(episodes)
    print(f"\n[failures] {n} non-trivial episodes, success {np.mean([e['success'] for e in episodes]):.1%}")
    print("\nSuccess by task type (what the expert does in its 25 steps):")
    res["by_type"] = {}
    for t in ("carry", "pick", "place", "move"):
        sub = [e for e in episodes if e["type"] == t]
        if sub:
            sr = float(np.mean([e["success"] for e in sub]))
            res["by_type"][t] = {"n": len(sub), "success": sr}
            print(f"  {t:6s} n={len(sub):3d}  success {sr:.0%}")
    failed = [e for e in episodes if not e["success"]]
    modes = Counter(e["mode"] for e in failed)
    res["failure_modes"] = {k: v / len(failed) for k, v in modes.items()}
    print(f"\nPhysical failure modes ({len(failed)} failed episodes):")
    for k, v in modes.most_common():
        sub = [e for e in failed if e["mode"] == k]
        types = Counter(e["type"] for e in sub)
        print(f"  {k:24s} {v:3d} ({v / len(failed):.0%})  task types {dict(types)}")
    arm = np.mean([e["arm_at_goal_without_cube"] for e in failed])
    res["arm_at_goal_without_cube"] = float(arm)
    print(f"  end-effector within {100 * GOAL_EE:.0f} cm of the goal while the cube is not: {arm:.0%} of failures")
    res["failed_final_vs_init"] = {
        "median_final_minus_init_cm": float(100 * np.median([e["final_dist"] - e["init_dist"] for e in failed])),
        "frac_cube_moved_lt_2cm": float(np.mean([e["cube_moved"] < MOVED for e in failed])),
    }
    print(f"  failed episodes: median final - initial distance {res['failed_final_vs_init']['median_final_minus_init_cm']:+.1f} cm; "
          f"cube moved < 2 cm in {res['failed_final_vs_init']['frac_cube_moved_lt_2cm']:.0%}")

    first_failed = [p for p in plans if p["t"] == 0 and not p["episode_success"]]
    first_ok = [p for p in plans if p["t"] == 0 and p["episode_success"]]
    halluc = np.mean([p["pred_cube_to_goal"] < 2 * SUCCESS_RADIUS and p["real_cube_to_goal"] > SUCCESS_RADIUS
                      for p in first_failed])
    res["model_view_first_plans"] = {
        "failed": {
            "n": len(first_failed),
            "median_pred_cube_to_goal_cm": float(100 * np.median([p["pred_cube_to_goal"] for p in first_failed])),
            "median_real_cube_to_goal_cm": float(100 * np.median([p["real_cube_to_goal"] for p in first_failed])),
            "median_pred_cube_err_cm": float(100 * np.median([p["pred_cube_err"] for p in first_failed])),
            "median_pred_ee_err_cm": float(100 * np.median([p["pred_ee_err"] for p in first_failed])),
            "frac_model_predicts_cube_within_8cm_of_goal": float(halluc),
        },
        "succeeded": {
            "n": len(first_ok),
            "median_pred_cube_err_cm": float(100 * np.median([p["pred_cube_err"] for p in first_ok])),
            "median_pred_ee_err_cm": float(100 * np.median([p["pred_ee_err"] for p in first_ok])),
        },
        "goal_decode_median_err_cm": float(100 * np.median([p["goal_decode_err"] for p in plans])),
    }
    m = res["model_view_first_plans"]
    print("\nWhat the model predicted for the first plan (probe-decoded final latent):")
    print(f"  failed: cube predicted {m['failed']['median_pred_cube_to_goal_cm']:.1f} cm from the goal, really "
          f"{m['failed']['median_real_cube_to_goal_cm']:.1f} cm; cube error {m['failed']['median_pred_cube_err_cm']:.1f} cm, "
          f"end-effector error {m['failed']['median_pred_ee_err_cm']:.1f} cm; model put the cube within 8 cm of the goal "
          f"in {m['failed']['frac_model_predicts_cube_within_8cm_of_goal']:.0%}")
    print(f"  succeeded: cube error {m['succeeded']['median_pred_cube_err_cm']:.1f} cm, end-effector error "
          f"{m['succeeded']['median_pred_ee_err_cm']:.1f} cm (goal frames decode to {m['goal_decode_median_err_cm']:.1f} cm)")
    (out / "failures.json").write_text(json.dumps({"summary": res, "episodes": episodes}, indent=1, default=float))
    print(f"\n[failures] wrote {out / 'failures.json'}")


if __name__ == "__main__":
    main()
