"""Step 1 / data collection on Cube (docs/step1_preregistration.md).

    python scripts/step1_collect.py                          # all stages
    python scripts/step1_collect.py step1.stage=closed_loop  # one stage

Stages (each resumable; outputs under results/cube/step1/):

1. ``closed_loop``  baseline LeWM on 50 non-trivial episodes per seed, logging
   every replan (MuJoCo state, embeddings, CEM populations of iterations 1 and
   10, executed plan) and the per-step cube positions.
2. ``replay_plans`` replay every executed plan from its saved state, rendering
   after each action block -> real embeddings z_1..z_5.
3. ``replay_candidates`` pick 200 replan states (100 from failed, 100 from
   successful episodes) and replay all 600 logged candidates of each ->
   real terminal embedding z_5 and the cube trajectory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import build_world, evaluate_policy, pick_device, seed_everything  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.common import encode_frames, out_dir, to_env_actions  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS, sample_nontrivial_start_states  # noqa: E402
from ua.step1.recording import LoggingCEMSolver, LoggingPolicy  # noqa: E402
from ua.step1.replay import replay_many  # noqa: E402

BLOCK = 5  # env steps per action block


def closed_loop(cfg, out: Path, device: str) -> None:
    todo = [s for s in cfg.step1.seeds if not (out / "closed_loop" / f"seed_{s}.pt").exists()]
    if not todo:
        print("[closed_loop] all seeds done")
        return
    (out / "closed_loop").mkdir(exist_ok=True)
    world, dataset, transform, process = build_world(cfg)
    base = load_base(cfg.policy, device)
    for seed in todo:
        seed_everything(int(seed))
        eps, starts, rows, init_dist = sample_nontrivial_start_states(
            dataset, int(cfg.eval.num_eval), int(cfg.eval.goal_offset_steps), int(seed),
            float(cfg.step1.min_init_dist),
        )
        solver = LoggingCEMSolver(
            model=base, batch_size=1, num_samples=int(cfg.solver.num_samples),
            var_scale=float(cfg.solver.var_scale), n_steps=int(cfg.solver.n_steps),
            topk=int(cfg.solver.topk), device=device, seed=int(seed),
            record_steps=tuple(cfg.step1.record_iters),
        )
        policy = LoggingPolicy(
            solver=solver, config=swm.PlanConfig(**cfg.plan_config),
            process=process, transform=transform,
        )
        t0 = time.time()
        metrics = evaluate_policy(cfg, world, dataset, policy, eps, starts)
        log = policy.finish()
        succ = np.asarray(metrics["episode_successes"], dtype=bool)
        # sanity: the env goal is the dataset cube position 25 steps later
        goal_ds = np.asarray(dataset.get_row_data(rows + int(cfg.eval.goal_offset_steps))["privileged_block_0_pos"])
        assert np.allclose(goal_ds, log["goal_pos"], atol=1e-6), "goal mismatch"
        init_cl = np.linalg.norm(log["cube_traj"][:, 0] - log["goal_pos"], axis=1)
        assert np.allclose(init_cl, init_dist, atol=1e-6), "initial distance mismatch"
        payload = {
            "seed": int(seed), "episodes": eps, "start_steps": starts, "rows": rows,
            "init_dist": init_dist, "success": succ, **log,
            "action_mean": process["action"].mean_, "action_scale": process["action"].scale_,
        }
        torch.save(payload, out / "closed_loop" / f"seed_{seed}.pt")
        print(
            f"[closed_loop] seed {seed}: success {succ.mean():.0%} on non-trivial starts, "
            f"{len(log['records'])} replans, {time.time() - t0:.0f}s", flush=True,
        )
    world.close()


def _world_cfg(cfg) -> dict:
    w = OmegaConf.to_container(cfg.world, resolve=True)
    w["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    return w


def replay_plans(cfg, out: Path, device: str) -> None:
    (out / "plan_replays").mkdir(exist_ok=True)
    seeds = [s for s in cfg.step1.seeds if not (out / "plan_replays" / f"seed_{s}.pt").exists()]
    if not seeds:
        print("[replay_plans] all seeds done")
        return
    _, _, transform, process = _world_pieces(cfg)
    base = load_base(cfg.policy, device)
    scaler = process["action"]
    for seed in seeds:
        log = torch.load(out / "closed_loop" / f"seed_{seed}.pt", weights_only=False)
        recs = log["records"]
        steps = [BLOCK * k for k in range(1, int(cfg.plan_config.horizon) + 1)]
        jobs = (
            {"key": i, "snapshot": r["snapshot"], "actions": to_env_actions(r["plan"][None], scaler),
             "render_steps": steps}
            for i, r in enumerate(recs)
        )
        pos = [None] * len(recs)
        z = [None] * len(recs)
        t0 = time.time()
        for res in replay_many(jobs, _world_cfg(cfg), int(cfg.step1.workers)):
            i = res["key"]
            pos[i] = res["pos"][0]
            z[i] = encode_frames(base, res["frames"][0], transform["pixels"], device)
        pos, z = np.stack(pos), torch.stack(z)
        check = _determinism(log, recs, pos, z)
        torch.save({"pos": pos, "z_real": z, "check": check}, out / "plan_replays" / f"seed_{seed}.pt")
        print(f"[replay_plans] seed {seed}: {len(recs)} plans in {time.time() - t0:.0f}s; {check}", flush=True)


def _determinism(log, recs, pos, z) -> dict:
    """Replayed first-segment end vs. the closed loop at the next replan."""
    by_key = {(r["t"], r["env"]): i for i, r in enumerate(recs)}
    dpos, demb = [], []
    for i, r in enumerate(recs):
        nxt = by_key.get((r["t"] + pos.shape[1], r["env"]))
        if nxt is None:
            continue
        dpos.append(np.abs(pos[i, -1] - log["cube_traj"][r["env"], r["t"] + pos.shape[1]]).max())
        demb.append((z[i, -1] - recs[nxt]["emb"][-1]).abs().max().item())
    return {
        "n": len(dpos),
        "max_abs_pos_m": float(max(dpos)) if dpos else float("nan"),
        "max_abs_emb": float(max(demb)) if demb else float("nan"),
    }


def _world_pieces(cfg):
    """Transforms and normalizers without building 50 envs."""
    from sklearn import preprocessing

    from evaluate import get_dataset, img_transform

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    scaler = preprocessing.StandardScaler()
    values = dataset.get_col_data("action")
    scaler.fit(values[~np.isnan(values).any(axis=1)])
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    return None, dataset, transform, {"action": scaler}


def select_states(cfg, out: Path) -> list[tuple[int, int]]:
    """(seed, record index) of the pre-registered 200 replay states."""
    failed, succeeded = [], []
    for seed in cfg.step1.seeds:
        log = torch.load(out / "closed_loop" / f"seed_{seed}.pt", weights_only=False)
        for i, r in enumerate(log["records"]):
            (succeeded if log["success"][r["env"]] else failed).append((int(seed), i))
    rng = np.random.default_rng(int(cfg.step1.select_seed))
    nf, ns = int(cfg.step1.n_states_failed), int(cfg.step1.n_states_success)
    nf_take, ns_take = min(nf, len(failed)), min(ns, len(succeeded))
    short = nf + ns - nf_take - ns_take  # a short group is filled from the other one
    extra = min(short, len(failed) - nf_take)
    nf_take += extra
    ns_take += min(short - extra, len(succeeded) - ns_take)
    pick_f = rng.choice(len(failed), size=nf_take, replace=False)
    pick_s = rng.choice(len(succeeded), size=ns_take, replace=False)
    chosen = [failed[i] for i in sorted(pick_f)] + [succeeded[i] for i in sorted(pick_s)]
    print(f"[select] {nf_take} states from failed episodes, {ns_take} from successful "
          f"(available {len(failed)} / {len(succeeded)})")
    return chosen


def replay_candidates(cfg, out: Path, device: str) -> None:
    state_dir = out / "candidate_replays"
    state_dir.mkdir(exist_ok=True)
    chosen = select_states(cfg, out)
    torch.save(chosen, out / "replay_states.pt")
    todo = [(s, i) for s, i in chosen if not (state_dir / f"s{s}_r{i}.pt").exists()]
    if not todo:
        print("[replay_candidates] all states done")
        return
    _, _, transform, process = _world_pieces(cfg)
    base = load_base(cfg.policy, device)
    scaler = process["action"]
    logs = {s: torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in {s for s, _ in todo}}
    iters = list(cfg.step1.record_iters)
    horizon_steps = BLOCK * int(cfg.plan_config.horizon)

    def jobs():
        for s, i in todo:
            r = logs[s]["records"][i]
            cands = torch.cat([r[f"cand_{it}"] for it in iters])  # (600, 5, 25)
            yield {"key": (s, i), "snapshot": r["snapshot"], "actions": to_env_actions(cands, scaler),
                   "render_steps": [horizon_steps]}

    t0 = time.time()
    for n, res in enumerate(replay_many(jobs(), _world_cfg(cfg), int(cfg.step1.workers)), start=1):
        s, i = res["key"]
        z5 = encode_frames(base, res["frames"][:, 0], transform["pixels"], device)
        torch.save({"pos": res["pos"].astype(np.float32), "z5": z5, "iters": iters},
                   state_dir / f"s{s}_r{i}.pt")
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"[replay_candidates] {n}/{len(todo)} states, {el:.0f}s "
                  f"(eta {el / n * (len(todo) - n):.0f}s)", flush=True)


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    stage = str(cfg.step1.stage)
    print(f"[step1_collect] stage={stage} device={device} out={out} radius={SUCCESS_RADIUS}")
    if stage in ("closed_loop", "all"):
        closed_loop(cfg, out, device)
    if stage in ("replay_plans", "all"):
        replay_plans(cfg, out, device)
    if stage in ("replay_candidates", "all"):
        replay_candidates(cfg, out, device)


if __name__ == "__main__":
    main()
