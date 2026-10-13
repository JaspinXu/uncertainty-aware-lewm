"""Replay recorded CEM candidates in the simulator (L1 ground truth).

    python scripts/cube/oracle_replay.py results/cube/records/baseline_s42.pt \
        --out results/cube/oracle --workers 16

For every recorded planning call, the executed plan, the elites and the random
non-elite candidates kept by the recorder are executed open loop from the exact
planning-time simulator state. Output ``<out>/<run_id>.pt`` holds, aligned with
each record's ``subset``:

  success_step (K,)     first env step (1-based) solving the task, 0 = never
  cube_dist   (K, T)    cube-target distance after every env step [m]
  true_lat    (K, B, D) encoder latent of the frame at the end of each block
  pred_err    (K, B)    ||z_pred_k - z_true_k||^2 (model prediction error)
  true_cost   (K,)      ||z_true_H - z_goal||^2 (cost the planner should have seen)
  start_dist            cube-target distance at planning time

plus a fidelity check: replaying the executed first plan must land on the
state the evaluation itself recorded 25 steps later.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cube.oracle import normalized_to_env_actions, run_jobs  # noqa: E402


def goal_keys(state: dict) -> tuple[str, str | None]:
    pos = [k for k in state if k.startswith("goal_privileged") and k.endswith("_pos")]
    quat = [k for k in state if k.startswith("goal_privileged") and k.endswith("_quat")]
    if not pos:
        raise KeyError(f"no goal position in recorded state keys {sorted(state)}")
    return pos[0], (quat[0] if quat else None)


def build_jobs(payload: dict, plans_filter: str):
    meta = payload["meta"]
    block = int(meta["plan_config"]["action_block"])
    for i, rec in enumerate(payload["plans"]):
        if plans_filter == "first" and rec["plan_idx"] != 0:
            continue
        pos_k, quat_k = goal_keys(rec["state"])
        yield {
            "key": i,
            "qpos": rec["state"]["qpos"],
            "qvel": rec["state"]["qvel"],
            "goal_pos": rec["state"][pos_k],
            "goal_quat": None if quat_k is None else rec["state"][quat_k],
            "env_actions": normalized_to_env_actions(
                rec["subset_actions"], block, meta["action_mean"], meta["action_scale"]
            ),
            "render_every": block,
        }


class FrameEncoder:
    """Frames (uint8 HWC) -> LeWM latents, with the planner's exact image transform."""

    def __init__(self, policy: str, device: str, img_size: int = 224):
        from omegaconf import OmegaConf

        from evaluate import img_transform
        from planners import load_base

        self.device = device
        self.model = load_base(policy, device)
        self.transform = img_transform(OmegaConf.create({"eval": {"img_size": img_size}}))
        self.dtype = next(self.model.parameters()).dtype

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 512) -> np.ndarray:
        from torchvision import tv_tensors

        flat = frames.reshape(-1, *frames.shape[-3:])
        out = []
        for s in range(0, len(flat), batch):
            x = torch.from_numpy(flat[s : s + batch]).permute(0, 3, 1, 2)
            x = torch.stack([self.transform(tv_tensors.Image(f)) for f in x])
            x = x.to(self.device, self.dtype)[:, None]  # (b, 1, C, H, W)
            z = self.model.encode({"pixels": x})["emb"][:, 0]
            out.append(z.float().cpu())
        return torch.cat(out).numpy().reshape(*frames.shape[:-3], -1)


def replay_file(path: Path, out_dir: Path, workers: int, device: str, plans_filter: str, policy: str | None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    plans = payload["plans"]
    env_kwargs = {
        k: v
        for k, v in meta["world"].items()
        if k not in ("env_name", "num_envs", "max_episode_steps")
    }
    encoder = FrameEncoder(policy or meta["policy"], device)
    jobs = list(build_jobs(payload, plans_filter))
    print(f"[oracle] {path.name}: {len(jobs)} plans x {len(plans[0]['subset'])} candidates, workers={workers}")

    results = {}
    t0 = time.time()
    for n_done, res in enumerate(run_jobs(jobs, workers, env_kwargs), start=1):
        rec = plans[res["key"]]
        true_lat = encoder(res.pop("frames"))  # (K, B, D)
        pred = rec["subset_pred"].astype(np.float32)  # (K, P, D)
        nb = min(true_lat.shape[1], pred.shape[1])
        pred_err = ((pred[:, :nb] - true_lat[:, :nb]) ** 2).sum(-1)
        true_cost = ((true_lat[:, -1] - rec["z_goal"][None]) ** 2).sum(-1)
        results[res["key"]] = {
            "env": rec["env"],
            "plan_idx": rec["plan_idx"],
            "t": rec["t"],
            "subset": rec["subset"],
            "success_step": res["success_step"],
            "cube_dist": res["cube_dist"],
            "final_qpos": res["final_qpos"],
            "start_dist": res["start_cube_dist"],
            "true_lat": true_lat.astype(np.float16),
            "pred_err": pred_err.astype(np.float32),
            "true_cost": true_cost.astype(np.float32),
        }
        if n_done % 10 == 0 or n_done == len(jobs):
            print(f"[oracle]   {n_done}/{len(jobs)} plans ({time.time() - t0:.0f}s)", flush=True)

    fidelity = fidelity_check(plans, results)
    print(f"[oracle] fidelity: {fidelity}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / path.name
    torch.save({"meta": {**meta, "fidelity": fidelity, "plans_filter": plans_filter}, "results": results}, out)
    print(f"[oracle] wrote {out} ({time.time() - t0:.0f}s)")
    return out


def fidelity_check(plans: list, results: dict) -> dict:
    """Executed plan of segment p replayed vs. the state recorded at segment p+1."""
    by_env = {(r["env"], r["plan_idx"]): i for i, r in enumerate(plans)}
    diffs = []
    for key, res in results.items():
        nxt = by_env.get((res["env"], res["plan_idx"] + 1))
        if nxt is None:
            continue
        if int(res["success_step"][0]) > 0:  # executed plan solved the task -> env stopped
            continue
        diffs.append(np.abs(res["final_qpos"][0] - plans[nxt]["state"]["qpos"]).max())
    if not diffs:
        return {"n": 0}
    diffs = np.asarray(diffs)
    return {"n": int(len(diffs)), "max_abs_qpos_diff": float(diffs.max()), "median": float(np.median(diffs))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/oracle")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plans", choices=("all", "first"), default="all")
    ap.add_argument("--policy", default=None, help="override checkpoint used to encode frames")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    for path in args.records:
        if args.skip_existing and (args.out / path.name).exists():
            print(f"[oracle] skip {path.name} (exists)")
            continue
        replay_file(path, args.out, args.workers, args.device, args.plans, args.policy)
    os._exit(0)  # avoid EGL teardown noise in worker-less runs


if __name__ == "__main__":
    main()
