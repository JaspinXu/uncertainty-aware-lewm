"""Diagnostic: do live-rendered Cube frames look like the dataset frames to LeWM?

    python scripts/cube/check_domain_gap.py [--seeds 42 43] [--policy cube/lewm]

The evaluation starts every episode from a dataset state and gives the planner
the *dataset* frame of that state and a *dataset* goal frame; every later
observation is rendered live by the simulator. The oracle replays (L1) also
render live frames. If live frames were encoded differently from dataset
frames, the replays would measure that gap on top of the model's error, and
second plans (made from a live frame) would be handicapped.

For the evaluation start states of the given seeds (the rows the planning
panel uses) this script
  1. sets the simulator to the dataset state exactly as the evaluation does
     (qpos/qvel + goal target) and renders the frame the way World does;
  2. replays the dataset's own next 25 actions open loop, rendering every 5
     steps (both possible action alignments are tried; the one that follows
     the dataset's cube trajectory more closely is used);
  3. encodes dataset and live frames with the pretrained encoder.

Reported in squared latent distance (the planner's cost unit):
  render gap        live frame vs dataset frame of the same state (k = 0)
  segment gap       live vs dataset frame along the replayed segment (k = 1..5;
                    includes the replay's drift from the dataset trajectory)
  prediction error  LeWM's open-loop prediction from the dataset start frame
                    against dataset frames (in distribution) and against live
                    frames (what the oracle replays measure), and from the
                    live start frame against live frames
  references        distance between dataset frames 5 and 25 steps apart
Writes <out>/domain_gap.json, domain_gap.md and fig_domain_gap.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/cube"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cube import data as cube_data  # noqa: E402
from cube.clips import dataset_path  # noqa: E402
from cube.oracle import cube_qpos_slice, make_cube_env, render_like_world  # noqa: E402

GOAL_OFFSET, BLOCK, H = 25, 5, 5


class H5Columns:
    """The two methods of the swm dataset that cube/data.py needs."""

    def __init__(self, f: h5py.File):
        self.f = f
        self.column_names = list(f.keys())

    def get_col_data(self, col: str) -> np.ndarray:
        return self.f[col][:]


def start_frame(env, qpos, qvel, goal_pos, goal_quat) -> np.ndarray:
    """State setup of cube/oracle.replay (= the evaluation's episode start), then render."""
    env._reset_next_step = False
    env.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
    env.set_target_pos(0, np.asarray(goal_pos, np.float64), np.asarray(goal_quat, np.float64))
    env.pre_step()
    env.post_step()
    return render_like_world(env)


def replay_segment(env, row: dict, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames every BLOCK steps and the cube position after every step."""
    start_frame(env, row["qpos"], row["qvel"], row["goal_pos"], row["goal_quat"])
    sl = cube_qpos_slice(env)
    frames, cube = [], []
    for t, a in enumerate(actions):
        env.step(np.asarray(a, np.float32))
        cube.append(env._data.qpos[sl].copy())
        if (t + 1) % BLOCK == 0:
            frames.append(render_like_world(env))
    return np.stack(frames), np.stack(cube)


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "mean": float(x.mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--policy", default="cube/lewm")
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/diag")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from oracle_replay import FrameEncoder

    from cube.rollout import rollout_from_latents

    f = h5py.File(dataset_path(args.dataset), "r")
    cols = H5Columns(f)
    rows = np.unique(np.concatenate([
        cube_data.sample_start_rows(cols, args.num_eval, s, GOAL_OFFSET) for s in args.seeds]))
    act_all = f["action"][:]
    ep_idx = f[cube_data.episode_column(cols)][:]
    scaler = cube_data.fit_scaler(act_all)
    print(f"[gap] {len(rows)} start states from seeds {args.seeds}; rendering ...", flush=True)

    env = make_cube_env()
    sl = cube_qpos_slice(env)
    px = f["pixels"]
    ds_frames, live0, live_seg, ds_seg, drift, chosen, acts_norm = [], [], [], [], [], [], []
    for r in rows:
        r = int(r)
        row = {"qpos": f["qpos"][r], "qvel": f["qvel"][r],
               "goal_pos": f["privileged_block_0_pos"][r + GOAL_OFFSET],
               "goal_quat": f["privileged_block_0_quat"][r + GOAL_OFFSET]}
        ds_frames.append(px[r])
        live0.append(start_frame(env, row["qpos"], row["qvel"], row["goal_pos"], row["goal_quat"]))
        # cube trajectory of the dataset; the qpos column holds the cube joint too
        cube_ds = f["qpos"][r + 1 : r + 1 + GOAL_OFFSET][:, sl]
        best = None
        for shift in (0, 1):  # action[r] or action[r + 1] is the first action taken from row r
            if ep_idx[r + shift + GOAL_OFFSET - 1] != ep_idx[r]:
                continue  # would run into the next episode
            a = act_all[r + shift : r + shift + GOAL_OFFSET]
            frames, cube = replay_segment(env, row, a)
            err = float(np.abs(cube - cube_ds).max())
            if best is None or err < best[0]:
                best = (err, shift, frames, cube, a)
        err, shift, frames, cube, a = best
        chosen.append(shift)
        drift.append(100 * np.linalg.norm(cube[BLOCK - 1 :: BLOCK] - cube_ds[BLOCK - 1 :: BLOCK], axis=-1))
        live_seg.append(frames)
        ds_seg.append(np.stack([px[r + BLOCK * (k + 1)] for k in range(H)]))
        acts_norm.append(scaler.transform(a).astype(np.float32).reshape(H, BLOCK * a.shape[1]))
    print(f"[gap] rendered in {time.time() - t0:.0f} s; encoding ...", flush=True)

    enc = FrameEncoder(args.policy, args.device)
    z_ds0 = enc(np.stack(ds_frames))  # (N, D)
    z_live0 = enc(np.stack(live0))
    z_ds = enc(np.stack(ds_seg))  # (N, H, D)
    z_live = enc(np.stack(live_seg))
    A = torch.from_numpy(np.stack(acts_norm))[:, None]  # (N, 1, H, 25)

    @torch.no_grad()
    def predict(z0: np.ndarray) -> np.ndarray:
        dt = enc.dtype
        out = rollout_from_latents(enc.model, torch.from_numpy(z0)[:, None].to(args.device, dt),
                                   A.to(args.device, dt))
        return out[:, 0].float().cpu().numpy()  # (N, H, D)

    zh_ds, zh_live = predict(z_ds0), predict(z_live0)
    sq = lambda a, b: ((a - b) ** 2).sum(-1)  # noqa: E731
    pix = np.abs(np.stack(ds_frames).astype(np.int16) - np.stack(live0).astype(np.int16))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    res = {
        "policy": args.policy, "seeds": args.seeds, "n_states": int(len(rows)),
        "action_alignment": {"action[r]": int(np.sum(np.array(chosen) == 0)),
                             "action[r+1]": int(np.sum(np.array(chosen) == 1))},
        "pixel_mae_start": quantiles(pix.mean(axis=(1, 2, 3))),
        "pixel_frac_changed_start": quantiles((pix.max(-1) > 8).mean(axis=(1, 2))),
        "render_gap_start": quantiles(sq(z_live0, z_ds0)),
        "segment_gap": [quantiles(sq(z_live[:, k], z_ds[:, k])) for k in range(H)],
        "replay_cube_drift_cm": [quantiles(np.stack(drift)[:, k]) for k in range(H)],
        "pred_err_dataset_vs_dataset": [quantiles(sq(zh_ds[:, k], z_ds[:, k])) for k in range(H)],
        "pred_err_dataset_vs_live": [quantiles(sq(zh_ds[:, k], z_live[:, k])) for k in range(H)],
        "pred_err_live_vs_live": [quantiles(sq(zh_live[:, k], z_live[:, k])) for k in range(H)],
        "goal_cost_expert_actions": {
            "from_dataset_frame": quantiles(sq(zh_ds[:, -1], z_ds[:, -1])),
            "from_live_frame": quantiles(sq(zh_live[:, -1], z_ds[:, -1]))},
        "ref_dataset_5_steps": quantiles(sq(z_ds0, z_ds[:, 0])),
        "ref_dataset_25_steps": quantiles(sq(z_ds0, z_ds[:, -1])),
        "ref_random_pairs": quantiles(sq(z_ds0, z_ds0[perm])),
        "minutes": (time.time() - t0) / 60,
    }
    (args.out / "domain_gap.json").write_text(json.dumps(res, indent=1))

    med = lambda q: f"{q['median']:.2f} [{q['p10']:.2f}, {q['p90']:.2f}]"  # noqa: E731
    L = ["# Live rendering vs dataset frames (diagnostic)\n",
         f"Generated by `scripts/cube/check_domain_gap.py`: {len(rows)} evaluation start states "
         f"(seeds {', '.join(map(str, args.seeds))}), encoder `{args.policy}`. "
         "Squared latent distances, median [10th, 90th percentile].\n",
         f"- pixels, live vs dataset frame of the same state: mean abs. difference {med(res['pixel_mae_start'])} "
         f"(0-255 scale); share of pixels differing by more than 8: {med(res['pixel_frac_changed_start'])}",
         f"- **render gap** (latent, same state): {med(res['render_gap_start'])}",
         f"- references: dataset frames 5 steps apart {med(res['ref_dataset_5_steps'])}, "
         f"25 steps apart {med(res['ref_dataset_25_steps'])}, random pairs {med(res['ref_random_pairs'])}",
         f"- action alignment that reproduces the dataset: {res['action_alignment']}\n",
         "| block k | segment gap (live vs dataset) | replay drift of the cube [cm] | pred. error: dataset start → dataset frames "
         "| dataset start → live frames | live start → live frames |",
         "|---:|---|---|---|---|---|"]
    for k in range(H):
        L.append(f"| {k + 1} | {med(res['segment_gap'][k])} | {med(res['replay_cube_drift_cm'][k])} | "
                 f"{med(res['pred_err_dataset_vs_dataset'][k])} | {med(res['pred_err_dataset_vs_live'][k])} | "
                 f"{med(res['pred_err_live_vs_live'][k])} |")
    g = res["goal_cost_expert_actions"]
    L.append(f"\nPlanning cost of the expert's own 25 actions against the goal frame: from the dataset start frame "
             f"{med(g['from_dataset_frame'])}, from the live start frame {med(g['from_live_frame'])}.\n")
    (args.out / "domain_gap.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)

    try:
        from PIL import Image

        idx = np.argsort(sq(z_live0, z_ds0))[[0, len(rows) // 2, -1]]  # best, median, worst
        tiles = []
        for i in idx:
            d = np.clip(pix[i].astype(np.int32) * 4, 0, 255).astype(np.uint8)
            tiles.append(np.concatenate([ds_frames[i], live0[i], d], axis=1))
        Image.fromarray(np.concatenate(tiles, axis=0)).save(args.out / "fig_domain_gap.png")
    except Exception as e:  # the figure is optional
        print(f"[gap] figure skipped: {e}")


if __name__ == "__main__":
    main()
    os._exit(0)
