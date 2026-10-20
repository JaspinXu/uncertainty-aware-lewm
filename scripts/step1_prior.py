"""Step 1 / train the behaviour-cloning prior and check it offline (exploratory).

    python scripts/step1_prior.py                                              # goal 25 steps ahead
    python scripts/step1_prior.py +prior_name=prior_var "+prior_offsets=[1,100]" +prior_steps=20000

1. Train pi(z_t, z_goal) -> next 25 expert actions (ua/step1/prior.py) on the
   cached frozen latents (results/cube/step1/ensemble/latents.pt), with the
   goal k steps ahead, k uniform in prior_offsets (default exactly 25). The 400
   Step-1 evaluation episodes are left out of training, so the prior cannot
   memorise their solutions; the 200 held-out cache episodes give the
   validation loss. (The priors of the method comparison are trained by
   scripts/cube_train.py.)
2. At the 400 Step-1 start states: the prior's proposal against the expert's
   actions, and the first 25 steps of the prior, the CEM plan and the expert
   replayed open loop in the simulator: whether each reaches the goal, the
   world model's predicted goal distance d_pred, and the real one d_real
   (the frame actually reached, encoded). Where the prior reaches the goal and
   the CEM plan does not, d_pred vs d_real tells whether the planner's choice
   is a prediction error or a cost (latent goal distance) that disagrees with
   the task.

Writes results/cube/step1/prior/<prior_name>.pt and offline.json (offline_<prior_name>.json
for other names). Closed-loop use: scripts/step1_pilot.py +pilot=bc_init | bc_only |
bc_only_rh1 [+prior=<prior_name>], scripts/step1_longgoal.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import img_transform, pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.common import encode_frames, out_dir  # noqa: E402
from ua.step1.prior import (  # noqa: E402
    GOAL_OFFSET, HORIZON, PriorData, eval_mse, load_prior, train_prior,
)
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402
from ua.step1.replay import ReplayEnv  # noqa: E402
from ua.step1.rollout import deterministic_rollouts  # noqa: E402


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    pdir = out / "prior"
    pdir.mkdir(exist_ok=True)
    seeds = [int(s) for s in cfg.step1.seeds]
    logs = {s: torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in seeds}
    mean, scale = logs[seeds[0]]["action_mean"], logs[seeds[0]]["action_scale"]

    # ---- data: cached latents, Step-1 episodes left out ----
    cache = torch.load(out / "ensemble" / "latents.pt", weights_only=False)
    episodes = list(cache["episodes"])
    n_val = int(cfg.step1.ensemble.val_episodes)
    step1_eps = {int(e) for log in logs.values() for e in log["episodes"]}
    train_rows = [i for i, e in enumerate(episodes[:-n_val]) if e not in step1_eps]
    val_rows = list(range(len(episodes) - n_val, len(episodes)))
    assert not step1_eps & {episodes[i] for i in train_rows}
    name = str(cfg.get("prior_name", "prior"))
    offsets = tuple(int(k) for k in cfg.get("prior_offsets", [GOAL_OFFSET, GOAL_OFFSET]))
    train = PriorData(cache, train_rows, mean, scale, device, offsets)
    val = PriorData(cache, val_rows, mean, scale, device, offsets)
    print(f"[prior] {name}: goal {offsets[0]}-{offsets[1]} steps ahead; {len(train_rows)} training episodes, "
          f"{len(step1_eps)} Step-1 episodes left out, {len(val_rows)} validation episodes")

    ckpt = pdir / f"{name}.pt"
    if ckpt.exists():
        prior = load_prior(ckpt, device)
        train_log = torch.load(ckpt, weights_only=False)["train_log"]
    else:
        prior, train_log = train_prior(train, val, steps=int(cfg.get("prior_steps", 8_000)))
        torch.save({"state_dict": prior.state_dict(), "train_log": train_log, "action_mean": mean, "action_scale": scale,
                    "offsets": offsets, "excluded_episodes": sorted(step1_eps),
                    "train_episodes": sorted(int(episodes[i]) for i in train_rows)}, ckpt)
    val_metrics = eval_mse(prior, val)
    print(f"[prior] validation MSE {val_metrics['mse']:.3f} (normalized units; per dim "
          + ", ".join(f"{x:.2f}" for x in val_metrics["mse_per_dim"]) + ")")

    # ---- the 400 Step-1 start states ----
    base = load_base(cfg.policy, device)
    scores = torch.load(out / "scores.pt", weights_only=False)
    failures = {(e["seed"], e["env"]): e for e in json.loads((out / "failures.json").read_text())["episodes"]}
    h5 = h5py.File(ROOT / "datasets" / f"{cfg.eval.dataset_name}.h5", "r")
    act_col = h5["action"]
    env = ReplayEnv(OmegaConf.to_container(cfg.world, resolve=True) | {"max_episode_steps": 100})
    rows = []
    for s in seeds:
        for i, r in enumerate(logs[s]["records"]):
            if r["t"] != 0:
                continue
            row = int(logs[s]["rows"][r["env"]])
            expert = (act_col[row : row + GOAL_OFFSET] - mean) / scale
            rows.append({"seed": s, "env": r["env"], "i": i, "rec": r, "expert": expert.reshape(HORIZON, -1)})
    z = torch.stack([x["rec"]["emb"][-1] for x in rows]).to(device)
    g = torch.stack([x["rec"]["goal_emb"] for x in rows]).to(device)
    with torch.no_grad():
        prop = prior(z, g, torch.full((len(z),), GOAL_OFFSET, device=device)).cpu()
    expert = torch.as_tensor(np.stack([x["expert"] for x in rows]), dtype=torch.float32)
    emb0 = torch.stack([x["rec"]["emb"] for x in rows]).to(device)

    def d_pred(blocks):
        zh = deterministic_rollouts(base, emb0, blocks.to(device))[:, -1]
        return ((zh - g) ** 2).sum(-1).cpu().numpy()

    # replay the prior's and the expert's 25 actions from the exact start states;
    # the CEM's first plans were replayed by step1_collect (plan_replays) and scored by step1_score
    tf = img_transform(cfg)
    goal_pos = np.stack([x["rec"]["goal_pos"] for x in rows])
    plans = {}
    for k, blocks in (("prior", prop), ("expert", expert)):
        pos, frames = [], []
        for x, b in zip(rows, blocks):
            p, f = env.run(x["rec"]["snapshot"], b.numpy().reshape(-1, len(mean)) * scale + mean, [GOAL_OFFSET])
            pos.append(p)
            frames.append(f[0])
        z_real = encode_frames(base, np.stack(frames), tf, device)
        plans[k] = {"pos": np.stack(pos), "d_real": ((z_real - g.cpu()) ** 2).sum(-1).numpy(), "d_pred": d_pred(blocks)}
    cem_pos = {s: torch.load(out / "plan_replays" / f"seed_{s}.pt", weights_only=False)["pos"] for s in seeds}
    plans["cem"] = {
        "pos": np.stack([cem_pos[x["seed"]][x["i"]] for x in rows]),
        "d_real": np.array([float(scores["plans"][x["seed"]]["d_real"][x["i"]]) for x in rows]),
        "d_pred": np.array([float(scores["plans"][x["seed"]]["d_pred"][x["i"]]) for x in rows]),
    }
    for v in plans.values():  # the segment reaches the goal if the cube comes within the radius at any step
        v["success"] = np.linalg.norm(v["pos"] - goal_pos[:, None], axis=-1).min(1) <= SUCCESS_RADIUS

    types = np.array([failures[(x["seed"], x["env"])]["type"] for x in rows])
    pr, cem = plans["prior"], plans["cem"]
    won = pr["success"] & ~cem["success"]  # the prior's segment reaches the goal, the CEM's does not
    res = {
        "n_train_episodes": len(train_rows), "n_excluded_step1_episodes": len(step1_eps),
        "offsets": list(offsets), "val": val_metrics, "train_log": train_log,
        "start_states": {
            "n": len(rows),
            "mse_vs_expert": float(((prop - expert) ** 2).mean()),
            "first_segment_success": {k: float(v["success"].mean()) for k, v in plans.items()},
            "prior_first_segment_success_by_type": {t: float(pr["success"][types == t].mean())
                                                    for t in ("carry", "pick", "place")},
            "median_d_pred": {k: float(np.median(v["d_pred"])) for k, v in plans.items()},
            "median_d_real": {k: float(np.median(v["d_real"])) for k, v in plans.items()},
            "frac_pred_cost_prefers_prior_over_cem": float(np.mean(pr["d_pred"] < cem["d_pred"])),
            "frac_real_cost_prefers_prior_over_cem": float(np.mean(pr["d_real"] < cem["d_real"])),
            "prior_reaches_cem_does_not": {
                "n": int(won.sum()),
                "pred_cost_prefers_prior": float(np.mean(pr["d_pred"][won] < cem["d_pred"][won])),
                "real_cost_prefers_prior": float(np.mean(pr["d_real"][won] < cem["d_real"][won])),
            },
        },
    }
    (pdir / ("offline.json" if name == "prior" else f"offline_{name}.json")).write_text(json.dumps(res, indent=1))
    st = res["start_states"]
    fs, w = st["first_segment_success"], st["prior_reaches_cem_does_not"]
    print(f"[prior] {st['n']} Step-1 start states, MSE vs expert {st['mse_vs_expert']:.3f}. First 25 steps replayed "
          f"open loop reach the goal: prior {fs['prior']:.0%}, CEM plan {fs['cem']:.0%}, expert {fs['expert']:.0%}")
    for key in ("median_d_pred", "median_d_real"):
        print(f"[prior] {key}: " + ", ".join(f"{k} {v:.1f}" for k, v in st[key].items()))
    print(f"[prior] cost prefers the prior over the CEM plan: predicted {st['frac_pred_cost_prefers_prior_over_cem']:.0%}, "
          f"real latent {st['frac_real_cost_prefers_prior_over_cem']:.0%}; where only the prior reaches the goal "
          f"(n={w['n']}): predicted {w['pred_cost_prefers_prior']:.0%}, real latent {w['real_cost_prefers_prior']:.0%}")

if __name__ == "__main__":
    main()
