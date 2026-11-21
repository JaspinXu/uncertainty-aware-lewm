"""Step 1 / scoring: predictions, real errors and both sigma^2 estimators.

    python scripts/step1_score.py

For every executed plan (all replans) and every replayed candidate (200
states x 600) computes, from the cached start embedding:

- deterministic predictions -> d_pred, e_k = ||z_hat_k - z_k||^2, d_real, d0
- MC-dropout sigma^2_k with N=20 (and N=5 = the first 5 samples)
- ensemble sigma^2_k over the trained heads, and the ensemble-mean errors
- task outcome o = max(min_t ||cube_t - goal|| - 0.04, 0) for candidates

Candidate costs are recomputed and checked against the CEM's logged costs.
Writes results/cube/step1/scores.pt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.common import out_dir  # noqa: E402
from ua.step1.ensemble import new_head  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402
from ua.step1.rollout import (  # noqa: E402
    deterministic_rollouts,
    ensemble_rollouts,
    mc_dropout_rollouts,
    sigma2,
)


def load_members(cfg, out: Path, device):
    paths = sorted((out / "ensemble").glob("member_*.pt"))
    members = []
    for p in paths:
        ck = torch.load(p, weights_only=False)
        head = new_head(ROOT / "checkpoints" / cfg.policy / "config.json", seed=ck["seed"])
        head.load_state_dict(ck["state_dict"])
        members.append(head.to(device).eval())
    return members


def sqdist(a, b):
    return ((a - b) ** 2).sum(-1)


def score_batch(base, members, emb0, actions, goal, z_real, mc_n, mc_seed, device):
    """All quantities for N sequences sharing nothing but the models.

    emb0 (N, H, D), actions (N, T, A), goal (N, D), z_real (N, K, D) real
    embeddings after the last K blocks (K=5 for plans, 1 for candidates).
    """
    emb0, actions, goal, z_real = (x.to(device) for x in (emb0, actions, goal, z_real))
    K = z_real.shape[1]
    z = deterministic_rollouts(base, emb0, actions)  # (N, P, D)
    mc = mc_dropout_rollouts(base, emb0, actions, n=mc_n, seed=mc_seed)  # (n, N, P, D)
    res = {
        "d_pred": sqdist(z[:, -1], goal),
        "d0": sqdist(emb0[:, -1], goal),
        "d_real": sqdist(z_real[:, -1], goal),
        "err": sqdist(z[:, -K:], z_real),  # (N, K)
        "s2_mc20": sigma2(mc),  # (N, P)
        "s2_mc5": sigma2(mc[:5]),
    }
    if members:
        ens = ensemble_rollouts(members, emb0, actions)  # (M, N, P, D)
        zm = ens.mean(0)
        res["s2_ens"] = sigma2(ens)
        res["err_ens"] = sqdist(zm[:, -K:], z_real)
        res["d_pred_ens"] = sqdist(zm[:, -1], goal)
    return {k: v.float().cpu() for k, v in res.items()}


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    base = load_base(cfg.policy, device)
    members = load_members(cfg, out, device)
    print(f"[score] {len(members)} ensemble members")
    mc_n, mc_seed = int(cfg.step1.mc_samples), int(cfg.step1.mc_seed)

    # ---- executed plans (every replan) ----
    plans = {}
    for seed in cfg.step1.seeds:
        log = torch.load(out / "closed_loop" / f"seed_{seed}.pt", weights_only=False)
        rep = torch.load(out / "plan_replays" / f"seed_{seed}.pt", weights_only=False)
        recs = log["records"]
        emb0 = torch.stack([r["emb"] for r in recs])
        goal = torch.stack([r["goal_emb"] for r in recs])
        acts = torch.stack([r["plan"] for r in recs])
        res = score_batch(base, members, emb0, acts, goal, rep["z_real"], mc_n, mc_seed + int(seed), device)
        res["t"] = torch.tensor([r["t"] for r in recs])
        res["env"] = torch.tensor([r["env"] for r in recs])
        plans[int(seed)] = res
        print(f"[score] seed {seed}: {len(recs)} plans")

    # ---- replayed candidates ----
    chosen = torch.load(out / "replay_states.pt", weights_only=False)
    logs = {}
    states = []
    worst = 0.0
    for n, (s, i) in enumerate(chosen):
        if s not in logs:
            logs[s] = torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False)
        r = logs[s]["records"][i]
        rep = torch.load(out / "candidate_replays" / f"s{s}_r{i}.pt", weights_only=False)
        iters = rep["iters"]
        cands = torch.cat([r[f"cand_{it}"] for it in iters])
        logged = torch.cat([r[f"cost_{it}"] for it in iters])
        N = cands.shape[0]
        res = score_batch(
            base, members, r["emb"][None].expand(N, -1, -1), cands,
            r["goal_emb"][None].expand(N, -1), rep["z5"][:, None], mc_n, mc_seed + 7919 * (n + 1), device,
        )
        worst = max(worst, ((res["d_pred"] - logged).abs() / logged.abs().clamp_min(1e-8)).max().item())
        dist = np.linalg.norm(rep["pos"] - r["goal_pos"][None, None], axis=-1)  # (N, 25)
        res["o"] = torch.from_numpy(np.maximum(dist.min(axis=1) - SUCCESS_RADIUS, 0.0)).float()
        res["final_dist"] = torch.from_numpy(dist[:, -1]).float()
        res["pool"] = torch.repeat_interleave(torch.arange(len(iters)), N // len(iters))
        res["key"] = (s, i)
        states.append(res)
        if (n + 1) % 25 == 0:
            print(f"[score] {n + 1}/{len(chosen)} states")
    print(f"[score] max relative |recomputed - logged| candidate cost: {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError("recomputed candidate costs do not match the CEM logs")
    torch.save({"plans": plans, "states": states, "n_members": len(members), "mc_samples": mc_n},
               out / "scores.pt")
    print(f"[score] wrote {out / 'scores.pt'}")


if __name__ == "__main__":
    main()
