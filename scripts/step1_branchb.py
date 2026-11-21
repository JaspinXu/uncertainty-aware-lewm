"""Step 1 / pilot evidence for the Branch-B directions (exploratory, not pre-registered).

    python scripts/step1_branchb.py

Reuses the Step-1 data; writes results/cube/step1/branchb.json.

1. Expert plan vs CEM plan at the 400 first replans (start = dataset state):
   the expert's own 25 actions are replayed and scored by the model. Is the
   true solution ranked below the CEM's (hallucinated) plan?
2. Action-support score: Mahalanobis distance of each action block to the
   expert block distribution, used like sigma^2 in C1 / C2 / C3.
3. Cost alignment: does the latent cost rank the physical outcome of real
   end states, and does a cube-position probe cost (ridge on frozen latents)
   do better, on real and on predicted latents?
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
from sklearn.linear_model import Ridge  # noqa: E402

from evaluate import img_transform, pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1 import stats  # noqa: E402
from ua.step1.common import encode_frames, out_dir  # noqa: E402
from ua.step1.ensemble import new_head  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402
from ua.step1.replay import ReplayEnv  # noqa: E402
from ua.step1.rollout import deterministic_rollouts, ensemble_rollouts, mc_dropout_rollouts, sigma2  # noqa: E402

BLOCK, H = 5, 5


def sq(a, b):
    return ((a - b) ** 2).sum(-1)


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    seeds = [int(s) for s in cfg.step1.seeds]
    logs = {s: torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in seeds}
    scores = torch.load(out / "scores.pt", weights_only=False)
    base = load_base(cfg.policy, device)
    members = []
    for p in sorted((out / "ensemble").glob("member_*.pt")):
        ck = torch.load(p, weights_only=False)
        head = new_head(ROOT / "checkpoints" / cfg.policy / "config.json", seed=ck["seed"])
        head.load_state_dict(ck["state_dict"])
        members.append(head.to(device).eval())

    h5 = h5py.File(ROOT / "datasets" / f"{cfg.eval.dataset_name}.h5", "r")
    actions = h5["action"][:].astype(np.float32)
    cube = h5["privileged_block_0_pos"][:]
    ep_off = h5["ep_offset"][:]
    a_mean = np.nanmean(actions, axis=0)
    a_std = np.nanstd(actions, axis=0)
    res = {}

    # ------------------------------------------------------------------ #
    # 1. expert plan vs CEM plan at t = 0
    # ------------------------------------------------------------------ #
    env = ReplayEnv(OmegaConf.to_container(cfg.world, resolve=True) | {"max_episode_steps": 100})
    transform = img_transform(cfg)
    rows_out = []
    for s in seeds:
        log, sc = logs[s], scores["plans"][s]
        for i, r in enumerate(log["records"]):
            if r["t"] != 0:
                continue
            e = r["env"]
            row = int(log["rows"][e])
            raw = actions[row : row + H * BLOCK]
            norm = ((raw - a_mean) / a_std).reshape(H, BLOCK * raw.shape[1])
            pos, frames = env.run(r["snapshot"], raw, render_steps=[BLOCK * k for k in range(1, H + 1)])
            z_real = encode_frames(base, frames, transform, device)
            rows_out.append({
                "seed": s, "rec": i, "emb": r["emb"], "goal": r["goal_emb"],
                "norm": torch.from_numpy(norm), "z_real": z_real, "pos": pos,
                "goal_pos": r["goal_pos"], "ds_pos": cube[row + 1 : row + 1 + H * BLOCK],
                "cost_9": r["cost_9"].numpy(),
                "cem_d_pred": float(sc["d_pred"][i]), "cem_d_real": float(sc["d_real"][i]),
                "cem_e5": float(sc["err"][i, -1]), "cem_s2_ens": float(sc["s2_ens"][i, -1]),
                "cem_s2_mc": float(sc["s2_mc20"][i, -1]),
                "cem_seg_success": bool(
                    np.linalg.norm(log["cube_traj"][e, 1 : 1 + H * BLOCK] - log["goal_pos"][e], axis=1).min()
                    <= SUCCESS_RADIUS),
            })
    emb0 = torch.stack([x["emb"] for x in rows_out]).to(device)
    goal = torch.stack([x["goal"] for x in rows_out]).to(device)
    acts = torch.stack([x["norm"] for x in rows_out]).float().to(device)
    zhat = deterministic_rollouts(base, emb0, acts)
    zr = torch.stack([x["z_real"] for x in rows_out]).to(device)
    exp_d_pred = sq(zhat[:, -1], goal).cpu().numpy()
    exp_d_real = sq(zr[:, -1], goal).cpu().numpy()
    exp_e5 = sq(zhat[:, -1], zr[:, -1]).cpu().numpy()
    exp_s2_mc = sigma2(mc_dropout_rollouts(base, emb0, acts, n=20, seed=0))[:, -1].cpu().numpy()
    exp_s2_ens = sigma2(ensemble_rollouts(members, emb0, acts))[:, -1].cpu().numpy()
    dist = np.array([np.linalg.norm(x["pos"] - x["goal_pos"], axis=1) for x in rows_out])
    exp_success = dist.min(axis=1) <= SUCCESS_RADIUS
    replay_vs_dataset = float(np.max([np.abs(x["pos"] - x["ds_pos"]).max() for x in rows_out]))
    cem = {k: np.array([x[k] for x in rows_out]) for k in ("cem_d_pred", "cem_d_real", "cem_e5", "cem_s2_ens", "cem_s2_mc", "cem_seg_success")}
    rank = np.array([np.mean(x["cost_9"] < d) for x, d in zip(rows_out, exp_d_pred)])
    res["expert_vs_cem"] = {
        "n": len(rows_out),
        "replay_vs_dataset_max_abs_m": replay_vs_dataset,
        "expert_replay_success": float(exp_success.mean()),
        "cem_first_segment_success": float(cem["cem_seg_success"].mean()),
        "median_d_pred": {"expert": float(np.median(exp_d_pred)), "cem": float(np.median(cem["cem_d_pred"]))},
        "median_d_real": {"expert": float(np.median(exp_d_real)), "cem": float(np.median(cem["cem_d_real"]))},
        "median_e5": {"expert": float(np.median(exp_e5)), "cem": float(np.median(cem["cem_e5"]))},
        "median_s2_mc": {"expert": float(np.median(exp_s2_mc)), "cem": float(np.median(cem["cem_s2_mc"]))},
        "median_s2_ens": {"expert": float(np.median(exp_s2_ens)), "cem": float(np.median(cem["cem_s2_ens"]))},
        "frac_model_prefers_cem_plan": float(np.mean(exp_d_pred > cem["cem_d_pred"])),
        "expert_rank_in_last_pool_median": float(np.median(rank)),
        "frac_expert_better_than_all_last_pool": float(np.mean(rank == 0)),
        "frac_expert_worse_than_all_last_pool": float(np.mean(rank == 1)),
    }
    print("[branchb] 1.", json.dumps(res["expert_vs_cem"], indent=1))

    # ------------------------------------------------------------------ #
    # 2. action-support score (Mahalanobis to expert action blocks)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(0)
    norm_all = np.nan_to_num((actions - a_mean) / a_std, nan=0.0)
    starts = rng.integers(0, len(actions) - BLOCK, 200000)
    blocks = norm_all[starts[:, None] + np.arange(BLOCK)].reshape(len(starts), -1)
    mu, cov = blocks.mean(0), np.cov(blocks, rowvar=False)
    prec = torch.from_numpy(np.linalg.inv(cov + 1e-4 * np.eye(cov.shape[0]))).float()
    mu_t = torch.from_numpy(mu).float()

    def support(a):  # (N, H, 25) normalized blocks -> (N,) mean Mahalanobis^2
        d = a.float().cpu() - mu_t
        return torch.einsum("nhi,ij,nhj->nh", d, prec, d).mean(-1).numpy()

    exp_sup = support(acts)
    sf, ef, sl, el, o, dp, ss, dr = [], [], [], [], [], [], [], []
    probe_inputs = []
    for st in scores["states"]:
        s, i = st["key"]
        r = logs[s]["records"][i]
        cands = torch.cat([r["cand_0"], r["cand_9"]])
        sup = support(cands)
        first, last = (st["pool"] == 0).numpy(), (st["pool"] == 1).numpy()
        e5 = st["err"][:, -1].numpy()
        sf.append(sup[first]), ef.append(e5[first]), sl.append(sup[last]), el.append(e5[last])
        o.append(st["o"].numpy()[last]), dp.append(st["d_pred"].numpy()[last])
        ss.append(sup[last]), dr.append(st["d_real"].numpy()[last])
        probe_inputs.append((r, cands, last))
    c1 = stats.criterion_c1(sf, ef, sl, el)
    c2 = stats.criterion_c2(o, dp, ss, dr)
    plan_sup, y, d_pred, d0, groups = [], [], [], [], []
    for s in seeds:
        log, sc = logs[s], scores["plans"][s]
        plan_sup.append(support(torch.stack([r["plan"] for r in log["records"]])))
        for i, r in enumerate(log["records"]):
            t0, e = r["t"], r["env"]
            seg = np.linalg.norm(log["cube_traj"][e, t0 + 1 : t0 + 26] - log["goal_pos"][e], axis=1)
            y.append(int(seg.min() > SUCCESS_RADIUS))
            d_pred.append(float(sc["d_pred"][i])), d0.append(float(sc["d0"][i])), groups.append(s * 1000 + e)
    c3 = stats.criterion_c3(np.array(y), np.concatenate(plan_sup), np.array(d_pred), np.array(d0), np.array(groups))
    res["action_support"] = {
        "expert_plan_median": float(np.median(exp_sup)),
        "cem_plan_median": float(np.median(np.concatenate(plan_sup))),
        "expert_block_reference_median": float(np.median(support(torch.from_numpy(blocks[:2000].reshape(-1, 1, blocks.shape[1]))))),
        "c1_median_rho": c1.median, "c1_frac_positive": c1.frac_positive,
        "c1_median_first": float(np.median(c1.rho_first)), "c1_median_last": float(np.median(c1.rho_last)),
        "c2_share": c2.share, "c2_ci": c2.ci,
        "c3_auroc": c3.auroc, "c3_delta": c3.delta, "c3_delta_ci": c3.delta_ci,
    }
    print("[branchb] 2.", json.dumps(res["action_support"], indent=1, default=float))

    # ------------------------------------------------------------------ #
    # 3. cost alignment: latent L2 vs a cube-position probe cost
    # ------------------------------------------------------------------ #
    cache = torch.load(out / "ensemble" / "latents.pt", weights_only=False)
    glob_rows = ep_off[cache["ep"]] + cache["step"]
    Z, P = cache["emb"].astype(np.float32), cube[glob_rows]
    n_val = int(cfg.step1.ensemble.val_episodes)
    val_mask = np.isin(cache["ep"], cache["episodes"][-n_val:])
    tr = np.nonzero(~val_mask)[0]
    tr = rng.choice(tr, size=min(200000, len(tr)), replace=False)
    probe = Ridge(alpha=1.0).fit(Z[tr], P[tr])
    pv = probe.predict(Z[val_mask])
    r2 = 1 - ((pv - P[val_mask]) ** 2).sum(0) / ((P[val_mask] - P[val_mask].mean(0)) ** 2).sum(0)
    err_cm = 100 * np.linalg.norm(pv - P[val_mask], axis=1)
    W = torch.from_numpy(probe.coef_.T).float().to(device)
    bias = torch.from_numpy(probe.intercept_).float().to(device)

    def pcube(z):
        return z @ W + bias

    rho_lat, rho_probe_real, o_l, o_pr, o_pp, d_l, s_pr, s_pp, d_r = [], [], [], [], [], [], [], [], []
    for st, (r, cands, last) in zip(scores["states"], probe_inputs):
        s, i = st["key"]
        rep = torch.load(out / "candidate_replays" / f"s{s}_r{i}.pt", weights_only=False)
        g = r["goal_emb"].to(device)
        z5_real = rep["z5"].to(device)
        zp = deterministic_rollouts(base, r["emb"][None].expand(len(cands), -1, -1).to(device), cands.to(device))[:, -1]
        final = st["final_dist"].numpy()
        c_probe_real = torch.linalg.norm(pcube(z5_real) - pcube(g[None]), dim=-1).cpu().numpy()
        c_probe_pred = torch.linalg.norm(pcube(zp) - pcube(g[None]), dim=-1).cpu().numpy()
        d_real = st["d_real"].numpy()
        rho_lat.append(stats.spearman(d_real[last], final[last]))
        rho_probe_real.append(stats.spearman(c_probe_real[last], final[last]))
        o_l.append(st["o"].numpy()[last])
        d_l.append(st["d_pred"].numpy()[last])
        s_pr.append(c_probe_real[last])
        s_pp.append(c_probe_pred[last])
        d_r.append(d_real[last])

    def capture(scores_):  # share of oracle headroom captured by argmin of a score
        base_o = np.array([oi[np.argmin(di)] for oi, di in zip(o_l, d_l)])
        orc = np.array([oi.min() for oi in o_l])
        new = np.array([oi[np.argmin(si)] for oi, si in zip(o_l, scores_)])
        return float((base_o - new).sum() / (base_o - orc).sum()), float(np.mean(new == 0))

    c2_probe = stats.criterion_c2(o_l, d_l, s_pp, d_r)  # z(d_pred) + lambda * z(probe cost)
    res["cost_alignment"] = {
        "probe_r2_xyz": r2.tolist(), "probe_val_err_cm_median": float(np.median(err_cm)),
        "within_state_rho_latent_cost_vs_physical": float(np.median(rho_lat)),
        "within_state_rho_probe_cost_vs_physical": float(np.median(rho_probe_real)),
        "capture_latent_oracle": capture(d_r),
        "capture_probe_oracle_real_latents": capture(s_pr),
        "capture_probe_cost_predicted_latents": capture(s_pp),
        "c2_probe_penalty_on_predicted": {"share": c2_probe.share, "ci": c2_probe.ci},
    }
    print("[branchb] 3.", json.dumps(res["cost_alignment"], indent=1, default=float))
    (out / "branchb.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
