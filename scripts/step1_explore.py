"""Step 1 / exploratory follow-ups (NOT pre-registered; cannot change the decision).

    python scripts/step1_explore.py

Questions that separate the two Branch-B options:

B-1 (learned failure-risk model): can planning-time features predict real
    outcomes better than the cost alone?
    - per replan: out-of-fold gradient boosting on [cost, d0, sigma^2_1..5 (MC and
      ensemble), CEM population statistics] -> AUROC(segment failure) vs cost-only;
    - per candidate: a cross-fitted re-ranker of the last CEM pool -> share of
      the oracle headroom it captures (C2 with a learned score instead of sigma^2).
B-2 (sigma^2 as a data-collection signal): is the model worse on planner-visited
    states than on expert data, and does sigma^2 rank states by that error?
    - on-policy 1-block error of executed plans vs the expert validation loss;
    - across-state Spearman(sigma^2_k, e_k) over executed plans.

Writes results/cube/step1/explore.md and explore.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from ua.step1.common import out_dir  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402
from ua.step1.stats import N_BOOT, _cluster_boot_idx, oof_logistic, spearman  # noqa: E402

D = 192  # latent dim (per-element MSE = summed error / D)


def _z(x):
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def replan_features(logs, scores, have_ens):
    """Planning-time features of every replan. The replan time is returned
    separately: a second plan fails 97% of the time, so it would dominate."""
    X, Xc, y, g, t = [], [], [], [], []
    for seed, log in logs.items():
        sc = scores["plans"][seed]
        traj, goal = log["cube_traj"], log["goal_pos"]
        for i, r in enumerate(log["records"]):
            t0, e = r["t"], r["env"]
            seg = np.linalg.norm(traj[e, t0 + 1 : t0 + 26] - goal[e], axis=1)
            y.append(int(seg.min() > SUCCESS_RADIUS))
            g.append(seed * 1000 + e)
            t.append(t0)
            c9 = r["cost_9"].numpy()
            c0 = r["cost_0"].numpy()
            f = [np.log(float(sc["d_pred"][i])), np.log(float(sc["d0"][i])), np.log(r["elite_cost"]),
                 np.log(c9.std() + 1e-8), np.log(c0.min()), np.log(np.median(c0))]
            f += np.log(sc["s2_mc20"][i].numpy()).tolist()
            if have_ens:
                f += np.log(sc["s2_ens"][i].numpy()).tolist()
            X.append(f)
            Xc.append([np.log(float(sc["d_pred"][i])), np.log(float(sc["d0"][i]))])
    return np.array(X), np.array(Xc), np.array(y), np.array(g), np.array(t)


def oof_gbm(X, y, groups):
    p = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def boot_delta(y, p_new, p_ref, groups, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        idx = _cluster_boot_idx(groups, rng)
        if y[idx].min() == y[idx].max():
            continue
        out.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def candidate_reranker(states, have_ens, seed=0):
    """Cross-fitted (5 folds over states) GBM predicting o from within-state
    z-scored planning-time features; re-rank the last pool by the prediction."""
    feats, targets, sid = [], [], []
    for k, st in enumerate(states):
        last = (st["pool"] == 1).numpy()
        cols = [_z(st["d_pred"].numpy()[last])]
        for key in ["s2_mc20"] + (["s2_ens"] if have_ens else []):
            for j in range(st[key].shape[1]):
                cols.append(_z(np.log(st[key][:, j].numpy()[last])))
        feats.append(np.column_stack(cols))
        targets.append(st["o"].numpy()[last])
        sid.append(np.full(last.sum(), k))
    X, y, g = np.concatenate(feats), np.concatenate(targets), np.concatenate(sid)
    pred = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                            l2_regularization=1.0, random_state=seed)
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    o_base, o_orc, o_new = [], [], []
    for k in range(len(states)):
        m = g == k
        ok, dk = y[m], feats[k][:, 0]
        o_base.append(ok[np.argmin(dk)])
        o_orc.append(ok.min())
        o_new.append(ok[np.argmin(pred[m])])
    o_base, o_orc, o_new = map(np.array, (o_base, o_orc, o_new))
    head = o_base - o_orc
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(head), len(head))
        if head[idx].sum() > 0:
            boot.append((o_base[idx] - o_new[idx]).sum() / head[idx].sum())
    return {
        "share": float((o_base - o_new).sum() / head.sum()),
        "ci": tuple(np.percentile(boot, [2.5, 97.5])),
        "success_base": float(np.mean(o_base == 0)),
        "success_reranked": float(np.mean(o_new == 0)),
        "success_oracle": float(np.mean(o_orc == 0)),
    }


@torch.no_grad()
def expert_first_block_error(cfg, cache_path):
    """Released predictor, held-out expert clips: per-element error of the first
    predicted block from one context frame (the e_1 of an executed plan)."""
    from evaluate import get_dataset, pick_device
    from planners import load_base
    from sklearn import preprocessing

    from ua.step1.ensemble import ClipSampler, base_head
    from ua.step1.rollout import rollout_from_emb

    device = pick_device()
    cache = torch.load(cache_path, weights_only=False)
    n_val = int(cfg.step1.ensemble.val_episodes)
    rows = np.arange(len(cache["lengths"]) - n_val, len(cache["lengths"]))
    values = get_dataset(cfg, cfg.eval.dataset_name).get_col_data("action")
    scaler = preprocessing.StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
    sampler = ClipSampler(cache, rows, scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32), device, 0)
    head = base_head(load_base(cfg.policy, device)).eval()
    idx = torch.linspace(0, len(sampler) - 1, min(20000, len(sampler))).long().to(device)
    errs = []
    for i in range(0, len(idx), 2048):
        emb, act = sampler.batch(0, idx[i : i + 2048])
        pred = rollout_from_emb(head, emb[:, :1], act[:, :1])  # (B, 1, D)
        errs.append(((pred[:, 0] - emb[:, 1]) ** 2).mean(-1).cpu())
    e = torch.cat(errs).numpy()
    return {"median": float(np.median(e)), "mean": float(e.mean())}


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    out = out_dir(cfg, ROOT)
    logs = {int(s): torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in cfg.step1.seeds}
    scores = torch.load(out / "scores.pt", weights_only=False)
    have_ens = "s2_ens" in next(iter(scores["plans"].values()))
    res = {}

    # ---- B-1: richer planning-time features -> failure of the FIRST plan ----
    X, Xc, y, g, t = replan_features(logs, scores, have_ens)
    res["segment_failure_rate_by_replan"] = {int(v): float(y[t == v].mean()) for v in np.unique(t)}
    res["n_by_replan"] = {int(v): int((t == v).sum()) for v in np.unique(t)}
    first = t == 0
    X, Xc, y, g = X[first], Xc[first], y[first], g[first]
    p_cost = oof_logistic(Xc, y, g)
    p_gbm = oof_gbm(X, y, g)
    res["first_plan_auroc_cost_only"] = float(roc_auc_score(y, p_cost))
    res["first_plan_auroc_gbm"] = float(roc_auc_score(y, p_gbm))
    res["first_plan_auroc_gbm_delta_ci"] = boot_delta(y, p_gbm, p_cost, g)
    s2_cols = {"s2_mc20": 6 + 4} | ({"s2_ens": 6 + 5 + 4} if have_ens else {})
    for key, col in s2_cols.items():  # sigma^2 at the terminal step, raw score
        res[f"first_plan_auroc_{key}"] = float(roc_auc_score(y, X[:, col]))
    res["candidate_reranker"] = candidate_reranker(scores["states"], have_ens)

    # ---- B-2: on-policy error vs expert, and across-state ranking ----
    e1 = np.concatenate([scores["plans"][s]["err"][:, 0].numpy() for s in logs]) / D
    res["onpolicy_err_block1_per_elem_median"] = float(np.median(e1))
    res["onpolicy_err_block1_per_elem_mean"] = float(np.mean(e1))
    log_path = out / "ensemble" / "train_log.json"
    if log_path.exists():
        res["expert_val_loss_released"] = json.loads(log_path.read_text()).get("released_val_loss")
    cache_path = out / "ensemble" / "latents.pt"
    if cache_path.exists():  # same quantity as e_1 on expert data: 1 context frame, block 0
        res["expert_err_block1_per_elem"] = expert_first_block_error(cfg, cache_path)
    for key in ["s2_mc20"] + (["s2_ens"] if have_ens else []):
        s2 = np.concatenate([scores["plans"][s][key].numpy() for s in logs])
        err = np.concatenate([scores["plans"][s]["err"].numpy() for s in logs])
        res[f"across_state_rho_{key}"] = [spearman(s2[:, k], err[:, k]) for k in range(err.shape[1])]

    (out / "explore.json").write_text(json.dumps(res, indent=2, default=float))
    L = ["# Step 1, exploratory follow-ups (not pre-registered)\n",
         "These results cannot change the Step-1 decision. They inform the choice *within* Branch B.\n",
         "## B-1 feasibility: can a learned model predict real outcomes?\n",
         "- Segment failure rate by replan: "
         + ", ".join(f"t={k}: {v:.1%} (n={res['n_by_replan'][k]})" for k, v in res["segment_failure_rate_by_replan"].items())
         + ". A second plan almost never recovers, so only first plans are scored below.",
         f"- First plan fails within 25 steps: cost-only logistic AUROC **{res['first_plan_auroc_cost_only']:.3f}**; "
         f"gradient boosting on cost + σ²₁…₅ + CEM statistics AUROC **{res['first_plan_auroc_gbm']:.3f}** "
         f"(Δ 95% CI [{res['first_plan_auroc_gbm_delta_ci'][0]:+.3f}, {res['first_plan_auroc_gbm_delta_ci'][1]:+.3f}]); "
         + ", ".join(f"raw {k}: {res[f'first_plan_auroc_{k}']:.3f}" for k in s2_cols) + ".",
         f"- Per candidate, a learned re-ranker of the last CEM pool captures "
         f"**{res['candidate_reranker']['share']:+.1%}** of the oracle headroom "
         f"(95% CI [{res['candidate_reranker']['ci'][0]:+.1%}, {res['candidate_reranker']['ci'][1]:+.1%}]); "
         f"success of the picked candidate {res['candidate_reranker']['success_base']:.0%} (baseline) → "
         f"{res['candidate_reranker']['success_reranked']:.0%} (re-ranked), oracle {res['candidate_reranker']['success_oracle']:.0%}.\n",
         "## B-2 premise: is the model worse on planner-visited states?\n",
         f"- First-block prediction error per latent element (one context frame): executed plans "
         f"median {res['onpolicy_err_block1_per_elem_median']:.4f} / mean {res['onpolicy_err_block1_per_elem_mean']:.4f}"
         + (f"; held-out expert clips median {res['expert_err_block1_per_elem']['median']:.4f} / "
            f"mean {res['expert_err_block1_per_elem']['mean']:.4f} "
            f"(×{res['onpolicy_err_block1_per_elem_median'] / res['expert_err_block1_per_elem']['median']:.0f} at the median)."
            if res.get("expert_err_block1_per_elem") else "."),
         ]
    for key in ["s2_mc20"] + (["s2_ens"] if have_ens else []):
        L.append(f"- Across executed plans, Spearman(σ²_k, e_k) for k = 1..5 ({key}): "
                 + ", ".join(f"{v:+.2f}" for v in res[f"across_state_rho_{key}"]))
    (out / "explore.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
