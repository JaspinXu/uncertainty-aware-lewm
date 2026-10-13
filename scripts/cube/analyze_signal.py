"""L1 signal test: is MC-dropout sigma^2 a useful risk signal for Cube planning?

    python scripts/cube/analyze_signal.py                       # all recorded runs
    python scripts/cube/analyze_signal.py --cal-seeds 910 911 912

Inputs : results/cube/records/<run>.pt  (evaluate.py record.enabled=true)
         results/cube/oracle/<run>.pt   (scripts/cube/oracle_replay.py)
Outputs: results/cube/l1/signal_summary.json, signal_report.md, fig_*.png,
         candidates.csv.gz (joined per-candidate table)

Questions (pre-registered; "non-trivial" = cube not already within 4 cm of the
goal at planning time, the only plans where actions matter):

Q1 cost validity   does the planner's own cost rank candidates by real outcome?
Q2 sigma ~ error   does sigma^2 track the model's true prediction error?
Q3 false positives among the CEM elites (candidates the model believes in), does
                   sigma^2 flag the ones that actually fail?  (AUROC, plan-
                   clustered bootstrap CI)  + does it add to the cost itself?
Q4 episodes        does sigma^2 of the executed first plan predict episode failure?
Q5 re-ranking      picking argmin(mc_cost + lambda * sigma^2_H) among the
                   replayed candidates: lambda chosen on the calibration seeds,
                   reported on the locked test seeds (McNemar vs lambda = 0).

Verdict: GO if Q3's AUROC CI excludes 0.5 and Q5 improves on the test seeds;
WEAK-SIGNAL if only Q3 holds; NO-GO otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cube.metrics import (  # noqa: E402
    auroc,
    cluster_bootstrap,
    mcnemar_exact,
    spearman,
    within_group_mean,
)
from cube.tables import SUCCESS_RADIUS, load_runs  # noqa: E402

TEST_SEEDS = [42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055]
CAL_SEEDS = [910, 911, 912]
LAMBDA_MULTS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def auroc_ci(df: pd.DataFrame, score: str, label: str, group: str = "plan_id", n_boot: int = 1000) -> dict:
    s, y, g = df[score].to_numpy(), df[label].to_numpy(), df[group].to_numpy()
    pt, lo, hi = cluster_bootstrap(lambda i: auroc(s[i], y[i]), g, n_boot=n_boot)
    return {"auroc": _f(pt), "ci": [_f(lo), _f(hi)], "n": int(len(df)), "n_pos": int(y.sum()), "n_groups": int(len(np.unique(g)))}


def spearman_ci(df: pd.DataFrame, x: str, y: str, group: str = "plan_id", n_boot: int = 1000) -> dict:
    a, b, g = df[x].to_numpy(), df[y].to_numpy(), df[group].to_numpy()
    pt, lo, hi = cluster_bootstrap(lambda i: spearman(a[i], b[i]), g, n_boot=n_boot)
    within, n = within_group_mean(lambda i: spearman(a[i], b[i]), g)
    return {"pooled": _f(pt), "ci": [_f(lo), _f(hi)], "within_plan_mean": _f(within), "n_plans": n}


# ---------------------------------------------------------------------- #
def q1_cost_validity(c: pd.DataFrame) -> dict:
    out = {"spearman_detcost_finaldist": spearman_ci(c, "det_cost", "final_dist")}
    c = c.assign(failure=~c.success)
    within, n = within_group_mean(
        lambda i: auroc(c.det_cost.to_numpy()[i], c.failure.to_numpy()[i]), c.plan_id.to_numpy()
    )
    out["auroc_detcost_failure"] = {**auroc_ci(c, "det_cost", "failure"), "within_plan_mean": _f(within), "n_plans_mixed": n}
    out["spearman_detcost_truecost"] = spearman_ci(c, "det_cost", "true_cost")
    return out


def q2_sigma_vs_error(c: pd.DataFrame, P: int) -> dict:
    out = {}
    for k in range(1, P + 1):
        if f"pred_err_{k}" in c:
            out[f"step_{k}"] = spearman_ci(c, f"sig2_{k}", f"pred_err_{k}", n_boot=300)
    c = c.assign(cost_gap=(c.det_cost - c.true_cost).abs())
    out["sig2_H_vs_cost_gap"] = spearman_ci(c, "sig2_H", "cost_gap", n_boot=300)
    return out


def cv_auc_gain(df: pd.DataFrame, base: list[str], extra: list[str], label: str, group: str = "plan_id", seed: int = 0) -> dict:
    """Grouped 5-fold CV AUROC of a logistic model with/without extra features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = df.dropna(subset=base + extra + [label])
    y = df[label].to_numpy().astype(int)
    g = df[group].to_numpy()
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 5:
        return {"base_auc": None, "full_auc": None, "gain": None}
    X0 = np.log1p(np.clip(df[base].to_numpy(dtype=float), 0, None))
    X1 = np.log1p(np.clip(df[base + extra].to_numpy(dtype=float), 0, None))
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X0, y, g):
        for X, p in ((X0, p0), (X1, p1)):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            if len(np.unique(y[tr])) < 2:
                p[te] = 0.5
                continue
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
    gain, lo, hi = cluster_bootstrap(lambda i: auroc(p1[i], y[i]) - auroc(p0[i], y[i]), g, n_boot=500, seed=seed)
    return {"base_auc": _f(auroc(p0, y)), "full_auc": _f(auroc(p1, y)), "gain": _f(gain), "gain_ci": [_f(lo), _f(hi)]}


def q3_false_positives(c: pd.DataFrame) -> dict:
    e = c[c.is_elite].assign(failure=lambda d: ~d.success)
    out = {"elite_failure_rate": _f(e.failure.mean()) if len(e) else None}
    for score in ("sig2_H", "sig2_1", "mc_cost_std", "det_cost"):
        within, n = within_group_mean(
            lambda i, s=score: auroc(e[s].to_numpy()[i], e.failure.to_numpy()[i]), e.plan_id.to_numpy()
        )
        out[score] = {**auroc_ci(e, score, "failure"), "within_plan_mean": _f(within), "n_plans_mixed": n}
    out["logit_gain_sig2_over_cost"] = cv_auc_gain(e, ["det_cost"], ["sig2_H"], "failure")
    return out


def q4_episodes(ep: pd.DataFrame) -> dict:
    ep = ep.assign(failure=~ep.success)
    out = {"n": int(len(ep)), "failure_rate": _f(ep.failure.mean()) if len(ep) else None}
    for score in ("first_exec_sig2_H", "first_exec_det_cost", "first_exec_mc_cost_std", "first_elite_cost_std", "first_start_dist"):
        if score in ep and ep[score].notna().any():
            out[score] = auroc_ci(ep.assign(gid=ep.run + "/" + ep.env.astype(str)), score, "failure", group="gid")
    ep2 = ep.assign(gid=ep.run + "/" + ep.env.astype(str))
    out["logit_gain_sig2_over_cost_and_difficulty"] = cv_auc_gain(
        ep2, ["first_exec_det_cost", "first_start_dist"], ["first_exec_sig2_H"], "failure", group="gid"
    )
    return out


def rerank_success(c: pd.DataFrame, lam: float, cost: str = "mc_cost") -> pd.Series:
    """Per-plan success of argmin(cost + lam * sig2_H) among replayed candidates."""
    score = c[cost] + lam * c["sig2_H"]
    idx = score.groupby(c.plan_id).idxmin()
    return c.loc[idx.values].set_index("plan_id")["success"]


def q5_rerank(c: pd.DataFrame, cal_runs: list[str], test_runs: list[str]) -> dict:
    cal, test = c[c.run.isin(cal_runs)], c[c.run.isin(test_runs)]
    ref = cal if len(cal) else c
    ratio = float(np.median(ref.mc_cost) / max(np.median(ref.sig2_H), 1e-12))
    grid = [m * ratio for m in LAMBDA_MULTS]
    out = {"ratio_cost_over_sigma2": ratio, "lambda_grid": grid, "multipliers": LAMBDA_MULTS}
    out["cal_success"] = [_f(rerank_success(cal, l).mean()) if len(cal) else None for l in grid]
    out["test_success"] = [_f(rerank_success(test, l).mean()) if len(test) else None for l in grid]
    if len(cal):
        vals = np.array([v if v is not None else -1 for v in out["cal_success"]])
        best = int(np.flatnonzero(vals == vals.max())[0])  # ties -> smallest lambda
    else:
        best = 0
    out["lambda_star"] = grid[best]
    out["lambda_star_multiplier"] = LAMBDA_MULTS[best]
    for name, df in (("cal", cal), ("test", test)):
        if not len(df):
            continue
        s0, s1 = rerank_success(df, 0.0), rerank_success(df, grid[best])
        ex = df[df.is_exec].set_index("plan_id")["success"]
        upper = df.groupby("plan_id")["success"].any()
        out[name] = {
            "n_plans": int(len(s0)),
            "exec_success": _f(ex.mean()),
            "lambda0_success": _f(s0.mean()),
            "lambda_star_success": _f(s1.loc[s0.index].mean()),
            "oracle_best_in_pool": _f(upper.mean()),
            "mcnemar_lambda_star_vs_0": mcnemar_exact(s0.to_numpy(), s1.loc[s0.index].to_numpy()),
        }
    return out


def verdict(q3: dict, q5: dict) -> str:
    sig = q3.get("sig2_H", {})
    ci = sig.get("ci", [None, None])
    signal = ci[0] is not None and ci[0] > 0.5
    test = q5.get("test", {})
    useful = (
        test.get("lambda_star_success") is not None
        and test.get("lambda0_success") is not None
        and q5.get("lambda_star_multiplier", 0) > 0
        and test["lambda_star_success"] > test["lambda0_success"]
    )
    if signal and useful:
        return "GO"
    if signal:
        return "WEAK-SIGNAL"
    return "NO-GO"


# ---------------------------------------------------------------------- #
def make_figures(c: pd.DataFrame, ep: pd.DataFrame, q5: dict, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    if "pred_err_H" in c and len(c):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sub = c.sample(min(len(c), 5000), random_state=0)
        for flag, col, lab in ((False, "#999999", "random/exec"), (True, "#d62728", "elite")):
            s = sub[sub.is_elite == flag]
            ax.scatter(s.sig2_H, s.pred_err_H, s=4, alpha=0.4, c=col, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"MC-dropout $\sigma^2_H$")
        ax.set_ylabel(r"true error $\|\hat z_H - z_H\|^2$")
        ax.set_title(f"Spearman {spearman(c.sig2_H, c.pred_err_H):.2f}")
        ax.legend(markerscale=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_sigma_vs_error.png", dpi=150)
        plt.close(fig)
        files.append("fig_sigma_vs_error.png")

        e = c[c.is_elite]
        if e.success.nunique() == 2:
            from sklearn.metrics import roc_curve

            fig, ax = plt.subplots(figsize=(4.2, 4))
            for score in ("sig2_H", "mc_cost_std", "det_cost"):
                fpr, tpr, _ = roc_curve(~e.success, e[score])
                ax.plot(fpr, tpr, label=f"{score} ({auroc(e[score], ~e.success):.2f})")
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlabel("false positive rate")
            ax.set_ylabel("true positive rate")
            ax.set_title("elite candidates: predicting failure")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_elite_failure_roc.png", dpi=150)
            plt.close(fig)
            files.append("fig_elite_failure_roc.png")

    if q5.get("lambda_grid"):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        x = [max(m, 1e-3) for m in q5["multipliers"]]
        for key, lab in (("cal_success", "calibration seeds"), ("test_success", "test seeds")):
            y = [np.nan if v is None else 100 * v for v in q5[key]]
            ax.plot(x, y, marker="o", label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ / (median cost / median $\sigma^2$)   (0 plotted at 1e-3)")
        ax.set_ylabel("success of re-ranked choice [%]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_rerank_lambda.png", dpi=150)
        plt.close(fig)
        files.append("fig_rerank_lambda.png")

    if len(ep) and ep.first_exec_sig2_H.notna().any():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        bins = np.logspace(np.log10(max(ep.first_exec_sig2_H.min(), 1e-6)), np.log10(ep.first_exec_sig2_H.max() + 1e-6), 25)
        for ok, col in ((True, "#2ca02c"), (False, "#d62728")):
            ax.hist(ep[ep.success == ok].first_exec_sig2_H, bins=bins, alpha=0.6, color=col, label="success" if ok else "failure")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma^2_H$ of executed first plan")
        ax.set_ylabel("episodes (non-trivial)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_episode_sigma.png", dpi=150)
        plt.close(fig)
        files.append("fig_episode_sigma.png")
    return files


def fmt(v, pct=False, nd=3):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def fci(ci):
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_report(summary: dict, out: Path) -> None:
    s = summary
    q3, q5, q4 = s["q3_false_positives"], s["q5_rerank"], s["q4_episodes"]
    L = []
    L.append("# L1 signal test: MC-dropout sigma^2 on OGBench-Cube\n")
    L.append(f"Generated by `scripts/cube/analyze_signal.py` from {s['n_runs']} recorded baseline runs "
             f"({s['n_episodes']} episodes, {s['n_plans']} planning calls, {s['n_candidates']} replayed candidates).\n")
    L.append(f"**Verdict: {s['verdict']}**\n")
    L.append("## Protocol checks\n")
    L.append(f"- Oracle fidelity (replayed executed plan vs. state the evaluation reached): {s['fidelity']}")
    d = s["difficulty"]
    L.append(f"- Trivial first plans (cube already within {SUCCESS_RADIUS * 100:.0f} cm): {fmt(d['trivial_first_plan_frac'], True)}")
    L.append(f"- Baseline success: all {fmt(d['success_all'], True)}, non-trivial {fmt(d['success_nontrivial'], True)}, "
             f"trivial {fmt(d['success_trivial'], True)}\n")
    q1 = s["q1_cost_validity"]
    L.append("## Q1 Is the planner's cost a good ranking of real outcomes?\n")
    L.append(f"- Spearman(det cost, final cube distance): pooled {fmt(q1['spearman_detcost_finaldist']['pooled'])} "
             f"CI {fci(q1['spearman_detcost_finaldist']['ci'])}, within-plan mean {fmt(q1['spearman_detcost_finaldist']['within_plan_mean'])}")
    a = q1["auroc_detcost_failure"]
    L.append(f"- AUROC(det cost -> failure): pooled {fmt(a['auroc'])} CI {fci(a['ci'])}, within-plan {fmt(a['within_plan_mean'])} ({a['n_plans_mixed']} mixed plans)\n")
    L.append("## Q2 Does sigma^2 track the true prediction error?\n")
    L.append("| block k | pooled Spearman | 95% CI | within-plan mean |")
    L.append("|---|---:|---|---:|")
    for k, v in s["q2_sigma_vs_error"].items():
        if k.startswith("step_"):
            L.append(f"| {k[5:]} | {fmt(v['pooled'])} | {fci(v['ci'])} | {fmt(v['within_plan_mean'])} |")
    g = s["q2_sigma_vs_error"]["sig2_H_vs_cost_gap"]
    L.append(f"\nSpearman(sigma^2_H, |det cost - true cost|): {fmt(g['pooled'])} CI {fci(g['ci'])}\n")
    L.append("## Q3 Among CEM elites, does sigma^2 flag the candidates that fail?\n")
    L.append(f"Elite failure rate (non-trivial plans): {fmt(q3['elite_failure_rate'], True)}\n")
    L.append("| score | AUROC | 95% CI (plan bootstrap) | within-plan mean | plans with mixed outcome |")
    L.append("|---|---:|---|---:|---:|")
    for k in ("sig2_H", "sig2_1", "mc_cost_std", "det_cost"):
        v = q3[k]
        L.append(f"| {k} | {fmt(v['auroc'])} | {fci(v['ci'])} | {fmt(v['within_plan_mean'])} | {v['n_plans_mixed']} |")
    lg = q3["logit_gain_sig2_over_cost"]
    L.append(f"\nAdding sigma^2_H to the cost in a logistic failure model (grouped 5-fold CV): "
             f"AUROC {fmt(lg['base_auc'])} -> {fmt(lg['full_auc'])} (gain {fmt(lg['gain'])}, CI {fci(lg.get('gain_ci'))})\n")
    L.append("## Q4 Episode level (executed first plan, non-trivial episodes)\n")
    L.append(f"Episodes: {q4['n']}, failure rate {fmt(q4['failure_rate'], True)}\n")
    L.append("| predictor | AUROC (failure) | 95% CI |")
    L.append("|---|---:|---|")
    for k, v in q4.items():
        if isinstance(v, dict) and "auroc" in v:
            L.append(f"| {k} | {fmt(v['auroc'])} | {fci(v['ci'])} |")
    lg = q4["logit_gain_sig2_over_cost_and_difficulty"]
    L.append(f"\nsigma^2 on top of cost + start distance (CV): AUROC {fmt(lg['base_auc'])} -> {fmt(lg['full_auc'])}\n")
    L.append("## Q5 Re-ranking the replayed candidates with a sigma^2 penalty\n")
    L.append(f"lambda grid = multiplier x {q5['ratio_cost_over_sigma2']:.3g} (median cost / median sigma^2 on calibration seeds); "
             f"selected multiplier {q5['lambda_star_multiplier']} (lambda* = {q5['lambda_star']:.4g}).\n")
    L.append("| multiplier | " + " | ".join(str(m) for m in q5["multipliers"]) + " |")
    L.append("|---|" + "---:|" * len(q5["multipliers"]))
    L.append("| calibration | " + " | ".join(fmt(v, True) for v in q5["cal_success"]) + " |")
    L.append("| test | " + " | ".join(fmt(v, True) for v in q5["test_success"]) + " |")
    for name in ("cal", "test"):
        if name in q5:
            v = q5[name]
            m = v["mcnemar_lambda_star_vs_0"]
            L.append(f"\n{name}: {v['n_plans']} non-trivial plans; executed plan {fmt(v['exec_success'], True)}, "
                     f"argmin cost {fmt(v['lambda0_success'], True)}, lambda* {fmt(v['lambda_star_success'], True)}, "
                     f"best in pool (upper bound) {fmt(v['oracle_best_in_pool'], True)}; McNemar p = {m['p_value']:.3g} "
                     f"({m['b_only']} gained / {m['a_only']} lost)")
    if s.get("figures"):
        L.append("\n## Figures\n")
        for f in s["figures"]:
            L.append(f"![{f}]({f})")
    out.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=ROOT / "results/cube/records")
    ap.add_argument("--oracle", type=Path, default=ROOT / "results/cube/oracle")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/l1")
    ap.add_argument("--method", default="baseline", help="only runs named <method>_s<seed>")
    ap.add_argument("--cal-seeds", type=int, nargs="*", default=CAL_SEEDS)
    ap.add_argument("--test-seeds", type=int, nargs="*", default=TEST_SEEDS)
    args = ap.parse_args()

    runs = [f"{args.method}_s{s}" for s in args.cal_seeds + args.test_seeds]
    data = load_runs(args.records, args.oracle, runs)
    c, plans, ep = data["candidates"], data["plans"], data["episodes"]
    if not len(c):
        sys.exit(f"no records found for {runs} in {args.records}")
    if "success" not in c:
        sys.exit("no oracle results joined -- run scripts/cube/oracle_replay.py first")
    c = c.dropna(subset=["success"])
    P = max(int(k.split("_")[1]) for k in c.columns if k.startswith("sig2_") and k[5:].isdigit())

    first = plans[plans.plan_idx == 0]
    trivial_first = first.start_dist <= SUCCESS_RADIUS
    ep_nt = ep[ep.first_start_dist > SUCCESS_RADIUS]
    nt = c[c.start_dist > SUCCESS_RADIUS]
    cal_runs = [f"{args.method}_s{s}" for s in args.cal_seeds]
    test_runs = [f"{args.method}_s{s}" for s in args.test_seeds]

    q3 = q3_false_positives(nt)
    q5 = q5_rerank(nt, cal_runs, test_runs)
    summary = {
        "runs": sorted(c.run.unique().tolist()),
        "n_runs": int(c.run.nunique()),
        "n_episodes": int(len(ep)),
        "n_plans": int(c.plan_id.nunique()),
        "n_candidates": int(len(c)),
        "fidelity": _oracle_fidelity(args.oracle, runs),
        "difficulty": {
            "trivial_first_plan_frac": _f(trivial_first.mean()) if len(first) else None,
            "success_all": _f(ep.success.mean()),
            "success_nontrivial": _f(ep_nt.success.mean()) if len(ep_nt) else None,
            "success_trivial": _f(ep[ep.first_start_dist <= SUCCESS_RADIUS].success.mean()) if len(ep) else None,
            "per_seed_success": {str(k): _f(v) for k, v in ep.groupby("seed").success.mean().items()},
        },
        "q1_cost_validity": q1_cost_validity(nt),
        "q2_sigma_vs_error": q2_sigma_vs_error(nt, P),
        "q3_false_positives": q3,
        "q4_episodes": q4_episodes(ep_nt),
        "q5_rerank": q5,
    }
    summary["verdict"] = verdict(q3, q5)
    args.out.mkdir(parents=True, exist_ok=True)
    summary["figures"] = make_figures(nt, ep_nt, q5, args.out)
    (args.out / "signal_summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, args.out / "signal_report.md")
    c.to_csv(args.out / "candidates.csv.gz", index=False)
    print(json.dumps({k: summary[k] for k in ("verdict", "n_runs", "n_plans", "difficulty")}, indent=1))
    print(f"wrote {args.out / 'signal_report.md'}")


def _oracle_fidelity(oracle_dir: Path, runs: list[str]) -> dict:
    import torch

    worst, n = 0.0, 0
    for r in runs:
        p = oracle_dir / f"{r}.pt"
        if not p.exists():
            continue
        f = torch.load(p, map_location="cpu", weights_only=False)["meta"].get("fidelity", {})
        if f.get("n"):
            n += f["n"]
            worst = max(worst, f["max_abs_qpos_diff"])
    return {"n_checked": n, "max_abs_qpos_diff": worst}


if __name__ == "__main__":
    main()
    os._exit(0)
