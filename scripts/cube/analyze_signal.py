"""L1 signal test: is sigma^2 a useful risk signal for Cube planning?

    python scripts/cube/analyze_signal.py                       # all recorded runs
    python scripts/cube/analyze_signal.py --cal-seeds 910 911 912

Inputs : results/cube/records/<run>.pt   (evaluate.py record.enabled=true)
         results/cube/oracle/<run>.pt    (scripts/cube/oracle_replay.py)
         results/cube/ensemble/<run>.pt  (scripts/cube/score_ensemble.py; optional)
Outputs: results/cube/l1/signal_summary.json, signal_report.md, fig_*.png,
         candidates.csv.gz (joined per-candidate table)

Two uncertainty signals are tested side by side when available:
  mc   MC-dropout variance of the pretrained predictor (8 masks)
  ens  disagreement of the 5-member latent dynamics ensemble (ua/ensemble.py)

Questions (pre-registered; "non-trivial" = cube not already within 4 cm of the
goal at planning time, the only plans where actions matter):

Q1 cost validity   does the planner's own cost rank candidates by real outcome?
Q2 sigma ~ error   does sigma^2 track the planner's true prediction error?
Q3 false positives among the CEM elites (candidates the model believes in), does
                   sigma^2 flag the ones that actually fail?  (AUROC, plan-
                   clustered bootstrap CI)  + does it add to the cost itself?
Q4 episodes        does sigma^2 of the executed first plan predict episode failure?
Q5 re-ranking      argmin(cost + lambda * sigma^2_H) among the replayed
                   candidates: lambda chosen on the calibration seeds, reported
                   on the locked test seeds (McNemar vs lambda = 0).
Q6 failure modes   when the executed plan fails, did a better candidate exist
                   in the replayed pool (selection) or not (search / model)?

Verdict per signal: GO if Q3's AUROC CI excludes 0.5 and Q5 improves on the
test seeds; WEAK-SIGNAL if only Q3 holds; NO-GO otherwise.
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
# signal -> (terminal column, per-block column prefix, planner cost it is added to)
SIGNALS = {
    "mc": {"label": "MC-dropout", "H": "sig2_H", "step": "sig2_", "cost": "mc_cost"},
    "ens": {"label": "latent ensemble", "H": "sig2ens_H", "step": "sig2ens_", "cost": "det_cost"},
}


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def available(c: pd.DataFrame) -> list[str]:
    return [s for s, v in SIGNALS.items() if v["H"] in c and c[v["H"]].notna().any()]


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
def fidelity(c: pd.DataFrame, ep: pd.DataFrame, oracle_meta: dict) -> dict:
    """Replay fidelity + agreement of the oracle with the real closed-loop outcome.

    The real episode succeeds iff one of its executed plans, replayed from the
    state where it was planned, solves the task (plans are executed open loop).
    """
    ex = c[c.is_exec]
    by_ep = ex.groupby(["run", "env"]).success.any()
    real = ep.set_index(["run", "env"]).success
    common = by_ep.index.intersection(real.index)
    agree = float((by_ep.loc[common] == real.loc[common]).mean()) if len(common) else None
    return {**oracle_meta, "episodes_compared": int(len(common)), "oracle_vs_closed_loop_agreement": agree}


def q1_cost_validity(c: pd.DataFrame) -> dict:
    out = {"spearman_detcost_finaldist": spearman_ci(c, "det_cost", "final_dist")}
    c = c.assign(failure=~c.success)
    within, n = within_group_mean(
        lambda i: auroc(c.det_cost.to_numpy()[i], c.failure.to_numpy()[i]), c.plan_id.to_numpy()
    )
    out["auroc_detcost_failure"] = {**auroc_ci(c, "det_cost", "failure"), "within_plan_mean": _f(within), "n_plans_mixed": n}
    out["spearman_detcost_truecost"] = spearman_ci(c, "det_cost", "true_cost")
    return out


def q2_sigma_vs_error(c: pd.DataFrame, P: int, signals: list[str]) -> dict:
    out = {}
    c = c.assign(cost_gap=(c.det_cost - c.true_cost).abs())
    for s in signals:
        v = SIGNALS[s]
        d = {}
        for k in range(1, P + 1):
            col = f"{v['step']}{k}"
            if f"pred_err_{k}" in c and col in c:
                d[f"step_{k}"] = spearman_ci(c, col, f"pred_err_{k}", n_boot=300)
        d["H_vs_cost_gap"] = spearman_ci(c, v["H"], "cost_gap", n_boot=300)
        out[s] = d
    if "ens_pred_err_H" in c:
        out["prediction_error_H"] = {
            "pretrained_median": _f(c.pred_err_H.median()),
            "ensemble_mean_median": _f(c.ens_pred_err_H.median()),
        }
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
        return {"base_auc": None, "full_auc": None, "gain": None, "gain_ci": [None, None]}
    X0 = np.log1p(np.clip(df[base].to_numpy(dtype=float), 0, None))
    X1 = np.log1p(np.clip(df[base + extra].to_numpy(dtype=float), 0, None))
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X0, y, g):
        for X, p in ((X0, p0), (X1, p1)):
            if len(np.unique(y[tr])) < 2:
                p[te] = 0.5
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
    gain, lo, hi = cluster_bootstrap(lambda i: auroc(p1[i], y[i]) - auroc(p0[i], y[i]), g, n_boot=500, seed=seed)
    return {"base_auc": _f(auroc(p0, y)), "full_auc": _f(auroc(p1, y)), "gain": _f(gain), "gain_ci": [_f(lo), _f(hi)]}


def q3_false_positives(c: pd.DataFrame, signals: list[str]) -> dict:
    e = c[c.is_elite].assign(failure=lambda d: ~d.success)
    out = {"elite_failure_rate": _f(e.failure.mean()) if len(e) else None, "scores": {}, "logit_gain": {}}
    scores = [SIGNALS[s]["H"] for s in signals] + ["sig2_1", "mc_cost_std", "det_cost"]
    for score in scores:
        within, n = within_group_mean(
            lambda i, s=score: auroc(e[s].to_numpy()[i], e.failure.to_numpy()[i]), e.plan_id.to_numpy()
        )
        out["scores"][score] = {**auroc_ci(e, score, "failure"), "within_plan_mean": _f(within), "n_plans_mixed": n}
    for s in signals:
        out["logit_gain"][s] = cv_auc_gain(e, ["det_cost"], [SIGNALS[s]["H"]], "failure")
    return out


def q4_episodes(ep: pd.DataFrame, signals: list[str]) -> dict:
    ep = ep.assign(failure=~ep.success, gid=ep.run + "/" + ep.env.astype(str))
    out = {"n": int(len(ep)), "failure_rate": _f(ep.failure.mean()) if len(ep) else None, "scores": {}, "logit_gain": {}}
    cols = [f"first_exec_{SIGNALS[s]['H']}" for s in signals]
    cols += ["first_exec_det_cost", "first_exec_mc_cost_std", "first_elite_cost_std", "first_start_dist"]
    for score in cols:
        if score in ep and ep[score].notna().any():
            out["scores"][score] = auroc_ci(ep, score, "failure", group="gid")
    for s in signals:
        out["logit_gain"][s] = cv_auc_gain(
            ep, ["first_exec_det_cost", "first_start_dist"], [f"first_exec_{SIGNALS[s]['H']}"], "failure", group="gid"
        )
    return out


def rerank_success(c: pd.DataFrame, lam: float, cost: str, sig: str) -> pd.Series:
    """Per-plan success of argmin(cost + lam * sig) among replayed candidates."""
    score = c[cost] + lam * c[sig]
    idx = score.groupby(c.plan_id).idxmin()
    return c.loc[idx.values].set_index("plan_id")["success"]


def q5_rerank(c: pd.DataFrame, cal_runs: list[str], test_runs: list[str], signal: str) -> dict:
    cost, sig = SIGNALS[signal]["cost"], SIGNALS[signal]["H"]
    cal, test = c[c.run.isin(cal_runs)], c[c.run.isin(test_runs)]
    ref = cal if len(cal) else c
    ratio = float(np.median(ref[cost]) / max(np.median(ref[sig]), 1e-12))
    grid = [m * ratio for m in LAMBDA_MULTS]
    out = {"cost": cost, "ratio_cost_over_sigma2": ratio, "lambda_grid": grid, "multipliers": LAMBDA_MULTS}
    out["cal_success"] = [_f(rerank_success(cal, l, cost, sig).mean()) if len(cal) else None for l in grid]
    out["test_success"] = [_f(rerank_success(test, l, cost, sig).mean()) if len(test) else None for l in grid]
    best = 0
    if len(cal):
        vals = np.array([v if v is not None else -1 for v in out["cal_success"]])
        best = int(np.flatnonzero(vals == vals.max())[0])  # ties -> smallest lambda
    out["lambda_star"] = grid[best]
    out["lambda_star_multiplier"] = LAMBDA_MULTS[best]
    for name, df in (("cal", cal), ("test", test)):
        if not len(df):
            continue
        s0 = rerank_success(df, 0.0, cost, sig)
        s1 = rerank_success(df, grid[best], cost, sig).loc[s0.index]
        ex = df[df.is_exec].set_index("plan_id")["success"]
        upper = df.groupby("plan_id")["success"].any()
        out[name] = {
            "n_plans": int(len(s0)),
            "exec_success": _f(ex.mean()),
            "lambda0_success": _f(s0.mean()),
            "lambda_star_success": _f(s1.mean()),
            "oracle_best_in_pool": _f(upper.mean()),
            "mcnemar_lambda_star_vs_0": mcnemar_exact(s0.to_numpy(), s1.to_numpy()),
        }
    return out


def q6_failure_modes(c: pd.DataFrame) -> dict:
    """Why do executed plans fail? (non-trivial plans)

    selection failure: some replayed candidate solves the task, the planner picked another
    search failure:    none of the 51 replayed candidates (exec, elites, randoms) solves it
    """
    ex = c[c.is_exec].set_index("plan_id")
    any_ok = c.groupby("plan_id").success.any().loc[ex.index]
    out = {}
    for name, mask in (("all", np.ones(len(ex), bool)), ("first_plan", (ex.plan_idx == 0).to_numpy()),
                       ("second_plan", (ex.plan_idx == 1).to_numpy())):
        e, a = ex[mask], any_ok[mask]
        fail = ~e.success
        out[name] = {
            "n_plans": int(len(e)),
            "exec_success": _f(e.success.mean()) if len(e) else None,
            "n_exec_failures": int(fail.sum()),
            "selection_failure_frac": _f(a[fail].mean()) if fail.any() else None,
            "search_failure_frac": _f((~a[fail]).mean()) if fail.any() else None,
            "exec_true_minus_pred_cost_median_failed": _f((e.true_cost - e.det_cost)[fail].median()) if fail.any() else None,
            "exec_true_minus_pred_cost_median_succeeded": _f((e.true_cost - e.det_cost)[~fail].median()) if (~fail).any() else None,
        }
    return out


def verdict(q3: dict, q5: dict, signal: str) -> str:
    ci = q3["scores"].get(SIGNALS[signal]["H"], {}).get("ci", [None, None])
    signal_ok = ci[0] is not None and ci[0] > 0.5
    test = q5.get("test", {})
    useful = (
        test.get("lambda_star_success") is not None
        and test.get("lambda0_success") is not None
        and q5.get("lambda_star_multiplier", 0) > 0
        and test["lambda_star_success"] > test["lambda0_success"]
    )
    if signal_ok and useful:
        return "GO"
    return "WEAK-SIGNAL" if signal_ok else "NO-GO"


# ---------------------------------------------------------------------- #
def make_figures(c: pd.DataFrame, ep: pd.DataFrame, q5: dict, signals: list[str], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    if "pred_err_H" in c and len(c):
        fig, axes = plt.subplots(1, len(signals), figsize=(4.5 * len(signals), 4), squeeze=False)
        sub = c.sample(min(len(c), 5000), random_state=0)
        for ax, s in zip(axes[0], signals):
            col = SIGNALS[s]["H"]
            for flag, colr, lab in ((False, "#999999", "random/exec"), (True, "#d62728", "elite")):
                d = sub[sub.is_elite == flag]
                ax.scatter(d[col], d.pred_err_H, s=4, alpha=0.4, c=colr, label=lab)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(rf"{SIGNALS[s]['label']} $\sigma^2_H$")
            ax.set_ylabel(r"true error $\|\hat z_H - z_H\|^2$")
            ax.set_title(f"Spearman {spearman(c[col], c.pred_err_H):.2f}")
            ax.legend(markerscale=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_sigma_vs_error.png", dpi=150)
        plt.close(fig)
        files.append("fig_sigma_vs_error.png")

        e = c[c.is_elite]
        if e.success.nunique() == 2:
            from sklearn.metrics import roc_curve

            fig, ax = plt.subplots(figsize=(4.2, 4))
            for score in [SIGNALS[s]["H"] for s in signals] + ["mc_cost_std", "det_cost"]:
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

    if q5:
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        for s, r in q5.items():
            x = [max(m, 1e-3) for m in r["multipliers"]]
            for key, ls in (("cal_success", "--"), ("test_success", "-")):
                y = [np.nan if v is None else 100 * v for v in r[key]]
                ax.plot(x, y, ls, marker="o", label=f"{s}: {'cal' if key.startswith('cal') else 'test'}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ / (median cost / median $\sigma^2$)   (0 plotted at 1e-3)")
        ax.set_ylabel("success of re-ranked choice [%]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_rerank_lambda.png", dpi=150)
        plt.close(fig)
        files.append("fig_rerank_lambda.png")

    if len(ep) and "first_exec_sig2_H" in ep and ep.first_exec_sig2_H.notna().any():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        v = ep.first_exec_sig2_H.dropna()
        bins = np.logspace(np.log10(max(v.min(), 1e-6)), np.log10(v.max() + 1e-6), 25)
        for ok, colr in ((True, "#2ca02c"), (False, "#d62728")):
            ax.hist(ep[ep.success == ok].first_exec_sig2_H.dropna(), bins=bins, alpha=0.6, color=colr,
                    label="success" if ok else "failure")
        ax.set_xscale("log")
        ax.set_xlabel(r"MC $\sigma^2_H$ of executed first plan")
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


def write_report(s: dict, out: Path) -> None:
    sigs = s["signals"]
    L = ["# L1 signal test: is sigma^2 a useful risk signal on OGBench-Cube?\n"]
    L.append(f"Generated by `scripts/cube/analyze_signal.py` from {s['n_runs']} recorded baseline runs "
             f"({s['n_episodes']} episodes, {s['n_plans']} planning calls, {s['n_candidates']} replayed candidates). "
             f"Signals: " + ", ".join(f"{k} = {SIGNALS[k]['label']}" for k in sigs) + ".\n")
    L.append("**Verdict: " + ", ".join(f"{SIGNALS[k]['label']} {v}" for k, v in s["verdicts"].items()) + "**\n")
    f = s["fidelity"]
    d = s["difficulty"]
    L.append("## Protocol checks\n")
    L.append(f"- Oracle replay fidelity: executed plan replayed vs. state the evaluation reached, max |Δqpos| = "
             f"{f.get('max_abs_qpos_diff')} over {f.get('n_checked')} transitions")
    L.append(f"- Oracle vs. closed-loop outcome: {fmt(f.get('oracle_vs_closed_loop_agreement'), True)} agreement over "
             f"{f.get('episodes_compared')} episodes")
    L.append(f"- Trivial first plans (cube already within {SUCCESS_RADIUS * 100:.0f} cm): {fmt(d['trivial_first_plan_frac'], True)}")
    L.append(f"- Baseline success: all {fmt(d['success_all'], True)}, non-trivial {fmt(d['success_nontrivial'], True)}, "
             f"trivial {fmt(d['success_trivial'], True)}\n")
    q1 = s["q1_cost_validity"]
    L.append("## Q1 Is the planner's cost a good ranking of real outcomes?\n")
    sp = q1["spearman_detcost_finaldist"]
    L.append(f"- Spearman(LeWM cost, final cube distance): pooled {fmt(sp['pooled'])} CI {fci(sp['ci'])}, within-plan mean {fmt(sp['within_plan_mean'])}")
    a = q1["auroc_detcost_failure"]
    L.append(f"- AUROC(LeWM cost -> failure): pooled {fmt(a['auroc'])} CI {fci(a['ci'])}, within-plan {fmt(a['within_plan_mean'])} ({a['n_plans_mixed']} mixed plans)\n")

    q2 = s["q2_sigma_vs_error"]
    L.append("## Q2 Does sigma^2 track the planner's true prediction error?\n")
    L.append("Spearman between sigma^2 at block k and ||ẑ_k − z_k||² (pooled, plan-bootstrap 95% CI; within-plan mean).\n")
    L.append("| block k | " + " | ".join(SIGNALS[k]["label"] for k in sigs) + " |")
    L.append("|---|" + "---|" * len(sigs))
    steps = sorted({key for k in sigs for key in q2[k] if key.startswith("step_")}, key=lambda x: int(x[5:]))
    for st in steps:
        cells = []
        for k in sigs:
            v = q2[k].get(st)
            cells.append(f"{fmt(v['pooled'])} {fci(v['ci'])}; {fmt(v['within_plan_mean'])}" if v else "n/a")
        L.append(f"| {st[5:]} | " + " | ".join(cells) + " |")
    L.append("| |cost − true cost| | " + " | ".join(
        f"{fmt(q2[k]['H_vs_cost_gap']['pooled'])} {fci(q2[k]['H_vs_cost_gap']['ci'])}" for k in sigs) + " |")
    if "prediction_error_H" in q2:
        pe = q2["prediction_error_H"]
        L.append(f"\nMedian terminal prediction error: pretrained {fmt(pe['pretrained_median'])}, ensemble mean {fmt(pe['ensemble_mean_median'])}\n")

    q3 = s["q3_false_positives"]
    L.append("\n## Q3 Among CEM elites, does sigma^2 flag the candidates that fail?\n")
    L.append(f"Elite failure rate (non-trivial plans): {fmt(q3['elite_failure_rate'], True)}\n")
    L.append("| score | AUROC | 95% CI (plan bootstrap) | within-plan mean | plans with mixed outcome |")
    L.append("|---|---:|---|---:|---:|")
    for k, v in q3["scores"].items():
        L.append(f"| {k} | {fmt(v['auroc'])} | {fci(v['ci'])} | {fmt(v['within_plan_mean'])} | {v['n_plans_mixed']} |")
    for k, lg in q3["logit_gain"].items():
        L.append(f"\n{SIGNALS[k]['label']}: adding sigma^2_H to the LeWM cost in a logistic failure model (grouped 5-fold CV): "
                 f"AUROC {fmt(lg['base_auc'])} -> {fmt(lg['full_auc'])} (gain {fmt(lg['gain'])}, CI {fci(lg.get('gain_ci'))})")

    q4 = s["q4_episodes"]
    L.append("\n## Q4 Episode level (executed first plan, non-trivial episodes)\n")
    L.append(f"Episodes: {q4['n']}, failure rate {fmt(q4['failure_rate'], True)}\n")
    L.append("| predictor | AUROC (failure) | 95% CI |")
    L.append("|---|---:|---|")
    for k, v in q4["scores"].items():
        L.append(f"| {k} | {fmt(v['auroc'])} | {fci(v['ci'])} |")
    for k, lg in q4["logit_gain"].items():
        L.append(f"\n{SIGNALS[k]['label']} on top of cost + start distance (CV): AUROC {fmt(lg['base_auc'])} -> {fmt(lg['full_auc'])}")

    L.append("\n## Q5 Re-ranking the replayed candidates with a sigma^2 penalty\n")
    for k, q5 in s["q5_rerank"].items():
        L.append(f"### {SIGNALS[k]['label']}: argmin({q5['cost']} + lambda * sigma^2_H)\n")
        L.append(f"lambda = multiplier x {q5['ratio_cost_over_sigma2']:.3g} (median cost / median sigma^2, calibration seeds); "
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
        L.append("")
    q6 = s.get("q6_failure_modes", {})
    if q6:
        L.append("## Q6 Why do executed plans fail?\n")
        L.append("Selection failure = another replayed candidate (of the executed plan, 30 elites, 20 randoms) solves the task; "
                 "search failure = none does. Cost gap = true latent cost − predicted cost of the executed plan (median).\n")
        L.append("| plans | n | executed success | failures | selection failures | search failures | cost gap (failed) | cost gap (succeeded) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, v in q6.items():
            L.append(f"| {name} | {v['n_plans']} | {fmt(v['exec_success'], True)} | {v['n_exec_failures']} | "
                     f"{fmt(v['selection_failure_frac'], True)} | {fmt(v['search_failure_frac'], True)} | "
                     f"{fmt(v['exec_true_minus_pred_cost_median_failed'])} | {fmt(v['exec_true_minus_pred_cost_median_succeeded'])} |")
        L.append("")
    if s.get("figures"):
        L.append("## Figures\n")
        for fig in s["figures"]:
            L.append(f"![{fig}]({fig})")
    out.write_text("\n".join(L) + "\n")


def oracle_fidelity_meta(oracle_dir: Path, runs: list[str]) -> dict:
    import torch

    worst, n = 0.0, 0
    for r in runs:
        p = oracle_dir / f"{r}.pt"
        if not p.exists():
            continue
        fm = torch.load(p, map_location="cpu", weights_only=False)["meta"].get("fidelity", {})
        if fm.get("n"):
            n += fm["n"]
            worst = max(worst, fm["max_abs_qpos_diff"])
    return {"n_checked": n, "max_abs_qpos_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=ROOT / "results/cube/records")
    ap.add_argument("--oracle", type=Path, default=ROOT / "results/cube/oracle")
    ap.add_argument("--ensemble", type=Path, default=ROOT / "results/cube/ensemble")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/l1")
    ap.add_argument("--method", default="baseline", help="only runs named <method>_s<seed>")
    ap.add_argument("--cal-seeds", type=int, nargs="*", default=CAL_SEEDS)
    ap.add_argument("--test-seeds", type=int, nargs="*", default=TEST_SEEDS)
    args = ap.parse_args()

    runs = [f"{args.method}_s{s}" for s in args.cal_seeds + args.test_seeds]
    data = load_runs(args.records, args.oracle, runs, ensemble_dir=args.ensemble)
    c, plans, ep = data["candidates"], data["plans"], data["episodes"]
    if not len(c):
        sys.exit(f"no records found for {runs} in {args.records}")
    if "success" not in c:
        sys.exit("no oracle results joined -- run scripts/cube/oracle_replay.py first")
    c = c.dropna(subset=["success"])
    signals = available(c)
    P = max(int(k.split("_")[1]) for k in c.columns if k.startswith("sig2_") and k[5:].isdigit())

    first = plans[plans.plan_idx == 0]
    ep_nt = ep[ep.first_start_dist > SUCCESS_RADIUS]
    nt = c[c.start_dist > SUCCESS_RADIUS]
    cal_runs = [f"{args.method}_s{s}" for s in args.cal_seeds]
    test_runs = [f"{args.method}_s{s}" for s in args.test_seeds]

    q3 = q3_false_positives(nt, signals)
    q5 = {s: q5_rerank(nt, cal_runs, test_runs, s) for s in signals}
    summary = {
        "runs": sorted(c.run.unique().tolist()),
        "signals": signals,
        "n_runs": int(c.run.nunique()),
        "n_episodes": int(len(ep)),
        "n_plans": int(c.plan_id.nunique()),
        "n_candidates": int(len(c)),
        "fidelity": fidelity(c, ep, oracle_fidelity_meta(args.oracle, runs)),
        "difficulty": {
            "trivial_first_plan_frac": _f((first.start_dist <= SUCCESS_RADIUS).mean()) if len(first) else None,
            "success_all": _f(ep.success.mean()),
            "success_nontrivial": _f(ep_nt.success.mean()) if len(ep_nt) else None,
            "success_trivial": _f(ep[ep.first_start_dist <= SUCCESS_RADIUS].success.mean()) if len(ep) else None,
            "per_seed_success": {str(k): _f(v) for k, v in ep.groupby("seed").success.mean().items()},
        },
        "q1_cost_validity": q1_cost_validity(nt),
        "q2_sigma_vs_error": q2_sigma_vs_error(nt, P, signals),
        "q3_false_positives": q3,
        "q4_episodes": q4_episodes(ep_nt, signals),
        "q5_rerank": q5,
        "q6_failure_modes": q6_failure_modes(nt),
        "verdicts": {s: verdict(q3, q5[s], s) for s in signals},
    }
    # backwards-compatible top-level fields used by l2_params.py / make_report.py
    summary["verdict"] = summary["verdicts"].get("mc", next(iter(summary["verdicts"].values()), "n/a"))
    args.out.mkdir(parents=True, exist_ok=True)
    summary["figures"] = make_figures(nt, ep_nt, q5, signals, args.out)
    (args.out / "signal_summary.json").write_text(json.dumps(summary, indent=1))
    write_report(summary, args.out / "signal_report.md")
    c.to_csv(args.out / "candidates.csv.gz", index=False)
    print(json.dumps({k: summary[k] for k in ("verdicts", "n_runs", "n_plans", "difficulty", "fidelity")}, indent=1))
    print(f"wrote {args.out / 'signal_report.md'}")


if __name__ == "__main__":
    main()
    os._exit(0)
