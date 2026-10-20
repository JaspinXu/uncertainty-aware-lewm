"""Step 1 / decision: apply the pre-registered criteria and write the report.

    python scripts/step1_decide.py

Reads results/cube/step1/{closed_loop,plan_replays,candidate_replays}/ and
scores.pt; writes results/cube/step1/{decision.json, report.md, fig_*.png}.
Criteria and thresholds: docs/step1_preregistration.md, ua/step1/stats.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
from dataclasses import asdict  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from ua.step1 import stats  # noqa: E402
from ua.step1.common import out_dir  # noqa: E402
from ua.step1.protocol import SUCCESS_RADIUS  # noqa: E402

ESTIMATORS = {"mc20": "MC-dropout (N=20)", "ens": "Ensemble (M=5)", "mc5": "MC-dropout (N=5)"}
SIGMA_KEY = {"mc20": "s2_mc20", "ens": "s2_ens", "mc5": "s2_mc5"}

# figure style (dataviz reference palette, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"mc20": "#2a78d6", "ens": "#eb6834", "mc5": "#2a78d6"}


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def load_all(cfg, out):
    logs = {int(s): torch.load(out / "closed_loop" / f"seed_{s}.pt", weights_only=False) for s in cfg.step1.seeds}
    reps = {int(s): torch.load(out / "plan_replays" / f"seed_{s}.pt", weights_only=False) for s in cfg.step1.seeds}
    scores = torch.load(out / "scores.pt", weights_only=False)
    return logs, reps, scores


def plan_table(logs, reps, scores):
    """One row per executed plan (replan) with labels and scores."""
    rows = {k: [] for k in ("seed", "env", "t", "y", "group", "d_pred", "d0", "d_real", "e5",
                            "s2_mc20", "s2_mc5", "s2_ens", "final_dist_replay")}
    agree = []
    for seed, log in logs.items():
        sc, rep = scores["plans"][seed], reps[seed]
        traj, goal = log["cube_traj"], log["goal_pos"]
        for i, r in enumerate(log["records"]):
            t0, e = r["t"], r["env"]
            seg = np.linalg.norm(traj[e, t0 + 1 : t0 + 26] - goal[e], axis=1)
            y = int(seg.min() > SUCCESS_RADIUS)  # segment failure
            rep_d = np.linalg.norm(rep["pos"][i] - r["goal_pos"], axis=1)
            agree.append(int(rep_d.min() > SUCCESS_RADIUS) == y)
            rows["seed"].append(seed)
            rows["env"].append(e)
            rows["t"].append(t0)
            rows["y"].append(y)
            rows["group"].append(seed * 1000 + e)
            rows["final_dist_replay"].append(rep_d[-1])
            for k in ("d_pred", "d0", "d_real"):
                rows[k].append(float(sc[k][i]))
            rows["e5"].append(float(sc["err"][i, -1]))
            for k in ("s2_mc20", "s2_mc5", "s2_ens"):
                rows[k].append(float(sc[k][i, -1]) if k in sc else np.nan)
    table = {k: np.asarray(v) for k, v in rows.items()}
    return table, float(np.mean(agree))


def episode_table(logs, plans):
    eps = []
    for seed, log in logs.items():
        for e in range(len(log["success"])):
            m = (plans["seed"] == seed) & (plans["env"] == e)
            order = np.argsort(plans["t"][m])
            eps.append({
                "seed": seed, "env": e, "success": bool(log["success"][e]),
                "init_dist": float(log["init_dist"][e]),
                "final_dist": float(np.linalg.norm(log["cube_traj"][e, -1] - log["goal_pos"][e])),
                "d_pred": plans["d_pred"][m][order].tolist(),
                "d_real": plans["d_real"][m][order].tolist(),
                "d0_first": float(plans["d0"][m][order][0]),
                "d_pred_first": float(plans["d_pred"][m][order][0]),
                **{f"max_{k}": float(np.max(plans[k][m])) for k in ("s2_mc20", "s2_mc5", "s2_ens")},
            })
    return eps


# --------------------------------------------------------------------------- #
# criteria per estimator
# --------------------------------------------------------------------------- #
def run_criteria(est, plans, states, err_key="err"):
    sk = SIGMA_KEY[est]
    if sk not in states[0]:
        return None
    sf, ef, sl, el, o, d, s, dr = [], [], [], [], [], [], [], []
    for st in states:
        first, last = st["pool"] == 0, st["pool"] == 1
        sig, err = st[sk][:, -1].numpy(), st[err_key][:, -1].numpy()
        sf.append(sig[first]), ef.append(err[first]), sl.append(sig[last]), el.append(err[last])
        o.append(st["o"].numpy()[last]), d.append(st["d_pred"].numpy()[last])
        s.append(sig[last]), dr.append(st["d_real"].numpy()[last])
    c1 = stats.criterion_c1(sf, ef, sl, el)
    c2 = stats.criterion_c2(o, d, s, dr)
    c3 = stats.criterion_c3(plans["y"], plans[sk], plans["d_pred"], plans["d0"], plans["group"])
    return {"c1": c1, "c2": c2, "c3": c3, "passed": bool(c1.passed and c2.passed and c3.passed)}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    return x


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(res, plans, tax, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ests = [e for e in ("mc20", "ens") if res.get(e)]
    # C1: per-state rho, one panel per estimator
    fig, axes = plt.subplots(1, len(ests), figsize=(4.6 * len(ests), 3.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, ests):
        c1 = res[e]["c1"]
        _style(ax)
        ax.hist(c1.rho_state, bins=np.linspace(-1, 1, 41), color=COLOR[e], rwidth=0.85)
        top = ax.get_ylim()[1]
        ax.axvline(c1.median, color=INK, linewidth=1.2)
        ax.axvline(stats.RHO_MEDIAN_MIN, color=INK2, linewidth=1.2)
        # median label left of its line, threshold label right of its line
        ax.text(min(c1.median, stats.RHO_MEDIAN_MIN), top * 0.95, f"median {c1.median:.2f} ",
                color=INK2, fontsize=8, va="top", ha="right")
        ax.text(max(c1.median, stats.RHO_MEDIAN_MIN), top * 0.95, " threshold 0.30",
                color=INK2, fontsize=8, va="top", ha="left")
        ax.set_title(f"{ESTIMATORS[e]}: within-state ρ(σ², e₅)", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("per-state Spearman ρ (mean of first/last CEM pools)", color=INK2, fontsize=9)
    axes[0].set_ylabel("states", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_c1_within_state_rho.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # C3: failure rate by sigma^2 decile of the executed plan
    fig, ax = plt.subplots(figsize=(5.4, 3.4), facecolor=SURFACE)
    _style(ax)
    ends = []
    for e in ests:
        s = plans[SIGMA_KEY[e]]
        q = np.quantile(s, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(q, s, side="right") - 1, 0, 9)
        rate = [plans["y"][b == k].mean() for k in range(10)]
        ax.plot(np.arange(1, 11), rate, color=COLOR[e], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=ESTIMATORS[e])
        ends.append(rate[-1])
    for k, v in enumerate(ends):  # end labels, nudged apart when they would touch
        y = v + (0.03 if k and abs(v - ends[0]) < 0.04 else 0.0)
        ax.text(10.25, y, f"{v:.0%}", color=INK2, fontsize=8, va="center")
    base_rate = plans["y"].mean()
    ax.axhline(base_rate, color=INK2, linewidth=1)
    ax.text(5.5, base_rate + 0.025, f"all replans {base_rate:.0%}", color=INK2, fontsize=8, ha="center")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("σ² decile of the executed plan (1 = lowest)", color=INK2, fontsize=9)
    ax.set_ylabel("segment failure rate", color=INK2, fontsize=9)
    ax.set_title("Per-replan failure rate vs σ²", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_c3_failure_by_sigma2.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # gate G: failure taxonomy
    fig, ax = plt.subplots(figsize=(5.4, 2.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    names = {"a": "(a) model exploited", "b": "(b) cost misaligned", "c": "(c) search failed", "d": "(d) ran out of time"}
    keys = ["a", "b", "c", "d"][::-1]
    vals = [tax.shares[k] for k in keys]
    ax.barh([names[k] for k in keys], vals, height=0.55, color=COLOR["mc20"])
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.0%} ({tax.counts[keys[i]]})", va="center", color=INK2, fontsize=8)
    ax.axvline(stats.CATEGORY_A_MIN, color=INK2, linewidth=1)
    ax.text(stats.CATEGORY_A_MIN, -0.35, " gate: (a) ≥ 20%", color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    ax.set_title(f"Failure categories ({sum(tax.counts.values())} failed non-trivial episodes)",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_gate_taxonomy.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt_ci(ci, pct=False):
    return f"[{ci[0]:+.1%}, {ci[1]:+.1%}]" if pct else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _mark(ok):
    return "✅" if ok else "❌"


def report(cfg, res, branch, reason, tax, episodes, plans, checks, extra, out):
    L = []
    L.append("# Step 1 result: does σ² separate failures on Cube?\n")
    L.append(f"**Decision: {branch}**: {reason}.\n")
    L.append("Protocol and thresholds: [docs/step1_preregistration.md](../../../docs/step1_preregistration.md). "
             "All numbers below come from `scripts/step1_decide.py`.\n")

    succ = np.array([e["success"] for e in episodes])
    L.append("## Closed loop (baseline LeWM, non-trivial starts)\n")
    L.append("| seed | " + " | ".join(str(s) for s in cfg.step1.seeds) + " | all |")
    L.append("|---|" + "---:|" * (len(cfg.step1.seeds) + 1))
    per = [np.mean([e["success"] for e in episodes if e["seed"] == s]) for s in cfg.step1.seeds]
    L.append("| success | " + " | ".join(f"{p:.0%}" for p in per) + f" | **{succ.mean():.1%}** ({succ.sum()}/{len(succ)}) |")
    L.append(f"\n{len(plans['y'])} replans; {int(plans['y'].sum())} segment failures. "
             f"Replay determinism: {checks['determinism']}; segment labels from closed loop and replay "
             f"agree on {checks['label_agreement']:.1%} of replans.\n")

    L.append("## Pre-registered criteria\n")
    cols = [e for e in ("mc20", "ens", "mc5") if res.get(e)]
    L.append("| criterion | threshold | " + " | ".join(ESTIMATORS[e] + (" *(sensitivity)*" if e == "mc5" else "") for e in cols) + " |")
    L.append("|---|---|" + "---|" * len(cols))

    def row(name, thr, f):
        L.append(f"| {name} | {thr} | " + " | ".join(f(res[e]) for e in cols) + " |")

    row("C1 median within-state ρ", "≥ 0.30",
        lambda r: f"{_mark(r['c1'].median >= stats.RHO_MEDIAN_MIN)} {r['c1'].median:.3f} {_fmt_ci(r['c1'].median_ci)}")
    row("C1 states with ρ > 0", "≥ 70%",
        lambda r: f"{_mark(r['c1'].frac_positive >= stats.RHO_POSITIVE_MIN)} {r['c1'].frac_positive:.0%}")
    row("C2 headroom share", "≥ 20%, CI > 0",
        lambda r: f"{_mark(r['c2'].passed)} {r['c2'].share:+.1%} {_fmt_ci(r['c2'].ci, True)}")
    row("C3 AUROC(σ² → segment failure)", "≥ 0.65",
        lambda r: f"{_mark(r['c3'].auroc >= stats.AUROC_MIN)} {r['c3'].auroc:.3f} {_fmt_ci(r['c3'].auroc_ci)}")
    row("C3 ΔAUROC over cost-only", "CI > 0",
        lambda r: f"{_mark(r['c3'].delta_ci[0] > 0)} {r['c3'].delta:+.3f} [{r['c3'].delta_ci[0]:+.3f}, {r['c3'].delta_ci[1]:+.3f}]")
    row("**passes C1–C3**", "", lambda r: f"**{_mark(r['passed'])}**")

    L.append(f"\nCost-only AUROC (log d_pred, log d₀): {res['mc20']['c3'].auroc_cost:.3f}. "
             f"Base rate: {res['mc20']['c3'].positives}/{res['mc20']['c3'].n} segments fail.\n")

    L.append("## Gate G: failure categories\n")
    L.append(f"τ (90th pct of d_real over {checks['n_at_goal']} replays that end at the goal) = {tax.tau:.2f}.\n")
    L.append("| category | count | share |")
    L.append("|---|---:|---:|")
    names = {"a": "(a) model exploited: d_pred ≤ τ < d_real", "b": "(b) cost misaligned: d_real ≤ τ",
             "c": "(c) search failed", "d": "(d) ran out of time: final ≤ 50% initial"}
    for k in "abcd":
        L.append(f"| {names[k]} | {tax.counts[k]} | {tax.shares[k]:.0%} |")
    L.append(f"\nGate (a) ≥ 20%: **{_mark(tax.passed)}**\n")

    L.append("## Headroom (C2) details, last CEM pool\n")
    L.append("| | " + " | ".join(ESTIMATORS[e] for e in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for label, f in (
        ("mean headroom o(base) − o(oracle), cm", lambda r: f"{100 * r['c2'].headroom_mean:.1f}"),
        ("states with headroom > 0", lambda r: f"{r['c2'].extra['states_with_headroom']}/{len(r['c1'].rho_state)}"),
        ("success: baseline pick / physical oracle", lambda r: f"{r['c2'].extra['base_success_rate']:.0%} / {r['c2'].extra['oracle_success_rate']:.0%}"),
        ("share captured by latent oracle (argmin d_real)", lambda r: f"{r['c2'].captured_latent_oracle:+.1%}"),
        ("λ chosen (cross-fit counts)", lambda r: ", ".join(f"{k}:{v}" for k, v in r["c2"].lam_counts.items() if v)),
        ("decision-change rate at modal λ", lambda r: f"{r['c2'].decision_change_rate:.0%}"),
    ):
        L.append(f"| {label} | " + " | ".join(f(res[e]) for e in cols) + " |")

    L.append("\n## Sensitivity (does not change the decision)\n")
    L.append("| | " + " | ".join(ESTIMATORS[e] for e in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    L.append("| median ρ, first CEM pool | " + " | ".join(f"{np.median(res[e]['c1'].rho_first):.3f}" for e in cols) + " |")
    L.append("| median ρ, last CEM pool | " + " | ".join(f"{np.median(res[e]['c1'].rho_last):.3f}" for e in cols) + " |")
    L.append("| episode AUROC (max σ² → failure) | " + " | ".join(
        f"{extra[f'episode_{e}'][0]:.3f} {_fmt_ci(extra[f'episode_{e}'][1])}" if f"episode_{e}" in extra else "-" for e in cols) + " |")
    if "ens_mean_c1" in extra:
        L.append(f"\nEnsemble with errors of its own mean prediction: median within-state ρ "
                 f"{extra['ens_mean_c1'].median:.3f}, ρ > 0 in {extra['ens_mean_c1'].frac_positive:.0%} of states.\n")
    if "ensemble_val" in extra:
        L.append(f"\nEnsemble members, held-out 1-step loss: {extra['ensemble_val']} "
                 f"(released predictor {extra['released_val']:.4f}).\n")

    L.append("\n## Figures\n")
    for fname in ("fig_c1_within_state_rho.png", "fig_c3_failure_by_sigma2.png", "fig_gate_taxonomy.png"):
        L.append(f"![{fname}]({fname})\n")
    (out / "report.md").write_text("\n".join(L))


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    out = out_dir(cfg, ROOT)
    logs, reps, scores = load_all(cfg, out)
    plans, agreement = plan_table(logs, reps, scores)
    episodes = episode_table(logs, plans)
    states = scores["states"]

    res = {e: run_criteria(e, plans, states) for e in ESTIMATORS}

    # gate G
    at_goal = [plans["d_real"][plans["final_dist_replay"] <= SUCCESS_RADIUS]]
    for st in states:
        at_goal.append(st["d_real"].numpy()[st["final_dist"].numpy() <= SUCCESS_RADIUS])
    at_goal = np.concatenate(at_goal)
    failed = [e for e in episodes if not e["success"]]
    tax = stats.failure_taxonomy(failed, at_goal)

    branch, reason = stats.decide(res["mc20"]["passed"], res["ens"]["passed"] if res.get("ens") else None, tax.passed)

    extra = {}
    y_ep = np.array([not e["success"] for e in episodes])
    for e in ESTIMATORS:
        if res.get(e):
            extra[f"episode_{e}"] = stats.episode_auroc(y_ep, np.array([ep[f"max_{SIGMA_KEY[e]}"] for ep in episodes]))
    if res.get("ens") and "err_ens" in states[0]:
        sf, ef, sl, el = [], [], [], []
        for st in states:
            first, last = st["pool"] == 0, st["pool"] == 1
            sig, err = st["s2_ens"][:, -1].numpy(), st["err_ens"][:, -1].numpy()
            sf.append(sig[first]), ef.append(err[first]), sl.append(sig[last]), el.append(err[last])
        extra["ens_mean_c1"] = stats.criterion_c1(sf, ef, sl, el)
    log_path = out / "ensemble" / "train_log.json"
    if log_path.exists():
        tl = json.loads(log_path.read_text())
        extra["released_val"] = tl.get("released_val_loss", float("nan"))
        extra["ensemble_val"] = ", ".join(
            f"{tl[k][-1]['val']:.4f}" for k in sorted(tl) if k.startswith("member_")
        )

    checks = {
        "determinism": {s: reps[s]["check"] for s in reps},
        "label_agreement": agreement,
        "n_at_goal": int(len(at_goal)),
    }
    figures(res, plans, tax, out)
    report(cfg, res, branch, reason, tax, episodes, plans, checks, extra, out)

    decision = {
        "branch": branch,
        "reason": reason,
        "gate_category_a": asdict(tax),
        "criteria": {
            e: None if r is None else {
                "passed": r["passed"],
                "c1": {k: v for k, v in asdict(r["c1"]).items() if k not in ("rho_state", "rho_first", "rho_last")},
                "c2": asdict(r["c2"]),
                "c3": asdict(r["c3"]),
            }
            for e, r in res.items()
        },
        "closed_loop_success": float(np.mean([e["success"] for e in episodes])),
        "n_episodes": len(episodes),
        "n_replans": int(len(plans["y"])),
        "checks": checks,
        "episode_auroc": {k: v for k, v in extra.items() if k.startswith("episode_")},
    }
    (out / "decision.json").write_text(json.dumps(_jsonable(decision), indent=2))
    print(f"[decide] {branch}: {reason}")
    print(f"[decide] wrote {out / 'report.md'} and {out / 'decision.json'}")


if __name__ == "__main__":
    main()
