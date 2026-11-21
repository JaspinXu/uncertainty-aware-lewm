"""Figures 2-5 of the CS5478 Team 16 final report, from the result files committed in this repository.

    python scripts/make_report_figures.py                 # -> docs/figures/paper/fig_*.pdf and .png
    python scripts/make_report_figures.py --out /tmp/figs

Needs only numpy, scipy and matplotlib (no GPU, dataset or checkpoint). It also prints every number the
report quotes, recomputed from the per-episode results wherever they exist (paired exact McNemar tests,
Wilson and bootstrap intervals), so the report can be checked against the result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

R = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(R / "docs" / "figures" / "paper"))
OUT = Path(ap.parse_args().out)
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8c8a84", "#e9e7e1"
GRAY, LIGHT, DARK = "#9a9893", "#d3d1ca", "#4a4945"
BLUE, BLUE_L, BLUE_D, ORANGE = "#2a78d6", "#86b6ef", "#184f95", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
W = 5.5  # NeurIPS text width (in)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- helpers
def md_tables(path):
    """Every pipe table of a markdown file, as a list of rows (dict header -> cell)."""
    tables, cur = [], []
    for line in Path(path).read_text().splitlines() + [""]:
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        if len(cur) > 2:
            tables.append([dict(zip(cur[0], r)) for r in cur[2:]])
        cur = []
    return tables


def num(cell):
    """First number of a table cell such as '0.555 [0.508, 0.605]; 0.278' or '**65.6 ± 6.0**'."""
    import re
    return float(re.search(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", cell.replace("−", "-")).group())


def bracket(cell):
    import re
    lo, hi = re.search(r"\[(-?[\d.]+),\s*(-?[\d.]+)\]", cell.replace("−", "-")).groups()
    return float(lo), float(hi)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def mcnemar(a, b):
    """Paired 0/1 outcomes of a reference a and a method b -> gained, lost, exact two-sided p."""
    g, l = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
    return g, l, (binomtest(min(g, l), g + l, 0.5).pvalue if g + l else 1.0)


def boot(a, b, reps=10_000, seed=0):
    """Bootstrap 95% CI (pp) of the paired difference b - a, resampling episodes."""
    d = (b - a).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(d), (reps, len(d)))
    return np.percentile(100 * d[idx].mean(1), [2.5, 97.5])


def compare_run(k, run, base="results/cube/compare"):
    d = R / base / f"k{k}" / run
    succ, types, keys = [], [], []
    for f in sorted(d.glob("seed_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        j = json.loads(f.read_text())
        succ += list(j["episode_successes"])
        types += list(j["types"])
        keys += [(j["seed"], r) for r in j["rows"]]
    assert len(set(keys)) == len(keys)
    return {"succ": np.array(succ, int), "types": np.array(types), "keys": keys, "name": run}


A_SEEDS = (42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055)  # panel A test seeds


def panel_a_run(method):
    succ, keys = [], []
    for s in A_SEEDS:
        j = json.loads((R / "results/cube/runs" / f"{method}_s{s}.json").read_text())
        succ += list(j["episode_successes"])
        keys += [(s, i, int(e), int(t)) for i, (e, t) in enumerate(zip(j["episodes"], j["start_steps"]))]
    return {"succ": np.array(succ, int), "keys": keys, "name": method}


def pair(a, b):
    ib = {k: i for i, k in enumerate(b["keys"])}
    common = [(i, ib[k]) for i, k in enumerate(a["keys"]) if k in ib]
    assert len(common) == len(a["keys"]) == len(b["keys"]), (a["name"], b["name"], len(common))
    ia, jb = map(np.array, zip(*common))
    return a["succ"][ia], b["succ"][jb]


def rate(run, typ=None):
    s = run["succ"] if typ is None else run["succ"][run["types"] == typ]
    return 100 * s.mean(), len(s)


def compare(a, b, label):
    xa, xb = pair(a, b)
    g, l, p = mcnemar(xa, xb)
    lo, hi = boot(xa, xb)
    d = 100 * (xb.mean() - xa.mean())
    print(f"  {label:<58s} {100 * xb.mean():5.1f} vs {100 * xa.mean():5.1f}  Δ {d:+5.1f} "
          f"[{lo:+.1f}, {hi:+.1f}]  {g:3d}/{l:<3d} p={p:.2g}")
    return d, p


# --------------------------------------------------------------------------- numbers
print("== Panels B and C (held out), success by task type")
B = {r: compare_run(25, r) for r in ("lewm", "mc5", "gated", "gated_tc", "bc_only", "prior_cem_s1",
                                     "prior_cem_sconst", "prior_cem_sp", "bc_select_khat")}
C_RUNS = ("lewm", "mc5", "gated", "gated_tc", "bc_only", "prior_cem_s1", "prior_cem_sconst", "prior_cem_sp",
          "bc_only_var", "prior_cem_s1_var", "prior_cem_sconst_var", "prior_cem_sp_var",
          "prior_cem_sconst_var_a01", "prior_cem_sp_var_a01", "bc_select_khat", "lewm_khat",
          "prior_cem_sp_var_khat", "prior_cem_sconst_var_khat", "prior_cem_s1_var_khat",
          "bc_select_khat_m1", "bc_select_khat_sp_m4", "bc_select_khat_sconst_m4", "bc_select_khat_sp_m16",
          "bc_select_khat_m1_ens", "bc_select_khat_sp_m4_ens", "bc_select_khat_m1_ens3",
          "bc_select_khat_sp_m4_ens3")
C = {r: compare_run(50, r) for r in C_RUNS}
for name, panel, types in (("B", B, ("pick", "carry", "place")), ("C", C, ("pick", "place", "move", "carry"))):
    print(f" panel {name}: n by type", {t: int((panel['lewm']['types'] == t).sum()) for t in types})
    for r, run in panel.items():
        all_, n = rate(run)
        cells = "  ".join(f"{t} {rate(run, t)[0]:5.1f}" for t in types)
        _, lo, hi = wilson(run["succ"].sum(), n)
        print(f"  {r:<28s} {all_:5.1f} [{lo:.1f}, {hi:.1f}] (n={n})  {cells}")

print("== Paired comparisons, panel B")
for a, b in (("lewm", "prior_cem_sp"), ("prior_cem_s1", "prior_cem_sp"), ("prior_cem_sconst", "prior_cem_sp"),
             ("bc_only", "prior_cem_sp"), ("bc_only", "bc_select_khat"), ("prior_cem_sp", "bc_select_khat"),
             ("lewm", "mc5"), ("lewm", "gated"), ("lewm", "gated_tc"), ("lewm", "bc_only")):
    compare(B[a], B[b], f"{b} vs {a}")
print("== Paired comparisons, panel C")
for a, b in (("lewm", "prior_cem_sp_var_a01"), ("bc_only_var", "prior_cem_sp_var_a01"),
             ("prior_cem_sconst_var_a01", "prior_cem_sp_var_a01"), ("bc_only_var", "prior_cem_sconst_var_a01"),
             ("prior_cem_sconst_var", "prior_cem_sp_var"), ("prior_cem_s1_var", "prior_cem_sp_var"),
             ("bc_only_var", "prior_cem_sp_var"), ("prior_cem_sconst", "prior_cem_sp"), ("bc_only", "prior_cem_sp"),
             ("bc_only", "bc_only_var"),
             ("lewm", "bc_select_khat"), ("bc_only_var", "bc_select_khat"), ("prior_cem_sp_var_a01", "bc_select_khat"),
             ("lewm", "lewm_khat"), ("prior_cem_sp_var", "prior_cem_sp_var_khat"),
             ("bc_select_khat", "bc_select_khat_m1"), ("bc_select_khat_m1", "bc_select_khat_sp_m4"),
             ("bc_select_khat_m1", "bc_select_khat_sp_m16"), ("bc_select_khat_m1", "bc_select_khat_m1_ens"),
             ("bc_select_khat_sp_m4", "bc_select_khat_sp_m4_ens"),
             ("lewm", "mc5"), ("lewm", "gated"), ("lewm", "gated_tc"), ("lewm", "bc_only_var")):
    compare(C[a], C[b], f"{b} vs {a}")

print("== Panel A (9 test seeds x 50, 25-step goals, 30 CEM iterations)")
A = {m: panel_a_run(m) for m in ("baseline", "mc5", "gated", "goal_cvar", "penalty", "ens_penalty", "risk",
                                 "l4_lewm", "l4_ms", "l4_prop", "l4_prop_aux", "l4_prop_ms")}
A_D = {}
for m, run in A.items():
    k, n = int(run["succ"].sum()), len(run["succ"])
    p, lo, hi = wilson(k, n)
    line = f"  {m:<12s} {p:5.1f} [{lo:.1f}, {hi:.1f}] (n={n})"
    if m != "baseline":
        xa, xb = pair(A["baseline"], run)
        g, l, pv = mcnemar(xa, xb)
        A_D[m] = (100 * (xb.mean() - xa.mean()), pv)
        line += f"  Δ {A_D[m][0]:+.1f}  {g}/{l}  p={pv:.2g}"
    print(line)

print("== Tuning panel (seeds 100-104, 50-step goals)")
T = {r: compare_run(50, r, base="results/cube/compare_tune") for r in (
    "bc_only_var", "prior_cem_sp_var_a01", "prior_cem_sp_var_a025", "prior_cem_sp_var_a05", "prior_cem_sp_var",
    "bc_select_khat", "bc_select_khat_sp_m4", "bc_select_khat_sp_m4_ens03", "bc_select_khat_sp_m4_ens")}
for r, run in T.items():
    p, lo, hi = wilson(run["succ"].sum(), len(run["succ"]))
    print(f"  {r:<28s} {p:5.1f} [{lo:.1f}, {hi:.1f}] (n={len(run['succ'])})")

D = json.loads((R / "action-prior/results/cube/summary.json").read_text())
print("== Panel D (Cube, 50 trajectories x 5 planner seeds)")
for g, v in D["groups"].items():
    print(f"  {g:<12s} {v['mean_percent']:5.1f} ± {v['std_percent']:.1f}")
for k, v in D["paired"].items():
    print(f"  {k:<32s} Δ {v['delta_pp']:+.1f} CI {v['trajectory_cluster_bootstrap_95_pp']}")

# --------------------------------------------------------------------------- Figure 2: diagnosis
sig = md_tables(R / "results/cube/l1/signal_report.md")
q2 = next(t for t in sig if "block k" in t[0])
q6 = {r["plans"]: r for r in next(t for t in sig if "plans" in t[0] and "search failures" in t[0])}
bb = json.loads((R / "results/cube/step1/branchb.json").read_text())["expert_vs_cem"]
pil = {r[""]: r for r in md_tables(R / "results/cube/step1/pilots/overview.md")[0]}
base_row = next(v for k, v in pil.items() if k.startswith("baseline"))
exp_row = pil["`expert_init`"]
print("== Diagnosis")
print("  Q6", {k: (v["n"], v["failures"], v["search failures"]) for k, v in q6.items()})
print("  expert vs CEM: d_pred", bb["median_d_pred"], "d_real", bb["median_d_real"], "e5", bb["median_e5"],
      "prefers CEM", bb["frac_model_prefers_cem_plan"], "expert better than all", bb["frac_expert_better_than_all_last_pool"])
print("  expert_init", exp_row, "\n  baseline", base_row)

fig, axs = plt.subplots(1, 3, figsize=(W, 1.55), gridspec_kw={"width_ratios": [1.2, 0.95, 1.15], "wspace": 0.72})
ax = axs[0]
rows = [("all", "all failed\nplans"), ("first_plan", "first plan"), ("second_plan", "after a failed\nfirst plan")]
for i, (key, lab) in enumerate(rows):
    srch = num(q6[key]["search failures"])
    ax.barh(i, srch, color=DARK, height=0.6, edgecolor="white", lw=1)
    ax.barh(i, 100 - srch, left=srch, color=LIGHT, height=0.6, edgecolor="white", lw=1)
    ax.text(srch / 2, i, f"{srch:.0f}%", color="white", ha="center", va="center", fontsize=6.5, fontweight="bold")
ax.set_yticks(range(3), [f"{lab}\n(n = {int(num(q6[k]['failures']))})" for k, lab in rows])
ax.set_ylim(2.45, -1.25)
ax.set_xlim(0, 100)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("share of failed plans (%)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("(a) No candidate works", x=-0.5)
ax.legend(handles=[Patch(color=DARK, label="search failure"), Patch(color=LIGHT, label="selection failure")],
          loc="upper left", ncol=2, handlelength=0.9, columnspacing=0.7, borderaxespad=0.0, handletextpad=0.4)

ax = axs[1]
x = np.arange(2)
pred = [bb["median_d_pred"]["expert"], bb["median_d_pred"]["cem"]]
real = [bb["median_d_real"]["expert"], bb["median_d_real"]["cem"]]
ax.bar(x - 0.19, pred, 0.36, color=GRAY, label="predicted by LeWM", edgecolor="white", lw=1, zorder=2)
ax.bar(x + 0.19, real, 0.36, color=DARK, label="real (replay)", edgecolor="white", lw=1, zorder=2)
for xi, v in zip(x - 0.19, pred):
    ax.text(xi, v * 1.35, f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=INK2)
for xi, v in zip(x + 0.19, real):
    ax.text(xi, v * 1.35, f"{v:.2f}" if v < 1 else f"{v:.0f}", ha="center", va="bottom", fontsize=6, color=INK2)
ax.set_yscale("log")
ax.set_ylim(5e-3, 3e5)
ax.set_yticks([1e-2, 1, 1e2, 1e4], ["0.01", "1", "100", r"$10^4$"])
ax.minorticks_off()
ax.set_xticks(x, ["expert's\nplan", "CEM's\nplan"])
ax.set_ylabel(r"median cost $\|z_5 - z_g\|^2$")
grid(ax)
ax.set_title("(b) CEM exploits model error", x=-0.42)
ax.legend(loc="upper left", handlelength=0.9, borderaxespad=0.1)

ax = axs[2]
cats = [("pick", "pick"), ("carry", "carry"), ("place", "place"), ("success", "all")]
x = np.arange(len(cats))
b_vals = [num(base_row[c]) for c, _ in cats]
e_vals = [num(exp_row[c]) for c, _ in cats]
ax.bar(x - 0.19, b_vals, 0.36, color=GRAY, label="LeWM's CEM", edgecolor="white", lw=1, zorder=2)
ax.bar(x + 0.19, e_vals, 0.36, color=ORANGE, label="CEM started at the expert's actions", edgecolor="white",
       lw=1, zorder=2)
for xi, v in zip(x - 0.19, b_vals):
    ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK2)
for xi, v in zip(x + 0.19, e_vals):
    ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK2)
ax.set_xticks(x, [c[1] for c in cats])
ax.set_ylim(0, 148)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("success (%)")
grid(ax)
ax.set_title("(c) The start of the search decides", x=-0.3)
ax.legend(loc="upper left", ncol=1, handlelength=0.9, borderaxespad=0.0, fontsize=6.0)
save(fig, "fig_diagnosis")

# --------------------------------------------------------------------------- Figure 3: main results
fig, axs = plt.subplots(1, 3, figsize=(W, 1.58), gridspec_kw={"width_ratios": [1.45, 0.95, 0.95], "wspace": 0.55})
ax = axs[0]
bars = [("lewm", "LeWM (CEM)", GRAY), ("gated", r"+ $\sigma^2$ gated truncation", LIGHT),
        ("bc_only_var", "prior alone, no planning", ORANGE),
        ("prior_cem_sconst_var_a01", "prior + CEM, fixed width", BLUE_L),
        ("prior_cem_sp_var_a01", r"$\sigma_p$-CEM (ours)", BLUE),
        ("bc_select_khat", r"+ time-to-goal selection (ours)", BLUE_D)]
for i, (r, lab, col) in enumerate(bars):
    run = C[r]
    p, lo, hi = wilson(run["succ"].sum(), len(run["succ"]))
    ax.barh(i, p, color=col, height=0.66, edgecolor="white", lw=1, zorder=2)
    ax.plot([lo, hi], [i, i], color=INK2, lw=0.8, zorder=3)
    ax.text(hi + 1.5, i, f"{p:.1f}", va="center", ha="left", fontsize=6.3, color=INK)
ax.set_yticks(range(len(bars)), [b[1] for b in bars])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("success (%), Wilson 95% CI")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
grid(ax, "x")
ax.set_title("(a) Panel C: 50-step goals, held out", x=-0.9)

ax = axs[1]
order = [("bc_only_var", 0), ("prior_cem_sp_var_a01", 0.1), ("prior_cem_sp_var_a025", 0.25),
         ("prior_cem_sp_var_a05", 0.5), ("prior_cem_sp_var", 1.0)]
xs = np.arange(len(order))
vals = [wilson(T[r]["succ"].sum(), len(T[r]["succ"])) for r, _ in order]
ax.plot(xs[1:], [v[0] for v in vals[1:]], color=BLUE, lw=1.2, zorder=2)
for xi, (p, lo, hi), (r, a) in zip(xs, vals, order):
    col = ORANGE if a == 0 else BLUE
    ax.plot([xi, xi], [lo, hi], color=col, lw=0.8, zorder=2)
    ax.plot(xi, p, "o", ms=4.2, color=col, mec="white", mew=0.8, zorder=3)
ax.annotate("chosen", (1, vals[1][2]), xytext=(1, vals[1][2] + 3.2), ha="center", fontsize=6, color=BLUE,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
ax.set_xticks(xs, ["0\n(prior)", "0.1", "0.25", "0.5", "1"])
ax.set_xlabel(r"width scale $\alpha$ in $\sigma_0 = \alpha\,\sigma_p$")
ax.set_ylabel("success (%)")
ax.set_ylim(62, 90)
grid(ax)
ax.set_title(r"(b) Tuning seeds: choose $\alpha$", x=-0.32)

ax = axs[2]
its = [5, 10, 20, 30]
for key, lab, col, mk in (("guided", "BC-guided CEM", BLUE, "o"), ("baseline", "original CEM", GRAY, "s")):
    m = np.array([D["groups"][f"{key}-{k}"]["mean_percent"] for k in its])
    s = np.array([D["groups"][f"{key}-{k}"]["std_percent"] for k in its])
    ax.fill_between(its, m - s, m + s, color=col, alpha=0.18, lw=0, zorder=1)
    ax.plot(its, m, marker=mk, ms=3.6, color=col, lw=1.2, mec="white", mew=0.7, label=lab, zorder=3)
ax.axhline(D["groups"]["bc-5"]["mean_percent"], color=ORANGE, lw=1.0, ls=(0, (3, 2)), zorder=2)
ax.text(30.5, D["groups"]["bc-5"]["mean_percent"] - 1.2, "BC alone", color=ORANGE, fontsize=6, ha="right",
        va="top")
ax.set_xticks(its)
ax.set_xlabel("CEM iterations")
ax.set_ylabel("success (%)")
ax.set_ylim(62, 100)
grid(ax)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), handlelength=1.4, borderaxespad=0.1)
ax.set_title("(c) Panel D: CEM iterations", x=-0.32)
save(fig, "fig_results")

# --------------------------------------------------------------------------- Figure 4: the alternatives
rep = md_tables(R / "results/cube/l5/representation.md")
err5 = {r["checkpoint"]: num(r["k=5"]) for r in rep[2]}
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32})
ax = axs[0]
ks = np.arange(1, 6)
for col_name, lab, col, mk in (("latent ensemble", "ensemble", DARK, "s"), ("MC-dropout", "MC-dropout", GRAY, "o")):
    m = np.array([num(r[col_name]) for r in q2[:5]])
    lo, hi = np.array([bracket(r[col_name]) for r in q2[:5]]).T
    ax.fill_between(ks, lo, hi, color=col, alpha=0.2, lw=0)
    ax.plot(ks, m, marker=mk, ms=3.4, color=col, lw=1.2, mec="white", mew=0.7, label=lab)
ax.axvspan(4.6, 5.4, color=BLUE, alpha=0.09, lw=0)
ax.text(5.0, 0.74, "CEM\nscores\nhere", color=BLUE, fontsize=5.8, ha="center", va="top")
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xticks(ks)
ax.set_xlim(0.6, 5.5)
ax.set_ylim(-0.1, 0.78)
ax.set_xlabel("predicted block k")
ax.set_ylabel(r"Spearman $\rho(\sigma^2_k$, true error$_k)$")
grid(ax)
ax.legend(loc="lower left", handlelength=1.4, borderaxespad=0.1)
ax.set_title(r"(a) $\sigma^2$ tracks error only a few blocks ahead", x=-0.22)

ax = axs[1]
var_tab = {r["label"]: r for r in md_tables(R / "results/cube/summary/variants_panel.md")[0]}
pts = [("cube/lewm", "baseline", "pretrained"), ("cube/l4_lewm", "l4_lewm", "control"),
       ("cube/l4_ms", "l4_ms", "+ms"), ("cube/l4_prop", "l4_prop", "+prop"),
       ("cube/l4_prop_aux", "l4_prop_aux", "+prop_aux"), ("cube/l4_prop_ms", "l4_prop_ms", "+prop_ms")]
offs = {"pretrained": (4, 3), "control": (4, 3), "+ms": (4, 3), "+prop": (4, -6), "+prop_aux": (4, 2),
        "+prop_ms": (4, -6)}
for ck, lab_tab, lab in pts:
    succ = A["baseline" if lab_tab == "baseline" else lab_tab]["succ"]
    p, lo, hi = wilson(succ.sum(), len(succ))
    x0 = err5[ck]
    col = DARK if lab == "pretrained" else GRAY
    ax.plot([x0, x0], [lo, hi], color=col, lw=0.7, alpha=0.8)
    ax.plot(x0, p, "o", ms=4.2, color=col, mec="white", mew=0.8, zorder=3)
    ax.annotate(lab, (x0, p), xytext=offs[lab], textcoords="offset points", fontsize=5.9, color=INK2)
ax.set_xlim(0.0, 0.056)
ax.set_xticks([0, 0.02, 0.04])
ax.set_ylim(55, 73)
ax.set_xlabel("5-block latent error on expert data")
ax.set_ylabel("panel A success (%)")
grid(ax)
ax.set_title("(b) Lower prediction error, not higher success", x=-0.2)
save(fig, "fig_alternatives")

fig, ax = plt.subplots(figsize=(W - 1.35, 1.12))
rows = ["score", "latent", "search", "prior"]
row_lab = {"score": r"world-model $\sigma^2$ in the score (Sec. 4.5)", "latent": "retrained latent (Sec. 4.6)",
           "search": "prior-guided search (ours)", "prior": "prior alone, no planning"}
row_col = {"score": GRAY, "latent": GRAY, "search": BLUE, "prior": ORANGE}
marks = {"A": "o", "B": "s", "C": "^", "D": "D"}
pts = []  # (row, panel, delta, p)
for m in ("mc5", "gated", "goal_cvar", "penalty", "ens_penalty", "risk"):
    pts.append(("score", "A", *A_D[m]))
for m in ("l4_lewm", "l4_ms", "l4_prop", "l4_prop_aux", "l4_prop_ms"):
    pts.append(("latent", "A", *A_D[m]))
for panel, runs, tag in ((B, ("mc5", "gated", "gated_tc"), "B"), (C, ("mc5", "gated", "gated_tc"), "C")):
    for r in runs:
        xa, xb = pair(panel["lewm"], panel[r])
        pts.append(("score", tag, 100 * (xb.mean() - xa.mean()), mcnemar(xa, xb)[2]))
for panel, runs, tag in ((B, ("prior_cem_s1", "prior_cem_sconst", "prior_cem_sp", "bc_select_khat"), "B"),
                         (C, ("prior_cem_s1_var", "prior_cem_sconst_var", "prior_cem_sp_var",
                              "prior_cem_sconst_var_a01", "prior_cem_sp_var_a01", "bc_select_khat"), "C")):
    for r in runs:
        xa, xb = pair(panel["lewm"], panel[r])
        pts.append(("search", tag, 100 * (xb.mean() - xa.mean()), mcnemar(xa, xb)[2]))
for panel, r, tag in ((B, "bc_only", "B"), (C, "bc_only_var", "C")):
    xa, xb = pair(panel["lewm"], panel[r])
    pts.append(("prior", tag, 100 * (xb.mean() - xa.mean()), mcnemar(xa, xb)[2]))
for k in (5, 10, 20, 30):
    pv = D["paired"].get(f"guided-{k}_minus_baseline-{k}")
    d = D["groups"][f"guided-{k}"]["mean_percent"] - D["groups"][f"baseline-{k}"]["mean_percent"]
    pts.append(("search", "D", d, 0.0 if pv is None or pv["trajectory_cluster_bootstrap_95_pp"][0] > 0 else 1.0))
def panel_d(mode, k=5):  # per-trajectory success averaged over the 5 planner seeds (tasks are shared)
    runs = [json.loads((R / f"action-prior/results/cube/test-{mode}-k{k}-s{s}-rho0.25.json").read_text())
            for s in range(62, 67)]
    assert all(r["tasks"]["episode_ids"] == runs[0]["tasks"]["episode_ids"] for r in runs)
    return np.mean([np.array(r["metrics"]["episode_successes"], float) for r in runs], axis=0)


d_bc = panel_d("bc") - panel_d("baseline")  # trajectory-cluster bootstrap, as in action-prior's summary
ci = np.percentile(100 * d_bc[np.random.default_rng(0).integers(0, len(d_bc), (10_000, len(d_bc)))].mean(1),
                   [2.5, 97.5])
print(f"== Panel D: BC alone vs original CEM (5 iterations) {100 * d_bc.mean():+.1f} pp, cluster CI {ci}")
pts.append(("prior", "D", 100 * d_bc.mean(), 0.0 if ci[0] > 0 else 1.0))
rng = np.random.default_rng(3)
for i, row in enumerate(rows):
    sel = [p for p in pts if p[0] == row]
    lo, hi = min(p[2] for p in sel), max(p[2] for p in sel)
    ax.plot([lo, hi], [i, i], color=row_col[row], lw=4.5, alpha=0.16, solid_capstyle="round", zorder=1)
    for _, tag, d, pv in sel:
        y = i + rng.uniform(-0.2, 0.2)
        face = row_col[row] if pv < 0.05 else "white"
        ax.plot(d, y, marker=marks[tag], ms=3.4, mfc=face, mec=row_col[row], mew=0.8, ls="none", zorder=3)
    ax.text(hi + 1.6, i, f"{lo:+.0f} to {hi:+.0f}", va="center", fontsize=6.2, color=INK2)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_yticks(range(len(rows)), [row_lab[r] for r in rows])
ax.invert_yaxis()
ax.set_xlim(-8, 58)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel(r"$\Delta$ success vs. the panel's LeWM (pp)")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
grid(ax, "x")
handles = [Line2D([], [], marker=mk, ls="none", mfc=MUTED, mec=MUTED, ms=3.4, label=f"panel {t}")
           for t, mk in marks.items()]
handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=MUTED, ms=3.4, label="p ≥ 0.05"))
ax.legend(handles=handles, loc="upper right", ncol=3, handletextpad=0.3, columnspacing=0.9,
          borderaxespad=0.2, fontsize=6.0, labelspacing=0.3)
save(fig, "fig_summary")

print("== Figure 4c points (row, panel, Δ, p)")
for row in rows:
    sel = [p for p in pts if p[0] == row]
    print(f"  {row:<7s} n={len(sel):2d}  range {min(p[2] for p in sel):+.1f} .. {max(p[2] for p in sel):+.1f}")
print("wrote", *sorted(p.name for p in OUT.glob("fig_*")))
