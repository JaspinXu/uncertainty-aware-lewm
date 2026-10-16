"""Figures for the README and the slides, drawn from the result files.

    python scripts/cube/make_readme_figures.py [--out docs/figures]

  planners.png        closed-loop success, paired difference to the baseline
                      (9 test seeds x 50 episodes, episode-bootstrap 95% CI)
  sigma_signal.png    Spearman(sigma^2_k, true error_k) per block, MC-dropout
                      vs latent ensemble (plan-bootstrap 95% CI)
  search_failures.png why executed plans fail (Q6) and the size of the model's
                      errors on one scale (planner candidates, expert actions,
                      rendering gap; from the L1 replays and the rendering check)
  representation.png  open-loop error vs planning success of the L4 variants

Colours follow the slide deck: purple = uncertainty, teal = representation,
orange = search, gray = not significant or reference, navy = ink.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/cube"
SEEDS = [42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055]
INK, INK2, MUTED, GRID = "#14213D", "#374151", "#6B7280", "#E5E7EB"
PURPLE, TEAL, GRAY, ORANGE = "#6C4DFF", "#0F9D94", "#A7B0BF", "#D9701C"


def setup_fonts() -> None:
    for f in ("/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
              "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": ["Poppins", "DejaVu Sans"] if "Poppins" in names else ["DejaVu Sans"],
        "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK2, "axes.titlecolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def episodes(label: str) -> np.ndarray:
    return np.concatenate([np.asarray(json.loads((R / f"runs/{label}_s{s}.json").read_text())["episode_successes"], float)
                           for s in SEEDS])


def boot_ci(diff: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    m = diff[idx].mean(1) * 100
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def clean(ax, xgrid=True) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def planners(out: Path) -> None:
    panel = json.loads((R / "summary/planning_panel.json").read_text())
    names = {"gated": "Gated truncation", "ens_penalty": "Ensemble σ² penalty", "penalty": "MC-dropout σ² penalty",
             "goal_cvar": "Goal-CVaR penalty", "risk": "Learned risk head", "mc5": "MC averaging (MC-5)"}
    base = episodes("baseline")
    rows = []
    for lab, name in names.items():
        c = panel["comparisons"][lab]
        d = episodes(lab) - base
        rows.append((name, 100 * d.mean(), boot_ci(d), c["mcnemar"]["p_value"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=200)
    y = np.arange(len(rows))
    for i, (name, dm, (lo, hi), p) in enumerate(rows):
        sig = p < 0.01
        ax.barh(i, dm, height=0.46, color=PURPLE if sig else GRAY, zorder=2)
        ax.plot([lo, hi], [i, i], color=INK2, linewidth=1.2, zorder=3)
        ptxt = f"p = {p:.4f}" if p < 0.01 else f"p = {p:.2f}"
        ax.text(max(hi, dm) + 0.35, i, f"{dm:+.1f} pp   {ptxt}", va="center", fontsize=10.5,
                color=INK if sig else MUTED, fontweight="medium" if sig else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-3.5, 14.5)
    ax.set_xlabel("success minus baseline (percentage points), 95% CI")
    clean(ax)
    base_rate = panel["labels"]["baseline"]["pooled"]
    fig.suptitle(f"Closed-loop planning on OGBench-Cube: 450 paired episodes, baseline {base_rate:.1f}%",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "p: exact McNemar test, uncorrected · purple: significant after Holm correction over the six planners",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out / "planners.png")
    plt.close(fig)


def sigma_signal(out: Path) -> None:
    q2 = json.loads((R / "l1/signal_summary.json").read_text())["q2_sigma_vs_error"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    k = np.arange(1, 6)
    for key, name, col in (("ens", "latent ensemble", TEAL), ("mc", "MC-dropout", PURPLE)):
        v = np.array([q2[key][f"step_{i}"]["pooled"] for i in k])
        lo = np.array([q2[key][f"step_{i}"]["ci"][0] for i in k])
        hi = np.array([q2[key][f"step_{i}"]["ci"][1] for i in k])
        ax.fill_between(k, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(k, v, color=col, linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.5, label=name, zorder=3)
        ax.text(5.12, v[-1], f"{name}  {v[-1]:.2f}", va="center", fontsize=10, color=INK2)
        ax.text(1.1, v[0] + 0.035, f"{v[0]:.2f}", va="bottom", ha="left", fontsize=10, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(k, [f"block {i}" for i in k])
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(-0.08, 0.8)
    ax.set_ylabel("Spearman ρ (σ² vs. true error)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.set_title("σ² tracks the model's error one block ahead,\nnot at block 5, where the planning cost is scored",
                 loc="left", fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(out / "sigma_signal.png")
    plt.close(fig)


def failures(out: Path) -> None:
    """Q6 failure modes (left) and the size of the model's errors on one scale (right)."""
    sig = json.loads((R / "l1/signal_summary.json").read_text())
    q6 = sig["q6_failure_modes"]
    hw = json.loads((R / "diag_hw/domain_gap.json").read_text())
    sw = json.loads((R / "diag/domain_gap.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.9), dpi=200, gridspec_kw={"width_ratios": [1, 1.25]})

    rows = [("all failed plans", "all"), ("failed first plans", "first_plan"), ("failed second plans", "second_plan")]
    y = np.arange(len(rows))[::-1]
    for i, (name, key) in zip(y, rows):
        s = 100 * q6[key]["search_failure_frac"]
        a1.barh(i, s, height=0.5, color=ORANGE, zorder=2)
        a1.barh(i, 100 - s, left=s + 0.6, height=0.5, color=GRAY, zorder=2)
        a1.text(s - 2, i, f"{s:.0f}%", va="center", ha="right", color="white", fontsize=10.5, fontweight="medium")
        a1.text(101.5, i, f"n = {q6[key]['n_exec_failures']}", va="center", ha="left", color=MUTED, fontsize=9.5)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 112)
    a1.set_xticks([0, 25, 50, 75, 100])
    a1.set_xlabel("share of failed executed plans (%)")
    clean(a1)
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=GRAY)],
              labels=["search failure: none of 51 replayed candidates succeeds", "selection failure: one of them does"],
              frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    a1.set_title("Why executed plans fail (non-trivial calls)", loc="left", fontsize=11.5)

    def med(x):
        return float(x["median"])
    items = [  # (label, value, colour, is reference)
        ("rendering gap: live vs dataset frame\nof the same state (software renderer)", med(sw["render_gap_start"]), TEAL, False),
        ("expert's own 25 actions,\nerror after 5 blocks", med(hw["goal_cost_expert_actions"]["from_dataset_frame"]), TEAL, False),
        ("for scale: one block of real motion\n(dataset frames 5 steps apart)", med(hw["ref_dataset_5_steps"]), GRAY, True),
        ("planner's candidates,\nerror after 5 blocks", sig["q2_sigma_vs_error"]["prediction_error_H"]["pretrained_median"], ORANGE, False),
        ("for scale: random pairs of frames", med(hw["ref_random_pairs"]), GRAY, True),
    ]
    y = np.arange(len(items))[::-1]
    for i, (name, v, col, ref) in zip(y, items):
        a2.plot([1e-3, v], [i, i], color=GRID if ref else col, linewidth=1.2 if ref else 2, zorder=2,
                linestyle=(0, (2, 2)) if ref else "-")
        a2.plot(v, i, "o", markersize=9, color=col, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        a2.text(v * 1.6, i, f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}", va="center",
                fontsize=10.5, color=INK if not ref else MUTED, fontweight="medium" if not ref else "normal")
    a2.set_xscale("log")
    a2.set_xlim(3e-3, 3e3)
    a2.set_yticks(y, [it[0] for it in items], fontsize=9.5)
    a2.set_xlabel("median squared latent distance (log scale)")
    clean(a2)
    a2.set_title("The model predicts its planner's candidates\nfar worse than the expert's motion",
                 loc="left", fontsize=11.5)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out / "search_failures.png")
    plt.close(fig)


def representation(out: Path) -> None:
    names = {"l4_lewm": "control", "l4_ms": "multi-step", "l4_prop": "proprio", "l4_prop_aux": "proprio + aux",
             "l4_prop_ms": "proprio + aux\n+ multi-step"}
    err = {v: json.loads((R / f"l5/repr_cube__{v}.json").read_text())["kstep"][4]["rel_latent_err"] for v in names}
    comp = json.loads((R / "summary/variants_vs_control.json").read_text())["comparisons"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=200, sharey=True)
    y = np.arange(len(names))[::-1]
    for i, v in zip(y, names):
        a1.barh(i, err[v], height=0.46, color=GRAY if v == "l4_lewm" else TEAL, zorder=2)
        a1.text(err[v] + 0.0008, i, f"{err[v]:.3f}", va="center", fontsize=10, color=INK2)
        if v == "l4_lewm":
            a2.text(0.3, i, "reference", va="center", fontsize=10, color=MUTED)
            continue
        c = comp[v]
        dm, p = c["delta_mean_pp"], c["mcnemar"]["p_value"]
        a2.barh(i, dm, height=0.46, color=TEAL, zorder=2)
        x = dm + (0.3 if dm >= 0 else -0.3)
        ptxt = f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
        a2.text(x, i, f"{dm:+.1f} pp  (p = {ptxt})", va="center", ha="left" if dm >= 0 else "right",
                fontsize=10, color=INK2)
    a1.set_yticks(y, list(names.values()))
    a1.set_xlim(0, 0.045)
    a1.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    a1.set_xlabel("open-loop latent error after 5 blocks\n(relative to latent variance, lower is better)")
    a2.axvline(0, color=INK2, linewidth=1)
    a2.set_xlim(-13, 8)
    a2.set_xticks([-10, -5, 0, 5])
    a2.set_xlabel("planning success vs. control (pp)")
    clean(a1)
    clean(a2)
    a1.set_title("Prediction on dataset sequences", loc="left", fontsize=11.5)
    a2.set_title("Planning (450 paired episodes)", loc="left", fontsize=11.5)
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "representation.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    planners(args.out)
    sigma_signal(args.out)
    failures(args.out)
    representation(args.out)
    print(f"[figures] -> {args.out}")


if __name__ == "__main__":
    main()
