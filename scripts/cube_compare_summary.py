"""Summarize the multi-seed comparison of scripts/cube_compare.py.

    python scripts/cube_compare_summary.py                    # goal offset compare.offset (25)
    python scripts/cube_compare_summary.py compare.offset=50

Reads <compare.out_dir>/k<offset>/<run>/seed_*.json and writes k<offset>/summary.md and per_seed.csv:
  1. every run: success mean ± std over seeds, pooled success with a Wilson 95% CI, success by task
     type, and against lewm on the same start states: Δ (pp), episodes gained / lost, exact McNemar p
     (Holm-adjusted over the runs) and an exact sign-flip p over the per-seed differences;
  2. the ablation pairs (gated vs mc5, gated_tc vs gated, the three prior_cem widths, ...);
  3. planner diagnostics (fraction of CEM candidates truncated, scored blocks, sigma^2);
  4. the per-seed success rates of every run.
A comparison only pairs the seeds both runs have, and checks that their start states are identical.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ua.step1.stats import mcnemar  # noqa: E402

REFERENCE = "lewm"
ORDER = ("lewm", "mc5", "gated", "gated_tc", "goal_cvar", "bc_only",
         "prior_cem_s1", "prior_cem_sconst", "prior_cem_sp", "bc_select_khat")
PAIRS = (  # (run, against): the questions the comparison is meant to answer
    ("gated", "mc5"), ("gated_tc", "mc5"), ("gated_tc", "gated"),
    ("prior_cem_sconst", "prior_cem_s1"), ("prior_cem_sp", "prior_cem_s1"), ("prior_cem_sp", "prior_cem_sconst"),
    ("prior_cem_sp", "bc_only"), ("bc_select_khat", "bc_only"), ("bc_select_khat", "prior_cem_sp"),
)
TYPES = ("carry", "pick", "place", "move")


def load_runs(kdir: Path) -> dict[str, dict[int, dict]]:
    runs = {}
    for d in sorted(p for p in kdir.iterdir() if p.is_dir() and p.name != "calibration"):
        seeds = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in d.glob("seed_*.json")}
        if seeds:
            runs[d.name] = seeds
    rank = {n: i for i, n in enumerate(ORDER)}
    return dict(sorted(runs.items(), key=lambda kv: (rank.get(kv[0].split("-")[0], len(ORDER)), kv[0])))


def successes(run: dict[int, dict], seeds) -> np.ndarray:
    return np.concatenate([np.asarray(run[s]["episode_successes"], bool) for s in seeds])


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_flip(diffs: np.ndarray, n_mc: int = 200_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip (randomization) test of mean(diffs) = 0; exact up to 16 seeds."""
    d = np.asarray(diffs, float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(list(itertools.product((1, -1), repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice((1, -1), size=(n_mc, len(d)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def paired(a: dict[int, dict], b: dict[int, dict]):
    """Seeds both runs have, their pooled successes, and the per-seed success-rate differences a - b."""
    seeds = sorted(set(a) & set(b))
    for s in seeds:
        assert a[s]["episodes"] == b[s]["episodes"] and a[s]["start_steps"] == b[s]["start_steps"], \
            f"seed {s}: start states differ"
    xa, xb = successes(a, seeds), successes(b, seeds)
    diffs = np.array([np.mean(a[s]["episode_successes"]) - np.mean(b[s]["episode_successes"]) for s in seeds])
    return seeds, xa, xb, diffs


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


@hydra.main(version_base=None, config_path="../config/eval", config_name="compare")
def main(cfg: DictConfig) -> None:
    offset = int(cfg.compare.offset)
    kdir = ROOT / cfg.compare.out_dir / f"k{offset}"
    runs = load_runs(kdir)
    if not runs:
        print(f"[summary] no runs under {kdir}")
        return
    ref = runs.get(REFERENCE)
    md = [f"# Cube comparison, goal {offset} steps ahead (budget {2 * offset} steps)\n",
          "Non-trivial start states from episodes no learned component was trained on; success = the cube "
          "within 4 cm of the goal. Mean ± std over seeds (50 episodes each); pooled success with a Wilson 95% "
          f"CI. Against `{REFERENCE}` on the same start states: Δ, episodes gained / lost, exact McNemar p "
          "(Holm-adjusted over the runs) and an exact sign-flip p over the per-seed differences.\n"]

    # ---- 1. every run ----
    rows, raw_p = [], []
    for name, run in runs.items():
        seeds = sorted(run)
        rates = np.array([np.mean(run[s]["episode_successes"]) for s in seeds])
        x = successes(run, seeds)
        types = np.concatenate([np.asarray(run[s]["types"]) for s in seeds])
        lo, hi = wilson(int(x.sum()), len(x))
        row = {"name": name, "seeds": len(seeds), "n": len(x), "mean": rates.mean(),
               "std": rates.std(ddof=1) if len(rates) > 1 else 0.0, "pooled": x.mean(), "ci": (lo, hi),
               "types": [f"{x[types == t].mean():.0%}" if (types == t).any() else "–" for t in TYPES]}
        if ref is not None and name != REFERENCE and set(run) & set(ref):
            ps, xa, xb, diffs = paired(run, ref)
            g, l_, p = mcnemar(xb, xa)
            row.update(delta=100 * (xa.mean() - xb.mean()), gl=f"{g} / {l_}", p=p, p_seed=sign_flip(diffs),
                       paired_seeds=len(ps))
            raw_p.append(p)
        rows.append(row)
    adj = iter(holm(raw_p)) if raw_p else iter(())
    md.append("| run | seeds | episodes | success, mean ± std over seeds | pooled [95% CI] | "
              + " | ".join(TYPES) + f" | Δ vs {REFERENCE} (pp) | gained / lost | McNemar p (Holm) | sign-flip p |")
    md.append("|---|---:|---:|---:|---:|" + "---:|" * len(TYPES) + "---:|---:|---:|---:|")
    for r in rows:
        cmp = (f"{r['delta']:+.1f} | {r['gl']} | {fmt_p(next(adj))} | {fmt_p(r['p_seed'])}"
               if "delta" in r else "| | |")
        md.append(f"| `{r['name']}` | {r['seeds']} | {r['n']} | {100 * r['mean']:.1f} ± {100 * r['std']:.1f} | "
                  f"{100 * r['pooled']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}] | "
                  + " | ".join(r["types"]) + f" | {cmp} |")
    first = next(iter(runs.values()))
    types = np.concatenate([np.asarray(v["types"]) for v in first.values()])
    md.append(f"\nTask types (cube held at the start / at the goal; seeds of `{next(iter(runs))}`): "
              + ", ".join(f"{t} {int((types == t).sum())}" for t in TYPES) + ".\n")

    # ---- 2. ablation pairs ----
    pair_rows = [(a, b) for a, b in PAIRS if a in runs and b in runs]
    if pair_rows:
        md.append("## Ablations (paired on the seeds both runs have)\n")
        md.append("| comparison | seeds | success | Δ (pp) | gained / lost | McNemar p | sign-flip p |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for a, b in pair_rows:
            seeds, xa, xb, diffs = paired(runs[a], runs[b])
            if not seeds:
                continue
            g, l_, p = mcnemar(xb, xa)
            md.append(f"| `{a}` vs `{b}` | {len(seeds)} | {100 * xa.mean():.1f} vs {100 * xb.mean():.1f} | "
                      f"{100 * (xa.mean() - xb.mean()):+.1f} | {g} / {l_} | {fmt_p(p)} | {fmt_p(sign_flip(diffs))} |")
        md.append("")

    # ---- 3. planner diagnostics ----
    diag_rows = []
    for name, run in runs.items():
        keys = ("truncated_frac", "mean_scored_blocks", "mean_sigma2")
        vals = {k: [run[s]["diagnostics"][k] for s in run if k in run[s].get("diagnostics", {})] for k in keys}
        if any(vals.values()):
            diag_rows.append(f"| `{name}` | " + " | ".join(f"{np.mean(v):.3f}" if v else "–" for v in vals.values())
                             + " |")
    if diag_rows:
        md.append("## Planner diagnostics (mean over planning calls and seeds)\n")
        md.append("| run | fraction of candidates truncated | scored blocks (of 5) | mean sigma^2 |")
        md.append("|---|---:|---:|---:|")
        md += diag_rows + [""]

    # ---- 4. per-seed success rates ----
    all_seeds = sorted(set().union(*[set(r) for r in runs.values()]))
    md.append("## Success rate per seed (%)\n")
    md.append("| run | " + " | ".join(str(s) for s in all_seeds) + " |")
    md.append("|---|" + "---:|" * len(all_seeds))
    for name, run in runs.items():
        md.append(f"| `{name}` | " + " | ".join(f"{100 * np.mean(run[s]['episode_successes']):.0f}" if s in run
                                                else "–" for s in all_seeds) + " |")
    (kdir / "summary.md").write_text("\n".join(md) + "\n")
    with (kdir / "per_seed.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "episodes", "successes", "success_rate"])
        for name, run in runs.items():
            for s in sorted(run):
                x = np.asarray(run[s]["episode_successes"], bool)
                w.writerow([name, s, len(x), int(x.sum()), f"{x.mean():.4f}"])
    print("\n".join(md))
    print(f"\n[summary] wrote {kdir / 'summary.md'} and {kdir / 'per_seed.csv'}")


if __name__ == "__main__":
    main()
