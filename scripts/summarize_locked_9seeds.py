#!/usr/bin/env python
"""Build the final 9-seed success-rate table (report Table 1 / slides Fig. 4-5).

Inputs (all produced by the locked protocol: n=50, CEM 300/30/30, RH=5):
  seeds 42-45      -> pusht/locked_s42_45/results/{baseline,mc_n5,gated_n5,goal_a0p1}_s<seed>.txt
  seeds 1011-5055  -> pusht/fair_ua_final/results/test_{baseline,mc_n5,gated_n5}_s<seed>.txt
                      pusht/locked_s42_45/results/goal_a0p1_s<seed>.txt

Output (also printed to the terminal):
  pusht/locked_s42_45/combined_9seeds.md

Run from the repo root:
  python scripts/summarize_locked_9seeds.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "pusht" / "locked_s42_45" / "results"
FAIR = ROOT / "pusht" / "fair_ua_final" / "results"
OUT_MD = ROOT / "pusht" / "locked_s42_45" / "combined_9seeds.md"

SEEDS_A = [42, 43, 44, 45]
SEEDS_B = [1011, 2022, 3033, 4044, 5055]
SEEDS = SEEDS_A + SEEDS_B

# method label -> (file pattern for seeds 42-45, file pattern for seeds 1011-5055)
METHODS = {
    "Baseline": (LOCKED / "baseline_s{}.txt", FAIR / "test_baseline_s{}.txt"),
    "MC-5": (LOCKED / "mc_n5_s{}.txt", FAIR / "test_mc_n5_s{}.txt"),
    "Gated N=5": (LOCKED / "gated_n5_s{}.txt", FAIR / "test_gated_n5_s{}.txt"),
    "Goal-CVaR α=0.1": (LOCKED / "goal_a0p1_s{}.txt", LOCKED / "goal_a0p1_s{}.txt"),
}


def success_rate(path: Path) -> float:
    text = path.read_text(errors="ignore")
    m = re.search(r"['\"]success_rate['\"]\s*:\s*([0-9.]+)", text)
    if m is None:
        raise ValueError(f"no success_rate in {path}")
    return float(m.group(1))


def collect() -> dict[str, np.ndarray]:
    table = {}
    for name, (pat_a, pat_b) in METHODS.items():
        vals = [success_rate(Path(str(pat_a).format(s))) for s in SEEDS_A]
        vals += [success_rate(Path(str(pat_b).format(s))) for s in SEEDS_B]
        table[name] = np.asarray(vals, dtype=float)
    return table


def fmt_stat(v: np.ndarray) -> str:
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def write_markdown(table: dict[str, np.ndarray]) -> None:
    lines = ["# Combined locked panels (9 seeds)\n\n"]
    lines.append("Protocol: `pusht/lewm` | n=50 | CEM 300/30/30 | RH=5\n\n")
    head = "| Method | " + " | ".join(str(s) for s in SEEDS) + " | mean ± std (9) |\n"
    lines.append(head)
    lines.append("|---|" + "---:|" * (len(SEEDS) + 1) + "\n")
    for name, v in table.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{x:.1f}" for x in v) + f" | **{fmt_stat(v)}** |\n"
        )

    base = table["Baseline"]
    lines.append("\n## vs baseline (pp, over 9 seeds)\n\n| Method | Δ mean | per-seed Δ |\n|---|---:|---|\n")
    for name, v in table.items():
        if name == "Baseline":
            continue
        d = v - base
        lines.append(f"| {name} | {d.mean():+.2f} | {', '.join(f'{x:+.1f}' for x in d)} |\n")

    lines.append("\n## Subgroup means\n\n| Method | mean 42–45 | mean 1011–5055 | mean all 9 |\n|---|---:|---:|---:|\n")
    na = len(SEEDS_A)
    for name, v in table.items():
        lines.append(f"| {name} | {fmt_stat(v[:na])} | {fmt_stat(v[na:])} | {fmt_stat(v)} |\n")

    OUT_MD.write_text("".join(lines))
    print("".join(lines))
    print(f"[write] {OUT_MD}")


if __name__ == "__main__":
    table = collect()
    write_markdown(table)
