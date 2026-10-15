"""Print how many evaluation runs cluster/submit.sh may put into one Slurm job.

    python3 scripts/cube/runs_per_job.py record|plan|l5   ->  "TARGET MAX"

TARGET runs fill about TARGET_MIN minutes, MAX runs about MAX_MIN (jobs on the
`gpu` partition may run 180 min, and a job may get a slower GPU type than the
smoke test; the job runner stops early rather than start a run that would not
fit). The estimate comes from the GPU timings of the smoke
test (results/cube/smoke/timing.json, 10 episodes):

    planning_s   wall time of the recorded baseline run (incl. start-up)
    eval_s       time inside the evaluation loop
    cem_s        time inside the CEM solver
    oracle_s     wall time of the oracle replay

A 50-episode run is assumed to cost 5x the per-episode parts (conservative:
batched CEM on the GPU grows slower than that); the L2/L3 planners score every
candidate with 5-8 stochastic rollouts (PLAN_FACTOR x the CEM time). Without a
GPU smoke test conservative defaults are used. Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_MIN, MAX_MIN = 45, 120
DEFAULT = {"record": (3, 4), "plan": (4, 6), "l5": (6, 8)}
PLAN_FACTOR = {"record": 1.0, "plan": 8.0, "l5": 1.5}


def run_minutes(kind: str, t: dict) -> float:
    scale = 50.0 / float(t["episodes"])
    eval_s = float(t["eval_s"])
    cem_s = min(float(t.get("cem_s") or eval_s), eval_s)
    overhead = max(0.0, float(t["planning_s"]) - eval_s)
    loop = scale * ((eval_s - cem_s) + PLAN_FACTOR[kind] * cem_s)
    if kind == "record":
        loop += scale * float(t["oracle_s"])
    return (overhead + loop) / 60.0 + 1.0


def runs_per_job(kind: str, timing: dict | None) -> tuple[int, int]:
    try:
        if not timing or not timing.get("gpu"):
            raise ValueError("no GPU timing")
        minutes = run_minutes(kind, timing)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return DEFAULT[kind]
    target = max(1, min(24, int(TARGET_MIN // minutes)))
    most = max(target, min(24, int(MAX_MIN // minutes)))
    return target, most


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if kind not in DEFAULT:
        sys.exit(f"unknown kind {kind!r} (record|plan|l5)")
    path = ROOT / "results/cube/smoke/timing.json"
    timing = json.loads(path.read_text()) if path.exists() else None
    print(*runs_per_job(kind, timing))


if __name__ == "__main__":
    main()
