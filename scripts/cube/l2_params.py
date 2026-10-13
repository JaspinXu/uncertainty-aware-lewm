"""Print Cube-specific planner parameters for the L2 panel (used by job scripts).

    python scripts/cube/l2_params.py gated     -> "<threshold> <omega>"
    python scripts/cube/l2_params.py penalty   -> "<penalty_weight>"

gated:   threshold = median planning-time sigma^2_0 on Cube calibration episodes
         (results/calibration/gated_cube.json); omega is rescaled so that
         omega * threshold equals the PushT setting (omega=1 at threshold 0.2535),
         i.e. the gate truncates equally aggressively in units of the median sigma^2.
penalty: lambda* selected on the calibration seeds by the L1 re-ranking analysis
         (results/cube/l1/signal_summary.json). If the selection is lambda = 0
         (no sigma^2 penalty helps), the pre-specified equal-scale value
         lambda = median(cost) / median(sigma^2) is used instead, so the panel still
         measures a non-trivial penalty; the choice is printed to stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUSHT_OMEGA_TIMES_THRESHOLD = 1.0 * 0.2535046935081482


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "gated":
        cal = json.loads((ROOT / "results/calibration/gated_cube.json").read_text())
        thr = float(cal["recommended_threshold"])
        print(f"{thr:.6g} {PUSHT_OMEGA_TIMES_THRESHOLD / thr:.6g}")
    elif what == "penalty":
        s = json.loads((ROOT / "results/cube/l1/signal_summary.json").read_text())["q5_rerank"]
        if s["lambda_star_multiplier"] > 0:
            lam, why = s["lambda_star"], f"L1 selection (multiplier {s['lambda_star_multiplier']})"
        else:
            lam, why = s["ratio_cost_over_sigma2"], "L1 selected lambda=0 -> equal-scale default"
        print(f"[l2_params] penalty_weight={lam:.6g} ({why})", file=sys.stderr)
        print(f"{lam:.6g}")
    else:
        sys.exit("usage: l2_params.py gated|penalty")


if __name__ == "__main__":
    main()
