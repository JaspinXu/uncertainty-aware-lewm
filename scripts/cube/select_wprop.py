"""Pick w_prop for a proprio variant from its calibration-seed runs.

    python scripts/cube/select_wprop.py prop_ms     -> prints e.g. 0.1

Reads results/cube/runs/l4_<variant>_w<w>_s<seed>.json (42_l5_plan.sbatch select),
returns the w with the highest mean success over the calibration seeds; ties go
to the larger w (closest to the unweighted latent distance). Writes the table to
results/cube/l5/wprop_<variant>.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    v = sys.argv[1]
    pat = re.compile(rf"^l4_{re.escape(v)}_w(?P<w>[0-9.]+)_s(?P<s>\d+)$")
    table: dict[float, list[float]] = {}
    for p in (ROOT / "results/cube/runs").glob(f"l4_{v}_w*_s*.json"):
        m = pat.match(p.stem)
        if m:
            table.setdefault(float(m["w"]), []).append(json.loads(p.read_text())["success_rate"])
    if not table:
        sys.exit(f"no calibration runs for variant {v}")
    means = {w: float(np.mean(s)) for w, s in table.items()}
    best = max(means.values())
    w_star = max(w for w, m in means.items() if m == best)
    out = ROOT / "results/cube/l5" / f"wprop_{v}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variant": v, "mean_success": {str(k): m for k, m in sorted(means.items())},
                               "n_seeds": {str(k): len(s) for k, s in table.items()}, "w_prop": w_star}, indent=1))
    print(f"{w_star:g}")


if __name__ == "__main__":
    main()
