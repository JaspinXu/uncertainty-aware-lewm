"""L6: gather every Cube result into results/cube/RESULTS.md (numbers only).

    python scripts/cube/make_report.py

Re-generates the panels from the per-run JSON files, summarises the L4
training logs (with a training-curve figure) and links the L1/L3/L5 reports.
Sections whose inputs do not exist yet are listed as pending, so the file can
be regenerated at any point of the project.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TEST_SEEDS = [42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055]
PLANNERS = ["baseline", "mc5", "gated", "goal_cvar", "penalty", "risk"]
VARIANTS = ["lewm", "ms", "prop", "prop_aux", "prop_ms"]
R = Path(os.environ.get("CUBE_RESULTS", ROOT / "results/cube"))  # override for tests


def demote(md: str) -> str:
    """Embed a markdown fragment one heading level deeper."""
    return md.replace("\n## ", "\n### ")


def summarize(name: str, labels: list[str], title: str) -> str | None:
    have = [lab for lab in labels if any((R / "runs").glob(f"{lab}_s*.json"))]
    if len(have) < 1:
        return None
    cmd = [sys.executable, str(ROOT / "scripts/cube/summarize_runs.py"), "--name", name, "--title", title,
           "--runs", str(R / "runs"), "--out", str(R / "summary"),
           "--methods", *have, "--seeds", *map(str, TEST_SEEDS), "--reference", "baseline"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return (R / "summary" / f"{name}.md").read_text()


def training_section() -> str | None:
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / "lewm_data")) / "checkpoints/cube"
    rows, curves = [], {}
    for v in VARIANTS:
        log = home / f"l4_{v}" / "log.csv"
        if not log.exists():
            continue
        with log.open() as f:
            recs = [r for r in csv.DictReader(f)]
        if not recs:
            continue

        def last(key):
            vals = [float(r[key]) for r in recs if r.get(key) not in (None, "")]
            return vals[-1] if vals else float("nan")

        done = (home / f"l4_{v}" / "DONE").exists()
        rows.append((v, int(recs[-1]["step"]), done, last("pred"), last("val_pred"), last("val_ar"), last("val_redund"),
                     last("val_aux"), last("samples_per_s")))
        curves[v] = [(int(r["step"]), float(r["val_pred"])) for r in recs if r.get("val_pred")]
    if not rows:
        return None
    L = ["| variant | steps | finished | train pred | val pred (1-step) | val 5-step | val redundancy R² | val aux | samples/s |",
         "|---|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for v, st, done, p, vp, va, vr, vx, sps in rows:
        f = lambda x: "" if x != x else f"{x:.4f}"  # noqa: E731
        L.append(f"| {v} | {st} | {'yes' if done else 'no'} | {f(p)} | {f(vp)} | {f(va)} | {f(vr)} | {f(vx)} | {sps:.0f} |")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.5))
        for v, c in curves.items():
            if c:
                ax.plot(*zip(*c), label=v)
        ax.set_xlabel("step")
        ax.set_ylabel("validation 1-step prediction loss")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        fig.tight_layout()
        (R / "l4").mkdir(parents=True, exist_ok=True)
        fig.savefig(R / "l4/fig_training.png", dpi=150)
        L.append("\n![training curves](l4/fig_training.png)")
    except Exception as e:  # plotting is optional
        L.append(f"\n(figure skipped: {e})")
    return "\n".join(L)


def main() -> None:
    out = ["# OGBench-Cube results (auto-generated)\n",
           "Produced by `scripts/cube/make_report.py`; every number comes from the files it links. "
           "Test panel = seeds " + ", ".join(map(str, TEST_SEEDS)) + " (50 episodes each).\n"]
    pending = []

    info = next((p for p in (R / "dataset_info.json", R / "smoke/dataset_info.json") if p.exists()), None)
    if info:
        d = json.loads(info.read_text())
        out.append("## Dataset and protocol\n")
        triv = [v["trivial_frac"] for v in d.get("protocol", {}).get("per_seed", {}).values()]
        out.append(f"- {d['episodes']} episodes, {d['rows']} frames ({d['path'].split('/')[-1]}), episode length {d['episode_length']}")
        if triv:
            out.append(f"- start states already within 4 cm of the goal (any action succeeds): {100 * sum(triv) / len(triv):.1f}% on average over the protocol seeds\n")
    else:
        pending.append("dataset summary (cluster/jobs/02_smoke_test.sbatch)")

    l1 = R / "l1/signal_summary.json"
    if l1.exists():
        s = json.loads(l1.read_text())
        q3 = s["q3_false_positives"]["sig2_H"]
        out.append("## L1 sigma^2 signal test\n")
        out.append(f"Verdict **{s['verdict']}** (details: [l1/signal_report.md](l1/signal_report.md)). "
                   f"Elite-failure AUROC of sigma^2_H = {q3['auroc']} (95% CI {q3['ci']}).\n")
    else:
        pending.append("L1 signal test (bash cluster/submit.sh l1)")

    panel = summarize("planning_panel", PLANNERS, "Planning on Cube: uncertainty-aware planners (L2) and learned risk (L3)")
    if panel:
        out.append("## L2 / L3 planners\n")
        out.append(demote(panel.split("\n", 2)[2]))
    else:
        pending.append("L2/L3 planning panel")
    if (R / "l3/risk_report.md").exists():
        out.append("Offline risk-head report: [l3/risk_report.md](l3/risk_report.md)\n")

    tr = training_section()
    if tr:
        out.append("## L4 representation training\n")
        out.append(tr + "\n")
    else:
        pending.append("L4 training (bash cluster/submit.sh l4)")

    rep = R / "l5/representation.md"
    if rep.exists():
        out.append("## L5 representations\n")
        out.append(demote(rep.read_text().split("\n", 2)[2]))
    vpanel = summarize("variants_panel", ["baseline"] + [f"l4_{v}" for v in VARIANTS],
                       "Planning with the L4 representation variants")
    if vpanel and "l4_" in vpanel:
        out.append("## L5 planning with the variants\n")
        out.append("`baseline` = pretrained LeWM-Cube; `l4_lewm` = our retrained control.\n")
        out.append(demote(vpanel.split("\n", 2)[2]))
        sel = sorted((R / "l5").glob("wprop_*.json"))
        if sel:
            out.append("w_prop selected on the calibration seeds: " + ", ".join(
                f"{json.loads(p.read_text())['variant']}: {json.loads(p.read_text())['w_prop']}" for p in sel) + "\n")
    else:
        pending.append("L5 planning panel")

    if pending:
        out.append("## Pending\n")
        out.extend(f"- {p}" for p in pending)
    (R / "RESULTS.md").write_text("\n".join(out) + "\n")
    print((R / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
    os._exit(0)
