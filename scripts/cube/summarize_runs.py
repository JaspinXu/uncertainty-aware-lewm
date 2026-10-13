"""Success-rate panel for Cube evaluation runs (any set of planners / checkpoints).

    python scripts/cube/summarize_runs.py                        # every run in results/cube/runs
    python scripts/cube/summarize_runs.py --methods baseline mc5 gated goal_cvar \
        --reference baseline --name l2_panel

A run is ``results/cube/runs/<label>_s<seed>.json`` written by evaluate.py;
``label`` (run id without the seed) identifies the planner/checkpoint variant.
Episodes of different labels with the same seed share start/goal states, so
labels are compared per episode (exact McNemar test on the pooled pairs).

With the dataset available, each episode is also tagged *trivial* when the
cube starts within the 4 cm success radius of its goal (any action succeeds);
success on the non-trivial episodes is reported separately.

Writes results/cube/summary/<name>.md and <name>.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from cube.metrics import mcnemar_exact, wilson_ci  # noqa: E402

RUN_RE = re.compile(r"^(?P<label>.+)_s(?P<seed>\d+)$")
SUCCESS_RADIUS = 0.04


def load_runs(run_dir: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for p in sorted(run_dir.glob("*.json")):
        m = RUN_RE.match(p.stem)
        if not m:
            continue
        d = json.loads(p.read_text())
        runs.setdefault(m["label"], {})[int(m["seed"])] = d
    return runs


def start_distances(runs: dict, dataset: str, goal_offset: int) -> dict[tuple[int, int, int], float]:
    """(episode, start_step) -> cube-goal distance at the start state."""
    try:
        import stable_worldmodel as swm

        from cube import data as cube_data

        ds = swm.data.HDF5Dataset(dataset, keys_to_cache=[], cache_dir=swm.data.utils.get_cache_dir())
    except Exception as e:  # dataset not available (e.g. laptop)
        print(f"[summary] no dataset ({type(e).__name__}); skipping trivial/non-trivial split")
        return {}
    pos_col = cube_data.first_present(ds, ["privileged_block_0_pos", "privileged/block_0_pos"])
    pos = np.asarray(ds.get_col_data(pos_col)).reshape(-1, 3)
    offsets = np.asarray(ds.offsets)
    out = {}
    for by_seed in runs.values():
        for d in by_seed.values():
            for ep, st in zip(d["episodes"], d["start_steps"]):
                if (ep, st) not in out:
                    row = offsets[ep] + st
                    out[(ep, st)] = float(np.linalg.norm(pos[row] - pos[row + goal_offset]))
    return out


def summarize(runs: dict, labels: list[str], reference: str | None, dist: dict) -> dict:
    out = {"labels": {}, "comparisons": {}}
    for label in labels:
        by_seed = runs.get(label, {})
        if not by_seed:
            continue
        seeds = sorted(by_seed)
        per_seed = {s: 100.0 * np.mean(by_seed[s]["episode_successes"]) for s in seeds}
        allep = np.concatenate([np.asarray(by_seed[s]["episode_successes"], bool) for s in seeds])
        lo, hi = wilson_ci(int(allep.sum()), len(allep))
        row = {
            "seeds": seeds,
            "per_seed": per_seed,
            "mean": float(np.mean(list(per_seed.values()))),
            "std": float(np.std(list(per_seed.values()), ddof=1)) if len(seeds) > 1 else 0.0,
            "pooled": float(100 * allep.mean()),
            "pooled_ci": [100 * lo, 100 * hi],
            "n_episodes": int(len(allep)),
            "eval_time_mean_s": float(np.mean([by_seed[s].get("evaluation_time", np.nan) for s in seeds])),
        }
        if dist:
            nt = np.concatenate(
                [
                    [dist[(e, t)] > SUCCESS_RADIUS for e, t in zip(by_seed[s]["episodes"], by_seed[s]["start_steps"])]
                    for s in seeds
                ]
            ).astype(bool)
            row["nontrivial_n"] = int(nt.sum())
            row["nontrivial_success"] = float(100 * allep[nt].mean()) if nt.any() else None
            row["trivial_success"] = float(100 * allep[~nt].mean()) if (~nt).any() else None
        out["labels"][label] = row

    if reference and reference in runs:
        ref = runs[reference]
        for label in labels:
            if label == reference or label not in runs:
                continue
            common = sorted(set(ref) & set(runs[label]))
            if not common:
                continue
            a = np.concatenate([np.asarray(ref[s]["episode_successes"], bool) for s in common])
            b = np.concatenate([np.asarray(runs[label][s]["episode_successes"], bool) for s in common])
            deltas = [100 * (np.mean(runs[label][s]["episode_successes"]) - np.mean(ref[s]["episode_successes"])) for s in common]
            cmp = {
                "seeds": common,
                "delta_mean_pp": float(np.mean(deltas)),
                "delta_per_seed_pp": [float(x) for x in deltas],
                "mcnemar": mcnemar_exact(a, b),
            }
            if dist:
                nt = np.concatenate(
                    [[dist[(e, t)] > SUCCESS_RADIUS for e, t in zip(ref[s]["episodes"], ref[s]["start_steps"])] for s in common]
                ).astype(bool)
                cmp["nontrivial_delta_pp"] = float(100 * (b[nt].mean() - a[nt].mean())) if nt.any() else None
                cmp["nontrivial_mcnemar"] = mcnemar_exact(a[nt], b[nt]) if nt.any() else None
            out["comparisons"][label] = cmp
    return out


def to_markdown(s: dict, reference: str | None, title: str) -> str:
    L = [f"# {title}\n", "Generated by `scripts/cube/summarize_runs.py`. Success rates in %, 50 episodes per seed.\n"]
    labels = list(s["labels"])
    if not labels:
        return "\n".join(L + ["(no runs)"])
    all_seeds = sorted({x for v in s["labels"].values() for x in v["seeds"]})
    L.append("| label | " + " | ".join(str(x) for x in all_seeds) + " | mean ± std | pooled [95% CI] | non-trivial |")
    L.append("|---|" + "---:|" * len(all_seeds) + "---:|---:|---:|")
    for lab in labels:
        v = s["labels"][lab]
        cells = [f"{v['per_seed'][x]:.0f}" if x in v["per_seed"] else "" for x in all_seeds]
        nt = v.get("nontrivial_success")
        nt_s = f"{nt:.1f} (n={v['nontrivial_n']})" if nt is not None else "n/a"
        L.append(
            f"| {lab} | " + " | ".join(cells)
            + f" | **{v['mean']:.1f} ± {v['std']:.1f}** | {v['pooled']:.1f} [{v['pooled_ci'][0]:.1f}, {v['pooled_ci'][1]:.1f}] | {nt_s} |"
        )
    if s["comparisons"]:
        L.append(f"\n## Paired comparison against `{reference}` (same start/goal states)\n")
        L.append("| label | seeds | Δ mean (pp) | gained / lost episodes | McNemar p | Δ non-trivial (pp) | p (non-trivial) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for lab, c in s["comparisons"].items():
            m = c["mcnemar"]
            ntd = c.get("nontrivial_delta_pp")
            ntp = (c.get("nontrivial_mcnemar") or {}).get("p_value")
            L.append(
                f"| {lab} | {len(c['seeds'])} | {c['delta_mean_pp']:+.2f} | {m['b_only']} / {m['a_only']} | {m['p_value']:.3g} | "
                + (f"{ntd:+.2f}" if ntd is not None else "n/a") + " | " + (f"{ntp:.3g}" if ntp is not None else "n/a") + " |"
            )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "results/cube/runs")
    ap.add_argument("--methods", nargs="*", default=None, help="labels to include (default: all)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="restrict to these seeds")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--name", default="panel")
    ap.add_argument("--title", default="OGBench-Cube planning success")
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/summary")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if args.seeds is not None:
        keep = set(args.seeds)
        runs = {k: {s: d for s, d in v.items() if s in keep} for k, v in runs.items()}
    labels = args.methods or sorted(runs)
    dist = start_distances({k: runs[k] for k in labels if k in runs}, args.dataset, args.goal_offset)
    s = summarize(runs, labels, args.reference, dist)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.name}.json").write_text(json.dumps(s, indent=1))
    md = to_markdown(s, args.reference, args.title)
    (args.out / f"{args.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
    os._exit(0)
