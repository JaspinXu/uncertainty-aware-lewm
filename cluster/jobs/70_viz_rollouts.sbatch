#!/bin/bash
#SBATCH --job-name=lewm_viz
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
# Planning runs for the rollout videos: baseline and gated on a few test seeds,
# saving the executed actions (results/cube/viz_runs/rollouts/*.npz, ~0.3 MB each).
# Rendering happens elsewhere (scripts/cube/viz_blender.py needs Blender's bpy).
#     sbatch cluster/jobs/70_viz_rollouts.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/runs.sh
job_banner
require_gpu || exit 1
for seed in "${VIZ_SEEDS[@]}"; do
  for method in baseline gated; do
    if done_viz "$method" "$seed"; then echo "[skip] viz_${method}_s${seed}"; continue; fi
    run_viz "$method" "$seed"
  done
done
# do the re-runs reproduce the panel's episode outcomes? (GPU type and batch can change CEM details)
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("results/cube/viz_runs").glob("viz_*.json")):
    v = json.loads(p.read_text())
    ref = pathlib.Path("results/cube/runs") / f"{v['method']}_s{v['seed']}.json"
    if ref.exists():
        r = json.loads(ref.read_text())
        same = sum(a == b for a, b in zip(v["episode_successes"], r["episode_successes"]))
        print(f"{p.stem}: success {v['success_rate']:.0f}% (panel {r['success_rate']:.0f}%), same outcome in {same}/{len(r['episode_successes'])} episodes")
PY
