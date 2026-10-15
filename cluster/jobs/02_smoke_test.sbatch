#!/bin/bash
#SBATCH --job-name=lewm_smoke
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# End-to-end check on 10 Cube episodes: GPU, rendering, dataset layout,
# pretrained checkpoint, baseline planning with recording, oracle replay
# (+ fidelity of start-of-episode and mid-episode plans), training throughput.
# Writes results/cube/smoke/timing.json, which cluster/submit.sh uses to decide
# how many runs go into one job.
#     bash cluster/submit.sh smoke        (or: sbatch cluster/jobs/02_smoke_test.sbatch)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
require_gpu
SMOKE=results/cube/smoke
mkdir -p $SMOKE

echo "--- GPU nodes of partition $LEWM_GPU_PARTITION (node, GRES, features, state)"
sinfo -h -N -p "$LEWM_GPU_PARTITION" -o "%N %G %f %t" 2>/dev/null | sort -u | head -80 || true

echo "--- render check"
python scripts/cube/check_render.py --out $SMOKE/render_check.png

echo "--- dataset"
python scripts/cube/inspect_dataset.py --out $SMOKE/dataset_info.json

echo "--- baseline planning, 10 episodes, full CEM budget, recording on"
START=$(date +%s)
python evaluate.py --config-name cube method=baseline seed=0 eval.num_eval=10 \
  record.enabled=true output.run_id=smoke_baseline \
  output.dir=$SMOKE/runs record.dir=$SMOKE/records 2>&1 | tee $SMOKE/plan.log
PLAN_S=$(( $(date +%s) - START ))
echo "planning wall time: ${PLAN_S}s"

echo "--- oracle replay"
START=$(date +%s)
python scripts/cube/oracle_replay.py $SMOKE/records/smoke_baseline.pt \
  --out $SMOKE/oracle --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))
ORACLE_S=$(( $(date +%s) - START ))
echo "oracle wall time: ${ORACLE_S}s"

echo "--- training throughput (largest variant, at most 10 min, discarded afterwards)"
set +e
python train_cube.py --variant prop_ms --max-steps 150 --run-name smoke_speed --log-every 50 \
  --val-every 100000 --ckpt-every 100000 --warmup-steps 50 --time-limit-min 10 2>&1 | tee $SMOKE/train_speed.log
status=${PIPESTATUS[0]}
set -e
rm -rf "$STABLEWM_HOME/checkpoints/cube/smoke_speed"
[[ $status -eq 0 || $status -eq 3 ]] || { echo "training check failed (exit $status)"; exit "$status"; }

python - "$SMOKE" "$PLAN_S" "$ORACLE_S" <<'PY'
import json, re, sys
from pathlib import Path

import torch

smoke, plan_s, oracle_s = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
speeds = re.findall(r"samples_per_s=([0-9.]+)", (smoke / "train_speed.log").read_text())
cem = [float(x) for x in re.findall(r"CEM solve time: ([0-9.]+)", (smoke / "plan.log").read_text())]
run = json.loads((smoke / "runs/smoke_baseline.json").read_text())
fid = torch.load(smoke / "oracle/smoke_baseline.pt", map_location="cpu", weights_only=False)["meta"]["fidelity"]
timing = {
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "episodes": 10,
    "planning_s": plan_s,
    "eval_s": round(float(run["evaluation_time"]), 1),
    "cem_s": round(sum(cem), 1) if cem else None,
    "oracle_s": oracle_s,
    "success_rate": run["success_rate"],
    "train_samples_per_s": float(speeds[-1]) if len(speeds) > 1 else None,  # skip the warm-up value
    "fidelity": fid,
}
(smoke / "timing.json").write_text(json.dumps(timing, indent=1))
print("timing:", json.dumps(timing))
PY

echo "--- smoke test finished OK"
head -c 400 $SMOKE/runs/smoke_baseline.json; echo
