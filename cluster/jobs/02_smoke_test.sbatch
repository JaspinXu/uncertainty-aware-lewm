#!/bin/bash
#SBATCH --job-name=lewm_smoke
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# End-to-end check on 10 Cube episodes: rendering, dataset layout, pretrained
# checkpoint, baseline planning with recording, oracle replay (+ fidelity).
#     sbatch cluster/jobs/02_smoke_test.sbatch
# Pick a GPU type with e.g.  sbatch --gpus=a100-40:1 cluster/jobs/02_smoke_test.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
SMOKE=results/cube/smoke

echo "--- render check"
python scripts/cube/check_render.py --out $SMOKE/render_check.png

echo "--- dataset"
python scripts/cube/inspect_dataset.py --out $SMOKE/dataset_info.json

echo "--- baseline planning, 10 episodes, full CEM budget, recording on"
START=$(date +%s)
python evaluate.py --config-name cube method=baseline seed=0 eval.num_eval=10 \
  record.enabled=true output.run_id=smoke_baseline \
  output.dir=$SMOKE/runs record.dir=$SMOKE/records
echo "planning wall time: $(( $(date +%s) - START ))s"

echo "--- oracle replay"
START=$(date +%s)
python scripts/cube/oracle_replay.py $SMOKE/records/smoke_baseline.pt \
  --out $SMOKE/oracle --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "oracle wall time: $(( $(date +%s) - START ))s"

echo "--- smoke test finished OK"
cat $SMOKE/runs/smoke_baseline.json | head -c 400; echo
