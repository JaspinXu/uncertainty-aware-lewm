#!/bin/bash
#SBATCH --job-name=lewm_l3
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out
# L3 closed-loop panel: risk-aware planner (selection from risk_selection.json) x 9 test seeds.
#     sbatch --array=0-8 cluster/jobs/31_l3_panel.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/seeds.sh
job_banner
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${TEST_SEEDS[$IDX]}"
RUN="risk_s${SEED}"
if [[ -f "results/cube/runs/$RUN.json" ]]; then echo "[skip] $RUN exists"; exit 0; fi
read -r HEAD MODE BETA < <(python - <<'PY'
import json
s = json.load(open("results/cube/l3/risk_selection.json"))
print(s["head_path"], s["mode"], s["beta"])
PY
)
echo "seed=$SEED head=$HEAD mode=$MODE beta=$BETA"
python evaluate.py --config-name cube method=risk seed="$SEED" output.run_id="$RUN" \
  method.head_path="$HEAD" method.mode="$MODE" method.beta="$BETA"
