#!/bin/bash
#SBATCH --job-name=lewm_l1
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%A_%a.out
# Baseline LeWM planning on Cube with every planning call recorded, followed by
# the oracle replay of the recorded candidates. One array task = one seed.
#
#   sbatch --array=0-11 cluster/jobs/10_record_oracle.sbatch l1      # 3 cal + 9 test seeds
#   sbatch --array=0-15 cluster/jobs/10_record_oracle.sbatch train   # L3 risk-head data
#
# Optional second argument: checkpoint (default cube/lewm), third: run prefix.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/seeds.sh
job_banner

GROUP="${1:-l1}"
POLICY="${2:-cube/lewm}"
PREFIX="${3:-baseline}"
read -r -a SEEDS <<< "$(seed_group "$GROUP")"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#SEEDS[@]} )); then echo "array index $IDX out of range for group $GROUP"; exit 0; fi
SEED="${SEEDS[$IDX]}"
RUN="${PREFIX}_s${SEED}"
echo "group=$GROUP seed=$SEED run=$RUN policy=$POLICY"

if [[ -f results/cube/records/$RUN.pt && -f results/cube/runs/$RUN.json ]]; then
  echo "[skip] planning already recorded for $RUN"
else
  python evaluate.py --config-name cube method=baseline seed="$SEED" policy="$POLICY" \
    record.enabled=true output.run_id="$RUN"
fi
python scripts/cube/oracle_replay.py "results/cube/records/$RUN.pt" \
  --out results/cube/oracle --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) --skip-existing
echo "[done] $RUN"
