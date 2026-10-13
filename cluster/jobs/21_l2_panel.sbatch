#!/bin/bash
#SBATCH --job-name=lewm_l2
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out
# L2 locked panel on Cube: uncertainty-aware planners x 9 test seeds.
# One array task = (method, seed); the baseline comes from the L1 runs.
#     sbatch --array=0-35 cluster/jobs/21_l2_panel.sbatch
#
#   mc5        mean of 5 MC-dropout rollouts
#   gated      sigma^2 gate, threshold/omega calibrated for Cube (20_l2_calibrate)
#   goal_cvar  CVaR soft penalty (self-calibrating per planning call)
#   penalty    MC-5 + lambda * sigma^2_H, lambda selected in L1 (signal_summary.json)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/seeds.sh
job_banner

METHODS=(mc5 gated goal_cvar penalty)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NS=${#TEST_SEEDS[@]}
if (( IDX >= ${#METHODS[@]} * NS )); then echo "index $IDX out of range"; exit 0; fi
METHOD="${METHODS[$((IDX / NS))]}"
SEED="${TEST_SEEDS[$((IDX % NS))]}"
RUN="${METHOD}_s${SEED}"
if [[ -f "results/cube/runs/$RUN.json" ]]; then echo "[skip] $RUN exists"; exit 0; fi

EXTRA=()
case "$METHOD" in
  gated)
    read -r THR OMEGA < <(python scripts/cube/l2_params.py gated)
    EXTRA=(method.truncation.threshold="$THR" method.truncation.omega="$OMEGA")
    ;;
  penalty)
    LAM=$(python scripts/cube/l2_params.py penalty)
    EXTRA=(method.truncation.penalty_weight="$LAM")
    ;;
esac
echo "method=$METHOD seed=$SEED extra=${EXTRA[*]:-none}"
python evaluate.py --config-name cube method="$METHOD" seed="$SEED" output.run_id="$RUN" ${EXTRA[@]+"${EXTRA[@]}"}
