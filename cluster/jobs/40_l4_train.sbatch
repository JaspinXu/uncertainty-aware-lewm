#!/bin/bash
#SBATCH --job-name=lewm_l4
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# L4: train one representation variant on Cube. Resumable and self-chaining:
# the run stops ~12 min before the job's time limit, saves its state and
# resubmits itself; when max-steps is reached it submits the L5 evaluation.
#     sbatch cluster/jobs/40_l4_train.sbatch <variant> [max_steps]
#     variants: lewm | ms | prop | prop_aux | prop_ms     (see train_cube.py)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
V="$1"
STEPS="${2:-${L4_MAX_STEPS:-60000}}"
LEFT=$(python scripts/cube/slurm_time_left.py)
python train_cube.py --variant "$V" --max-steps "$STEPS" --time-limit-min $(( LEFT > 30 ? LEFT - 12 : 18 ))
status=$?
if [[ $status -eq 3 ]]; then
  echo "[l4] $V not finished -> resubmitting"
  read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"
  sbatch ${EXTRA[@]+"${EXTRA[@]}"} cluster/jobs/40_l4_train.sbatch "$V" "$STEPS"
elif [[ $status -eq 0 ]]; then
  if [[ "${L4_SKIP_EVAL:-0}" != "1" ]]; then
    echo "[l4] $V finished -> submitting its L5 evaluation"
    bash cluster/submit.sh l5 "$V"
  fi
else
  echo "[l4] training failed with exit code $status"
  exit "$status"
fi
