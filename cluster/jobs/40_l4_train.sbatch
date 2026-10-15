#!/bin/bash
#SBATCH --job-name=lewm_l4
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
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
require_gpu || exit 1
V="$1"
STEPS="${2:-${L4_MAX_STEPS:-20000}}"
DATA_ARGS=()
# Optional: copy the dataset to node-local disk first when the shared file
# system is the bottleneck (the smoke test prints training samples/s).
#   L4_STAGE_DIR=/path/on/node/local/disk bash cluster/submit.sh l4
if [[ -n "${L4_STAGE_DIR:-}" ]]; then
  SRC="$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5"
  DST="$L4_STAGE_DIR/cube_single_expert.h5"
  NEED=$(stat -c %s "$SRC"); FREE=$(df --output=avail -B1 "$L4_STAGE_DIR" | tail -1)
  if (( FREE > NEED + 10000000000 )); then
    [[ -f "$DST" && $(stat -c %s "$DST") -eq $NEED ]] || cp "$SRC" "$DST"
    DATA_ARGS=(--data-file "$DST")
    echo "[l4] training from local copy $DST"
  else
    echo "[l4] not enough space in $L4_STAGE_DIR for a local copy; reading from $SRC"
  fi
fi
LEFT=$(python scripts/cube/slurm_time_left.py)   # after staging, which can take a while
python train_cube.py --variant "$V" --max-steps "$STEPS" --time-limit-min $(( LEFT > 30 ? LEFT - 12 : 18 )) ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
status=$?
if [[ $status -eq 3 ]]; then
  echo "[l4] $V not finished -> resubmitting"
  bash cluster/submit.sh l4-continue "$V" "$STEPS"
elif [[ $status -eq 0 ]]; then
  if [[ "${L4_SKIP_EVAL:-0}" != "1" ]]; then
    echo "[l4] $V finished -> submitting its L5 evaluation"
    bash cluster/submit.sh l5 "$V" || echo "[l4] could not submit the L5 evaluation now; later run: bash cluster/submit.sh l5 $V"
  fi
else
  echo "[l4] training failed with exit code $status"
  exit "$status"
fi
