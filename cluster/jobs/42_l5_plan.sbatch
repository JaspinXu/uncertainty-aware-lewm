#!/bin/bash
#SBATCH --job-name=lewm_l5_plan
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out
# L5: CEM planning with a trained variant (cube/l4_<variant>) as the cost model.
#   select: proprio variants only -- w_prop in {1, 0.1, 0} x calibration seeds (array 0-8)
#   test:   9 test seeds with the selected w_prop (array 0-8)
#     sbatch --array=0-8 cluster/jobs/42_l5_plan.sbatch <variant> select|test
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/seeds.sh
job_banner
V="$1"; MODE="$2"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
POLICY="cube/l4_$V"
ARGS=(--config-name cube method=baseline policy="$POLICY")
case "$V" in prop*) ARGS+=(map_proprio_keys=true) ;; esac
W_LIST=(1.0 0.1 0.0)
if [[ "$MODE" == select ]]; then
  NS=${#CAL_SEEDS[@]}
  (( IDX < ${#W_LIST[@]} * NS )) || { echo "index out of range"; exit 0; }
  W="${W_LIST[$((IDX / NS))]}"; SEED="${CAL_SEEDS[$((IDX % NS))]}"
  RUN="l4_${V}_w${W}_s${SEED}"
  ARGS+=(+model_overrides.w_prop="$W")
else
  SEED="${TEST_SEEDS[$IDX]}"
  RUN="l4_${V}_s${SEED}"
  case "$V" in
    prop*) W=$(python scripts/cube/select_wprop.py "$V"); ARGS+=(+model_overrides.w_prop="$W") ;;
  esac
fi
if [[ -f "results/cube/runs/$RUN.json" ]]; then echo "[skip] $RUN exists"; exit 0; fi
echo "variant=$V mode=$MODE seed=$SEED run=$RUN"
python evaluate.py "${ARGS[@]}" seed="$SEED" output.run_id="$RUN"
