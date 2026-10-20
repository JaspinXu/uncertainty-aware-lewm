#!/usr/bin/env bash
# Submit the Cube comparison as Slurm jobs that each fit the 3-hour gpu partition:
#   1. prep: train the missing learned components + calibrate the gates (scripts/run_cube_compare.sh)
#   2. one evaluation job per method, started once prep has succeeded
# Every job resumes where an earlier attempt stopped, so a job that hits the time limit is simply
# submitted again (rerun this script, or e.g. METHODS="mc5" bash scripts/submit_cube_compare.sh).
# When all jobs are done:  STAGES=summary bash scripts/run_cube_compare.sh
#
#   conda activate <env>; cd ~/ua-lewm; bash scripts/submit_cube_compare.sh
#   GPU=a100-80 METHODS="lewm gated gated_tc" bash scripts/submit_cube_compare.sh
#
# Environment: PARTITION [gpu]  GPU [h100-47] (any --gpus type, or "any")  TIME [03:00:00]
#   METHODS [lewm mc5 gated gated_tc bc_only prior_cem_s1 prior_cem_sconst prior_cem_sp bc_select_khat]
#   plus SEEDS / CAL_SEEDS / OFFSET / DATASET / EXTRA, passed through to scripts/run_cube_compare.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PARTITION="${PARTITION:-gpu}"
GPU="${GPU:-h100-47}"
TIME="${TIME:-03:00:00}"
ALL_METHODS="${METHODS:-lewm mc5 gated gated_tc bc_only prior_cem_s1 prior_cem_sconst prior_cem_sp bc_select_khat}"
command -v python >/dev/null || { echo "activate the Python environment first (conda activate ...)" >&2; exit 1; }
mkdir -p results/cube/compare/logs

gpus="$GPU:1"; [[ "$GPU" == "any" ]] && gpus=1
opts=(--partition="$PARTITION" --gpus="$gpus" --time="$TIME")
prep=$(STAGES="train calibrate" METHODS="$ALL_METHODS" sbatch --parsable "${opts[@]}" --job-name=cube-prep \
       scripts/run_cube_compare.sbatch)
echo "[submit] prep (train + calibrate): job $prep"
for m in $ALL_METHODS; do
  id=$(STAGES=eval METHODS="$m" sbatch --parsable "${opts[@]}" --dependency=afterok:"$prep" --job-name="cube-$m" \
       scripts/run_cube_compare.sbatch)
  echo "[submit] $m: job $id"
done
echo "[submit] queue: squeue -u $USER --start | logs: results/cube/compare/logs/"
echo "[submit] when every job is done: STAGES=summary bash scripts/run_cube_compare.sh"
