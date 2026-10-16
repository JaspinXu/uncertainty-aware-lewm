#!/bin/bash
#SBATCH --job-name=lewm_tasks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# Runs the task lines of a file one after another (functions in cluster/runs.sh):
#     sbatch cluster/jobs/run_tasks.sbatch logs/tasks/<file>.txt
# cluster/submit.sh writes these files and splits every phase into a few such
# jobs (the SoC cluster allows 16 running / 32 queued jobs per user, counting
# each array task as a job).
#
# A run whose outputs exist is skipped. Before each run the job checks that the
# run still fits into the remaining time (1.5 x the longest run so far + 10 min);
# if not, it stops with exit code 4 so that dependent jobs do not start on
# incomplete results. Submitting the phase again reruns only what is missing.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/runs.sh
job_banner
require_gpu || exit 1

TASKS="$1"
mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$TASKS")
echo "[tasks] ${#LINES[@]} runs from $TASKS"
failed=0
longest=0
for i in "${!LINES[@]}"; do
  line="${LINES[$i]}"
  read -r -a words <<< "$line"
  if "done_${words[0]#run_}" "${words[@]:1}"; then
    echo "[tasks] already done: $line"
    continue
  fi
  left=$(python scripts/cube/slurm_time_left.py)
  need=$(( longest * 3 / 2 + 10 ))
  if (( left < need )); then
    echo "[tasks] stopping: ${left} min left, a run takes up to ${longest} min; $(( ${#LINES[@]} - i )) run(s) not started."
    echo "[tasks] submit the phase again to run the rest (finished runs are skipped)."
    exit 4
  fi
  start=$(date +%s)
  echo "[tasks] ($(( i + 1 ))/${#LINES[@]}) $line"
  if "${words[@]}"; then
    mins=$(( ($(date +%s) - start + 59) / 60 ))
    (( mins > longest )) && longest=$mins
    echo "[tasks] finished in ${mins} min: $line"
  else
    echo "[tasks] FAILED: $line"
    failed=$(( failed + 1 ))
  fi
done
if (( failed > 0 )); then
  echo "[tasks] $failed run(s) failed"
  exit 1
fi
echo "[tasks] all runs done"
