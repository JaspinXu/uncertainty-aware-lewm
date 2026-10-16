#!/bin/bash
#SBATCH --job-name=lewm_retry
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x_%j.out
# Submits a phase again later; used by cluster/submit.sh when a job (an L4
# training job submitting its L5 evaluation, or its own continuation) finds the
# per-user queue full. Retries every 30 min until the submission fits.
#     sbatch --begin=now+30minutes cluster/jobs/91_retry_submit.sbatch l5 prop
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
echo "=== $(date '+%F %T') | retry: bash cluster/submit.sh $*"
LEWM_RETRY=1 bash cluster/submit.sh "$@"
