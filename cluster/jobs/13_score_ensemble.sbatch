#!/bin/bash
#SBATCH --job-name=lewm_ens_score
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Ensemble sigma^2 for every recorded + replayed candidate (post hoc).
#     sbatch cluster/jobs/13_score_ensemble.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
python scripts/cube/score_ensemble.py results/cube/records/*.pt
