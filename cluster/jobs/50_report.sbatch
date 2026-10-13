#!/bin/bash
#SBATCH --job-name=lewm_report
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
# L6: regenerate results/cube/RESULTS.md from everything produced so far.
#     sbatch cluster/jobs/50_report.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
python scripts/cube/make_report.py
