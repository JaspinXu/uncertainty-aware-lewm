#!/bin/bash
#SBATCH --job-name=lewm_setup
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
# One-time Python environment setup (CPU node is enough).
#     sbatch cluster/jobs/00_setup_env.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
bash cluster/setup_env.sh
