#!/bin/bash
#SBATCH --job-name=lewm_l2_cal
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
# Calibrate the sigma^2 gate for Cube (planning-time sigma^2_0 of MC-5 on
# 16 held-out episodes; start states drawn with seed+1000, disjoint from all panels).
#     sbatch cluster/jobs/20_l2_calibrate.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
require_gpu || exit 1
python scripts/calibrate_gated_threshold.py --config-name cube method=mc5 seed=910 \
  +num_eval=16 +output_filename=gated_cube.json
