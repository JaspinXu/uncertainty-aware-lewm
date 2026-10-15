#!/bin/bash
#SBATCH --job-name=lewm_l3_train
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
# L3: train the risk heads on oracle-labelled candidates (train seeds), select
# head/mode/beta on the calibration seeds, offline report on the test seeds.
#     sbatch cluster/jobs/30_l3_train_risk.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
require_gpu || exit 1
python scripts/cube/train_risk_head.py
cat results/cube/l3/risk_report.md
