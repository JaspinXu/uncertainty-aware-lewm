#!/bin/bash
#SBATCH --job-name=lewm_ens
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
# Latent dynamics ensemble on the frozen pretrained encoder (second uncertainty
# signal for L1/L2): encode every dataset frame once, then train 5 members.
#     sbatch cluster/jobs/12_ensemble.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
require_gpu || exit 1
python scripts/cube/encode_dataset.py --policy cube/lewm
# 12k steps per member: 5 members take ~100 min on an h100-47 slice (encoding the
# 2M frames once takes ~11 min more), well inside the 3 h limit
python train_ensemble.py --policy cube/lewm --members "${ENS_MEMBERS:-5}" --steps "${ENS_STEPS:-12000}"
mkdir -p results/cube/ensemble
cp "$STABLEWM_HOME/checkpoints/cube/ens_lewm/report.json" results/cube/ensemble/train_report.json
