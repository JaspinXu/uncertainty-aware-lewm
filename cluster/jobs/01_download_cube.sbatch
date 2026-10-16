#!/bin/bash
#SBATCH --job-name=lewm_download
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
# Download the Cube dataset (46 GB .tar.zst -> ~50 GB .h5) and the pretrained
# LeWM-Cube checkpoint into $STABLEWM_HOME. Resumable: if the job hits its time
# limit, just submit it again.
#     sbatch cluster/jobs/01_download_cube.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
df -h "$STABLEWM_HOME" | tail -1
python scripts/cube/download_assets.py
du -sh "$STABLEWM_HOME"/datasets "$STABLEWM_HOME"/checkpoints 2>/dev/null || true
