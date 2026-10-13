#!/bin/bash
#SBATCH --job-name=lewm_l5_repr
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
# L5: probes + open-loop prediction error of the given checkpoints.
#     sbatch cluster/jobs/41_l5_repr.sbatch cube/lewm cube/l4_lewm ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
python scripts/cube/eval_representation.py --policies "$@"
