#!/bin/bash
#SBATCH --job-name=lewm_diag
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
# Diagnostic: live-rendered frames vs dataset frames of the same state, as the
# encoder sees them.
#     sbatch cluster/jobs/60_diag_domain_gap.sbatch [OUT_DIR]
# The log and domain_gap.md name the OpenGL renderer: a MIG slice (e.g. a100-40 on
# xgph10-15, h100-47) renders in software (llvmpipe), a whole GPU in hardware.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
require_gpu || exit 1
python scripts/cube/check_domain_gap.py --seeds 42 43 --out "${1:-results/cube/diag}"
