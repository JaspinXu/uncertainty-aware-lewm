#!/bin/bash
#SBATCH --job-name=lewm_summary
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
# Success-rate panel for a set of run labels on the 9 test seeds.
#     sbatch cluster/jobs/90_summarize.sbatch NAME LABEL [LABEL ...]
# e.g. sbatch cluster/jobs/90_summarize.sbatch l2_panel baseline mc5 gated goal_cvar penalty
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
source cluster/seeds.sh
NAME="$1"; shift
python scripts/cube/summarize_runs.py --name "$NAME" --methods "$@" --seeds "${TEST_SEEDS[@]}" --reference "${REFERENCE:-baseline}"
