#!/bin/bash
#SBATCH --job-name=lewm_l1_analyze
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
# L1 analysis (CPU only): results/cube/l1/signal_report.md + figures.
#     sbatch cluster/jobs/11_l1_analyze.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh
job_banner
python scripts/cube/analyze_signal.py
python scripts/cube/summarize_runs.py --methods baseline --name l1_baseline_panel
