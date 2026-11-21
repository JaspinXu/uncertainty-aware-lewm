#!/bin/bash
# Slurm wrapper of scripts/run_cube_compare.sh: one GPU, the stages in $STAGES (default all), resumable.
# Submit from the repo root with the Python environment activated (the job inherits it). Usually
# through scripts/submit_cube_compare.sh, which splits the comparison into jobs of at most 3 hours:
#
#   mkdir -p results/cube/compare/logs
#   STAGES="eval" METHODS="lewm" sbatch --gpus=h100-47:1 scripts/run_cube_compare.sbatch
#
# Defaults below follow the SoC cluster (gpu partition: 3-hour limit); options given to sbatch override them.
#SBATCH --job-name=cube-compare
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/cube/compare/logs/slurm-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
export PYTHON="${PYTHON:-$(command -v python)}"
echo "[slurm] job ${SLURM_JOB_ID:-?} ${SLURM_JOB_NAME:-} on $(hostname); stages ${STAGES:-all}; methods ${METHODS:-default}"
echo "[slurm] python $PYTHON; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
bash scripts/run_cube_compare.sh
