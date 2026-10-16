#!/usr/bin/env bash
# One-time setup of the Python environment on the cluster.
#
# Run it on a compute node, not on the login node, e.g.
#     sbatch cluster/jobs/00_setup_env.sbatch
# or interactively
#     srun --time=01:00:00 --cpus-per-task=4 --mem=16G --pty bash cluster/setup_env.sh
#
# What it does (idempotent, safe to re-run):
#   1. fetch the pinned stable-worldmodel / stable-pretraining submodules
#   2. install Miniforge into $CONDA_ROOT (if no conda there yet)
#   3. create conda env $CONDA_ENV_NAME (python 3.11) and pip-install the exact
#      package set from cluster/requirements-cluster.txt + the two submodules
#   4. print a short import / CUDA / MuJoCo-rendering check
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh" || true
cd "$PROJECT_ROOT"

echo "[setup] submodules"
git submodule update --init --recursive

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "[setup] installing Miniforge into $CONDA_ROOT"
  installer="$(mktemp -d)/miniforge.sh"
  curl -fsSL -o "$installer" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$installer" -b -p "$CONDA_ROOT"
  rm -f "$installer"
fi
set +u  # conda's shell hooks reference unset variables
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "[setup] creating conda env $CONDA_ENV_NAME"
  conda create -y -n "$CONDA_ENV_NAME" python=3.11
fi
conda activate "$CONDA_ENV_NAME"

echo "[setup] pip packages"
python -m pip install --upgrade pip
python -m pip install -r cluster/requirements-cluster.txt
python -m pip install --no-deps -e ./stable-worldmodel -e ./stable-pretraining

echo "[setup] checks"
python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import torch, stable_worldmodel, stable_pretraining  # noqa: F401
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
echo "[setup] done -- next: sbatch cluster/jobs/01_download_cube.sbatch"
