#!/usr/bin/env bash
# Common environment for every cluster job.  Usage (from the repo root):
#     source cluster/env.sh
# Any variable can be overridden by exporting it beforehand (e.g. in ~/.bashrc).

# repository root = parent of this file's directory
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Large artifacts: dataset (~50 GB after extraction), checkpoints, HF cache.
# Must be visible from every node (home is; use scratch only if it is shared).
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/lewm_data}"
mkdir -p "$STABLEWM_HOME"

# Python environment created by cluster/setup_env.sh (Miniforge + conda env)
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-lewm}"

# headless MuJoCo rendering on GPU nodes (fallback: MUJOCO_GL=osmesa, see cluster/README.md)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

export HF_HOME="${HF_HOME:-$STABLEWM_HOME/hf_cache}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/stable-worldmodel:$PROJECT_ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  # conda's shell hooks reference unset variables: relax `set -u` while they run
  _lewm_flags="$-"
  set +u
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
    echo "[env] conda env '$CONDA_ENV_NAME' not found -- run cluster/setup_env.sh first" >&2
  fi
  [[ "$_lewm_flags" == *u* ]] && set -u
  unset _lewm_flags
else
  echo "[env] no conda at $CONDA_ROOT -- run cluster/setup_env.sh first" >&2
fi

cd "$PROJECT_ROOT" || return 1
mkdir -p "$PROJECT_ROOT/logs"

# small helper for job logs
job_banner() {
  echo "=== $(date '+%F %T') | job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+[$SLURM_ARRAY_TASK_ID]} on $(hostname) ==="
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || true
}
