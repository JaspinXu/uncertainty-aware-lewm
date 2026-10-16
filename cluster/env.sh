#!/usr/bin/env bash
# Common environment for every cluster job.  Usage (from the repo root):
#     source cluster/env.sh
# Any variable can be overridden by exporting it beforehand (e.g. in ~/.bashrc).

# repository root = parent of this file's directory
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# NUS SoC compute cluster (sinfo / sacctmgr, Oct 2026):
#   partitions  normal (default, CPU only, 3 h), long (CPU, 3 days),
#               gpu (3 h), gpu-long (3 days, fewer nodes)
#   GPU types   a100-40, a100-80, h100-47, h100-96, h200-141 (gpu only), nv (older cards:
#               Titan V, Titan RTX, T4); an untyped request (--gpus=1) only gets nv cards
#   per user    16 running / 32 queued jobs (array tasks count one by one),
#               8 GPUs: a100-40 4, a100-80 2, h100-47 2, h100-96 1, h200-141 1
#               MIG slices (h100-47; a100-40 on xgph10-15) render MuJoCo in software (llvmpipe)
#   storage     home: shared all-flash NFS; /mnt/scratch: shared HDD NFS; /tmp: node-local
# cluster/submit.sh puts GPU jobs on $LEWM_GPU_PARTITION, asks for a GPU type
# per job (rotating over $LEWM_GPU_TYPES) and keeps job counts low.
export LEWM_GPU_PARTITION="${LEWM_GPU_PARTITION:-gpu}"

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
# temporary files of jobs on the node's own disk: with TMPDIR on NFS, data-loader
# workers fail to clean up ("Device or resource busy: .nfs...") at every shutdown
[[ -n "${SLURM_JOB_ID:-}" ]] && export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "$(uname -m)" != x86_64 ]]; then
  echo "[env] $(hostname) is $(uname -m): the conda env is x86_64 -- submit through cluster/submit.sh (it excludes such nodes)" >&2
fi
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
  echo "=== $(date '+%F %T') | job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+[$SLURM_ARRAY_TASK_ID]} on $(hostname) (partition ${SLURM_JOB_PARTITION:-none}) ==="
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "STABLEWM_HOME=$STABLEWM_HOME"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || true
}

# GPU jobs call this first: stop at once (instead of running for hours on the
# CPU) when the job has no usable GPU. On the SoC cluster that happens when a
# job is submitted without --partition=gpu (the default partition has no GPUs)
# or lands on a GPU that the CUDA 12.8 PyTorch build does not support.
require_gpu() {
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("[env] no GPU in this job: submit GPU jobs to a GPU partition (cluster/submit.sh does)")
try:
    torch.ones(8, device="cuda").mul_(2).sum().item()  # fails on unsupported GPU architectures
except Exception as e:
    sys.exit(f"[env] GPU {torch.cuda.get_device_name()} not usable with torch {torch.__version__}: {e}")
cap = torch.cuda.get_device_capability()
print(f"[env] GPU {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
PY
}
