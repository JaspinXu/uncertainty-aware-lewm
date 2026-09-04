#!/usr/bin/env bash
# Locked 9-seed PushT panel behind the final comparison (slides Fig. 4-5):
#   methods baseline | mc5 | gated | goal_cvar
#   seeds   42 43 44 45 1011 2022 3033 4044 5055
#   protocol pusht/lewm | n=50 | CEM 300/30/30 | RH=5   (config/eval/pusht.yaml)
# Resumable: runs whose results/<method>_s<seed>.txt is complete are skipped.
# Ends by printing the table (scripts/summarize_locked_9seeds.py).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/opt/anaconda3/bin/activate}"
if [[ -z "${PYTHON:-}" && -f "$CONDA_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_ACTIVATE" lewm_env
fi
export PYTHON="${PYTHON:-python}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p results/logs
exec > >(tee -a results/logs/panel.log) 2>&1
echo "[panel-start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

FAIL=0
for SEED in 42 43 44 45 1011 2022 3033 4044 5055; do
  for METHOD in baseline mc5 gated goal_cvar; do
    bash scripts/run_single.sh "$METHOD" "$SEED" || FAIL=1
  done
done
echo "[panel-end] $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=$FAIL"

if [[ "$FAIL" -eq 0 ]]; then
  "$PYTHON" scripts/summarize_locked_9seeds.py
fi
exit "$FAIL"
