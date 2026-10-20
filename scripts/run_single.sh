#!/usr/bin/env bash
# One PushT evaluation run (method x seed), saved as results/<run_id>.txt.
#
#   bash scripts/run_single.sh METHOD SEED [hydra overrides...]
#   METHOD: baseline | mc5 | gated | goal_cvar   (config/eval/method/*.yaml)
#
# Optional environment:
#   RUN_ID      result name (default <METHOD>_s<SEED>)
#   NUM_EVAL=50 CEM_SAMPLES=300 CEM_STEPS=30 CEM_TOPK=30 REC_HORIZON=5 POLICY=pusht
#   FORCE=1     rerun even if a complete result exists (old files go to results/archive/)
#   PYTHON      interpreter (default: python)
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 METHOD SEED [hydra overrides...]" >&2
  exit 2
fi
METHOD="$1"; SEED="$2"; shift 2
case "$METHOD" in baseline|mc5|gated|goal_cvar) ;; *) echo "Unknown METHOD=$METHOD" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-${METHOD}_s${SEED}}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RUN_ID=$RUN_ID" >&2; exit 2; }

export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT}"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

RESULT="$ROOT/results/$RUN_ID.txt"
LOG="$ROOT/results/logs/$RUN_ID.log"
mkdir -p "$ROOT/results/logs"

is_complete() { [[ -f "$RESULT" ]] && grep -q "'success_rate':" "$RESULT"; }

if is_complete && [[ "${FORCE:-0}" != "1" ]]; then
  echo "[skip] $RUN_ID (complete result exists: $RESULT)"
  exit 0
fi
if [[ -e "$RESULT" || -e "$LOG" ]]; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "[error] incomplete artifacts exist for $RUN_ID; inspect $LOG or rerun with FORCE=1" >&2
    exit 3
  fi
  ARCHIVE="$ROOT/results/archive/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$ARCHIVE"
  [[ -e "$RESULT" ]] && mv "$RESULT" "$ARCHIVE/"
  [[ -e "$LOG" ]] && mv "$LOG" "$ARCHIVE/"
fi

CMD=(
  "$PYTHON" evaluate.py
  "method=$METHOD" "seed=$SEED" "policy=${POLICY:-pusht}"
  "eval.num_eval=${NUM_EVAL:-50}"
  "solver.num_samples=${CEM_SAMPLES:-300}" "solver.n_steps=${CEM_STEPS:-30}" "solver.topk=${CEM_TOPK:-30}"
  "plan_config.receding_horizon=${REC_HORIZON:-5}"
  "output.run_id=$RUN_ID"
  "$@"
)
echo "[run] $RUN_ID"
printf '%q ' "${CMD[@]}" | tee "$LOG"; echo | tee -a "$LOG"
START=$(date +%s)
"${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
if [[ "$STATUS" -ne 0 ]]; then
  echo "[failed] $RUN_ID exit=$STATUS (see $LOG)" >&2
  exit "$STATUS"
fi
if ! is_complete; then
  echo "[failed] $RUN_ID finished but $RESULT has no success_rate" >&2
  exit 5
fi
echo "[complete] $RUN_ID ($(( $(date +%s) - START ))s)"
