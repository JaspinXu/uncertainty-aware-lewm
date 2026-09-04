#!/usr/bin/env bash
# Run exactly one reproducible PushT evaluation and preserve every artifact.
#
# Usage:
#   bash scripts/run_fair_ua_single.sh RUN_ID METHOD SEED [hydra overrides...]
#
# Methods:
#   baseline, mc_only_n3, mc_only_n5, adaptive_n3, adaptive_n5, gated_n5
#
# Environment:
#   POLICY=pusht/lewm
#   EXPERIMENT_GROUP=fair_ua_final
#   DATA_SPLIT=validation
#   CEM_SAMPLES=300 CEM_STEPS=30 CEM_TOPK=30
#   REC_HORIZON=5 NUM_EVAL=50
#   ADAPTIVE_OMEGA=0.3
#   GATED_THRESHOLD_JSON=/path/to/calibration.json  (for gated_n5)
#   FORCE=1                                        (archive and rerun)

set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 RUN_ID METHOD SEED [hydra overrides...]" >&2
  exit 2
fi

RUN_ID="$1"
METHOD="$2"
SEED="$3"
shift 3
EXTRA_OVERRIDES=("$@")

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
POLICY="${POLICY:-pusht/lewm}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-fair_ua_final}"
DATA_SPLIT="${DATA_SPLIT:-validation}"
NUM_EVAL="${NUM_EVAL:-50}"
CEM_SAMPLES="${CEM_SAMPLES:-300}"
CEM_STEPS="${CEM_STEPS:-30}"
CEM_TOPK="${CEM_TOPK:-30}"
REC_HORIZON="${REC_HORIZON:-5}"
ADAPTIVE_OMEGA="${ADAPTIVE_OMEGA:-0.3}"
FORCE="${FORCE:-0}"

export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT}"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

OUT_ROOT="$STABLEWM_HOME/pusht/$EXPERIMENT_GROUP"
RUN_DIR="$OUT_ROOT/runs/$RUN_ID"
RESULT_REL="$EXPERIMENT_GROUP/results/$RUN_ID.txt"
RESULT_PATH="$STABLEWM_HOME/pusht/$RESULT_REL"
LOG_PATH="$OUT_ROOT/logs/$RUN_ID.log"

mkdir -p "$OUT_ROOT/results" "$OUT_ROOT/logs" "$OUT_ROOT/runs"

if [[ -f "$RESULT_PATH" ]] && grep -q "'success_rate':" "$RESULT_PATH"; then
  if [[ "$FORCE" != "1" ]]; then
    echo "[skip] complete result already exists: $RESULT_PATH"
    exit 0
  fi
fi

if [[ -e "$RUN_DIR" || -e "$RESULT_PATH" || -e "$LOG_PATH" ]]; then
  if [[ "$FORCE" != "1" ]]; then
    echo "Existing incomplete artifacts for $RUN_ID; inspect them or rerun with FORCE=1" >&2
    exit 3
  fi
  ARCHIVE="$OUT_ROOT/archive/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$ARCHIVE"
  [[ -e "$RUN_DIR" ]] && mv "$RUN_DIR" "$ARCHIVE/run"
  [[ -e "$RESULT_PATH" ]] && mv "$RESULT_PATH" "$ARCHIVE/result.txt"
  [[ -e "$LOG_PATH" ]] && mv "$LOG_PATH" "$ARCHIVE/stdout.log"
fi
mkdir -p "$RUN_DIR"

COMMON_OVERRIDES=(
  "policy=$POLICY"
  "seed=$SEED"
  "eval.num_eval=$NUM_EVAL"
  "world.num_envs=$NUM_EVAL"
  "solver.num_samples=$CEM_SAMPLES"
  "solver.n_steps=$CEM_STEPS"
  "solver.topk=$CEM_TOPK"
  "plan_config.receding_horizon=$REC_HORIZON"
  "output.filename=$RESULT_REL"
  "hydra.run.dir=."
  "hydra.output_subdir=null"
)

case "$METHOD" in
  baseline)
    CMD=("$PYTHON" eval.py --config-name=pusht)
    METHOD_OVERRIDES=()
    ;;
  mc_only_n3)
    CMD=("$PYTHON" ua_eval.py --config-name=ua_pusht)
    METHOD_OVERRIDES=(
      "ua.estimator=mcdropout" "ua.n_samples=3"
      "ua.truncation.mode=none" "ua.truncation.penalty_weight=0"
    )
    ;;
  mc_only_n5)
    CMD=("$PYTHON" ua_eval.py --config-name=ua_pusht)
    METHOD_OVERRIDES=(
      "ua.estimator=mcdropout" "ua.n_samples=5"
      "ua.truncation.mode=none" "ua.truncation.penalty_weight=0"
    )
    ;;
  adaptive_n3)
    CMD=("$PYTHON" ua_eval.py --config-name=ua_pusht)
    METHOD_OVERRIDES=(
      "ua.estimator=mcdropout" "ua.n_samples=3"
      "ua.truncation.mode=adaptive_length"
      "ua.truncation.omega=$ADAPTIVE_OMEGA"
      "ua.truncation.penalty_weight=0"
    )
    ;;
  adaptive_n5)
    CMD=("$PYTHON" ua_eval.py --config-name=ua_pusht)
    METHOD_OVERRIDES=(
      "ua.estimator=mcdropout" "ua.n_samples=5"
      "ua.truncation.mode=adaptive_length"
      "ua.truncation.omega=$ADAPTIVE_OMEGA"
      "ua.truncation.penalty_weight=0"
    )
    ;;
  gated_n5)
    CMD=("$PYTHON" ua_eval.py --config-name=ua_pusht)
    if [[ -n "${GATED_THRESHOLD_JSON:-}" ]]; then
      if [[ ! -f "$GATED_THRESHOLD_JSON" ]]; then
        echo "GATED_THRESHOLD_JSON not found: $GATED_THRESHOLD_JSON" >&2
        exit 4
      fi
      GATED_THRESHOLD="$("$PYTHON" -c \
        "import json; print(json.load(open('$GATED_THRESHOLD_JSON'))['recommended_threshold'])")"
    elif [[ -z "${GATED_THRESHOLD:-}" ]]; then
      echo "Set GATED_THRESHOLD_JSON or GATED_THRESHOLD for gated_n5" >&2
      exit 4
    fi
    METHOD_OVERRIDES=(
      "ua.estimator=mcdropout" "ua.n_samples=5"
      "ua.truncation.mode=gated"
      "ua.truncation.omega=$ADAPTIVE_OMEGA"
      "ua.truncation.threshold=$GATED_THRESHOLD"
      "ua.truncation.penalty_weight=0"
    )
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

FULL_CMD=("${CMD[@]}" "${COMMON_OVERRIDES[@]}" "${METHOD_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")
printf '%q ' "${FULL_CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
printf '%s\n' "${EXTRA_OVERRIDES[@]}" > "$RUN_DIR/extra_overrides.txt"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

export RUN_ID METHOD SEED POLICY EXPERIMENT_GROUP DATA_SPLIT NUM_EVAL
export CEM_SAMPLES CEM_STEPS CEM_TOPK REC_HORIZON ADAPTIVE_OMEGA
export START_UTC GIT_COMMIT GIT_DIRTY RESULT_PATH LOG_PATH
"$PYTHON" - <<'PY' > "$RUN_DIR/metadata.json"
import json
import os
import platform
import torch

keys = [
    "RUN_ID", "METHOD", "SEED", "POLICY", "EXPERIMENT_GROUP", "DATA_SPLIT",
    "NUM_EVAL",
    "CEM_SAMPLES", "CEM_STEPS", "CEM_TOPK", "REC_HORIZON",
    "ADAPTIVE_OMEGA", "START_UTC", "GIT_COMMIT", "GIT_DIRTY",
    "RESULT_PATH", "LOG_PATH",
]
payload = {key.lower(): os.environ.get(key) for key in keys}
payload.update(
    python=platform.python_version(),
    platform=platform.platform(),
    torch=torch.__version__,
    mps_available=bool(torch.backends.mps.is_available()),
    cuda_available=bool(torch.cuda.is_available()),
)
print(json.dumps(payload, indent=2))
PY

echo "[run] $RUN_ID method=$METHOD seed=$SEED"
echo "[save] result=$RESULT_PATH"
echo "[save] log=$LOG_PATH"
set +e
"${FULL_CMD[@]}" 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
DURATION="$((END_EPOCH - START_EPOCH))"
export STATUS END_UTC DURATION
"$PYTHON" - <<'PY' > "$RUN_DIR/status.json"
import json
import os

print(json.dumps({
    "exit_code": int(os.environ["STATUS"]),
    "started_at": os.environ["START_UTC"],
    "ended_at": os.environ["END_UTC"],
    "duration_seconds": int(os.environ["DURATION"]),
    "result_path": os.environ["RESULT_PATH"],
    "log_path": os.environ["LOG_PATH"],
}, indent=2))
PY

if [[ "$STATUS" -ne 0 ]]; then
  echo "[failed] $RUN_ID exit=$STATUS; artifacts preserved in $RUN_DIR" >&2
  exit "$STATUS"
fi
if ! grep -q "'success_rate':" "$RESULT_PATH"; then
  echo "[failed] process exited 0 but result has no success_rate: $RESULT_PATH" >&2
  exit 5
fi
echo "[complete] $RUN_ID (${DURATION}s)"
