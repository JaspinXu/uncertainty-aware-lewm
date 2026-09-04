#!/usr/bin/env bash
# Run one durable Soft Penalty v2 evaluation.
# Usage: bash scripts/run_risk_v2_single.sh RUN_ID MODE SEED ALPHA [overrides...]

set -uo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 RUN_ID MODE SEED ALPHA [hydra overrides...]" >&2
  exit 2
fi

RUN_ID="$1"
RISK_MODE="$2"
SEED="$3"
ALPHA="$4"
shift 4
EXTRA_OVERRIDES=("$@")

[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
case "$RISK_MODE" in
  baseline|mc_terminal|soft_v1_control|goal_cvar_eot|terminal_sigma_eot) ;;
  *) echo "Unknown MODE=$RISK_MODE" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }
NUMBER_RE='^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$'
[[ "$ALPHA" =~ $NUMBER_RE ]] || { echo "ALPHA must be numeric" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
POLICY="${POLICY:-pusht/lewm}"
DATA_SPLIT="${DATA_SPLIT:-validation}"
NUM_EVAL="${NUM_EVAL:-50}"
CEM_SAMPLES="${CEM_SAMPLES:-300}"
CEM_STEPS="${CEM_STEPS:-30}"
CEM_TOPK="${CEM_TOPK:-30}"
REC_HORIZON="${REC_HORIZON:-5}"
TRUNCATION_MODE="${TRUNCATION_MODE:-none}"
ADAPTIVE_OMEGA="${ADAPTIVE_OMEGA:-0.3}"
FORCE="${FORCE:-0}"

case "$TRUNCATION_MODE" in
  none|fixed|adaptive_length|adaptive_perstep|gated) ;;
  *) echo "Unknown TRUNCATION_MODE=$TRUNCATION_MODE" >&2; exit 2 ;;
esac
NUMBER_RE_OMEGA='^-?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$'
[[ "$ADAPTIVE_OMEGA" =~ $NUMBER_RE_OMEGA ]] || {
  echo "ADAPTIVE_OMEGA must be numeric" >&2
  exit 2
}

export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT}"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Keep every v2 experiment artifact under this one new output tree.
OUT_ROOT="$STABLEWM_HOME/pusht/risk_aware_v2"
RUN_DIR="$OUT_ROOT/runs/$RUN_ID"
RESULT_PATH="$OUT_ROOT/results/$RUN_ID.txt"
LOG_PATH="$OUT_ROOT/logs/$RUN_ID.log"
mkdir -p "$OUT_ROOT/results" "$OUT_ROOT/logs" "$OUT_ROOT/runs"

is_complete() {
  [[ -f "$RESULT_PATH" && -f "$RUN_DIR/status.json" ]] || return 1
  "$PYTHON" - "$RESULT_PATH" "$RUN_DIR/status.json" <<'PY'
import json
import re
import sys

result = open(sys.argv[1], errors="ignore").read()
status = json.load(open(sys.argv[2]))
ok = int(status.get("exit_code", 1)) == 0
ok = ok and re.search(r"['\"]success_rate['\"]\s*:\s*[0-9.eE+-]+", result) is not None
raise SystemExit(0 if ok else 1)
PY
}

if is_complete; then
  echo "[skip] complete run: $RUN_ID"
  exit 0
fi

if [[ -e "$RUN_DIR" || -e "$RESULT_PATH" || -e "$LOG_PATH" ]]; then
  if [[ "$FORCE" != "1" ]]; then
    echo "Incomplete artifacts exist for $RUN_ID; inspect them or use FORCE=1" >&2
    exit 3
  fi
  ARCHIVE="$OUT_ROOT/archive/${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  mkdir -p "$ARCHIVE"
  [[ -e "$RUN_DIR" ]] && mv "$RUN_DIR" "$ARCHIVE/run"
  [[ -e "$RESULT_PATH" ]] && mv "$RESULT_PATH" "$ARCHIVE/result.txt"
  [[ -e "$LOG_PATH" ]] && mv "$LOG_PATH" "$ARCHIVE/stdout.log"
fi
mkdir -p "$RUN_DIR"

# Both evaluators derive their video directory from STABLEWM_HOME and policy.
# Give each run an isolated home, while linking its read-only checkpoint cache.
RUNTIME_HOME="$RUN_DIR/runtime"
mkdir -p "$RUNTIME_HOME/pusht"
if [[ ! -e "$RUNTIME_HOME/checkpoints" ]]; then
  ln -s "$STABLEWM_HOME/checkpoints" "$RUNTIME_HOME/checkpoints"
fi
[[ -d "$RUNTIME_HOME/checkpoints" ]] || {
  echo "Checkpoint cache not found: $STABLEWM_HOME/checkpoints" >&2
  exit 4
}

COMMON_OVERRIDES=(
  "policy=$POLICY"
  "cache_dir=$STABLEWM_HOME"
  "seed=$SEED"
  "eval.num_eval=$NUM_EVAL"
  "world.num_envs=$NUM_EVAL"
  "solver.num_samples=$CEM_SAMPLES"
  "solver.n_steps=$CEM_STEPS"
  "solver.topk=$CEM_TOPK"
  "plan_config.receding_horizon=$REC_HORIZON"
  "output.filename=../../../../results/$RUN_ID.txt"
  "hydra.run.dir=."
  "hydra.output_subdir=null"
)

if [[ "$RISK_MODE" == "baseline" ]]; then
  [[ -f "$ROOT/eval.py" && -f "$ROOT/config/eval/pusht.yaml" ]] || {
    echo "Missing eval.py or config/eval/pusht.yaml" >&2
    exit 4
  }
  FULL_CMD=(
    env "STABLEWM_HOME=$RUNTIME_HOME"
    "$PYTHON" eval.py --config-name=pusht
    "${COMMON_OVERRIDES[@]}"
    "${EXTRA_OVERRIDES[@]}"
  )
else
  [[ -f "$ROOT/risk_eval_v2.py" && -f "$ROOT/config/eval/risk_v2_pusht.yaml" ]] || {
    echo "Missing assumed risk_eval_v2.py or config/eval/risk_v2_pusht.yaml" >&2
    exit 4
  }
  FULL_CMD=(
    env "STABLEWM_HOME=$RUNTIME_HOME"
    "$PYTHON" risk_eval_v2.py --config-name=risk_v2_pusht
    "${COMMON_OVERRIDES[@]}"
    "risk.objective.mode=$RISK_MODE"
    "risk.objective.alpha=$ALPHA"
    "risk.truncation.mode=$TRUNCATION_MODE"
    "risk.truncation.omega=$ADAPTIVE_OMEGA"
    "${EXTRA_OVERRIDES[@]}"
  )
fi

printf '%q ' "${FULL_CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
printf '%s\n' "${EXTRA_OVERRIDES[@]}" > "$RUN_DIR/extra_overrides.txt"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
export RUN_ID RISK_MODE SEED ALPHA POLICY DATA_SPLIT NUM_EVAL
export CEM_SAMPLES CEM_STEPS CEM_TOPK REC_HORIZON TRUNCATION_MODE ADAPTIVE_OMEGA
export START_UTC GIT_COMMIT GIT_DIRTY RESULT_PATH LOG_PATH
"$PYTHON" - <<'PY' > "$RUN_DIR/metadata.json"
import json
import os
import platform

try:
    import torch
except ImportError:
    torch = None

keys = [
    "RUN_ID", "RISK_MODE", "SEED", "ALPHA", "POLICY", "DATA_SPLIT",
    "NUM_EVAL", "CEM_SAMPLES", "CEM_STEPS", "CEM_TOPK", "REC_HORIZON",
    "TRUNCATION_MODE", "ADAPTIVE_OMEGA",
    "START_UTC", "GIT_COMMIT", "GIT_DIRTY", "RESULT_PATH", "LOG_PATH",
]
payload = {key.lower(): os.environ.get(key) for key in keys}
payload.update(
    python=platform.python_version(),
    platform=platform.platform(),
    torch=None if torch is None else torch.__version__,
    mps_available=False if torch is None else bool(torch.backends.mps.is_available()),
    cuda_available=False if torch is None else bool(torch.cuda.is_available()),
)
print(json.dumps(payload, indent=2))
PY

echo "[run] $RUN_ID mode=$RISK_MODE seed=$SEED alpha=$ALPHA trunc=$TRUNCATION_MODE omega=$ADAPTIVE_OMEGA"
set +e
"${FULL_CMD[@]}" 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e
END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DURATION="$(($(date +%s) - START_EPOCH))"
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
  echo "[failed] $RUN_ID exit=$STATUS" >&2
  exit "$STATUS"
fi
if ! is_complete; then
  echo "[failed] result missing a parseable success_rate: $RESULT_PATH" >&2
  exit 5
fi
echo "[complete] $RUN_ID (${DURATION}s)"
