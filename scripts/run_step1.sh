#!/usr/bin/env bash
# Step 1 on Cube: does sigma^2 separate failures?  (docs/step1_preregistration.md)
#
#   bash scripts/run_step1.sh [hydra overrides...]
#
# Stages (every one resumes from what is already in results/cube/step1/):
#   1. step1_collect.py   closed loop (8 seeds x 50 non-trivial episodes) + replays
#   2. step1_ensemble.py  latent cache + 5 predictor heads on the frozen encoder
#   3. step1_score.py     predictions, real errors, MC-dropout and ensemble sigma^2
#   4. step1_decide.py    criteria C1-C3, gate G, decision -> report.md / decision.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT}"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$(uname)" == "Darwin" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"   # egl is not available on macOS; dm_control rejects cgl
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
OUT="$ROOT/results/cube/step1"
mkdir -p "$OUT"

for stage in step1_collect step1_ensemble step1_score step1_decide; do
  echo "[run_step1] $stage"
  "$PYTHON" "scripts/$stage.py" "$@" 2>&1 | tee -a "$OUT/$stage.log"
done
