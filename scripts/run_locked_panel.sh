#!/usr/bin/env bash
# Locked 9-seed test panel behind report Fig. 4 / Table 1 (slides Fig. 4-5).
#
# Protocol: pusht/lewm | n=50 | CEM 300/30/30 | RH=5 | split=test
# Methods : baseline, mc_only_n5, gated_n5 (via run_fair_ua_single.sh)
#           goal_cvar_eot alpha=0.1  (via run_risk_v2_single.sh)
# Seeds   : 42-45           -> pusht/locked_s42_45/results/<method>_s<seed>.txt
#           1011-5055       -> baseline/mc/gated: pusht/fair_ua_final/results/test_<method>_s<seed>.txt
#                              (skipped if complete)
#                              goal_cvar: pusht/locked_s42_45/results/goal_a0p1_s<seed>.txt
#
# Every single run is skipped when a complete result exists, so the script can be
# re-launched to resume. Finishes by regenerating combined_9seeds.md and the figure.
#
# Usage: bash scripts/run_locked_panel.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/opt/anaconda3/bin/activate}"
if [[ -z "${PYTHON:-}" && -f "$CONDA_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_ACTIVATE" lewm_env
fi
export PYTHON="${PYTHON:-python}"
export STABLEWM_HOME="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

export POLICY=pusht/lewm
export NUM_EVAL=50 CEM_SAMPLES=300 CEM_STEPS=30 CEM_TOPK=30 REC_HORIZON=5 DATA_SPLIT=test
export GATED_THRESHOLD_JSON="$ROOT/pusht/calibration/gated_lewm_strong_rh5.json"
export ADAPTIVE_OMEGA=1.0

LOCKED="$ROOT/pusht/locked_s42_45"
mkdir -p "$LOCKED/results" "$LOCKED/logs" "$LOCKED/runs"
exec > >(tee -a "$LOCKED/logs/orchestrator_panel.log") 2>&1
echo "[panel-start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"

[[ -f "$GATED_THRESHOLD_JSON" ]] || {
  echo "Missing $GATED_THRESHOLD_JSON (see README: Gated-threshold calibration)" >&2
  exit 4
}

FAIL=0

# Goal-CVaR alpha=0.1: run in the risk_v2 tree, then mirror the result into locked_s42_45.
run_goal() {
  local seed="$1" run_id="goal_a0p1_s$1"
  if [[ -f "$LOCKED/results/${run_id}.txt" ]] && grep -q "success_rate" "$LOCKED/results/${run_id}.txt"; then
    echo "[skip] $run_id"; return 0
  fi
  bash scripts/run_risk_v2_single.sh "$run_id" goal_cvar_eot "$seed" 0.1 || FAIL=1
  if [[ -f "$ROOT/pusht/risk_aware_v2/results/${run_id}.txt" ]]; then
    cp "$ROOT/pusht/risk_aware_v2/results/${run_id}.txt" "$LOCKED/results/${run_id}.txt"
  fi
}

echo "=== seeds 42-45 (group locked_s42_45) ==="
export EXPERIMENT_GROUP=locked_s42_45
for SEED in 42 43 44 45; do
  bash scripts/run_fair_ua_single.sh "baseline_s${SEED}" baseline   "$SEED" || FAIL=1
  bash scripts/run_fair_ua_single.sh "mc_n5_s${SEED}"    mc_only_n5 "$SEED" || FAIL=1
  bash scripts/run_fair_ua_single.sh "gated_n5_s${SEED}" gated_n5   "$SEED" || FAIL=1
  run_goal "$SEED"
done

echo "=== seeds 1011-5055 (group fair_ua_final, ids test_*) ==="
export EXPERIMENT_GROUP=fair_ua_final
for SEED in 1011 2022 3033 4044 5055; do
  bash scripts/run_fair_ua_single.sh "test_baseline_s${SEED}" baseline   "$SEED" || FAIL=1
  bash scripts/run_fair_ua_single.sh "test_mc_n5_s${SEED}"    mc_only_n5 "$SEED" || FAIL=1
  bash scripts/run_fair_ua_single.sh "test_gated_n5_s${SEED}" gated_n5   "$SEED" || FAIL=1
  run_goal "$SEED"
done

echo "[panel-end] $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  "$PYTHON" scripts/summarize_locked_9seeds.py
fi
exit "$FAIL"
