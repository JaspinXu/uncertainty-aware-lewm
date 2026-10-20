#!/usr/bin/env bash
# Multi-seed Cube comparison (config/eval/compare.yaml), locally or on a GPU server:
#   train      scripts/cube_train.py                          sigma prior, time-conditioned prior, k̂
#   calibrate  scripts/cube_compare.py +planner=calibrate     gate thresholds of gated / gated_tc
#   eval       scripts/cube_compare.py +planner=<method>      every method in METHODS, every seed in SEEDS
#   summary    scripts/cube_compare_summary.py                -> results/cube/compare/k<OFFSET>/summary.md
#
#   bash scripts/run_cube_compare.sh
#   STAGES="eval summary" METHODS="lewm gated_tc" SEEDS="0 1 2" bash scripts/run_cube_compare.sh
#   OFFSET=50 bash scripts/run_cube_compare.sh      # long-goal protocol (budget 100 steps)
#
# Environment (defaults in brackets):
#   STAGES     [train calibrate eval summary]
#   METHODS    [lewm mc5 gated gated_tc bc_only prior_cem_s1 prior_cem_sconst prior_cem_sp bc_select_khat]
#              goal_cvar is available too; see scripts/cube_compare.py for what each one is
#   SEEDS      [0 1 2 3 4 5 6 7 8 9]   evaluation seeds, 50 non-trivial episodes each
#   CAL_SEEDS  [900 901]               calibration seeds of the gates, disjoint from SEEDS
#   OFFSET     [25]                    goal offset in env steps (LeWM protocol); budget = 2 x OFFSET
#   JOBS       [1]   methods evaluated in parallel. Each process holds 50 simulated envs plus the world
#              model (about 20 GB of host memory on a 16 GB Mac, mostly the envs' renderers); raise it
#              only where memory allows
#   DATASET    [cube_single_expert]    eval.dataset_name: $STABLEWM_HOME/datasets/<DATASET>.h5
#   PYTHON [python]  MUJOCO_GL [egl on Linux, glfw on macOS]  STABLEWM_HOME [repo root]
#   EXTRA      extra hydra overrides for every python call, e.g. EXTRA="eval.num_eval=10"
# Resumable: trained components and finished seeds are skipped. Logs: results/cube/compare/logs/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-train calibrate eval summary}"
METHODS="${METHODS:-lewm mc5 gated gated_tc bc_only prior_cem_s1 prior_cem_sconst prior_cem_sp bc_select_khat}"
export SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CAL_SEEDS="${CAL_SEEDS:-900 901}"
export OFFSET="${OFFSET:-25}"
JOBS="${JOBS:-1}"
export DATASET="${DATASET:-cube_single_expert}"
export EXTRA="${EXTRA:-}"
if [[ "$(uname)" == "Darwin" ]]; then export MUJOCO_GL="${MUJOCO_GL:-glfw}"; else export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
export STABLEWM_HOME="${STABLEWM_HOME:-$ROOT}"
export PYTHONPATH="$ROOT:$ROOT/stable-worldmodel:$ROOT/stable-pretraining${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LOGS="$ROOT/results/cube/compare/logs"
mkdir -p "$LOGS"

has() { [[ " $STAGES " == *" $1 "* ]]; }
as_list() { echo "[$(echo $1 | tr ' ' ',')]"; }   # "0 1 2" -> [0,1,2]
export -f as_list

# common hydra overrides; hydra's own log file goes under logs/hydra/<name> instead of the repo root
common() {
  # shellcheck disable=SC2086
  echo compare.offset="$OFFSET" eval.dataset_name="$DATASET" hydra.run.dir="$LOGS/hydra/$1" $EXTRA
}
export -f common

run_method() {
  local m="$1" log="$LOGS/k${OFFSET}_$1.log"
  echo "[eval] $m -> $log"
  # shellcheck disable=SC2046
  if "$PYTHON" scripts/cube_compare.py +planner="$m" "+seeds=$(as_list "$SEEDS")" $(common "$m") >"$log" 2>&1; then
    grep "\[compare" "$log" | tail -n 3
  else
    echo "[failed] $m (see $log)" >&2
    return 1
  fi
}
export -f run_method

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) stages: $STAGES | offset $OFFSET | seeds: $SEEDS"
FAIL=0
if has train; then
  echo "[train] -> $LOGS/train.log"
  # shellcheck disable=SC2046
  "$PYTHON" scripts/cube_train.py $(common train) >"$LOGS/train.log" 2>&1 || { echo "[failed] train" >&2; exit 1; }
  grep "\[train\]" "$LOGS/train.log" | tail -n 8
fi
if has calibrate && [[ " $METHODS " == *" gated"* ]]; then
  echo "[calibrate] -> $LOGS/k${OFFSET}_calibrate.log"
  # shellcheck disable=SC2046
  "$PYTHON" scripts/cube_compare.py +planner=calibrate "+seeds=$(as_list "$CAL_SEEDS")" $(common calibrate) \
    >"$LOGS/k${OFFSET}_calibrate.log" 2>&1 || { echo "[failed] calibrate" >&2; exit 1; }
  grep "\[compare" "$LOGS/k${OFFSET}_calibrate.log" | tail -n 3
fi
if has eval; then
  if [[ "$JOBS" -le 1 ]]; then
    for m in $METHODS; do run_method "$m" || FAIL=1; done
  else
    printf '%s\n' $METHODS | xargs -P "$JOBS" -I{} bash -c 'run_method "$1"' _ {} || FAIL=1
  fi
fi
if has summary; then
  # shellcheck disable=SC2046
  "$PYTHON" scripts/cube_compare_summary.py $(common summary) | tail -n 4 || FAIL=1
fi
echo "[end] $(date -u +%Y-%m-%dT%H:%M:%SZ) fail=$FAIL"
exit "$FAIL"
