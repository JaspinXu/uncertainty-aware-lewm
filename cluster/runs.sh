#!/usr/bin/env bash
# One function per kind of evaluation run. cluster/jobs/run_tasks.sbatch runs
# task lines such as "run_record 42" or "run_l2 mc5 42" one after another, and
# cluster/submit.sh uses the matching done_* function to leave out finished
# runs, so resubmitting a phase only reruns what is missing.
#
#   run_record SEED [POLICY] [PREFIX]     L1: recorded baseline planning + oracle replay
#   run_l2 METHOD SEED                    L2: uncertainty-aware planner on a test seed
#   run_l3 SEED                           L3: risk-aware planner on a test seed
#   run_l5 VARIANT select W SEED          L5: w_prop selection run (calibration seed)
#   run_l5 VARIANT test SEED              L5: test run with the selected w_prop
# shellcheck disable=SC1091
source cluster/seeds.sh

# --- L1 ------------------------------------------------------------------------
done_record() { [[ -f "results/cube/oracle/${3:-baseline}_s$1.pt" ]]; }
run_record() {
  local seed="$1" policy="${2:-cube/lewm}" prefix="${3:-baseline}" run
  run="${prefix}_s${seed}"
  if [[ -f "results/cube/records/$run.pt" && -f "results/cube/runs/$run.json" ]]; then
    echo "[skip] planning already recorded for $run"
  else
    python evaluate.py --config-name cube method=baseline seed="$seed" policy="$policy" \
      record.enabled=true output.run_id="$run" || return 1
  fi
  python scripts/cube/oracle_replay.py "results/cube/records/$run.pt" \
    --out results/cube/oracle --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) --skip-existing
}

# --- L2 ------------------------------------------------------------------------
#   mc5          mean of 5 MC-dropout rollouts
#   gated        sigma^2 gate, threshold/omega calibrated for Cube (20_l2_calibrate)
#   goal_cvar    CVaR soft penalty (self-calibrating per planning call)
#   penalty      MC-5 + lambda * sigma^2_H, lambda selected in L1 (signal_summary.json)
#   ens_penalty  LeWM cost + lambda * latent-ensemble sigma^2_H, lambda selected in L1
L2_METHODS=(mc5 gated goal_cvar penalty ens_penalty)
done_l2() { [[ -f "results/cube/runs/$1_s$2.json" ]]; }
run_l2() {
  local method="$1" seed="$2" thr omega extra=()
  case "$method" in
    gated)
      read -r thr omega < <(python scripts/cube/l2_params.py gated) || return 1
      extra=(method.truncation.threshold="$thr" method.truncation.omega="$omega") ;;
    penalty) extra=(method.truncation.penalty_weight="$(python scripts/cube/l2_params.py penalty)") ;;
    ens_penalty) extra=(method.lam="$(python scripts/cube/l2_params.py ens_penalty)") ;;
  esac
  echo "method=$method seed=$seed extra=${extra[*]:-none}"
  python evaluate.py --config-name cube method="$method" seed="$seed" output.run_id="${method}_s${seed}" \
    ${extra[@]+"${extra[@]}"}
}

# --- L3 ------------------------------------------------------------------------
done_l3() { [[ -f "results/cube/runs/risk_s$1.json" ]]; }
run_l3() {
  local seed="$1" head mode beta
  read -r head mode beta < <(python -c 'import json; s = json.load(open("results/cube/l3/risk_selection.json")); print(s["head_path"], s["mode"], s["beta"])') || return 1
  echo "seed=$seed head=$head mode=$mode beta=$beta"
  python evaluate.py --config-name cube method=risk seed="$seed" output.run_id="risk_s$seed" \
    method.head_path="$head" method.mode="$mode" method.beta="$beta"
}

# --- L5 ------------------------------------------------------------------------
L5_W_PROP=(1.0 0.1 0.0)
done_l5() {
  if [[ "$2" == select ]]; then [[ -f "results/cube/runs/l4_$1_w$3_s$4.json" ]]
  else [[ -f "results/cube/runs/l4_$1_s$3.json" ]]; fi
}
run_l5() {
  local v="$1" mode="$2" seed w run args
  args=(--config-name cube method=baseline policy="cube/l4_$v")
  case "$v" in prop*) args+=(map_proprio_keys=true) ;; esac
  if [[ "$mode" == select ]]; then
    w="$3"; seed="$4"; run="l4_${v}_w${w}_s${seed}"
    args+=(+model_overrides.w_prop="$w")
  else
    seed="$3"; run="l4_${v}_s${seed}"
    case "$v" in
      prop*) w=$(python scripts/cube/select_wprop.py "$v") || return 1; args+=(+model_overrides.w_prop="$w") ;;
    esac
  fi
  echo "variant=$v mode=$mode seed=$seed run=$run"
  python evaluate.py "${args[@]}" seed="$seed" output.run_id="$run"
}

# --- visualisation -------------------------------------------------------------
# Same planner settings as the panels, plus the executed actions of every episode
# (cube/rollout_log.py), so episodes can be replayed exactly and rendered
# (scripts/cube/viz_export.py -> viz_blender.py -> viz_compose.py). Runs go to
# results/cube/viz_runs, not runs/, so they never enter the panel summaries.
VIZ_SEEDS=(42 2022 3033)
done_viz() { [[ -f "results/cube/viz_runs/rollouts/viz_$1_s$2.npz" ]]; }
run_viz() {
  local method="$1" seed="$2" thr omega extra=()
  case "$method" in
    gated)
      read -r thr omega < <(python scripts/cube/l2_params.py gated) || return 1
      extra=(method.truncation.threshold="$thr" method.truncation.omega="$omega") ;;
    baseline) ;;
    *) echo "run_viz: only baseline and gated" >&2; return 1 ;;
  esac
  python evaluate.py --config-name cube method="$method" seed="$seed" output.dir=results/cube/viz_runs \
    output.run_id="viz_${method}_s${seed}" ++output.save_rollout=true ${extra[@]+"${extra[@]}"}
}
