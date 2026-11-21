#!/usr/bin/env bash
# Tuning runs at goal offset 50 on seeds 100-104 (disjoint from the test seeds 0-9 and the calibration seeds),
# written to results/cube/compare_tune with the main run's trained models, so the test results stay untouched.
#
#   conda activate <env>; cd ~/ua-lewm; bash scripts/submit_cube_tune.sh
#   summary: OFFSET=50 STAGES=summary EXTRA="compare.out_dir=results/cube/compare_tune \
#            compare.models_dir=results/cube/compare/models" bash scripts/run_cube_compare.sh
#
# Selection rules, fixed before the runs (pooled success over the tuning seeds; ties go to the smaller value):
#   alpha*  = best of prior_cem_sp_var_a01 / _a025 / _a05 (sigma_p scaled by 0.1 / 0.25 / 0.5).
#             Confirm prior_cem_sp_var_a<alpha*> once on the test seeds; primary test: vs bc_only_var.
#   lambda* = best of bc_select_khat_sp_m4 / _sp_m4_ens03 / _sp_m4_ens (ensemble penalty 0 / 0.3 / 1 unit).
#             If lambda* = 0.3, confirm bc_select_khat_sp_m4_ens03 once on the test seeds; primary test:
#             vs bc_select_khat. (0 and 1 unit already have test-seed results.)
# bc_only_var, prior_cem_sp_var and bc_select_khat are the references on the same tuning seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
export OFFSET=50
export SEEDS="${SEEDS:-100 101 102 103 104}"
export EXTRA="compare.out_dir=results/cube/compare_tune compare.models_dir=results/cube/compare/models"
export LOGS="${LOGS:-results/cube/compare_tune/logs}"
export METHODS="${METHODS:-bc_only_var prior_cem_sp_var_a01 prior_cem_sp_var_a025 prior_cem_sp_var_a05 prior_cem_sp_var \
bc_select_khat bc_select_khat_sp_m4 bc_select_khat_sp_m4_ens03 bc_select_khat_sp_m4_ens}"
mkdir -p "$LOGS"
bash scripts/submit_cube_compare.sh
