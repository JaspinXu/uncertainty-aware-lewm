#!/usr/bin/env bash
# Submit experiment phases with their Slurm dependencies.
#
#   bash cluster/submit.sh bootstrap  # env setup -> data/checkpoint download -> smoke test (chained)
#   bash cluster/submit.sh planning   # L1 -> L2 -> L3 chained (recommended after the smoke test)
#   bash cluster/submit.sh l1         # record + oracle (cal/test/train seeds) + latent ensemble -> L1 analysis
#   bash cluster/submit.sh l2         # needs L1 results: calibrate gate -> 5 planners x 9 seeds -> panel
#   bash cluster/submit.sh l3         # needs L1 results: train risk head -> closed loop x 9 seeds -> panel
#   bash cluster/submit.sh l4 [variants]   # train representation variants (default: all five);
#                                          # each one submits its own L5 evaluation when done
#   bash cluster/submit.sh l5 <variant>    # (normally automatic) probes + planning for one variant
#   bash cluster/submit.sh report          # L6: collect every result into results/cube/RESULTS.md
#
# Extra sbatch options for every GPU job go through SBATCH_EXTRA, e.g.
#   SBATCH_EXTRA="--gpus=a100-80:1" bash cluster/submit.sh planning
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
# shellcheck disable=SC1091
source cluster/seeds.sh
read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"

n_of() { local -a s; read -r -a s <<< "$(seed_group "$1")"; echo "${#s[@]}"; }
sb() { sbatch --parsable ${EXTRA[@]+"${EXTRA[@]}"} "$@"; }
dep() { [[ -n "${1:-}" ]] && echo "--dependency=afterok:$1" || true; }

submit_l1() {  # -> prints "<analysis job id> <train-data job id>"
  local n_l1 n_tr j_rec j_tr j_ens j_sc j_an
  n_l1=$(n_of l1); n_tr=$(n_of train)
  j_rec=$(sb --array=0-$((n_l1 - 1)) cluster/jobs/10_record_oracle.sbatch l1)
  j_tr=$(sb --array=0-$((n_tr - 1)) cluster/jobs/10_record_oracle.sbatch train)
  j_ens=$(sb cluster/jobs/12_ensemble.sbatch)
  j_sc=$(sb --dependency=afterok:"$j_rec":"$j_tr":"$j_ens" cluster/jobs/13_score_ensemble.sbatch)
  # the analysis needs the records; it uses the ensemble scores if that branch succeeded
  j_an=$(sbatch --parsable --dependency=afterok:"$j_rec",afterany:"$j_sc" cluster/jobs/11_l1_analyze.sbatch)
  echo "L1: record/oracle $j_rec ($n_l1 seeds), risk-head data $j_tr ($n_tr seeds), ensemble $j_ens -> scoring $j_sc, analysis $j_an" >&2
  echo "$j_an $j_tr"
}

submit_l2() {  # $1 = dependency job id (optional)
  local j_cal j_pan j_sum n
  n=$(( 5 * $(n_of test) ))
  j_cal=$(sb $(dep "${1:-}") cluster/jobs/20_l2_calibrate.sbatch)
  j_pan=$(sb --dependency=afterok:"$j_cal" --array=0-$((n - 1)) cluster/jobs/21_l2_panel.sbatch)
  j_sum=$(sbatch --parsable --dependency=afterany:"$j_pan" cluster/jobs/90_summarize.sbatch \
          l2_panel baseline mc5 gated goal_cvar penalty ens_penalty)
  echo "L2: calibration $j_cal -> panel $j_pan ($n runs) -> summary $j_sum" >&2
  echo "$j_sum"
}

submit_l3() {  # $1 = dependency job ids "a:b" (optional)
  local j_tr j_pan j_sum n
  n=$(n_of test)
  j_tr=$(sb $(dep "${1:-}") cluster/jobs/30_l3_train_risk.sbatch)
  j_pan=$(sb --dependency=afterok:"$j_tr" --array=0-$((n - 1)) cluster/jobs/31_l3_panel.sbatch)
  j_sum=$(sbatch --parsable --dependency=afterany:"$j_pan" cluster/jobs/90_summarize.sbatch \
          planning_panel baseline mc5 gated goal_cvar penalty ens_penalty risk)
  echo "L3: risk head $j_tr -> panel $j_pan ($n runs) -> summary $j_sum" >&2
}

submit_l4() {
  local v
  for v in "$@"; do
    sb cluster/jobs/40_l4_train.sbatch "$v" > /dev/null
    echo "L4: training $v submitted" >&2
  done
  # the pretrained checkpoint is the reference for every L5 table
  [[ -f results/cube/l5/repr_cube__lewm.json ]] || sb cluster/jobs/41_l5_repr.sbatch cube/lewm > /dev/null
}

submit_l5() {  # $1 = variant
  local v="$1" j_sel j_test
  sb cluster/jobs/41_l5_repr.sbatch "cube/l4_$v" > /dev/null
  case "$v" in
    prop*)
      j_sel=$(sb --array=0-8 cluster/jobs/42_l5_plan.sbatch "$v" select)
      j_test=$(sb --dependency=afterok:"$j_sel" --array=0-8 cluster/jobs/42_l5_plan.sbatch "$v" test)
      echo "L5 $v: representation eval, w_prop selection $j_sel -> test panel $j_test" >&2 ;;
    *)
      j_test=$(sb --array=0-8 cluster/jobs/42_l5_plan.sbatch "$v" test)
      echo "L5 $v: representation eval, test panel $j_test" >&2 ;;
  esac
}

case "${1:-}" in
  bootstrap)
    J0=$(sbatch --parsable cluster/jobs/00_setup_env.sbatch)
    J1=$(sbatch --parsable --dependency=afterok:"$J0" cluster/jobs/01_download_cube.sbatch)
    J2=$(sb --dependency=afterok:"$J1" cluster/jobs/02_smoke_test.sbatch)
    echo "bootstrap: setup $J0 -> download $J1 -> smoke test $J2 (logs/lewm_smoke_$J2.out)" >&2
    ;;
  l1) submit_l1 > /dev/null ;;
  l2) submit_l2 > /dev/null ;;
  l3) submit_l3 ;;
  planning)
    read -r J_AN J_TR < <(submit_l1)
    submit_l2 "$J_AN" > /dev/null
    submit_l3 "$J_AN:$J_TR"
    ;;
  l4)
    shift
    if [[ $# -eq 0 ]]; then set -- lewm ms prop prop_aux prop_ms; fi
    submit_l4 "$@"
    ;;
  l5) submit_l5 "${2:?variant}" ;;
  report) sbatch --parsable cluster/jobs/50_report.sbatch > /dev/null; echo "L6 report job submitted" >&2 ;;
  *)
    sed -n '2,15p' "$0" >&2
    exit 2
    ;;
esac
squeue -u "${USER:-$(whoami)}" -o "%.14i %.20j %.9T %.10M %.20R" | head -50
