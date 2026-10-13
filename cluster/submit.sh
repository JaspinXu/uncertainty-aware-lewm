#!/usr/bin/env bash
# Submit experiment phases with their Slurm dependencies.
#
#   bash cluster/submit.sh planning   # L1 -> L2 -> L3 chained (recommended after the smoke test)
#   bash cluster/submit.sh l1         # record + oracle (cal/test/train seeds) -> L1 analysis
#   bash cluster/submit.sh l2         # needs L1 results: calibrate gate -> 4 planners x 9 seeds -> panel
#   bash cluster/submit.sh l3         # needs L1 results: train risk head -> closed loop x 9 seeds -> panel
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
  local n_l1 n_tr j_rec j_tr j_an
  n_l1=$(n_of l1); n_tr=$(n_of train)
  j_rec=$(sb --array=0-$((n_l1 - 1)) cluster/jobs/10_record_oracle.sbatch l1)
  j_tr=$(sb --array=0-$((n_tr - 1)) cluster/jobs/10_record_oracle.sbatch train)
  j_an=$(sbatch --parsable --dependency=afterok:"$j_rec" cluster/jobs/11_l1_analyze.sbatch)
  echo "L1: record/oracle $j_rec ($n_l1 seeds), risk-head data $j_tr ($n_tr seeds), analysis $j_an" >&2
  echo "$j_an $j_tr"
}

submit_l2() {  # $1 = dependency job id (optional)
  local j_cal j_pan j_sum n
  n=$(( 4 * $(n_of test) ))
  j_cal=$(sb $(dep "${1:-}") cluster/jobs/20_l2_calibrate.sbatch)
  j_pan=$(sb --dependency=afterok:"$j_cal" --array=0-$((n - 1)) cluster/jobs/21_l2_panel.sbatch)
  j_sum=$(sbatch --parsable --dependency=afterany:"$j_pan" cluster/jobs/90_summarize.sbatch \
          l2_panel baseline mc5 gated goal_cvar penalty)
  echo "L2: calibration $j_cal -> panel $j_pan ($n runs) -> summary $j_sum" >&2
  echo "$j_sum"
}

submit_l3() {  # $1 = dependency job ids "a:b" (optional)
  local j_tr j_pan j_sum n
  n=$(n_of test)
  j_tr=$(sb $(dep "${1:-}") cluster/jobs/30_l3_train_risk.sbatch)
  j_pan=$(sb --dependency=afterok:"$j_tr" --array=0-$((n - 1)) cluster/jobs/31_l3_panel.sbatch)
  j_sum=$(sbatch --parsable --dependency=afterany:"$j_pan" cluster/jobs/90_summarize.sbatch \
          planning_panel baseline mc5 gated goal_cvar penalty risk)
  echo "L3: risk head $j_tr -> panel $j_pan ($n runs) -> summary $j_sum" >&2
}

case "${1:-}" in
  l1) submit_l1 > /dev/null ;;
  l2) submit_l2 > /dev/null ;;
  l3) submit_l3 ;;
  planning)
    read -r J_AN J_TR < <(submit_l1)
    submit_l2 "$J_AN" > /dev/null
    submit_l3 "$J_AN:$J_TR"
    ;;
  *)
    sed -n '2,12p' "$0" >&2
    exit 2
    ;;
esac
squeue -u "$USER" -o "%.14i %.20j %.9T %.10M %.20R" | head -50
