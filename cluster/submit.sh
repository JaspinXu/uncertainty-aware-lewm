#!/usr/bin/env bash
# Submit experiment phases with their Slurm dependencies (run on the login node).
#
#   bash cluster/submit.sh bootstrap   # env setup -> data/checkpoint download -> smoke test
#                                      #   (steps already done are skipped)
#   bash cluster/submit.sh smoke       # smoke test only
#   bash cluster/submit.sh planning    # L1 -> L2 -> L3 chained (after a clean smoke test)
#   bash cluster/submit.sh l1|l2|l3    # one level (l2/l3 need the L1 results)
#   bash cluster/submit.sh l4 [variants]   # train representation variants (default: all five);
#                                          # each one submits its own L5 evaluation when done
#   bash cluster/submit.sh l5 <variant>    # (normally automatic) probes + planning for one variant
#   bash cluster/submit.sh report          # L6: results/cube/RESULTS.md, after all queued jobs
#   bash cluster/submit.sh diag            # ~10 min check: live-rendered vs dataset frames (results/cube/diag)
#   bash cluster/submit.sh viz             # ~20 min: baseline + gated runs that save executed actions (for videos)
#   bash cluster/submit.sh status          # what is done, what is queued
#   bash cluster/submit.sh results         # commit + push the small result files and recent job logs
#
# Every phase can be submitted again at any time: finished runs are left out.
# GPU jobs go to partition $LEWM_GPU_PARTITION (default gpu, 3 h). Each job asks
# for one GPU of an explicit type (an untyped --gpus=1 only gets the old `nv`
# cards on the SoC cluster): rendering jobs take turns over $LEWM_GPU_TYPES,
# training-type jobs over $LEWM_TRAIN_GPU_TYPES (defaults below).
# Extra sbatch options for every GPU job go through SBATCH_EXTRA.
set -euo pipefail
ARGS=("$@")
cd "$(dirname "$0")/.."
mkdir -p logs/tasks
# shellcheck disable=SC1091
source cluster/runs.sh   # done_* checks (and cluster/seeds.sh)
read -r -a EXTRA <<< "${SBATCH_EXTRA:-}"
STABLEWM_HOME="${STABLEWM_HOME:-$HOME/lewm_data}"
PART="${LEWM_GPU_PARTITION:-gpu}"
ME="$(id -un)"
# at most this many jobs per group of runs: GPU jobs beyond the per-user limit of
# 8 GPUs only wait in the queue, and every queued job uses one of 32 slots
MAXJ_L1=4 MAXJ_TRAIN=3 MAXJ_L2=6 MAXJ_L3=2 MAXJ_L5=2

# ---------------------------------------------------------------------- #
# sbatch helpers
# ---------------------------------------------------------------------- #
# One GPU type per job, in turn, from two lists. Jobs that render the simulator
# (planning runs, oracle replays) use whole GPUs: the h100-47 cards are MIG
# slices of an H100, which cannot render (MuJoCo falls back to slow software
# rendering). Training-type jobs start with the h100-47 slices. Together the
# lists match the per-user limits (a100-40 4, a100-80 2, h100-47 2, h100-96 1,
# h200-141 1; 8 GPUs at once in total).
read -r -a RENDER_TYPES <<< "${LEWM_GPU_TYPES:-a100-40 a100-80 a100-40 a100-80 a100-40 a100-40 h100-96 h200-141}"
read -r -a TRAIN_TYPES <<< "${LEWM_TRAIN_GPU_TYPES:-h100-47 h100-47 a100-40 h100-96 a100-80}"
RENDER_I=0
TRAIN_I=0
# Nodes whose CPUs are not x86_64 (xgpj0, an a100-80 node, is ARM): the conda
# env cannot run there ("Exec format error"), so every job leaves them out.
NON_X86=$(scontrol -o show nodes 2>/dev/null | awk '{n = a = ""; for (i = 1; i <= NF; i++) {
  if ($i ~ /^NodeName=/) n = substr($i, 10); if ($i ~ /^Arch=/) a = substr($i, 6) }
  if (n != "" && a != "" && a != "x86_64") print n}' | paste -sd, - || true)
NODE_FLAGS=()
[[ -n "${NON_X86:-}" ]] && NODE_FLAGS=(--exclude="$NON_X86")
_sb_gpu() {  # _sb_gpu TYPE VAR sbatch-args...
  local __t="$1" __var="$2" __id
  shift 2
  __id=$(sbatch --parsable --partition="$PART" --gres="gpu:$__t:1" ${NODE_FLAGS[@]+"${NODE_FLAGS[@]}"} \
         ${EXTRA[@]+"${EXTRA[@]}"} "$@") || return 1
  printf -v "$__var" '%s' "$__id"
}
sbg() {  # sbg VAR sbatch-args...  -> GPU job that renders; VAR = its job id
  local t=${RENDER_TYPES[$(( RENDER_I % ${#RENDER_TYPES[@]} ))]}
  RENDER_I=$(( RENDER_I + 1 ))
  _sb_gpu "$t" "$@"
}
sbt() {  # sbt VAR sbatch-args...  -> GPU job without rendering (training, encoding, probes)
  local t=${TRAIN_TYPES[$(( TRAIN_I % ${#TRAIN_TYPES[@]} ))]}
  TRAIN_I=$(( TRAIN_I + 1 ))
  _sb_gpu "$t" "$@"
}
sbc() { sbatch --parsable ${NODE_FLAGS[@]+"${NODE_FLAGS[@]}"} "$@"; }        # CPU job (default partition)
dep() { [[ -n "${2:-}" ]] && echo "--dependency=$1:$2" || true; }              # dep afterok "1:2" -> flag
idjoin() { local IFS=:; echo "$*" | sed -E 's/:+/:/g; s/^://; s/:$//'; }      # idjoin 1 "" 2:3 -> 1:2:3

QUEUE_LIMIT=$(sacctmgr -nP list associations user="$ME" format=maxsubmit 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1; exit}' || true)
QUEUE_LIMIT=${QUEUE_LIMIT:-32}
queued() { squeue -h -r -u "$ME" 2>/dev/null | wc -l; }
check_capacity() {  # check_capacity N WHAT
  local cur
  cur=$(queued)
  if (( cur + $1 > QUEUE_LIMIT )); then
    echo "$2 needs $1 job(s), but $cur of your $QUEUE_LIMIT queued-job slots are taken." >&2
    if [[ "${LEWM_RETRY:-0}" == 1 ]] && (( cur < QUEUE_LIMIT )); then
      # called from a job (nobody to retry by hand): try again in 30 minutes
      sbatch --parsable --begin=now+30minutes cluster/jobs/91_retry_submit.sbatch "${ARGS[@]}" >&2
      echo "Will try again in 30 minutes (job lewm_retry)." >&2
      exit 0
    fi
    echo "Submit it again when some jobs have finished (finished runs are skipped)." >&2
    exit 3
  fi
}

# todo LINE... -> TODO = the task lines whose outputs do not exist yet
todo() {
  local line words
  TODO=()
  for line in "$@"; do
    read -r -a words <<< "$line"
    "done_${words[0]#run_}" "${words[@]:1}" || TODO+=("$line")
  done
}

# chunk KIND MAX_JOBS -> PER (runs per job) and NJOBS for TODO. The run time
# estimate comes from the smoke-test timings (scripts/cube/runs_per_job.py:
# TARGET runs fill ~100 min, MAX ~150 min of the 3 h limit). Beyond MAX_JOBS
# jobs, runs are packed more densely, up to MAX per job.
chunk() {
  local n=${#TODO[@]} target most
  read -r target most < <(python3 scripts/cube/runs_per_job.py "$1" 2>/dev/null || echo "3 4")
  PER=$target
  NJOBS=0
  (( n > 0 )) || return 0
  NJOBS=$(( (n + PER - 1) / PER ))
  if (( NJOBS > $2 )); then
    PER=$(( (n + $2 - 1) / $2 ))
    (( PER > most )) && PER=$most
    NJOBS=$(( (n + PER - 1) / PER ))
  fi
  PER=$(( (n + NJOBS - 1) / NJOBS ))  # same number of runs in every job
  return 0
}

# submit_tasks NAME PER DEP_FLAG -> submits TODO in jobs of PER runs; IDS = "id:id:..." ("" if nothing to do)
submit_tasks() {
  local name="$1" per="$2" depflag="$3" ids=() f k=0 s stamp id
  stamp=$(date +%Y%m%d-%H%M%S)
  for (( s = 0; s < ${#TODO[@]}; s += per )); do
    f="logs/tasks/${name}_${stamp}_${k}.txt"
    printf '%s\n' "${TODO[@]:s:per}" > "$f"
    sbg id --job-name="lewm_$name" ${depflag:+"$depflag"} cluster/jobs/run_tasks.sbatch "$f"
    ids+=("$id")
    k=$(( k + 1 ))
  done
  IDS=$(idjoin ${ids[@]+"${ids[@]}"})
}

# ---------------------------------------------------------------------- #
# task lists
# ---------------------------------------------------------------------- #
record_lines() { local s; for s in $(seed_group "$1"); do echo "run_record $s"; done; }
l2_lines() { local s m; for s in "${TEST_SEEDS[@]}"; do for m in "${L2_METHODS[@]}"; do echo "run_l2 $m $s"; done; done; }
l3_lines() { local s; for s in "${TEST_SEEDS[@]}"; do echo "run_l3 $s"; done; }
l5_select_lines() { local w s; for w in "${L5_W_PROP[@]}"; do for s in "${CAL_SEEDS[@]}"; do echo "run_l5 $1 select $w $s"; done; done; }
l5_test_lines() { local s; for s in "${TEST_SEEDS[@]}"; do echo "run_l5 $1 test $s"; done; }

# ---------------------------------------------------------------------- #
# phases: prep_* decides what to run and how many jobs that takes (N_*),
# submit_* submits it
# ---------------------------------------------------------------------- #
J_AN=""; J_TR=""

prep_l1() {
  local lines
  mapfile -t lines < <(record_lines l1); todo "${lines[@]}"
  L1_TODO=("${TODO[@]}"); chunk record "$MAXJ_L1"; L1_PER=$PER; N_L1=$NJOBS
  mapfile -t lines < <(record_lines train); todo "${lines[@]}"
  TR_TODO=("${TODO[@]}"); chunk record "$MAXJ_TRAIN"; TR_PER=$PER; N_L1=$(( N_L1 + NJOBS ))
  NEED_ENS=1
  [[ -f "$STABLEWM_HOME/checkpoints/cube/ens_lewm/ensemble.pt" ]] && NEED_ENS=0
  N_L1=$(( N_L1 + NEED_ENS + 2 ))  # + scoring + analysis
}
submit_l1() {
  local j_rec j_tr j_ens="" j_sc d
  TODO=("${L1_TODO[@]}"); submit_tasks l1_record "$L1_PER" ""; j_rec=$IDS
  TODO=("${TR_TODO[@]}"); submit_tasks l1_train "$TR_PER" ""; j_tr=$IDS
  (( NEED_ENS )) && sbt j_ens cluster/jobs/12_ensemble.sbatch
  # scoring needs the records; it runs even if the ensemble job failed (then it
  # fails quickly, and the analysis goes ahead with MC-dropout only)
  d=$(idjoin "$j_rec" "$j_tr"); d=${d:+afterok:$d}
  [[ -n "$j_ens" ]] && d="${d:+$d,}afterany:$j_ens"
  sbt j_sc ${d:+--dependency=$d} cluster/jobs/13_score_ensemble.sbatch
  # the analysis needs the L1 records; it uses the ensemble scores if that branch succeeded
  d="afterany:$j_sc"
  [[ -n "$j_rec" ]] && d="afterok:$j_rec,$d"
  J_AN=$(sbc --dependency="$d" cluster/jobs/11_l1_analyze.sbatch)
  J_TR="$j_tr"
  echo "L1: ${#L1_TODO[@]} cal/test runs + ${#TR_TODO[@]} risk-head runs to do, ensemble ${j_ens:-already trained}, scoring $j_sc, analysis $J_AN" >&2
}

prep_l2() {
  local lines
  mapfile -t lines < <(l2_lines); todo "${lines[@]}"
  L2_TODO=("${TODO[@]}"); chunk plan "$MAXJ_L2"; L2_PER=$PER
  NEED_CAL=1
  [[ -f results/calibration/gated_cube.json ]] && NEED_CAL=0
  N_L2=$(( NJOBS + NEED_CAL + 1 ))
}
submit_l2() {  # $1 = job to wait for (optional)
  local j_cal="" j_pan j_sum d
  (( NEED_CAL )) && sbg j_cal $(dep afterok "${1:-}") cluster/jobs/20_l2_calibrate.sbatch
  d=$(idjoin "$j_cal" "${1:-}")
  TODO=("${L2_TODO[@]}"); submit_tasks l2 "$L2_PER" "$(dep afterok "$d")"; j_pan=$IDS
  d=$(idjoin "$j_cal" "$j_pan")
  j_sum=$(sbc $(dep afterany "$d") cluster/jobs/90_summarize.sbatch l2_panel baseline "${L2_METHODS[@]}")
  echo "L2: calibration ${j_cal:-done} -> ${#L2_TODO[@]} runs ($L2_PER per job) -> summary $j_sum" >&2
}

prep_l3() {
  local lines
  mapfile -t lines < <(l3_lines); todo "${lines[@]}"
  L3_TODO=("${TODO[@]}"); chunk plan "$MAXJ_L3"; L3_PER=$PER
  NEED_RISK=1
  [[ -f results/cube/l3/risk_selection.json ]] && NEED_RISK=0
  N_L3=$(( NJOBS + NEED_RISK + 1 ))
}
submit_l3() {  # $1 = jobs to wait for "a:b" (optional)
  local j_tr="" j_pan j_sum d
  (( NEED_RISK )) && sbt j_tr $(dep afterok "${1:-}") cluster/jobs/30_l3_train_risk.sbatch
  d=$(idjoin "$j_tr" "${1:-}")
  TODO=("${L3_TODO[@]}"); submit_tasks l3 "$L3_PER" "$(dep afterok "$d")"; j_pan=$IDS
  d=$(idjoin "$j_tr" "$j_pan")
  j_sum=$(sbc $(dep afterany "$d") cluster/jobs/90_summarize.sbatch planning_panel baseline "${L2_METHODS[@]}" risk)
  echo "L3: risk head ${j_tr:-done} -> ${#L3_TODO[@]} runs ($L3_PER per job) -> summary $j_sum" >&2
}

prep_l4() {  # variants...
  local v
  L4_TODO=()
  for v in "$@"; do
    if [[ -f "$STABLEWM_HOME/checkpoints/cube/l4_$v/DONE" ]]; then
      echo "L4: $v already trained (its evaluation: bash cluster/submit.sh l5 $v)" >&2
    else
      L4_TODO+=("$v")
    fi
  done
  NEED_REF=1
  [[ -f results/cube/l5/repr_cube__lewm.json ]] && NEED_REF=0
  N_L4=$(( ${#L4_TODO[@]} + NEED_REF ))
}
submit_l4() {
  local v id
  for v in ${L4_TODO[@]+"${L4_TODO[@]}"}; do
    sbt id cluster/jobs/40_l4_train.sbatch "$v"
    echo "L4: training $v submitted ($id)" >&2
  done
  # the pretrained checkpoint is the reference for every L5 table
  (( NEED_REF )) && sbt id cluster/jobs/41_l5_repr.sbatch cube/lewm
  return 0
}

prep_l5() {  # variant
  local lines
  SEL_TODO=()
  case "$1" in
    prop*) mapfile -t lines < <(l5_select_lines "$1"); todo "${lines[@]}"; SEL_TODO=(${TODO[@]+"${TODO[@]}"}) ;;
  esac
  TODO=(${SEL_TODO[@]+"${SEL_TODO[@]}"}); chunk l5 "$MAXJ_L5"; SEL_PER=$PER; N_L5=$(( NJOBS + 1 ))
  mapfile -t lines < <(l5_test_lines "$1"); todo "${lines[@]}"
  TEST_TODO=(${TODO[@]+"${TODO[@]}"}); chunk l5 "$MAXJ_L5"; TEST_PER=$PER; N_L5=$(( N_L5 + NJOBS ))
  NEED_REPR=1
  [[ -f "results/cube/l5/repr_cube__l4_$1.json" ]] && NEED_REPR=0
  N_L5=$(( N_L5 - 1 + NEED_REPR ))
}
submit_l5() {  # variant
  local v="$1" j_repr="" j_sel="" j_test
  (( NEED_REPR )) && sbt j_repr cluster/jobs/41_l5_repr.sbatch "cube/l4_$v"
  TODO=(${SEL_TODO[@]+"${SEL_TODO[@]}"}); submit_tasks "l5_${v}_select" "$SEL_PER" ""; j_sel=$IDS
  # with a selection step pending, the test runs wait for it (they read its result)
  TODO=(${TEST_TODO[@]+"${TEST_TODO[@]}"}); submit_tasks "l5_${v}_test" "$TEST_PER" "$(dep afterok "$j_sel")"; j_test=$IDS
  echo "L5 $v: representation ${j_repr:-done}, w_prop selection ${j_sel:-done or not needed} -> test runs ${j_test:-already done}" >&2
}

status() {
  local lines v f
  echo "== runs done"
  mapfile -t lines < <(record_lines l1); todo "${lines[@]}"; echo "L1 cal/test runs     $(( ${#lines[@]} - ${#TODO[@]} ))/${#lines[@]}"
  mapfile -t lines < <(record_lines train); todo "${lines[@]}"; echo "L1 risk-head runs    $(( ${#lines[@]} - ${#TODO[@]} ))/${#lines[@]}"
  echo "latent ensemble      $([[ -f "$STABLEWM_HOME/checkpoints/cube/ens_lewm/ensemble.pt" ]] && echo trained || echo 'not trained')"
  mapfile -t lines < <(l2_lines); todo "${lines[@]}"; echo "L2 panel runs        $(( ${#lines[@]} - ${#TODO[@]} ))/${#lines[@]}"
  mapfile -t lines < <(l3_lines); todo "${lines[@]}"; echo "L3 panel runs        $(( ${#lines[@]} - ${#TODO[@]} ))/${#lines[@]}"
  for v in lewm ms prop prop_aux prop_ms; do
    mapfile -t lines < <(l5_test_lines "$v"); todo "${lines[@]}"
    printf 'L4/L5 %-9s trained: %-3s  test runs %s/%s\n' "$v" \
      "$([[ -f "$STABLEWM_HOME/checkpoints/cube/l4_$v/DONE" ]] && echo yes || echo no)" \
      "$(( ${#lines[@]} - ${#TODO[@]} ))" "${#lines[@]}"
  done
  for f in results/cube/smoke/timing.json results/cube/l1/signal_report.md results/cube/summary/l2_panel.md \
           results/cube/l3/risk_report.md results/cube/summary/planning_panel.md results/cube/l5/representation.md \
           results/cube/diag/domain_gap.md results/cube/RESULTS.md; do
    [[ -f "$f" ]] && echo "written: $f"
  done
  echo "== queue: $(queued) of $QUEUE_LIMIT job slots in use"
}

case "${1:-}" in
  bootstrap)
    J0=""; J1=""
    [[ -x "${CONDA_ROOT:-$HOME/miniforge3}/envs/${CONDA_ENV_NAME:-lewm}/bin/python" ]] || J0=$(sbc cluster/jobs/00_setup_env.sbatch)
    if [[ ! -f "$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5" || ! -f "$STABLEWM_HOME/checkpoints/cube/lewm/weights.pt" ]]; then
      J1=$(sbc $(dep afterok "$J0") cluster/jobs/01_download_cube.sbatch)
    fi
    sbg J2 $(dep afterok "$(idjoin "$J0" "$J1")") cluster/jobs/02_smoke_test.sbatch
    echo "bootstrap: setup ${J0:-done} -> download ${J1:-done} -> smoke test $J2 (log: logs/lewm_smoke_$J2.out)" >&2
    ;;
  smoke)
    sbg J2 cluster/jobs/02_smoke_test.sbatch
    echo "smoke test $J2 (log: logs/lewm_smoke_$J2.out)" >&2
    ;;
  l1) prep_l1; check_capacity "$N_L1" L1; submit_l1 ;;
  l2) prep_l2; check_capacity "$N_L2" L2; submit_l2 ;;
  l3) prep_l3; check_capacity "$N_L3" L3; submit_l3 ;;
  planning)
    prep_l1; prep_l2; prep_l3
    check_capacity $(( N_L1 + N_L2 + N_L3 )) "planning (L1-L3)"
    submit_l1
    submit_l2 "$J_AN"
    submit_l3 "$(idjoin "$J_AN" "$J_TR")"
    ;;
  l4)
    shift
    if [[ $# -eq 0 ]]; then set -- lewm ms prop prop_aux prop_ms; fi
    prep_l4 "$@"; check_capacity "$N_L4" L4; submit_l4
    ;;
  l4-continue)  # used by cluster/jobs/40_l4_train.sbatch to resubmit itself
    check_capacity 1 "L4 continuation of ${2:?variant}"
    sbt J4 cluster/jobs/40_l4_train.sbatch "$2" "${3:-}"
    echo "L4: $2 continues in job $J4" >&2
    exit 0
    ;;
  l5) prep_l5 "${2:?variant}"; check_capacity "$N_L5" "L5 $2"; submit_l5 "$2" ;;
  report)  # runs after every lewm job that is queued now has ended
    W=$(squeue -h -u "$ME" -o "%i %j" 2>/dev/null | awk '$2 ~ /^lewm_/ && $2 != "lewm_report" {print $1}' | paste -sd: - || true)
    J6=$(sbc $(dep afterany "$W") cluster/jobs/50_report.sbatch)
    echo "L6 report job $J6${W:+ (after jobs $W)}" >&2
    ;;
  diag)  # live rendering vs dataset frames, as the encoder sees them
    sbg JD cluster/jobs/60_diag_domain_gap.sbatch
    echo "diagnostic job $JD (log: logs/lewm_diag_$JD.out)" >&2
    ;;
  viz)  # executed actions of baseline and gated runs, for rendering rollout videos
    sbg JV cluster/jobs/70_viz_rollouts.sbatch
    echo "visualisation runs: job $JV (log: logs/lewm_viz_$JV.out)" >&2
    ;;
  status) status ;;
  results)  # small result files + the tail of every job log of the last week -> git
    mkdir -p results/cube/logs
    while IFS= read -r f; do
      tr '\r' '\n' < "$f" | tail -n 400 > "results/cube/logs/$(basename "$f")"
    done < <(find logs -maxdepth 1 -name 'lewm_*.out' -mtime -7 2>/dev/null)
    git add -A results/cube results/calibration
    git diff --cached --quiet || git commit -qm "Cube results ($(date '+%F %H:%M'))"
    git pull --rebase -q && git push -q && echo "results pushed ($(git log -1 --format='%h %s'))" >&2
    exit 0
    ;;
  *)
    sed -n '2,20p' "$0" >&2
    exit 2
    ;;
esac
squeue -u "$ME" -o "%.12i %.22j %.9T %.10M %.10P %.28R" 2>/dev/null | head -40 || true
