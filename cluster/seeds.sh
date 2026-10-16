#!/usr/bin/env bash
# Seed groups shared by all job scripts (each seed = 50 evaluation episodes).
# shellcheck disable=SC2034

# locked evaluation panel (same seeds as the PushT pilot)
TEST_SEEDS=(42 43 44 45 1011 2022 3033 4044 5055)
# calibration / model selection (sigma^2 gate threshold, lambda, risk-head beta)
CAL_SEEDS=(910 911 912)
# extra recorded + replayed runs used only to train the L3 risk head
TRAIN_SEEDS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115)

seed_group() {  # seed_group NAME -> prints the seeds of that group
  case "$1" in
    test) echo "${TEST_SEEDS[@]}" ;;
    cal) echo "${CAL_SEEDS[@]}" ;;
    train) echo "${TRAIN_SEEDS[@]}" ;;
    l1) echo "${CAL_SEEDS[@]} ${TEST_SEEDS[@]}" ;;
    *) echo "unknown seed group '$1' (test|cal|train|l1)" >&2; return 1 ;;
  esac
}
