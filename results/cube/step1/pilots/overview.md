# Step-1 pilots, paired with the baseline (seeds 61-68, complete runs only)

Success rate overall and by task type; gained / lost episodes and exact McNemar p against the baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_init` | 400 | 89.5% | 100% | 72% | 100% | +38.0 | 153 / 1 | 1.4e-44 |
| `bc_khat-prior_tc-khat_e10` | 400 | 92.5% | 91% | 93% | 93% | +41.0 | 173 / 9 | 1.7e-40 |
| `bc_only` | 400 | 98.0% | 99% | 97% | 99% | +46.5 | 187 / 1 | 9.6e-55 |
| `bc_only-prior_var` | 400 | 79.0% | 89% | 60% | 92% | +27.5 | 136 / 26 | 3.5e-19 |
| `bc_only_rh1` | 400 | 48.8% | 27% | 50% | 65% | -2.8 | 77 / 88 | 0.44 |
| `bc_only_rh1-prior_var` | 400 | 50.5% | 68% | 31% | 57% | -1.0 | 76 / 80 | 0.81 |
| `bc_select-prior_tc` | 400 | 96.0% | 97% | 92% | 99% | +44.5 | 180 / 2 | 5.4e-51 |
| `bc_select_khat-prior_tc-khat_e10` | 400 | 94.0% | 96% | 89% | 99% | +42.5 | 174 / 4 | 2.2e-46 |
| `bc_select_khat_ua-prior_tc-khat_e10-b1` | 400 | 93.5% | 95% | 89% | 98% | +42.0 | 173 / 5 | 7.6e-45 |
| `bc_select_khat_ua-prior_tc-khat_e10-b10` | 400 | 92.8% | 96% | 89% | 94% | +41.2 | 176 / 11 | 2e-39 |
| `expert_init` | 400 | 90.8% | 100% | 75% | 100% | +39.2 | 159 / 2 | 8.9e-45 |
| `hold_goal` | 400 | 54.2% | 68% | 18% | 83% | +2.7 | 11 / 0 | 0.00098 |

Task types (n = 400): carry 113, pick 151, place 136.
