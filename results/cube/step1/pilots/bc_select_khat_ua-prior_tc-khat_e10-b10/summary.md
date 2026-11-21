# Pilot `bc_select_khat_ua-prior_tc-khat_e10-b10` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_select_khat_ua-prior_tc-khat_e10-b10` | 400 | 92.8% | 96% | 89% | 94% | +41.2 | 176 / 11 | 2e-39 |

Task types (n = 400): carry 113, pick 151, place 136.
