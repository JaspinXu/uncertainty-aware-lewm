# Pilot `bc_select_khat_ua-prior_tc-khat_e10-b1` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_select_khat_ua-prior_tc-khat_e10-b1` | 400 | 93.5% | 95% | 89% | 98% | +42.0 | 173 / 5 | 7.6e-45 |

Task types (n = 400): carry 113, pick 151, place 136.
