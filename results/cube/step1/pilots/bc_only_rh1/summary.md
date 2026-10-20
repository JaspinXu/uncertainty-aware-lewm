# Pilot `bc_only_rh1` (hold rule: None, replan every 1 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_only_rh1` | 400 | 48.8% | 27% | 50% | 65% | -2.8 | 77 / 88 | 0.44 |

Task types (n = 400): carry 113, pick 151, place 136.
