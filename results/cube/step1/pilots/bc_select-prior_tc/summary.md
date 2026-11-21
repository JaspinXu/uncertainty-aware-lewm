# Pilot `bc_select-prior_tc` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_select-prior_tc` | 400 | 96.0% | 97% | 92% | 99% | +44.5 | 180 / 2 | 5.4e-51 |

Task types (n = 400): carry 113, pick 151, place 136.
