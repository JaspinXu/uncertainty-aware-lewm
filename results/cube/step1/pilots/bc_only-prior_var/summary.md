# Pilot `bc_only-prior_var` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_only-prior_var` | 400 | 79.0% | 89% | 60% | 92% | +27.5 | 136 / 26 | 3.5e-19 |

Task types (n = 400): carry 113, pick 151, place 136.
