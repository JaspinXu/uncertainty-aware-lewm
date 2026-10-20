# Pilot `bc_init` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_init` | 400 | 89.5% | 100% | 72% | 100% | +38.0 | 153 / 1 | 1.4e-44 |

Task types (n = 400): carry 113, pick 151, place 136.
