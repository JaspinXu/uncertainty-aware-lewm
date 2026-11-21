# Pilot `bc_only` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | n | success | carry | pick | place | Δ (pp) | gained / lost | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (CEM, replan every 5 blocks) | 400 | 51.5% | 59% | 17% | 83% | | | |
| `bc_only` | 400 | 98.0% | 99% | 97% | 99% | +46.5 | 187 / 1 | 9.6e-55 |

Task types (n = 400): carry 113, pick 151, place 136.
