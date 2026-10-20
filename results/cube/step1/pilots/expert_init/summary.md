# Pilot `expert_init` (hold rule: None, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | success |
|---|---:|
| baseline (replan every 5 blocks) | 51.5% |
| expert_init | **90.8%** |

Δ = +39.2 pp, gained / lost episodes 159 / 2, exact McNemar p = 8.9e-45
