# Pilot `hold_goal` (hold rule: goal, replan every 5 block(s))

400 non-trivial episodes, paired with the Step-1 baseline on the same start states.

| | success |
|---|---:|
| baseline (replan every 5 blocks) | 51.5% |
| hold_goal | **54.2%** |

Δ = +2.7 pp, gained / lost episodes 11 / 0, exact McNemar p = 0.00098
