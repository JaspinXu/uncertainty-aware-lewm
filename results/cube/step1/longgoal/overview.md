# Long-goal protocol (exploratory)

Goal k expert steps ahead, budget 2k steps; non-trivial start states from episodes no prior was trained on. Task type = cube held at the start / at the goal (carry yes/yes, pick no/yes, place yes/no, move no/no). Δ, gained / lost and exact McNemar p are against `cem` on the same start states.

## Goal 50 steps ahead, budget 100 steps

| run | seeds | n | success | carry | pick | place | move | Δ vs cem (pp) | gained / lost | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cem` | 61,62,63,64 | 200 | 38.5% | – | 6% | 67% | 8% | | | |
| `bc_init-prior_var` | 61,62 | 100 | 63.0% | – | 19% | 96% | 45% | +24.0 | 26 / 2 | 3e-06 |
| `bc_khat-prior_tc-khat_e10` | 61,62,63,64 | 200 | 75.0% | – | 57% | 92% | 46% | +36.5 | 78 / 5 | 6.4e-18 |
| `bc_only-prior` | 61,62 | 100 | 56.0% | – | 44% | 75% | 0% | +17.0 | 20 / 3 | 0.00049 |
| `bc_only-prior_var` | 61,62,63,64 | 200 | 79.0% | – | 54% | 95% | 100% | +40.5 | 86 / 5 | 4e-20 |
| `bc_only_rh1-prior_var` | 61,62 | 100 | 44.0% | – | 31% | 53% | 45% | +5.0 | 22 / 17 | 0.52 |
| `bc_oracle_k-prior_tc` | 61,62 | 100 | 95.0% | – | 86% | 100% | 100% | +56.0 | 56 / 0 | 2.8e-17 |
| `bc_select-prior_tc` | 61,62,63,64 | 200 | 77.0% | – | 63% | 88% | 77% | +38.5 | 90 / 13 | 2.5e-15 |
| `bc_select_khat-prior_tc-khat_e10` | 61,62,63,64 | 200 | 85.5% | – | 67% | 98% | 100% | +47.0 | 95 / 1 | 2.4e-27 |
| `bc_select_khat_ua-prior_tc-khat_e10-b1` | 61,62,63,64 | 200 | 85.5% | – | 67% | 98% | 100% | +47.0 | 95 / 1 | 2.4e-27 |
| `bc_select_khat_ua-prior_tc-khat_e10-b10` | 61,62,63,64 | 200 | 84.5% | – | 69% | 94% | 100% | +46.0 | 96 / 4 | 6.4e-24 |

Task types over the 2 seed(s) of `bc_init-prior_var`: carry 0, pick 36, place 53, move 11.

## Goal 100 steps ahead, budget 200 steps

| run | seeds | n | success | carry | pick | place | move | Δ vs cem (pp) | gained / lost | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cem` | 61,62,63,64 | 200 | 33.0% | 39% | 10% | 97% | 0% | | | |
| `bc_khat-prior_tc-khat_e10` | 61,62,63,64 | 200 | 71.0% | 92% | 38% | 81% | 56% | +38.0 | 86 / 10 | 3.2e-16 |
| `bc_only-prior` | 61,62 | 100 | 66.0% | 100% | 47% | 100% | 4% | +29.0 | 30 / 1 | 3e-08 |
| `bc_only-prior_var` | 61,62,63,64 | 200 | 87.5% | 82% | 100% | 94% | 82% | +54.5 | 113 / 4 | 9.2e-29 |
| `bc_only_rh1-prior_var` | 61,62 | 100 | 76.0% | 87% | 82% | 100% | 38% | +39.0 | 42 / 3 | 8.7e-10 |
| `bc_oracle_k-prior_tc` | 61,62 | 100 | 92.0% | 95% | 100% | 100% | 77% | +55.0 | 55 / 0 | 5.6e-17 |
| `bc_select-prior_tc` | 61,62,63,64 | 200 | 66.5% | 86% | 87% | 59% | 24% | +33.5 | 83 / 16 | 4.4e-12 |
| `bc_select_khat-prior_tc-khat_e10` | 61,62,63,64 | 200 | 86.0% | 96% | 100% | 100% | 50% | +53.0 | 107 / 1 | 6.7e-31 |
| `bc_select_khat_ua-prior_tc-khat_e10-b1` | 61,62,63,64 | 200 | 85.5% | 95% | 100% | 100% | 50% | +52.5 | 106 / 1 | 1.3e-30 |
| `bc_select_khat_ua-prior_tc-khat_e10-b10` | 61,62,63,64 | 200 | 88.0% | 95% | 100% | 97% | 62% | +55.0 | 113 / 3 | 6.3e-30 |

Task types over the 4 seed(s) of `bc_khat-prior_tc-khat_e10`: carry 79, pick 39, place 32, move 50.
