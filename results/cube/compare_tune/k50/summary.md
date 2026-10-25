# Cube comparison, goal 50 steps ahead (budget 100 steps)

Non-trivial start states from episodes no learned component was trained on; success = the cube within 4 cm of the goal. Mean ± std over seeds (50 episodes each); pooled success with a Wilson 95% CI. Against `lewm` on the same start states: Δ, episodes gained / lost, exact McNemar p (Holm-adjusted over the runs) and an exact sign-flip p over the per-seed differences.

| run | seeds | episodes | success, mean ± std over seeds | pooled [95% CI] | carry | pick | place | move | Δ vs lewm (pp) | gained / lost | McNemar p (Holm) | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bc_only_var` | 5 | 250 | 74.8 ± 3.6 | 74.8 [69.1, 79.8] | 100% | 49% | 88% | 100% | | | | |
| `prior_cem_sp_var` | 5 | 250 | 72.0 ± 6.0 | 72.0 [66.1, 77.2] | 100% | 26% | 99% | 100% | | | | |
| `bc_select_khat` | 5 | 250 | 85.2 ± 3.0 | 85.2 [80.3, 89.1] | 100% | 66% | 96% | 100% | | | | |
| `bc_select_khat_sp_m4` | 5 | 250 | 80.4 ± 6.1 | 80.4 [75.0, 84.8] | 100% | 53% | 98% | 88% | | | | |
| `bc_select_khat_sp_m4_ens` | 5 | 250 | 84.4 ± 6.8 | 84.4 [79.4, 88.4] | 100% | 69% | 93% | 100% | | | | |
| `prior_cem_sp_var_a01` | 5 | 250 | 78.8 ± 5.0 | 78.8 [73.3, 83.4] | 100% | 46% | 98% | 100% | | | | |
| `prior_cem_sp_var_a025` | 5 | 250 | 77.2 ± 4.8 | 77.2 [71.6, 82.0] | 100% | 39% | 100% | 100% | | | | |
| `prior_cem_sp_var_a05` | 5 | 250 | 73.2 ± 5.2 | 73.2 [67.4, 78.3] | 100% | 29% | 99% | 100% | | | | |
| `bc_select_khat_sp_m4_ens03` | 5 | 250 | 82.8 ± 4.8 | 82.8 [77.6, 87.0] | 100% | 63% | 94% | 94% | | | | |

Task types (cube held at the start / at the goal; seeds of `bc_only_var`): carry 3, pick 93, place 137, move 17.

## Ablations (paired on the seeds both runs have)

| comparison | seeds | success | Δ (pp) | gained / lost | McNemar p | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| `prior_cem_sp_var` vs `bc_only_var` | 5 | 72.0 vs 74.8 | -2.8 | 22 / 29 | 0.4 | 0.44 |
| `bc_select_khat` vs `prior_cem_sp_var` | 5 | 85.2 vs 72.0 | +13.2 | 41 / 8 | 2.0e-06 | 0.062 |
| `bc_select_khat_sp_m4` vs `bc_select_khat` | 5 | 80.4 vs 85.2 | -4.8 | 7 / 19 | 0.029 | 0.25 |
| `bc_select_khat_sp_m4_ens` vs `bc_select_khat_sp_m4` | 5 | 84.4 vs 80.4 | +4.0 | 21 / 11 | 0.11 | 0.12 |
| `bc_select_khat_sp_m4_ens` vs `bc_select_khat` | 5 | 84.4 vs 85.2 | -0.8 | 14 / 16 | 0.86 | 0.88 |
| `prior_cem_sp_var_a01` vs `bc_only_var` | 5 | 78.8 vs 74.8 | +4.0 | 19 / 9 | 0.087 | 0.12 |
| `prior_cem_sp_var_a025` vs `bc_only_var` | 5 | 77.2 vs 74.8 | +2.4 | 25 / 19 | 0.45 | 0.5 |
| `prior_cem_sp_var_a05` vs `bc_only_var` | 5 | 73.2 vs 74.8 | -1.6 | 24 / 28 | 0.68 | 0.56 |
| `bc_select_khat_sp_m4_ens03` vs `bc_select_khat_sp_m4` | 5 | 82.8 vs 80.4 | +2.4 | 15 / 9 | 0.31 | 0.38 |
| `bc_select_khat_sp_m4_ens03` vs `bc_select_khat` | 5 | 82.8 vs 85.2 | -2.4 | 9 / 15 | 0.31 | 0.44 |

## Success rate per seed (%)

| run | 100 | 101 | 102 | 103 | 104 |
|---|---:|---:|---:|---:|---:|
| `bc_only_var` | 76 | 74 | 80 | 74 | 70 |
| `prior_cem_sp_var` | 78 | 66 | 78 | 66 | 72 |
| `bc_select_khat` | 90 | 86 | 84 | 84 | 82 |
| `bc_select_khat_sp_m4` | 80 | 78 | 88 | 72 | 84 |
| `bc_select_khat_sp_m4_ens` | 88 | 82 | 94 | 76 | 82 |
| `prior_cem_sp_var_a01` | 80 | 78 | 86 | 72 | 78 |
| `prior_cem_sp_var_a025` | 84 | 72 | 80 | 74 | 76 |
| `prior_cem_sp_var_a05` | 80 | 68 | 76 | 68 | 74 |
| `bc_select_khat_sp_m4_ens03` | 88 | 80 | 86 | 76 | 84 |
