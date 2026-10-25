# Cube comparison, goal 50 steps ahead (budget 100 steps)

Non-trivial start states from episodes no learned component was trained on; success = the cube within 4 cm of the goal. Mean ± std over seeds (50 episodes each); pooled success with a Wilson 95% CI. Against `lewm` on the same start states: Δ, episodes gained / lost, exact McNemar p (Holm-adjusted over the runs) and an exact sign-flip p over the per-seed differences.

| run | seeds | episodes | success, mean ± std over seeds | pooled [95% CI] | carry | pick | place | move | Δ vs lewm (pp) | gained / lost | McNemar p (Holm) | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lewm` | 10 | 500 | 41.8 ± 4.7 | 41.8 [37.6, 46.2] | 100% | 10% | 69% | 16% | | | | |
| `mc5` | 10 | 500 | 40.8 ± 6.3 | 40.8 [36.6, 45.2] | 100% | 11% | 67% | 13% | -1.0 | 16 / 21 | 1 | 0.56 |
| `gated` | 10 | 500 | 41.4 ± 6.7 | 41.4 [37.2, 45.8] | 100% | 14% | 66% | 13% | -0.4 | 28 / 30 | 1 | 0.89 |
| `gated_tc` | 10 | 500 | 38.4 ± 3.1 | 38.4 [34.2, 42.7] | 100% | 11% | 64% | 4% | -3.4 | 16 / 33 | 0.064 | 0.02 |
| `bc_only` | 10 | 500 | 53.2 ± 3.6 | 53.2 [48.8, 57.5] | 100% | 19% | 78% | 51% | +11.4 | 93 / 36 | 2.7e-06 | 0.002 |
| `prior_cem_s1` | 10 | 500 | 55.6 ± 6.5 | 55.6 [51.2, 59.9] | 100% | 6% | 95% | 31% | +13.8 | 95 / 26 | 1.4e-09 | 0.002 |
| `prior_cem_sconst` | 10 | 500 | 59.4 ± 5.3 | 59.4 [55.0, 63.6] | 100% | 10% | 98% | 40% | +17.6 | 112 / 24 | 7.8e-14 | 0.002 |
| `prior_cem_sp` | 10 | 500 | 60.6 ± 6.3 | 60.6 [56.3, 64.8] | 100% | 11% | 98% | 44% | +18.8 | 116 / 22 | 1.3e-15 | 0.002 |
| `bc_only_var` | 10 | 500 | 72.2 ± 6.9 | 72.2 [68.1, 75.9] | 100% | 48% | 85% | 98% | +30.4 | 186 / 34 | 2.0e-25 | 0.002 |
| `prior_cem_s1_var` | 10 | 500 | 60.0 ± 6.9 | 60.0 [55.6, 64.2] | 100% | 14% | 95% | 47% | +18.2 | 110 / 19 | 1.0e-15 | 0.002 |
| `prior_cem_sconst_var` | 10 | 500 | 64.2 ± 6.9 | 64.2 [59.9, 68.3] | 100% | 17% | 98% | 62% | +22.4 | 122 / 10 | 1.6e-24 | 0.002 |
| `prior_cem_sp_var` | 10 | 500 | 70.2 ± 7.7 | 70.2 [66.0, 74.0] | 100% | 22% | 99% | 100% | +28.4 | 150 / 8 | 7.0e-34 | 0.002 |
| `bc_select_khat` | 10 | 500 | 86.0 ± 6.0 | 86.0 [82.7, 88.8] | 100% | 70% | 95% | 98% | +44.2 | 229 / 8 | 5.3e-56 | 0.002 |
| `lewm_khat` | 10 | 500 | 37.4 ± 9.1 | 37.4 [33.3, 41.7] | 100% | 7% | 64% | 7% | -4.4 | 21 / 43 | 0.033 | 0.062 |
| `prior_cem_s1_var_khat` | 10 | 500 | 53.0 ± 6.5 | 53.0 [48.6, 57.3] | 100% | 10% | 88% | 27% | +11.2 | 80 / 24 | 1.9e-07 | 0.0039 |
| `prior_cem_sconst_var_khat` | 10 | 500 | 56.4 ± 8.4 | 56.4 [52.0, 60.7] | 100% | 13% | 93% | 22% | +14.6 | 98 / 25 | 1.7e-10 | 0.0039 |
| `prior_cem_sp_var_khat` | 10 | 500 | 62.4 ± 9.2 | 62.4 [58.1, 66.5] | 100% | 20% | 91% | 73% | +20.6 | 117 / 14 | 2.4e-20 | 0.002 |
| `bc_select_khat_m1` | 10 | 500 | 82.2 ± 5.2 | 82.2 [78.6, 85.3] | 0% | 66% | 92% | 96% | +40.4 | 214 / 12 | 1.1e-47 | 0.002 |
| `bc_select_khat_sp_m4` | 10 | 500 | 80.0 ± 8.5 | 80.0 [76.3, 83.3] | 100% | 56% | 95% | 89% | +38.2 | 201 / 10 | 4.7e-46 | 0.002 |
| `bc_select_khat_sconst_m4` | 10 | 500 | 76.2 ± 7.2 | 76.2 [72.3, 79.7] | 100% | 43% | 98% | 84% | +34.4 | 174 / 2 | 7.2e-48 | 0.002 |
| `bc_select_khat_sp_m16` | 10 | 500 | 82.0 ± 6.3 | 82.0 [78.4, 85.1] | 100% | 56% | 99% | 91% | +40.2 | 205 / 4 | 4.8e-54 | 0.002 |
| `bc_select_khat_m1_ens` | 10 | 500 | 84.2 ± 5.9 | 84.2 [80.7, 87.1] | 0% | 71% | 92% | 93% | +42.4 | 223 / 11 | 4.0e-51 | 0.002 |
| `bc_select_khat_sp_m4_ens` | 10 | 500 | 83.4 ± 8.6 | 83.4 [79.9, 86.4] | 0% | 67% | 94% | 91% | +41.6 | 216 / 8 | 2.6e-52 | 0.002 |
| `bc_select_khat_m1_ens3` | 10 | 500 | 83.0 ± 5.0 | 83.0 [79.5, 86.0] | 100% | 71% | 90% | 91% | +41.2 | 220 / 14 | 1.8e-47 | 0.002 |
| `bc_select_khat_sp_m4_ens3` | 10 | 500 | 79.8 ± 7.9 | 79.8 [76.1, 83.1] | 100% | 64% | 89% | 91% | +38.0 | 208 / 18 | 6.4e-41 | 0.002 |
| `prior_cem_sp_var_a01` | 10 | 500 | 77.4 ± 7.7 | 77.4 [73.5, 80.8] | 100% | 43% | 98% | 98% | +35.6 | 188 / 10 | 1.9e-42 | 0.002 |
| `prior_cem_sconst_var_a01` | 10 | 500 | 73.2 ± 9.2 | 73.2 [69.2, 76.9] | 100% | 34% | 97% | 96% | +31.4 | 168 / 11 | 4.9e-36 | 0.002 |

Task types (cube held at the start / at the goal; seeds of `lewm`): carry 1, pick 189, place 265, move 45.

## Ablations (paired on the seeds both runs have)

| comparison | seeds | success | Δ (pp) | gained / lost | McNemar p | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| `gated` vs `mc5` | 10 | 41.4 vs 40.8 | +0.6 | 18 / 15 | 0.73 | 0.77 |
| `gated_tc` vs `mc5` | 10 | 38.4 vs 40.8 | -2.4 | 16 / 28 | 0.096 | 0.26 |
| `gated_tc` vs `gated` | 10 | 38.4 vs 41.4 | -3.0 | 12 / 27 | 0.024 | 0.12 |
| `prior_cem_sconst` vs `prior_cem_s1` | 10 | 59.4 vs 55.6 | +3.8 | 31 / 12 | 0.0054 | 0.13 |
| `prior_cem_sp` vs `prior_cem_s1` | 10 | 60.6 vs 55.6 | +5.0 | 35 / 10 | 2.5e-04 | 0.047 |
| `prior_cem_sp` vs `prior_cem_sconst` | 10 | 60.6 vs 59.4 | +1.2 | 15 / 9 | 0.31 | 0.42 |
| `prior_cem_sp` vs `bc_only` | 10 | 60.6 vs 53.2 | +7.4 | 62 / 25 | 9.1e-05 | 0.0078 |
| `prior_cem_sconst_var` vs `prior_cem_s1_var` | 10 | 64.2 vs 60.0 | +4.2 | 29 / 8 | 7.5e-04 | 0.0039 |
| `prior_cem_sp_var` vs `prior_cem_s1_var` | 10 | 70.2 vs 60.0 | +10.2 | 57 / 6 | 1.6e-11 | 0.002 |
| `prior_cem_sp_var` vs `prior_cem_sconst_var` | 10 | 70.2 vs 64.2 | +6.0 | 36 / 6 | 2.8e-06 | 0.002 |
| `prior_cem_sp_var` vs `bc_only_var` | 10 | 70.2 vs 72.2 | -2.0 | 51 / 61 | 0.4 | 0.4 |
| `bc_only` vs `bc_only_var` | 10 | 53.2 vs 72.2 | -19.0 | 26 / 121 | 7.5e-16 | 0.002 |
| `bc_select_khat` vs `bc_only` | 10 | 86.0 vs 53.2 | +32.8 | 172 / 8 | 3.2e-41 | 0.002 |
| `bc_select_khat` vs `prior_cem_sp` | 10 | 86.0 vs 60.6 | +25.4 | 139 / 12 | 1.4e-28 | 0.002 |
| `bc_select_khat` vs `prior_cem_sp_var` | 10 | 86.0 vs 70.2 | +15.8 | 93 / 14 | 1.8e-15 | 0.002 |
| `prior_cem_sp_var_khat` vs `bc_select_khat` | 10 | 62.4 vs 86.0 | -23.6 | 10 / 128 | 3.1e-27 | 0.002 |
| `prior_cem_sp_var_khat` vs `prior_cem_sp_var` | 10 | 62.4 vs 70.2 | -7.8 | 16 / 55 | 3.8e-06 | 0.0039 |
| `prior_cem_sp_var_khat` vs `prior_cem_sconst_var_khat` | 10 | 62.4 vs 56.4 | +6.0 | 52 / 22 | 6.4e-04 | 0.0078 |
| `prior_cem_sconst_var_khat` vs `prior_cem_s1_var_khat` | 10 | 56.4 vs 53.0 | +3.4 | 30 / 13 | 0.014 | 0.049 |
| `prior_cem_sp_var_khat` vs `bc_only_var` | 10 | 62.4 vs 72.2 | -9.8 | 45 / 94 | 3.9e-05 | 0.0039 |
| `lewm_khat` vs `lewm` | 10 | 37.4 vs 41.8 | -4.4 | 21 / 43 | 0.0081 | 0.062 |
| `bc_select_khat_sp_m4` vs `bc_select_khat_m1` | 10 | 80.0 vs 82.2 | -2.2 | 26 / 37 | 0.21 | 0.38 |
| `bc_select_khat_sp_m4` vs `bc_select_khat_sconst_m4` | 10 | 80.0 vs 76.2 | +3.8 | 42 / 23 | 0.025 | 0.031 |
| `bc_select_khat_sp_m16` vs `bc_select_khat_sp_m4` | 10 | 82.0 vs 80.0 | +2.0 | 28 / 18 | 0.18 | 0.25 |
| `bc_select_khat_m1` vs `bc_select_khat` | 10 | 82.2 vs 86.0 | -3.8 | 19 / 38 | 0.016 | 0.074 |
| `bc_select_khat_sp_m4` vs `bc_select_khat` | 10 | 80.0 vs 86.0 | -6.0 | 20 / 50 | 4.4e-04 | 0.0039 |
| `bc_select_khat_sp_m4_ens` vs `bc_select_khat_sp_m4` | 10 | 83.4 vs 80.0 | +3.4 | 46 / 29 | 0.064 | 0.14 |
| `bc_select_khat_m1_ens` vs `bc_select_khat_m1` | 10 | 84.2 vs 82.2 | +2.0 | 38 / 28 | 0.27 | 0.25 |
| `bc_select_khat_sp_m4_ens` vs `bc_select_khat_m1` | 10 | 83.4 vs 82.2 | +1.2 | 43 / 37 | 0.58 | 0.59 |
| `bc_select_khat_sp_m4_ens3` vs `bc_select_khat_sp_m4_ens` | 10 | 79.8 vs 83.4 | -3.6 | 11 / 29 | 0.0064 | 0.039 |
| `bc_select_khat_sp_m4_ens` vs `bc_select_khat` | 10 | 83.4 vs 86.0 | -2.6 | 25 / 38 | 0.13 | 0.18 |
| `prior_cem_sp_var_a01` vs `bc_only_var` | 10 | 77.4 vs 72.2 | +5.2 | 53 / 27 | 0.0049 | 0.0078 |
| `prior_cem_sp_var_a01` vs `prior_cem_sconst_var_a01` | 10 | 77.4 vs 73.2 | +4.2 | 27 / 6 | 3.2e-04 | 0.016 |
| `prior_cem_sp_var_a01` vs `bc_select_khat` | 10 | 77.4 vs 86.0 | -8.6 | 34 / 77 | 5.5e-05 | 0.0039 |

## Planner diagnostics (mean over planning calls and seeds)

| run | fraction of candidates truncated | scored blocks (of 5) | mean sigma^2 |
|---|---:|---:|---:|
| `mc5` | 0.000 | 5.000 | 0.823 |
| `gated` | 0.092 | 4.904 | 0.804 |
| `gated_tc` | 0.125 | 4.633 | 0.687 |

## Success rate per seed (%)

| run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lewm` | 44 | 36 | 46 | 32 | 42 | 44 | 42 | 48 | 42 | 42 |
| `mc5` | 50 | 36 | 40 | 30 | 36 | 46 | 42 | 48 | 44 | 36 |
| `gated` | 46 | 38 | 38 | 26 | 38 | 42 | 46 | 48 | 48 | 44 |
| `gated_tc` | 40 | 38 | 42 | 32 | 38 | 34 | 40 | 40 | 40 | 40 |
| `bc_only` | 52 | 54 | 56 | 44 | 56 | 54 | 54 | 52 | 54 | 56 |
| `prior_cem_s1` | 60 | 52 | 62 | 48 | 52 | 48 | 62 | 56 | 66 | 50 |
| `prior_cem_sconst` | 60 | 66 | 64 | 50 | 58 | 52 | 64 | 62 | 56 | 62 |
| `prior_cem_sp` | 62 | 64 | 62 | 48 | 56 | 54 | 64 | 64 | 62 | 70 |
| `bc_only_var` | 74 | 66 | 72 | 70 | 66 | 60 | 82 | 74 | 80 | 78 |
| `prior_cem_s1_var` | 62 | 58 | 58 | 48 | 60 | 50 | 68 | 68 | 62 | 66 |
| `prior_cem_sconst_var` | 70 | 60 | 70 | 50 | 60 | 58 | 70 | 70 | 66 | 68 |
| `prior_cem_sp_var` | 78 | 66 | 76 | 56 | 62 | 64 | 76 | 78 | 72 | 74 |
| `bc_select_khat` | 94 | 90 | 88 | 76 | 84 | 78 | 94 | 86 | 86 | 84 |
| `lewm_khat` | 40 | 36 | 44 | 18 | 40 | 26 | 42 | 48 | 36 | 44 |
| `prior_cem_s1_var_khat` | 58 | 48 | 52 | 48 | 52 | 44 | 54 | 68 | 54 | 52 |
| `prior_cem_sconst_var_khat` | 56 | 52 | 60 | 46 | 58 | 42 | 58 | 72 | 58 | 62 |
| `prior_cem_sp_var_khat` | 68 | 58 | 60 | 46 | 62 | 50 | 70 | 76 | 66 | 68 |
| `bc_select_khat_m1` | 88 | 78 | 88 | 74 | 76 | 86 | 88 | 80 | 82 | 82 |
| `bc_select_khat_sp_m4` | 84 | 76 | 80 | 60 | 80 | 76 | 92 | 86 | 84 | 82 |
| `bc_select_khat_sconst_m4` | 80 | 78 | 80 | 58 | 70 | 76 | 80 | 80 | 82 | 78 |
| `bc_select_khat_sp_m16` | 88 | 84 | 80 | 68 | 76 | 80 | 86 | 88 | 84 | 86 |
| `bc_select_khat_m1_ens` | 88 | 90 | 92 | 74 | 80 | 82 | 90 | 78 | 86 | 82 |
| `bc_select_khat_sp_m4_ens` | 86 | 86 | 94 | 68 | 74 | 78 | 96 | 84 | 88 | 80 |
| `bc_select_khat_m1_ens3` | 82 | 84 | 90 | 78 | 80 | 82 | 86 | 74 | 90 | 84 |
| `bc_select_khat_sp_m4_ens3` | 84 | 84 | 90 | 62 | 74 | 76 | 86 | 80 | 78 | 84 |
| `prior_cem_sp_var_a01` | 76 | 74 | 86 | 70 | 70 | 64 | 82 | 86 | 84 | 82 |
| `prior_cem_sconst_var_a01` | 78 | 66 | 82 | 62 | 64 | 60 | 82 | 78 | 84 | 76 |
