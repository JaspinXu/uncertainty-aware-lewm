# Cube comparison, goal 25 steps ahead (budget 50 steps)

Non-trivial start states from episodes no learned component was trained on; success = the cube within 4 cm of the goal. Mean ± std over seeds (50 episodes each); pooled success with a Wilson 95% CI. Against `lewm` on the same start states: Δ, episodes gained / lost, exact McNemar p (Holm-adjusted over the runs) and an exact sign-flip p over the per-seed differences.

| run | seeds | episodes | success, mean ± std over seeds | pooled [95% CI] | carry | pick | place | move | Δ vs lewm (pp) | gained / lost | McNemar p (Holm) | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lewm` | 10 | 500 | 56.4 ± 8.4 | 56.4 [52.0, 60.7] | 64% | 22% | 82% | – | | | | |
| `mc5` | 10 | 500 | 55.6 ± 8.3 | 55.6 [51.2, 59.9] | 64% | 19% | 82% | – | -0.8 | 16 / 20 | 1 | 0.52 |
| `gated` | 10 | 500 | 55.8 ± 6.9 | 55.8 [51.4, 60.1] | 65% | 17% | 83% | – | -0.6 | 17 / 20 | 1 | 0.75 |
| `gated_tc` | 10 | 500 | 53.2 ± 7.3 | 53.2 [48.8, 57.5] | 60% | 16% | 82% | – | -3.2 | 13 / 29 | 0.059 | 0.051 |
| `bc_only` | 10 | 500 | 99.0 ± 1.4 | 99.0 [97.7, 99.6] | 99% | 100% | 98% | – | +42.6 | 215 / 2 | 1.3e-60 | 0.002 |
| `prior_cem_s1` | 10 | 500 | 94.4 ± 3.6 | 94.4 [92.0, 96.1] | 100% | 83% | 100% | – | +38.0 | 192 / 2 | 7.5e-54 | 0.002 |
| `prior_cem_sconst` | 10 | 500 | 98.2 ± 1.5 | 98.2 [96.6, 99.1] | 100% | 95% | 100% | – | +41.8 | 209 / 0 | 1.7e-62 | 0.002 |
| `prior_cem_sp` | 10 | 500 | 99.4 ± 1.0 | 99.4 [98.3, 99.8] | 100% | 98% | 100% | – | +43.0 | 215 / 0 | 3.0e-64 | 0.002 |
| `bc_select_khat` | 10 | 500 | 92.8 ± 2.3 | 92.8 [90.2, 94.8] | 93% | 87% | 98% | – | +36.4 | 189 / 7 | 1.6e-46 | 0.002 |

Task types (cube held at the start / at the goal; seeds of `lewm`): carry 157, pick 164, place 179, move 0.

## Ablations (paired on the seeds both runs have)

| comparison | seeds | success | Δ (pp) | gained / lost | McNemar p | sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| `gated` vs `mc5` | 10 | 55.8 vs 55.6 | +0.2 | 16 / 15 | 1 | 1 |
| `gated_tc` vs `mc5` | 10 | 53.2 vs 55.6 | -2.4 | 14 / 26 | 0.081 | 0.098 |
| `gated_tc` vs `gated` | 10 | 53.2 vs 55.8 | -2.6 | 15 / 28 | 0.066 | 0.078 |
| `prior_cem_sconst` vs `prior_cem_s1` | 10 | 98.2 vs 94.4 | +3.8 | 19 / 0 | 3.8e-06 | 0.0078 |
| `prior_cem_sp` vs `prior_cem_s1` | 10 | 99.4 vs 94.4 | +5.0 | 25 / 0 | 6.0e-08 | 0.0039 |
| `prior_cem_sp` vs `prior_cem_sconst` | 10 | 99.4 vs 98.2 | +1.2 | 7 / 1 | 0.07 | 0.062 |
| `prior_cem_sp` vs `bc_only` | 10 | 99.4 vs 99.0 | +0.4 | 5 / 3 | 0.73 | 0.75 |
| `bc_select_khat` vs `bc_only` | 10 | 92.8 vs 99.0 | -6.2 | 5 / 36 | 7.8e-07 | 0.002 |
| `bc_select_khat` vs `prior_cem_sp` | 10 | 92.8 vs 99.4 | -6.6 | 3 / 36 | 3.6e-08 | 0.002 |

## Planner diagnostics (mean over planning calls and seeds)

| run | fraction of candidates truncated | scored blocks (of 5) | mean sigma^2 |
|---|---:|---:|---:|
| `mc5` | 0.000 | 5.000 | 0.687 |
| `gated` | 0.082 | 4.915 | 0.673 |
| `gated_tc` | 0.112 | 4.668 | 0.589 |

## Success rate per seed (%)

| run | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lewm` | 60 | 66 | 60 | 50 | 54 | 52 | 70 | 60 | 50 | 42 |
| `mc5` | 62 | 62 | 60 | 48 | 48 | 52 | 68 | 62 | 52 | 42 |
| `gated` | 60 | 58 | 62 | 50 | 52 | 54 | 66 | 60 | 54 | 42 |
| `gated_tc` | 56 | 62 | 56 | 50 | 46 | 48 | 64 | 54 | 56 | 40 |
| `bc_only` | 98 | 100 | 100 | 100 | 100 | 98 | 96 | 98 | 100 | 100 |
| `prior_cem_s1` | 96 | 96 | 98 | 96 | 94 | 90 | 100 | 94 | 92 | 88 |
| `prior_cem_sconst` | 100 | 98 | 98 | 100 | 98 | 96 | 100 | 96 | 98 | 98 |
| `prior_cem_sp` | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 98 | 98 |
| `bc_select_khat` | 94 | 94 | 90 | 96 | 96 | 90 | 90 | 92 | 92 | 94 |
