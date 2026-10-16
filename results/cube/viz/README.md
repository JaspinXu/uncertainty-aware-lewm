# Rollout videos: LeWM baseline vs. uncertainty-gated planning

These are real rollouts from our planning runs, rendered with Blender (Cycles).
Nothing in them is generated or re-simulated with a different policy.

**How they were made.**
1. `bash cluster/submit.sh viz` re-ran the baseline and the gated planner on
   test seeds 42, 2022 and 3033 and logged the env actions every episode
   executed (`results/cube/viz_runs/`).
2. `scripts/cube/viz_export.py` replayed each selected episode from its start
   state and checked it against the cube positions logged on the cluster. The
   largest deviation was 1e-8 m, and every outcome and success step matched.
3. `scripts/cube/viz_blender.py` rendered the replay (every second env step).
4. `scripts/cube/viz_compose.py` composed the outputs.

**The panels.** Each output has three panels:
- the expert demonstration from the dataset (start → goal, 25 steps);
- LeWM (baseline);
- LeWM with uncertainty-gated planning.

All three start from the same state and have the same goal (green box).
Labels give the step, the cube–goal distance and the outcome (success = cube
within 4 cm of the goal).

**The episodes are selected.** We chose them because the two planners
disagree, mostly where gating succeeds and the baseline fails, plus one of the
reverse. They illustrate the difference; they are not evidence for it. Over
all 450 test episodes, success is 65.6% (baseline) against 70.7% (gated),
McNemar p = 0.0014 (`results/cube/summary/planning_panel.md`).

| file prefix | seed / episode | start distance | baseline | gated |
|---|---|---:|---|---|
| `out/s2022/ep40_*` | 2022 / 40 | 32.7 cm | fails (12.1 cm) | success at step 16 |
| `out/s2022/ep18_*` | 2022 / 18 | 23.8 cm | fails, pushes the cube away (50.8 cm) | success at step 18 |
| `out/s42/ep31_*` | 42 / 31 | 37.2 cm | fails (35.2 cm) | success at step 24 |
| `out/s3033/ep23_*` | 3033 / 23 | 22.5 cm | fails (22.7 cm) | success at step 23 |
| `out/s3033/ep48_*` | 3033 / 48 | 28.3 cm | success at step 46 | fails (38.0 cm) |

**Files per episode.**

| suffix | content |
|---|---|
| `_compare.mp4` | side-by-side video, 8 frames/s |
| `_compare.gif` | the same, smaller; not in git, sent separately |
| `_strip.png` | film strip, columns at steps 0, 10, …, 50 |
| `_final.png` | the last frame |

The rendered frames and scene files are not kept in git; re-create them with
the commands in `cluster/README.md` (section "Rollout videos").
