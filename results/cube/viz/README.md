# Rollout videos: LeWM baseline vs. uncertainty-gated planning

These are real rollouts of the two planners, replayed exactly and rendered with
Blender (Cycles). Nothing in them is generated or simulated with a different
policy.

## How the episodes were chosen (read before using them)

**Seeds.** 42, 2022 and 3033 are the three test seeds with the largest gated
gains in the panel (+10, +12 and +14 points; the mean over 9 seeds is +5.1).

**Re-runs.** The actions come from re-runs of the two planners on these seeds
(`bash cluster/submit.sh viz`; results in `results/cube/viz_runs/`). Re-runs do
not reproduce every episode outcome, because CEM details vary with the GPU a
job lands on. Same outcome as the panel run:

| planner | seed 42 | seed 2022 | seed 3033 |
|---|---:|---:|---:|
| baseline | 50/50 | 44/50 | 42/50 |
| gated | 44/50 | 50/50 | 50/50 |

**Episodes.** We kept only episodes whose outcome is the same in the panel run
and in the re-run, for both planners. Among those, we took the episodes where
gated succeeds and the baseline fails, farthest start first. One more episode,
where both fail, shows the search failures that remain.

**What the examples can and cannot show.** They illustrate the difference
between the planners; they are not evidence for it. Over all 450 test
episodes, success is 65.6% (baseline) against 70.7% (gated), McNemar
p = 0.0014 (`results/cube/summary/planning_panel.md`).

| file prefix | seed / episode | start distance | baseline | gated |
|---|---|---:|---|---|
| `out/s42/ep31_*` | 42 / 31 | 37.2 cm | drops the cube at the start, fails (35.2 cm) | success at step 24 |
| `out/s2022/ep22_*` | 2022 / 22 | 20.0 cm | fails (19.7 cm) | success at step 22 |
| `out/s2022/ep10_*` | 2022 / 10 | 17.0 cm | fails (17.1 cm) | success at step 17 |
| `out/s42/ep16_*` | 42 / 16 | 13.5 cm | fails (14.0 cm) | success at step 19 |
| `out/s42/ep22_*` | 42 / 22 | 30.8 cm | fails (30.1 cm) | fails (33.5 cm) |

## Pipeline

1. **Replay.** `scripts/cube/viz_export.py` replays each episode from its
   start state and checks it against the cube positions logged on the
   cluster. The largest deviation was 3·10⁻⁸ m, and every outcome and success
   step matched.
2. **Render.** `scripts/cube/viz_blender.py` renders every second env step.
3. **Compose.** `scripts/cube/viz_compose.py` builds three panels: the expert
   demonstration from the dataset (start → goal, 25 steps), LeWM (baseline),
   and LeWM with uncertainty-gated planning.

All three panels start from the same state and have the same goal (green
box). Labels give the step, the cube–goal distance and the outcome; success
means the cube is within 4 cm of the goal.

## Files per episode

| suffix | content |
|---|---|
| `_compare.mp4` | side-by-side video, 8 frames/s |
| `_compare.gif` | the same, smaller; not in git, sent separately |
| `_strip.png` | film strip, columns at steps 0, 10, …, 50 |
| `_final.png` | the last frame |

## Re-creating the files

Rendered frames and scene files are not kept in git. Re-create them with the
commands in `cluster/README.md` (section "Rollout videos"), passing the
episodes explicitly:
- seed 42: `--episodes 31 16 22`
- seed 2022: `--episodes 22 10`
