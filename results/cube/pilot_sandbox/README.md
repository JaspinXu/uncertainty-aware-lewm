# Sandbox pilot (pipeline check, not a result)

These files come from a small CPU-only run used to test the whole L1 pipeline
with the **real pretrained LeWM-Cube checkpoint** before the cluster runs.
They are not results of the study.

- **Data.** 24 synthetic expert episodes of 100 steps, generated with the
  same Cube environment and expert policy as the official data
  (`tests/make_synthetic_cube.py`). The official 46 GB dataset was not used.
- **Episodes.** 16 evaluation episodes, 8 per seed: seed 910 used as
  calibration, seed 42 as test.
- **Planner.** Full paper CEM budget (300 samples / 30 iterations / top-30).
  Recording on, MC-dropout with 8 masks, and an oracle replay of 51
  candidates for each of the 22 planning calls.

What it established about the pipeline:
- **Planning works.** The model plans normally on live-rendered frames:
  6/8 and 4/8 successes. A frame rendered after `set_state` encodes to
  within 0.02 (squared latent distance) of the dataset frame of the same
  state.
- **Recording does not change the planner.** Recorded costs equal the
  solver's costs exactly (mismatch 0).
- **Replay of start-of-episode plans is faithful.** Replaying the executed
  first plan reaches the state the evaluation reached to within
  max |Δqpos| ≈ 3e-7. The oracle agrees with the closed-loop outcome on
  16/16 episodes. This pilot only checked plans made at an episode start.
  Plans made in the middle of an episode were later found to replay
  inexactly when only qpos/qvel are restored (median error 0.15 on expert-like
  trajectories): the solver warm start and the end-effector pose that the next
  action is computed from are part of the state. Recording now stores both and
  the replay restores them, which makes every replay bit-exact
  (`cube/simstate.py`, `tests/test_cube_core.py::test_mid_episode_replay_is_exact`).
- **The ensemble branch runs end to end.** Encoding, training, post-hoc
  scoring and analysis all ran, but with a 30-step ensemble. Its numbers
  carry no information and are not reported here.

`signal_report.md` and the figures are the MC-dropout analysis of this pilot.
`representation.md` holds probes and open-loop errors of the pretrained model
on 12 of the synthetic episodes. With so few planning calls every interval
spans most of [0, 1], so nothing should be concluded from these numbers. The
analysis on the official data is produced by `bash cluster/submit.sh planning`
(see `cluster/README.md`).
