# Does Uncertainty Help a Latent World-Model Planner? A Ground-Truth Study of LeWorldModel on OGBench-Cube

*CS5478 final project. Every number below comes from the result files in
`results/cube/`, which `scripts/cube/make_report.py` collects into
`results/cube/RESULTS.md`.*

> **Status (2026-10-16).** Provisional numbers are marked †:
> - the L2 rows with fewer than 9 seeds;
> - the L5 rows of `prop_aux` and `prop_ms`;
> - the diagnostic in §5.6.
>
> The missing runs were lost to a node failure and are being re-run; the
> numbers will be updated when they finish. The σ² penalty with the ensemble
> (p = 0.058 on 7 seeds) is the result most likely to change.

## Abstract

LeWorldModel (LeWM) plans by cross-entropy-method (CEM) search for action
sequences whose predicted latent outcome lies closest to a goal image. If plans
fail because the planner exploits model errors, an uncertainty-aware planner
should help. On PushT, however, three such planners gained at most 2.2 points
(p > 0.1).

We test the premise directly on OGBench-Cube. We recorded 807 planning calls of
the pretrained LeWM and replayed 41,157 of the candidates it considered in the
simulator, starting from the exact planning state. The replays are bit-exact.
They give every candidate its real outcome and the model's real prediction
error.

**Uncertainty signals.** We compared MC-dropout variance with the disagreement
of a five-member latent dynamics ensemble.
- Both track the size of the prediction error one action block ahead
  (Spearman 0.55 and 0.69).
- Both barely track it at the fifth block, where the planning cost is evaluated
  (0.03 and 0.08).
- Within a planning call, neither separates the elite candidates that fail from
  those that succeed (AUROC 0.49 and 0.52).

**Why re-ranking cannot help.** 83% of failed plans are *search* failures: none
of the replayed alternatives solves the task. Consistent with this:
- re-ranking candidates by σ² does not raise success;
- re-ranking with a risk head trained on the replays does not either, although
  the head predicts failure well (AUROC 0.88);
- in closed loop, MC averaging, a CVaR penalty and two σ² penalties give no
  significant gain.

The one exception is uncertainty-gated horizon truncation (+5.3 points,
p = 0.002, significant after Holm correction). It reads σ² at the first block,
where σ² tracks the error best.

**Representation.** Representation changes trained under one budget show the
same pattern. Against an equally trained control, a multi-step rollout loss cuts
the five-block open-loop error on held-out dataset sequences 3.2-fold, yet
raises success by only 1.3 points. A factorised proprioceptive latent lowers
success by 5.1 points.

On the planner's own candidates, the model's prediction error is more than
an order of magnitude larger than on held-out dataset sequences. A check is
still pending on whether part of this comes from live rendering. If it does
not, the planner searches where the model is least reliable, and a better
search distribution, not a better uncertainty signal, is the lever that
matters.

## 1 Introduction

**Motivation.** LeWM [Maes et al., 2026] trains two parts jointly from pixels:
- a ViT encoder;
- an action-conditioned transformer predictor.

Training uses a next-latent prediction loss plus the SIGReg anti-collapse
regulariser. LeWM plans with CEM over five action blocks, minimising the
distance between the predicted final latent and the latent of a goal image.

On OGBench-Cube pick-and-place, LeWM reaches 74% success, behind DINO-WM (86%)
and goal-conditioned behaviour cloning (GCBC, 84%). A random policy reaches 48%,
because many evaluation episodes start with the cube already at its goal.

**The hypothesis inherited from the original project.** Plans fail because the
planner exploits model error, so penalising epistemic uncertainty should help
[Chua et al., 2018; Yu et al., 2020; Wu et al., 2022]. A PushT pilot tested this
over 9 seeds × 50 episodes:

| planner | success |
|---|---|
| baseline | 87.3 ± 3.5% |
| MC-5 | 89.6% |
| gated truncation | 88.9% |
| Goal-CVaR | 88.7% |

None of these differences was significant.

**This project.** Small or absent gains are consistent with two explanations
that call for different next steps:
- (a) the uncertainty signal does not identify the model's mistakes;
- (b) it does, but the planners use it poorly.

We separate the two with a ground-truth replay of the planner's decisions (L1).
We then test uncertainty-aware planners (L2), a learned risk model (L3), and
representation changes aimed at the model's errors (L4–L5). The L1 findings
frame how L2–L5 are read.

**Contributions.**
1. **A recorder and simulator oracle** that replay any candidate from its exact
   planning state.
   - The replays are bit-exact, including plans made mid-episode.
   - Mid-episode exactness requires restoring the solver warm start and the
     controller's cached end-effector pose, not only joint positions and
     velocities.
2. **A pre-registered signal test** (Q1–Q6) with plan-clustered bootstrap
   intervals, applied to MC-dropout and to a latent ensemble.
   - Both signals are informative *between* planning problems but not *within*
     a planning call, which is where CEM chooses.
   - Most failures are search failures, not selection failures.
3. **A paired comparison on the 9-seed test panel** of six uncertainty-aware
   planners on Cube, including a risk head learned from the replays (7–9 seeds
   per planner so far †). Only gated truncation helps.
4. **Five representation variants** of LeWM trained under one budget and
   evaluated three ways: probes, open-loop prediction error and closed-loop
   planning (two variants pending †). Better open-loop prediction on dataset
   sequences does not carry over to planning success.

## 2 Background

**LeWM.**
- The encoder maps a 224×224 frame to z ∈ ℝ¹⁹²: a ViT-tiny/14 CLS token
  followed by an MLP projector with BatchNorm.
- The predictor is a 6-layer causal transformer with AdaLN-zero action
  conditioning. It predicts z_{t+1} from up to 3 past latents and action
  blocks.
- One action block is 5 environment steps × 5 action dimensions, so 25 inputs.
- Training loss: next-latent MSE + 0.09 · SIGReg.

**Planning protocol (paper, Cube).**
- CEM with 300 samples, 30 iterations and the top 30 as elites.
- Horizon of 5 blocks (25 environment steps). The whole plan is executed before
  replanning, so a 50-step episode has 2 planning calls.
- Cost = ‖ẑ₅ − z_goal‖².
- The start and goal come from the expert dataset; the goal is the state 25
  steps after the start.
- Success means the cube is within 4 cm of its goal position.

**Uncertainty in model-based planning.** Epistemic uncertainty can be estimated
by MC-dropout [Gal & Ghahramani, 2016] or deep ensembles
[Lakshminarayanan et al., 2017]. Planners then use it in one of four ways:
- to aggregate rollouts;
- to penalise the objective [Chua et al., 2018; Yu et al., 2020];
- to truncate rollouts [Janner et al., 2019; Wu et al., 2022];
- to plan against a tail risk such as CVaR [Rockafellar & Uryasev, 2000].

## 3 Method

### 3.1 Recording planning calls and replaying them (L1)

`cube/recording.py` subclasses the CEM solver and the MPC policy. Its only side
effect is logging: in a pilot run, the recorded costs equal the solver's costs
exactly. At every planning call it stores four things.
- **Simulator state.** Joint positions and velocities; the solver warm start;
  the pinch-site pose that the controller caches from the previous step; and
  the goal target (`cube/simstate.py`).
- **Plans.** The executed plan (the final CEM mean) and the final population of
  300 candidates with their costs.
- **Per-candidate scores.** The deterministic cost, and the MC-dropout mean
  cost and spread.
- **Uncertainty profile.** The MC-dropout σ² at each of the 5 predicted blocks
  (8 masks).

A subset of 51 candidates per call is kept for replay: the executed plan, the
30 elites and 20 random non-elites. `cube/oracle.py` restores the stored state
in a fresh simulator and executes each kept candidate open loop, exactly as the
policy would. Actions take the policy's own float32 path through the dataset
action scaler.

The oracle logs the cube-to-goal distance after every step and renders the
frame at the end of every block. The pretrained encoder turns these frames into
the *true* latents z_k, which give two quantities:
- the model's true prediction error, ‖ẑ_k − z_k‖²;
- the cost the planner should have seen, ‖z₅ − z_goal‖².

**Fidelity.** Two checks confirm that the replays are faithful.
- *State check.* Replaying an executed plan must reach the state that the
  evaluation itself recorded 25 steps later (or at the end of the episode).
- *Outcome check.* An episode's real outcome must equal "one of its executed
  plans succeeds in replay".

Restoring only joint positions and velocities is enough for plans made at an
episode start. For mid-episode plans it is not. In our first smoke test, one of
five checked plans diverged by 0.14 in max |Δqpos|. Replays became exact once
the warm start and the cached pose were restored too
(`tests/test_cube_core.py::test_mid_episode_replay_is_exact`).

### 3.2 Uncertainty signals

- **MC-dropout (`mc`).** The pretrained predictor (dropout 0.1) runs with
  dropout active and normalisation statistics frozen, using 8 masks. σ²_k is
  the across-mask variance of ẑ_k, summed over latent dimensions.
- **Latent ensemble (`ens`, `ua/ensemble.py`).**
  - The pretrained encoder is frozen, and the latents of all dataset frames are
    computed once.
  - Five predictor heads with the LeWM architecture are trained, each from a
    different random initialisation on a random 80% of the training episodes.
  - Each head trains for 12,000 steps (batch 512, AdamW, lr 3·10⁻⁴).
  - The loss is the teacher-forced next-latent loss plus a 5-block open-loop
    rollout loss, so the members train on the rollout the planner uses.
  - σ²_ens is the members' disagreement on a candidate. The planner's own
    predictions and costs are unchanged.

### 3.3 Pre-registered questions

A plan is *non-trivial* when the cube is not yet within 4 cm of the goal at
planning time; only these plans enter Q1–Q6.

| | question |
|---|---|
| Q1 | Does the LeWM cost rank candidates by their real outcome (Spearman with final distance; AUROC for failure)? |
| Q2 | Does σ²_k track the true prediction error at each block k? |
| Q3 | Among the elites, does σ²_H separate the candidates that fail (the false-positive question)? Does σ² add to the cost in a cross-validated logistic model? |
| Q4 | Does σ² of the executed first plan predict episode failure, beyond cost and start distance? |
| Q5 | Re-ranking: choose argmin(cost + λσ²_H) among the replayed candidates. λ is selected on 3 calibration seeds and reported on 9 test seeds against λ = 0 (McNemar). |
| Q6 | When an executed plan fails, did a replayed alternative succeed (*selection failure*) or did none (*search failure*)? |

All intervals come from a bootstrap that resamples whole planning calls (or
episodes), never single candidates. Each AUROC is reported twice:
- **pooled** over all candidates;
- **within-plan**: the mean over planning calls with mixed outcomes.

**Verdict rule** (fixed before the analysis):
- **GO** if the Q3 AUROC interval excludes 0.5 and Q5 improves on the test
  seeds;
- **WEAK-SIGNAL** if only Q3 holds;
- **NO-GO** otherwise.

### 3.4 Uncertainty-aware planners on Cube (L2)

| planner | objective | Cube parameters |
|---|---|---|
| baseline | ‖ẑ₅ − z_g‖², one deterministic rollout | – |
| MC-5 | ‖mean of 5 MC-dropout rollouts − z_g‖² | – |
| gated | MC-5; a candidate whose first-block σ²₁ exceeds a threshold is scored at block k* = clamp(round(5 − ω σ²₁), 1, 5) instead of 5 | threshold = median planning-time σ²₁ (0.209), from 16 separate episodes (start seed 1910, disjoint from all panels); ω = 1.21 (ω × threshold as on PushT) |
| Goal-CVaR | MC-5 + elite-calibrated CVaR soft penalty | α = 0.1 |
| σ² penalty (MC) | MC-5 + λ σ²_H | λ = 6.74 |
| σ² penalty (ensemble) | LeWM cost + λ σ²_ens,H | λ = 5.37 (Q5 selection, 3 × median cost / median σ²) |

The MC penalty's λ needs a note: Q5 selected λ = 0 for MC-dropout, so the
pre-specified equal-scale value (median cost / median σ²) was used. This keeps
the comparison non-trivial.

### 3.5 Learned risk head (L3)

**Training data.** The candidates replayed on 16 training seeds: 54,825
candidates from 1,075 planning calls.

**Targets.** Three per candidate: the true terminal prediction error, the true
terminal cost, and segment success.

**Inputs.** Only features the planner already has: z₀, z_goal, the predicted
ẑ₅, the LeWM cost and the MC-dropout σ² profile. A second head without σ²
(`det`) measures whether σ² carries information a learned model can exploit.

**Use in planning.** The planner scores either J = cost + β·ê (`err` mode) or
J = ĉ_true. Head, mode and β were selected on the calibration seeds (142 non-trivial
plans): the `det` head, `err` mode, β = 0.5.

### 3.6 Representation variants (L4)

All variants share four settings:
- warm-started from the pretrained LeWM-Cube;
- trained for 20,000 steps;
- the LeWM recipe: batch 128, AdamW 5·10⁻⁵, bf16, SIGReg 0.09, frameskip 5;
- an episode-level train/validation split.

The `lewm` control is the pretrained model trained further with its original
loss, so each comparison isolates one change.

Factorised models keep the pretrained predictor as their core, behind linear
maps from and to [z_vis ; z_prop]. These maps start at [I | 0] and [I ; 0], so
at step 0 the visual prediction equals the pretrained model's (checked
numerically).

| variant | change |
|---|---|
| `lewm` | control: LeWM retrained under our budget |
| `ms` | + 5-block open-loop rollout loss (the planner's rollout) |
| `prop` | z = [z_vis ; z_prop], z_prop = MLP(proprioception, 19-d); a penalty (weight 0.5) on the share of z_vis variance that is linearly explained by the stop-gradient z_prop |
| `prop_aux` | + training-only cube-pose head on z_vis (position, cos 4θ, sin 4θ: the cube is 90° symmetric), weight 0.1 |
| `prop_ms` | all of the above |

Variants with a proprio block plan with the block-weighted cost
‖Δz_vis‖² + w_prop‖Δz_prop‖², with w_prop ∈ {1, 0.1, 0} selected on the
calibration seeds.

### 3.7 Representation evaluation (L5)

- **Probes.** Linear (ridge) and MLP probes are trained on 24,000 frames of
  held-out episodes and tested on 6,000 more. Targets: cube position, cube yaw
  (cos 4θ, sin 4θ), joint positions and velocities, end-effector position and
  gripper opening. We report Pearson r averaged over target dimensions, as in
  LeWM's Table 4.
- **Open-loop prediction.** We run the planner's rollout (1 context frame,
  k = 1…5 blocks) on held-out dataset sequences and report two errors:
  - the latent error divided by the total latent variance, against a
    "nothing moves" persistence baseline;
  - the cube position decoded from ẑ_k by the linear probe, in centimetres,
    which is comparable across models.
- **Planning.** The 9-seed protocol, with each variant as the cost model.

## 4 Experimental setup

- **Data and checkpoint.** `quentinll/lewm-cube`: the `cube_single_expert`
  dataset (10,000 expert episodes of 201 steps) and the paper's LeWM-Cube
  model.
- **Seeds**, each with 50 episodes; for a given seed, every method sees the
  same start and goal states:

  | role | seeds |
  |---|---|
  | test | 42–45, 1011, 2022, 3033, 4044, 5055 (the PushT panel) |
  | calibration | 910–912 |
  | risk-head training | 100–115 |

- **Statistics.**
  - Paired exact McNemar tests on episodes.
  - Holm correction across the L2/L3 planner family [Holm, 1979].
  - Non-trivial episodes are also reported separately.
- **Compute.** NUS SoC cluster, one GPU per job; commands are in
  `cluster/README.md`.
  - Rendering jobs (planning and replays) ran on whole A100, H100 or H200 GPUs.
  - Training jobs ran on H100 MIG slices and A100s.
  - Every job is resumable and skips finished runs.

## 5 Results

### 5.1 Protocol check

| quantity | value |
|---|---|
| Baseline success, 9 test seeds × 50 episodes | **65.6 ± 6.0%** (pooled 95% CI 61.1–69.8; per seed 58–80) |
| Reported in the paper | 74% |
| Episodes whose cube starts within 4 cm of the goal (12 recorded runs, 600 episodes) | 36.8% (they succeed 99.5% of the time) |
| Success on non-trivial episodes (same 600 episodes) | 46.2% |
| Replay fidelity | max \|Δqpos\| = **0** over 412 executed plans (207 start-of-episode, 205 mid-episode) |
| Oracle outcome = closed-loop outcome | 600 / 600 episodes |

Our baseline is below the published 74%, which lies outside our interval.
- We use the released checkpoint and the released evaluation configuration.
- Per-seed variation is large: 50-episode runs range from 58% to 80%.
- The cause of the gap is open; §5.6 tests one candidate.
- All comparisons below are paired within our own panel.

### 5.2 Is σ² a useful risk signal? (L1)

The analysis covers 12 recorded baseline runs (3 calibration + 9 test seeds;
600 episodes), 807 planning calls and 41,157 replayed candidates. Of these, 586
planning calls are non-trivial.

**Q1. The cost ranks problems, not candidates.**
- Over all candidates, the LeWM cost correlates with the final cube distance
  (Spearman 0.39, CI 0.34–0.44) and predicts failure (AUROC 0.64, CI
  0.61–0.67).
- Within a planning call the correlation vanishes (mean Spearman 0.03; AUROC
  0.56 on the 180 calls with mixed outcomes).

So the cost mostly tells hard start states from easy ones.

**Q2. σ² tracks the error at short horizon only.** Spearman between σ²_k and
‖ẑ_k − z_k‖², pooled with plan-bootstrap 95% CI; within-plan means in brackets:

| block k | MC-dropout | latent ensemble |
|---|---|---|
| 1 | 0.555 [0.508, 0.605] (0.28) | 0.686 [0.645, 0.722] (0.42) |
| 2 | 0.452 [0.394, 0.502] (0.17) | 0.592 [0.548, 0.638] (0.27) |
| 3 | 0.358 [0.297, 0.412] (0.10) | 0.517 [0.467, 0.563] (0.16) |
| 4 | 0.258 [0.189, 0.315] (0.06) | 0.400 [0.323, 0.453] (0.08) |
| 5 = H | 0.032 [−0.029, 0.092] (0.04) | 0.084 [0.007, 0.151] (0.06) |
| σ²_H vs \|cost − true cost\| | 0.040 [−0.023, 0.103] | 0.092 [0.012, 0.157] |

The ensemble is the better signal at every block. At the final block, where
the cost is evaluated, both correlations are close to zero. The same holds for
the error of the cost itself (Figure 1).

The size of the error matters as much as its ranking.
- On the planner's candidates, the median error at block 5 is 207 for the
  pretrained predictor and 206 for the ensemble mean.
- On held-out dataset sequences, the same predictor's mean 5-block error is
  8.3: 0.044 of the total latent variance (§5.5).
- On the candidates, the error is about 1.1 times the total latent variance.
  §5.6 checks how much of this comes from live rendering.

![Figure 1](../results/cube/l1/fig_sigma_vs_error.png)

*Figure 1. Terminal σ² vs the true terminal prediction error for replayed
candidates (red: elites). Left: MC-dropout; right: latent ensemble.*

**Q3. Within a call, σ² does not flag the failing elites.** 71.4% of the
elites of non-trivial plans fail; 158 calls have elites of both outcomes.

| score | pooled AUROC (failure) [95% CI] | within-plan AUROC |
|---|---:|---:|
| LeWM cost | 0.624 [0.588, 0.661] | 0.535 |
| MC σ²_H | 0.628 [0.592, 0.662] | 0.492 |
| ensemble σ²_H | 0.618 [0.584, 0.657] | 0.524 |
| MC σ²₁ | 0.453 [0.415, 0.494] | 0.523 |
| MC cost spread | 0.634 [0.600, 0.668] | 0.518 |

Adding σ²_H to the cost in a cross-validated logistic failure model changes the
AUROC by +0.003 for MC-dropout (CI −0.004 to 0.012) and by −0.002 for the
ensemble (CI −0.014 to 0.011).

**Q4. At the episode level, σ² flags a group of hopeless problems.** Over 379
non-trivial episodes (53.8% fail), the MC σ²_H of the executed first plan
predicts failure about as well as the start distance:

| predictor | AUROC | 95% CI |
|---|---:|---|
| MC σ²_H | 0.60 | 0.55–0.66 |
| start distance | 0.61 | 0.56–0.67 |
| LeWM cost | 0.57 | – |

Its distribution has a high-σ² mode (σ²_H ≳ 2.5) made up almost only of
failures (Figure 2). On top of cost and start distance:
- MC-dropout adds nothing (cross-validated AUROC 0.553 → 0.548);
- the ensemble adds a little (0.553 → 0.593).

![Figure 2](../results/cube/l1/fig_episode_sigma.png)

*Figure 2. MC-dropout σ²_H of the executed first plan, for non-trivial episodes
that succeed or fail.*

**Q5. Re-ranking does not help.**
- *MC-dropout.* The calibration seeds select λ = 0.
- *Ensemble.* They select λ* = 5.37. On the 444 non-trivial test plans this
  moves success of the re-ranked choice from 29.3% to 29.7% (14 plans gained,
  12 lost; p = 0.85).
- *Upper bound.* For comparison, the executed plans succeed 30.0% of the time,
  and an oracle that picks the best replayed candidate would reach 41.2%.

**Q6. Most failures are search failures.**

| plans (non-trivial) | n | executed plan succeeds | selection failures | search failures | cost gap, failed | cost gap, succeeded |
|---|---:|---:|---:|---:|---:|---:|
| all | 586 | 29.9% | 16.8% | **83.2%** | 252 | 45.5 |
| first plan | 379 | 45.6% | 29.1% | 70.9% | 236 | 45.5 |
| second plan | 207 | 1.0% | 4.4% | 95.6% | 266 | 81.2 |

- A failure is a *selection failure* if another replayed candidate (of 51)
  succeeds.
- The *cost gap* is the true minus the predicted cost of the executed plan
  (median).
- Second plans start after a failed first plan and almost never recover.
- For executed plans that fail, the model is far more optimistic than for
  those that succeed.

**Verdicts** (by the pre-registered rule):
- **MC-dropout: WEAK-SIGNAL.** The Q3 interval excludes 0.5; Q5 does not
  improve.
- **Latent ensemble: GO.** The Q3 interval excludes 0.5, and the Q5 test
  success improves, but by only 0.45 points (2 of 444 plans; not
  significant).

The rule used the pooled Q3 AUROC. The within-plan AUROC, which is what
re-ranking depends on, is ≈ 0.5 for both signals (§6.1).

**The ensemble itself.** On held-out episodes, the ensemble mean predicts
better than the pretrained predictor at every block. Its squared error over
blocks 1–5 is 0.95, 2.10, 3.67, 5.55, 7.17, against 1.05, 2.39, 4.31, 6.51,
8.31 for the pretrained predictor; single members are close to the pretrained
predictor. On these in-distribution sequences, σ²_ens tracks the
ensemble-mean error with Spearman 0.35–0.46 per block. Training took 95
minutes for all five members.

### 5.3 Uncertainty-aware planners in closed loop (L2, L3)

Paired against the baseline on the same start and goal states:

| planner | seeds | success (%) | Δ (pp) | gained / lost | McNemar p | Δ non-trivial (pp) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 9 | 65.6 ± 6.0 | – | – | – | – |
| MC-5 | 8† | 65.8 ± 5.6 | +0.25 | 18 / 17 | 1.00 | +0.0 |
| **gated** | 8† | **70.8 ± 5.1** | **+5.25** | **32 / 11** | **0.002** | **+8.1** (p = 0.003) |
| Goal-CVaR | 7† | 67.4 ± 6.4 | +0.86 | 19 / 16 | 0.74 | +0.9 |
| σ² penalty (MC) | 7† | 68.9 ± 6.9 | +2.29 | 18 / 10 | 0.18 | +3.7 |
| σ² penalty (ensemble) | 7† | 70.0 ± 6.8 | +3.43 | 23 / 11 | 0.058 | +5.1 |
| risk head (L3) | 9 | 66.2 ± 7.1 | +0.67 | 30 / 27 | 0.79 | +0.7 |

- Δ is computed on the seeds that both planners have.
- **Noise floor.** MC-5 flips 35 of 400 episodes, in both directions, for no
  net gain. Small changes to the objective reshuffle outcomes episode by
  episode, which is why the panel uses 9 seeds.
- **Gated truncation** is the only significant effect. Holm-adjusted over the
  six planners, p ≈ 0.011. It raises success in 5 of 8 seeds (+6 to +14
  points) and lowers it slightly in 3 (−2 to −4).
- **Truncation is rare.** The gated planner's mean scored horizon is 4.86
  blocks (4.73–4.94 per seed), so most candidates are still scored at block 5.
- **The penalties do steer the plans.** The MC σ² penalty lowers the mean σ² of
  the scored candidates from 0.93 (MC-5) to 0.74. Neither penalty's success
  change is significant.

### 5.4 The learned risk head (L3, offline)

| head | Spearman with true error (calibration / test) | AUROC failure (calibration / test) |
|---|---:|---:|
| with σ² (`full`) | 0.553 / 0.505 | 0.879 / 0.891 |
| without σ² (`det`) | 0.558 / 0.449 | 0.883 / 0.873 |
| LeWM cost | – | 0.637 / 0.643 |

- **Prediction.** The head predicts failure far better than the LeWM cost.
  Without σ² it is as good on the calibration seeds and slightly worse on the
  test seeds. So σ² adds a little to what a learned model extracts from the
  latents and the cost.
- **Re-ranking.** It still does not help. On the calibration seeds every head,
  mode and β gives between 26.8% and 29.6%, against 28.9% for the LeWM cost
  alone. On 444 test
  plans the risk-aware choice succeeds 29.3% of the time, exactly like the
  LeWM-cost choice (13 plans gained, 13 lost).
- **Closed loop.** The risk planner gains 0.7 points (p = 0.79; §5.3).

### 5.5 Representations (L4, L5)

**Training.** All five variants trained for 20,000 steps from the pretrained
model, taking 53–111 minutes each on one GPU. The training curves are plotted
in `results/cube/l4/fig_training.png` when `RESULTS.md` is generated (†).

**Probes** (Pearson r, linear / MLP, held-out episodes):

| checkpoint | dim | cube position | cube yaw (4θ) | joint velocity | effector position | gripper |
|---|---:|---:|---:|---:|---:|---:|
| pretrained LeWM | 192 | 0.996 / 0.999 | 0.19 / 0.36 | 0.04 / 0.08 | 0.989 / 0.999 | 0.96 / 0.99 |
| `lewm` (control) | 192 | 0.995 / 0.999 | 0.23 / 0.36 | 0.04 / 0.07 | 0.988 / 0.999 | 0.96 / 0.99 |
| `ms` | 192 | 0.997 / 0.999 | 0.02 / 0.21 | 0.05 / 0.08 | 0.992 / 0.999 | 0.96 / 0.99 |
| `prop` | 224 | 0.972 / 0.999 | 0.22 / 0.36 | 0.44 / 0.89 | 0.985 / 0.999 | 0.97 / 1.00 |
| `prop_aux` | 224 | 0.973 / 0.999 | **0.97 / 0.95** | 0.44 / 0.89 | 0.985 / 0.999 | 0.97 / 1.00 |
| `prop_ms` | 224 | 0.983 / 0.999 | 0.47 / 0.51 | 0.38 / 0.83 | 0.990 / 0.999 | 0.97 / 1.00 |

What the latents encode:
- **Pretrained LeWM.** Its latent encodes cube position, effector position and
  gripper opening linearly, but hardly the cube's yaw or the arm's velocity.
  Joint positions reach linear r ≈ 0.66–0.70 for every model, but R² ≤ 0.15
  (mostly ≤ 0), so they are not decodable in a useful sense.
- **The proprio block** makes joint velocity decodable, non-linearly (MLP
  r 0.83–0.89).
- **The cube-pose head** makes yaw linearly decodable (0.97).
- **The visual block alone (z_vis).** Probed by itself, it loses the linear
  cube position in the factorised models:
  - linear r = 0.29 (`prop`), 0.47 (`prop_aux`) and 0.74 (`prop_ms`);
  - the MLP still reaches 0.999 in all three;
  - effector position drops in the same way (linear 0.27 / 0.41 / 0.70).

**Open-loop prediction** on held-out dataset sequences:

| checkpoint | rel. latent error k = 1 | k = 5 | persistence k = 5 | cube error from ẑ₅ [cm] (from true z₅) |
|---|---:|---:|---:|---:|
| pretrained LeWM | 0.005 | 0.044 | 1.335 | 1.57 (1.42) |
| `lewm` (control) | 0.003 | 0.034 | 1.317 | 1.51 (1.50) |
| `ms` | 0.002 | **0.010** | 1.301 | **1.19** (1.13) |
| `prop` | 0.004 | 0.036 | 1.365 | 2.36 (3.39) |
| `prop_aux` | 0.004 | 0.034 | 1.340 | 2.28 (3.37) |
| `prop_ms` | 0.003 | 0.011 | 1.306 | 1.97 (2.74) |

- **The multi-step loss** cuts the 5-block latent error 3.2-fold against the
  control (4.2-fold against the pretrained model). It also lowers the decoded
  cube-position error.
- **Factorised models** have larger cube errors even from the true latent.
  A linear probe recovers the cube position less precisely from their latent
  (r 0.97–0.98, against 0.996).

**Planning** (9 test seeds; w_prop selected on calibration seeds):

| checkpoint | w_prop | success (%) | Δ vs pretrained (p) | Δ vs `lewm` control (p) |
|---|---:|---:|---:|---:|
| pretrained LeWM | – | 65.6 ± 6.0 | – | −1.1 (0.61) |
| `lewm` (control) | – | 66.7 ± 7.3 | +1.1 (0.61) | – |
| `ms` | – | 68.0 ± 5.8 | +2.4 (0.17) | +1.3 (0.54) |
| `prop` | 0 | 61.6 ± 7.9 | **−4.0 (0.027)** | **−5.1 (0.007)** |
| `prop_aux` | 1† | pending† | | |
| `prop_ms` | 0.1† | pending† | | |

- **w_prop selection.** Over the three calibration seeds, the w_prop values
  differ by at most 2.7 points for every variant. The values used for
  `prop_aux` and `prop_ms` (1 and 0.1) come from those calibration runs.
- **Results.** Retraining alone changes nothing, and the multi-step loss gives
  a small, non-significant gain. The proprio factorisation hurts.

### 5.6 Diagnostic: live-rendered vs dataset frames†

The evaluation gives the first plan dataset frames, for both the start and the
goal; every later observation is rendered live, and so are the replays. A
difference between live and dataset frames, as the encoder sees them, would
add to the measured prediction errors in §5.2 and would hurt second plans.

`scripts/cube/check_domain_gap.py` measures this gap on the evaluation start
states of two test seeds. *Result pending.*

## 6 Discussion

### 6.1 Is the signal uninformative, or is it used poorly?

**Mostly the former, in the sense that matters for the planner.** Both
signals measure something real.
- Their σ² at the first block ranks the model's errors (Spearman 0.55 and
  0.69).
- A high-σ² executed plan almost always fails (Figure 2).

But CEM's decision is a choice among the elites of one call, and there σ²
carries no information.
- Within-plan AUROCs are 0.49–0.52.
- σ²_H barely tracks the terminal error (0.03 and 0.08), and the error of the
  cost no better.

The pooled AUROCs of 0.62–0.63 look like a usable signal, but they mostly say
that hard start states have higher σ² and fail more often. The LeWM cost shows
the same between-problem pattern (pooled 0.64, within 0.56).

The learned risk head confirms this from another side. It detects failure much
better than either signal (0.88), with or without σ², yet it cannot choose a
better plan.

In hindsight, our pre-registered verdict rule should have used the within-plan
AUROC. Under the rule as written, the ensemble's GO rests on a 0.45-point,
non-significant Q5 gain.

### 6.2 Search failures dominate, and the model is optimistic exactly where it fails

**Selection is not the main problem.** Only 17% of failed plans had a
successful alternative among the replayed candidates, and only 4% of failed
second plans. Even a perfect selector over the replayed candidates would raise
non-trivial plan success only from 30% to 41%. This caps what any scoring
change can achieve through re-ranking.

**The model is over-optimistic about its failures.** For failed executed
plans, the true cost exceeds the predicted one by a median of 252, against 45.5
for successes. This is the optimizer's curse
[Smith & Winkler, 2006]: CEM picks the lowest predicted costs, and these are
disproportionately the most underestimated.

**The candidates lie where the model is least accurate.** At block 5:
- on the planner's candidates the error is ≈ 110% of the total latent
  variance;
- on held-out dataset sequences it is 4%.

A likely reason is the sampling. CEM draws each action independently from a
Gaussian in the normalised action space, so its candidates need not resemble
the smooth expert motions the model was trained on. We did not measure this
directly.

**A caveat.** Part of this gap could come from rendering, if live frames
differed from dataset frames as the encoder sees them. The diagnostic in §5.6
separates the two explanations.

### 6.3 Why does gating help when the penalties do not?

Every L2 planner changes CEM's scoring rule. The plan is the mean of the final
elites, so each of them acts on the search. Only gating helps: 70.8% against
65.8% for MC-5. It differs from MC-5 only in how a minority of candidates are
scored (the mean scored horizon is 4.86 blocks).

Two features of its rule fit the data.
1. **It reads σ² where σ² is informative.** The gate uses σ² at the first
   block, where σ² tracks the prediction error best (Q2: 0.55, against 0.03
   at block 5). The two penalties use σ²_H.
2. **It rewards plans that arrive early.** Scoring an uncertain candidate at
   an earlier block compares an intermediate prediction with the goal. This
   favours candidates predicted to reach the goal early, which leaves margin
   for prediction error later in the plan.

The records do not store intermediate-block costs, so we could not replay the
gated score on the recorded candidates. A recorded gated run with replays
would separate the two explanations, and would show whether the gate finds
better candidates or chooses better among them.

### 6.4 Better prediction is not better planning

**Multi-step training does what it targets but barely moves planning.** The
5-step loss improves the planner's own rollout on dataset sequences 3.2-fold
and lowers the decoded cube error, yet success rises by only 1.3 points over
the control. This is consistent with §6.2: the error that matters is the one
on the planner's candidates. That error is far larger (median 207, against a
mean of 8.3 on dataset sequences) than the in-distribution error this loss
reduces. §5.6 checks how much of the difference comes from rendering.

**The proprio factorisation hurts for a specific reason.**
- The de-redundancy penalty pushes z_vis to drop variance that the arm state
  explains linearly.
- In pick-and-place data the cube moves with the gripper, so its position is
  partly predictable from the arm state.
- The penalty therefore also removes the *linear* cube-position directions from
  z_vis: linear r drops from 0.996 to 0.29, while the MLP r stays at 0.999.
- With w_prop = 0, the planning cost is a squared distance in z_vis, so it no
  longer sees the cube position linearly, and success drops by 5 points.
- The cube-pose head restores yaw (0.97) and part of the linear position
  (0.47); adding the multi-step loss restores more (0.74). Their planning
  results are pending.

**What the representation lacks matters less than how it is searched.** The
proposal's motivation was that LeWM's latent "misses rotation and motion",
and the probes confirm this (yaw 0.19, joint velocity 0.04). Adding that
information did not help planning, at least not in the form we added it.

### 6.5 What would help

The evidence points to the search distribution and to the data the model sees,
not to the uncertainty signal.
- **Keep the search near the data.** Start CEM from a goal-conditioned BC prior
  (GCBC alone reaches 84% on Cube [Argenson & Dulac-Arnold, 2021; Hansen et
  al., 2022]) or sample temporally correlated noise [Pinneri et al., 2020].
  This addresses the search failures and the off-distribution error at once.
  It was the third lever of our proposal, and we did not reach it.
- **Train on the planner's own candidates.** The 41k replays are exactly such
  data, with true latents and outcomes, in the spirit of DAgger
  [Ross et al., 2011].
- **Use σ² for what it can do.** It flags problems the planner will not solve
  (Figure 2), which suits abstention, hand-off or more frequent replanning
  rather than plan selection.

## 7 Limitations

- **One environment, one checkpoint.** L4 variants were warm-started and
  trained for 20,000 steps, which may be below what some changes need.
- **The replays cover 51 of the 300 final candidates per call.** "Search
  failure" means that none of these succeeds. The oracle upper bound is
  therefore a lower bound on what a full re-ranking could reach.
- **Live rendering.** True latents come from live-rendered frames, and a
  rendering gap would inflate the measured errors (§5.6).
- **Incomplete runs.** At the time of writing, some L2 planners have 7–8 of
  the 9 test seeds (†).
- **Multiple comparisons.** p-values are raw McNemar values, plus a Holm
  adjustment for the planner family. The representation comparisons are not
  adjusted.
- **The gating mechanism** is inferred, not measured (§6.3).
- **Baseline gap.** Our baseline is below the published number, and we have
  not identified why.

## 8 Conclusion

On OGBench-Cube, uncertainty does not help LeWM's planner choose between plans,
because the choice is not where it fails.
- MC-dropout and ensemble variance track the model's short-horizon errors. They
  identify hard problems, but not the failing plans within a problem.
- 83% of failed plans had no successful alternative among the replayed
  candidates.
- On the candidates the search proposes, the model's errors are more than an
  order of magnitude larger than on dataset sequences (a rendering check is
  pending, §5.6).

The one uncertainty-aware planner that helps, gated truncation (+5 points),
reads σ² at the first block, where σ² is informative; its mechanism remains
to be isolated. Better open-loop prediction
on dataset sequences does not translate into planning success. The next lever
is the search distribution: keeping candidates near the data, or training the
model on the candidates the planner actually proposes.

## References

- L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, R. Balestriero. *LeWorldModel*. arXiv:2603.19312, 2026.
- G. Zhou, H. Pan, Y. LeCun, L. Pinto. *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*. ICML 2025.
- S. Park, K. Frans, B. Eysenbach, S. Levine. *OGBench: Benchmarking Offline Goal-Conditioned RL*. ICLR 2025.
- K. Chua, R. Calandra, R. McAllister, S. Levine. *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*. NeurIPS 2018.
- T. Yu, G. Thomas, L. Yu, S. Ermon, J. Zou, S. Levine, C. Finn, T. Ma. *MOPO: Model-based Offline Policy Optimization*. NeurIPS 2020.
- Y. Gal, Z. Ghahramani. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML 2016.
- B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS 2017.
- J. Wu, Z. Huang, C. Lv. *Uncertainty-Aware Model-Based Reinforcement Learning: Methodology and Application in Autonomous Driving*. IEEE T-IV, 2022.
- M. Janner, J. Fu, M. Zhang, S. Levine. *When to Trust Your Model: Model-Based Policy Optimization*. NeurIPS 2019.
- R. T. Rockafellar, S. Uryasev. *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2000.
- R. Balestriero, Y. LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*. 2025.
- J. E. Smith, R. L. Winkler. *The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis*. Management Science, 2006.
- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure*. Scandinavian Journal of Statistics, 1979.
- C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek, G. Martius. *Sample-efficient Cross-Entropy Method for Real-time Planning*. CoRL 2020.
- A. Argenson, G. Dulac-Arnold. *Model-Based Offline Planning*. ICLR 2021.
- N. Hansen, X. Wang, H. Su. *Temporal Difference Learning for Model Predictive Control*. ICML 2022.
- S. Ross, G. Gordon, D. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011.
