# Step 1 pre-registration: does σ² separate failures on Cube?

Fixed on 2026-10-16, before any Step-1 data was collected. Code:
`scripts/step1_*.py`, `ua/step1/`. Results go to `results/cube/step1/`.

## Setup

- Model: released LeWM checkpoint `checkpoints/cube`, frozen.
- Planner: baseline LeWM with CEM 300 samples / 10 iterations / top-30.
  Horizon 5 blocks, receding horizon 5, action block 5, budget 50 env steps.
  The goal is the expert state 25 env steps later (`config/eval/cube.yaml`).
- Non-trivial protocol: keep only start states whose initial cube–goal
  distance is > 4 cm. Success means the cube is within 4 cm of the goal.
- Validation seeds: 61–68, 50 episodes each, giving 400 non-trivial episodes.
  These are disjoint from the locked-test seeds (42–45, 1011–5055).
- Replay states: 200 replan states taken from replans where the episode was
  not yet successful. 100 come from episodes that end in failure and 100 from
  episodes that end in success, sampled with `numpy.random.default_rng(0)`.
- Candidate pools per replay state: all 300 candidates of CEM iteration 1
  ("first") and all 300 of iteration 10 ("last"). Each candidate is executed in
  the simulator from the saved MuJoCo state.
- Every executed plan of every replan is also replayed.

## Quantities

- ẑ_k: deterministic base-model prediction after k action blocks (k = 1..5).
- z_k: encoder embedding of the real frame after k blocks.
- z_g: goal embedding. z_0: embedding of the current observation.
- e_k = ‖ẑ_k − z_k‖² (real prediction error).
- d_pred = ‖ẑ_5 − z_g‖², d_real = ‖z_5 − z_g‖², d_0 = ‖z_0 − z_g‖².
- σ²_k (MC-dropout) is the sum over latent dimensions of the unbiased variance
  across N = 20 dropout rollouts. Dropout is 0.1 and normalization layers are
  frozen. This is the primary estimate; N = 5 is reported as a sensitivity check.
  "σ²" without a subscript means σ²_5, the step that enters the cost.
- σ²_ens is the same quantity across M = 5 predictors (see Ensemble).
- o(c), the task outcome of a replayed candidate: max(min_t ‖cube_t − goal‖ − 0.04, 0)
  over its 25 executed steps, in metres. 0 means it reached the goal.

## Criteria

These are applied separately to each estimator X ∈ {MC-dropout, ensemble}.

**C1 — within-state ρ.** For each replay state and each pool, compute the
Spearman correlation between σ²_X and e_5 over the pool's candidates. The
per-state ρ_s is the mean of the first-pool and last-pool values; an undefined
ρ counts as 0.
Pass: median_s ρ_s ≥ 0.30 **and** the fraction of states with ρ_s > 0 is ≥ 0.70.

**C2 — headroom.** Use the last-iteration pool.
- Baseline pick = argmin d_pred.
- Oracle pick = argmin o.
- Penalty pick = argmin z(d_pred) + λ·z(σ²_X), with z-scores taken within the pool.

λ ∈ {0.1, 0.25, 0.5, 1, 2, 4} is chosen by 2-fold cross-fitting over states:
pick λ on one half, score it on the other.
Share = Σ_s [o(base) − o(pen)] / Σ_s [o(base) − o(oracle)].
The 95% CI is a bootstrap over states (2000 reps), with the split and the λ
choice redone inside every rep.
Pass: share ≥ 0.20 **and** the CI lower bound is > 0.

**C3 — per-replan AUROC.**
- Units: replans in non-trivial episodes that were not yet successful at that replan.
- Label: segment failure, i.e. the cube never comes within 4 cm during the 25
  steps after the replan.
- Score: σ²_X of the executed plan.
- Gain over cost-only: ΔAUROC = AUROC(logit[log d_pred, log d_0, log σ²_X])
  − AUROC(logit[log d_pred, log d_0]). Scores are out-of-fold from 5-fold
  CV grouped by episode. The 95% CI is a bootstrap over episodes (2000 reps).

Pass: AUROC(σ²_X) ≥ 0.65 **and** the ΔAUROC CI lower bound is > 0.

**Gate G — share of category (a).** Classify every failed non-trivial episode.
τ is the 90th percentile of d_real over all replays whose final cube–goal
distance is ≤ 4 cm. The first rule that applies wins:
- (a) model exploited: some replan has d_pred ≤ τ < d_real;
- (b) cost misaligned: some replan has d_real ≤ τ;
- (d) ran out of time: the final cube–goal distance is ≤ 50% of the initial one;
- (c) search failed: everything else.

Pass: the share of (a) is ≥ 0.20.

## Decision

1. If G fails → Branch B.
2. Else if MC-dropout passes C1, C2 and C3 → GO: Branch A with MC-dropout σ².
3. Else if the ensemble passes C1, C2 and C3 → Branch A with ensemble σ².
4. Else → Branch B.

The following are reported but cannot change the decision:
- MC-dropout with N = 5;
- first-pool and last-pool ρ separately;
- episode-level AUROC (max σ² over replans → episode failure);
- errors of the ensemble-mean prediction;
- the share of headroom captured by the latent oracle (argmin d_real).

## Ensemble

- 5 members. Each is a predictor, action encoder and pred_proj with the released
  architecture, randomly initialised with seed m.
- Training: the LeWM 1-step teacher-forcing loss on frozen-encoder latents.
  Each member uses a bootstrap resample of 3,000 training episodes, which
  include all 400 Step-1 episodes.
- 200 held-out episodes give each member's 1-step validation MSE, which is
  compared with the released predictor.
- The training budget may be tuned on that validation MSE only, never on any
  Step-1 metric.
