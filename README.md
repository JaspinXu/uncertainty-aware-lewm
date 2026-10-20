# Uncertainty-Aware Planning for Frozen LeWorldModel

Course project built on top of [LeWorldModel](https://le-wm.github.io/)
(Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026; [paper](https://arxiv.org/pdf/2603.19312v1),
[checkpoints](https://huggingface.co/collections/quentinll/lewm)). The pretrained LeWM
checkpoint (`checkpoints/pusht/`) and the CEM planner are kept fixed; only the
trajectory-scoring function changes. The comparison below is on PushT; the
Cube Step-1 section further down tests σ² on OGBench Cube (`checkpoints/cube/`).

Methods compared (all n=50 episodes, CEM 300 candidates / 30 iterations / top-30,
receding horizon 5):

| Name | Config | Scoring rule |
|---|---|---|
| Baseline | `method=baseline` | one deterministic rollout, terminal latent distance to goal |
| MC-5 | `method=mc5` | mean of 5 MC-dropout rollouts, terminal distance |
| Gated N=5 | `method=gated` | MC-5 + horizon truncation when step-0 σ² exceeds a calibrated threshold |
| Goal-CVaR α=0.1 | `method=goal_cvar` | MC-5 distance + CVaR soft penalty on the worst dropout samples |

## Project layout

| Path | Role |
|---|---|
| `evaluate.py` | The one evaluation entry point: `python evaluate.py method=<name> seed=<n>` |
| `planners.py` | Builds the planner for `cfg.method` (baseline / `UncertaintyAwareJEPA` / `GoalCVaRJEPA`) |
| `ua/` | Library: `uncertainty.py` (MC-dropout rollouts, σ²), `truncation.py` (gated horizon rule), `ua_jepa.py` (MC aggregation planner), `goal_cvar.py` (soft-penalty planner), `calibrate.py` |
| `config/eval/pusht.yaml` | Shared protocol (dataset, start-state sampling, CEM budget, replanning) |
| `config/eval/method/*.yaml` | One file per method: `baseline`, `mc5`, `gated`, `goal_cvar` |
| `scripts/run_single.sh` | One run (method × seed) → `results/<method>_s<seed>.txt`, skipped if complete |
| `scripts/run_locked_panel.sh` | The 4 methods × 9 seeds panel, then the summary table |
| `scripts/summarize_locked_9seeds.py` | Reads `results/*.txt`, prints the table, writes `results/combined_9seeds.md` |
| `scripts/calibrate_gated_threshold.py` | Gated-threshold calibration → `results/calibration/` |
| `results/` | 36 result files behind the final table, `combined_9seeds.md`, `calibration/gated_lewm_strong_rh5.json` |
| `train.py`, `jepa.py`, `module.py`, `utils.py`, `config/train/`, `evaluate_wm.py` | LeWM training code (unchanged from upstream) |
| `stable-worldmodel/`, `stable-pretraining/` | Upstream dependencies (env, planner, trainer), git submodules |

## Environment

`stable-worldmodel/` and `stable-pretraining/` are git submodules pinned to the
upstream commits used for all experiments:

```bash
git clone --recurse-submodules <this repo>
cd uncertainty-aware-lewm
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export STABLEWM_HOME="$PWD"     # checkpoints/ and datasets/ live here
export PYTHONPATH="$PWD:$PWD/stable-worldmodel:$PWD/stable-pretraining:$PYTHONPATH"
export MUJOCO_GL=egl            # Cube on macOS: glfw (egl is unavailable, dm_control rejects cgl)
```

Data and checkpoints: neither the datasets nor the model weights are in the
repository (`*.pt` is git-ignored). Download them from the
[LeWM HuggingFace collection](https://huggingface.co/collections/quentinll/lewm)
and place them under `$STABLEWM_HOME`. Only the small `config.json` next to each
checkpoint is versioned.

| What | Path under `$STABLEWM_HOME` | Selected by |
|---|---|---|
| PushT dataset | `datasets/pusht_expert_train.h5` | `config/eval/pusht.yaml` |
| PushT checkpoint | `checkpoints/pusht/{config.json, weights.pt}` | `policy=pusht` |
| Cube dataset | `datasets/cube_single_expert.h5` | `config/eval/step1.yaml` |
| Cube checkpoint | `checkpoints/cube/{config.json, weights.pt}` | `policy=cube` |

The evaluators open `HDF5Dataset(eval.dataset_name, cache_dir=$STABLEWM_HOME)`,
which resolves to `<cache_dir>/datasets/<name>.h5`. `config/eval/cube.yaml`
names the Cube dataset `ogbench/cube_single_expert`; with the file at the path
above, pass `eval.dataset_name=cube_single_expert`, as `step1.yaml` does.

## Reproducing the final comparison

### 1. Gated-threshold calibration

```bash
python scripts/calibrate_gated_threshold.py method=mc5 seed=910 +num_eval=8 \
    +output_filename=gated_lewm_strong_rh5.json
```

Records step-0 σ² of every CEM candidate while planning 8 episodes (sampled
with seed 910+1000) under the locked CEM budget and writes
`results/calibration/gated_lewm_strong_rh5.json`. The gate threshold is its
`recommended_threshold` (the σ² median, 0.2535) and is stored in
`config/eval/method/gated.yaml`; this step is only needed to re-calibrate.

### 2. Locked 9-seed panel (slides Fig. 4-5)

```bash
bash scripts/run_locked_panel.sh
```

Runs `baseline`, `mc5`, `gated`, `goal_cvar` on seeds 42-45 and
1011-5055 (36 runs, `results/<method>_s<seed>.txt`; complete runs are
skipped, so the shipped results make this a no-op) and then calls

```bash
python scripts/summarize_locked_9seeds.py   # prints the table; also writes
                                            # results/combined_9seeds.md
```

which can be run on its own to rebuild the table from saved results.

### How the configuration was chosen

The selection protocols below were run once; their scripts and intermediate
results have since been removed, so this is a record only.

RH=5 and gated truncation (historical method ids: `mc_only_n5` = today's
`mc5`, `gated_n5` = `gated`; `adaptive_n5` = ungated adaptive truncation, no
longer a config):

1. **Replanning-frequency screening** — validation seeds 61/62/63, cheap CEM
   (100 / 20 / top-20), receding horizon RH ∈ {5, 2, 1}, methods `baseline` and
   `adaptive_n5`. `BEST_RH` = the RH with the highest mean baseline success
   (selected: **5**).
2. **Gated-threshold calibration** — step 1 above.
3. **Strong-CEM validation** — seeds 61/62/63, CEM 300/30/30, RH=5, methods
   `baseline`, `mc_only_n5`, `adaptive_n5`, `gated_n5` (ω=1.0). `BEST_METHOD` =
   whichever of `adaptive_n5` / `gated_n5` has the higher mean success
   (selected: **gated_n5**).
4. **Locked test** — seeds 1011/2022/3033/4044/5055 (fixed in advance, never
   used for selection), `baseline`, `mc_only_n5`, `gated_n5`. These are the
   `results/{baseline,mc5,gated}_s1011..5055.txt` files of the final panel.

α=0.1 for Goal-CVaR: `goal_cvar_eot` with α ∈ {0.1, 0.25, 0.5} was screened on
validation seeds 74/75 (n=20, CEM 100/20/20) and the α with the highest mean
success was frozen. Its later strong-validation gate (seeds 74-76, n=50,
CEM 300/30/30) did not pass (Goal-CVaR −0.67 pp vs MC), so that protocol's own
final seeds were never run; α=0.1 was instead evaluated on the locked nine-seed
panel alongside the other methods.

### Single runs by hand

```bash
bash scripts/run_single.sh METHOD SEED [hydra overrides...]
#   METHOD: baseline | mc5 | gated | goal_cvar
#   env:    NUM_EVAL=50 CEM_SAMPLES=300 CEM_STEPS=30 CEM_TOPK=30 REC_HORIZON=5 POLICY=pusht RUN_ID FORCE=1
python evaluate.py method=goal_cvar seed=42 method.alpha=0.25 output.run_id=goal_a0p25_s42
```

A run writes `results/<run_id>.txt` (resolved config + metrics, per-episode
successes included) and `results/logs/<run_id>.log`; videos are off by default
(`output.video=true` writes `results/videos/<run_id>/`).

## Cube Step 1: does σ² separate failures?

This step tests whether planning-time σ² flags the frozen Cube LeWM's real
failures. It runs before any planner is built on σ². Criteria, thresholds and
the decision rule were fixed in
[docs/step1_preregistration.md](docs/step1_preregistration.md) before the data
was collected.

The planner is the baseline LeWM (`config/eval/step1.yaml` = `cube.yaml`). It is
evaluated only on non-trivial start states, i.e. ones where the cube is not
already within 4 cm of the goal. Every replan saves the full MuJoCo state, so
each logged action sequence can be replayed exactly to obtain its real outcome.

```bash
bash scripts/run_step1.sh      # resumable; on macOS it sets MUJOCO_GL=glfw
```

| Stage | Script | Output in `results/cube/step1/` |
|---|---|---|
| Closed loop, 8 seeds × 50 episodes, plus replays of executed plans and of 200 states × 600 CEM candidates | `scripts/step1_collect.py` | `closed_loop/`, `plan_replays/`, `candidate_replays/` |
| 5 predictor heads on frozen-encoder latents (ensemble σ²) | `scripts/step1_ensemble.py` | `ensemble/` |
| Predictions, real errors, MC-dropout and ensemble σ² | `scripts/step1_score.py` | `scores.pt` |
| Criteria C1–C3, gate G, decision | `scripts/step1_decide.py` | `report.md`, `decision.json`, `fig_*.png` |

Library code lives in `ua/step1/`.

## Training (upstream LeWM, unchanged)

```bash
python train.py data=pusht     # config/train/lewm.yaml
```
