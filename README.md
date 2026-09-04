# Uncertainty-Aware Planning for Frozen LeWorldModel (ECE4880J project)

Course project built on top of [LeWorldModel](https://le-wm.github.io/)
(Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026; [paper](https://arxiv.org/pdf/2603.19312v1),
[checkpoints](https://huggingface.co/collections/quentinll/lewm)). The pretrained LeWM
checkpoint (`checkpoints/pusht/lewm`) and the CEM planner are kept fixed; only the
trajectory-scoring function changes. Everything is evaluated on PushT.

Methods compared (all n=50 episodes, CEM 300 candidates / 30 iterations / top-30,
receding horizon 5):

| Name | Entry point | Scoring rule |
|---|---|---|
| Baseline | `eval.py` | one deterministic rollout, terminal latent distance to goal |
| MC-5 (`mc_only_n5`) | `ua_eval.py` | mean of 5 MC-dropout rollouts, terminal distance |
| Gated N=5 (`gated_n5`) | `ua_eval.py` | MC-5 + horizon truncation when step-0 σ² exceeds a calibrated threshold |
| Goal-CVaR α=0.1 (`goal_cvar_eot`) | `risk_eval_v2.py` | MC-5 distance + CVaR soft penalty on the worst dropout samples |

## Project layout

| Path | Role |
|---|---|
| `eval.py`, `config/eval/pusht.yaml` | Deterministic baseline planner |
| `ua_eval.py`, `ua/`, `config/eval/ua_pusht.yaml` | MC-dropout aggregation and gated horizon truncation |
| `risk_eval_v2.py`, `risk_v2/`, `config/eval/risk_v2_pusht.yaml` | Goal-CVaR soft-penalty objective |
| `scripts/run_fair_ua_single.sh`, `scripts/run_risk_v2_single.sh` | One reproducible run (method × seed) with saved config/log/metadata |
| `scripts/calibrate_gated_threshold.py` | Gated-threshold calibration (writes `pusht/calibration/gated_lewm_strong_rh5.json`) |
| `scripts/run_locked_panel.sh`, `scripts/summarize_locked_9seeds.py` | Locked 9-seed panel (4 methods × 9 seeds) and its table |
| `pusht/` | Saved results behind the final table: 36 result files (`locked_s42_45/results/`, `fair_ua_final/results/test_*`), `locked_s42_45/combined_9seeds.md`, `calibration/gated_lewm_strong_rh5.json` |
| `train.py`, `jepa.py`, `module.py`, `utils.py`, `config/train/`, `evaluate_wm.py` | LeWM training code (unchanged from upstream) |
| `stable-worldmodel/`, `stable-pretraining/` | Upstream dependencies (env, planner, trainer) |

## Environment

`stable-worldmodel/` and `stable-pretraining/` are git submodules pinned to the
upstream commits used for all experiments:

```bash
git clone --recurse-submodules <this repo>
cd uncertainty-aware-lewm
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export STABLEWM_HOME="$PWD"     # checkpoints/, pusht/ and the dataset live here
export PYTHONPATH="$PWD:$PWD/stable-worldmodel:$PWD/stable-pretraining:$PYTHONPATH"
export MUJOCO_GL=egl
```

Data: download `pusht_expert_train` from the
[LeWM HuggingFace collection](https://huggingface.co/collections/quentinll/lewm)
and place it at `$STABLEWM_HOME/datasets/pusht_expert_train.h5` (the evaluators
open `HDF5Dataset(eval.dataset_name, cache_dir=$STABLEWM_HOME)`, which resolves
to `<cache_dir>/datasets/<name>.h5`). The dataset is not included in this
folder; the checkpoint is (`checkpoints/pusht/lewm/`).

## Reproducing the final comparison

### 1. Gated-threshold calibration

```bash
python scripts/calibrate_gated_threshold.py policy=pusht/lewm seed=910 ua.n_samples=5 \
    solver.num_samples=300 solver.n_steps=30 solver.topk=30 plan_config.receding_horizon=5 \
    +source=planning +num_eval=8 +output_filename=gated_lewm_strong_rh5.json \
    hydra.run.dir=. hydra.output_subdir=null
```

Records step-0 σ² of every CEM candidate while planning 8 episodes and writes
`pusht/calibration/gated_lewm_strong_rh5.json`; the gate threshold used by
`gated_n5` is its `recommended_threshold` (the σ² median). The file is already
in the repo, so this step is only needed to re-calibrate.

### 2. Locked 9-seed panel (slides Fig. 4-5)

```bash
bash scripts/run_locked_panel.sh
```

Runs Baseline, MC-5, Gated N=5 and Goal-CVaR α=0.1 on seeds 42-45
(into `pusht/locked_s42_45/`) and 1011-5055 (Baseline/MC-5/Gated into
`pusht/fair_ua_final/results/test_*`, Goal-CVaR into `pusht/locked_s42_45/`).
Resumable: every (method × seed) with a complete result is skipped. It then calls

```bash
python scripts/summarize_locked_9seeds.py   # prints the table; also writes
                                            # pusht/locked_s42_45/combined_9seeds.md
```

which can be run on its own to rebuild the table from saved results.

### How the configuration was chosen

The selection protocols below were run once; their scripts and intermediate
results have since been removed, so this is a record only.

RH=5 and `gated_n5`:

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
   `test_*` results reused by `run_locked_panel.sh`.

α=0.1 for Goal-CVaR: `goal_cvar_eot` with α ∈ {0.1, 0.25, 0.5} was screened on
validation seeds 74/75 (n=20, CEM 100/20/20) and the α with the highest mean
success was frozen. Its later strong-validation gate (seeds 74-76, n=50,
CEM 300/30/30) did not pass (Goal-CVaR −0.67 pp vs MC), so that protocol's own
final seeds were never run; α=0.1 was instead evaluated on the locked nine-seed
panel alongside the other methods.

### Single runs by hand

```bash
bash scripts/run_fair_ua_single.sh RUN_ID METHOD SEED [hydra overrides...]
#   METHOD: baseline | mc_only_n3 | mc_only_n5 | adaptive_n3 | adaptive_n5 | gated_n5
bash scripts/run_risk_v2_single.sh RUN_ID goal_cvar_eot SEED ALPHA [hydra overrides...]
#   env for both: DATA_SPLIT CEM_SAMPLES CEM_STEPS CEM_TOPK REC_HORIZON NUM_EVAL
#   gated_n5 also needs ADAPTIVE_OMEGA=1.0 GATED_THRESHOLD_JSON=pusht/calibration/gated_lewm_strong_rh5.json
```

Every run saves, under `pusht/<experiment_group>/`: the result with the full
resolved Hydra config (`results/<run_id>.txt`), stdout/stderr
(`logs/<run_id>.log`), the exact command, metadata and exit status
(`runs/<run_id>/`). An incomplete run is preserved and needs `FORCE=1` to
archive and rerun.

## Training (upstream LeWM, unchanged)

```bash
python train.py data=pusht     # config/train/lewm.yaml
```
