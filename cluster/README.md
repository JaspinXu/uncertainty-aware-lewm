# Running the Cube experiments on the SoC compute cluster (Slurm)

Everything below is run from the repository root on the cluster. Jobs write
their logs to `logs/` and their results to `results/cube/...`; large artifacts
(dataset, checkpoints, HF cache) live in `$STABLEWM_HOME` (default
`~/lewm_data`, see `cluster/env.sh`).

Do not run heavy work on the login node: everything goes through `sbatch`
(or an interactive `srun --pty bash`).

## 0. One-time setup

```bash
git clone --recursive https://github.com/JaspinXu/uncertainty-aware-lewm.git
cd uncertainty-aware-lewm
bash cluster/submit.sh bootstrap     # = steps 0-2 below, chained: setup -> download -> smoke test
```

or step by step:

```bash
sbatch cluster/jobs/00_setup_env.sbatch          # Miniforge + conda env "lewm" (~15 min)
```

Check `logs/lewm_setup_<jobid>.out`; it should end with `torch 2.8.0 ...` and
`[setup] done`. (A CPU node prints `cuda available: False`; that is fine.)

## 1. Data and checkpoint (L0)

```bash
sbatch cluster/jobs/01_download_cube.sbatch      # 46 GB archive -> ~50 GB .h5 + LeWM-Cube weights
```

Resumable: if it hits the time limit, submit it again. Needs ~100 GB free in
`$STABLEWM_HOME` during extraction (the archive is deleted afterwards).

## 2. Smoke test

```bash
sbatch cluster/jobs/02_smoke_test.sbatch         # ~15-30 min on one GPU
```

It checks headless rendering, prints the dataset columns, plans 10 episodes
with the pretrained model (recording on) and replays the recorded candidates in
the simulator. The log must end with `smoke test finished OK`, and the oracle
line `fidelity: {... 'max_abs_qpos_diff': ...}` should be tiny (< 1e-3).

Choosing a GPU type: any `#SBATCH` option can be overridden on the command
line, e.g. `sbatch --gpus=h100-96:1 cluster/jobs/02_smoke_test.sbatch`.
`sinfo -o "%P %G"` lists the partitions and GPU types you can request.

## 3. Experiments

Every phase is one command; jobs are chained with Slurm dependencies, finished
runs are skipped on resubmission, and all outputs go to `results/cube/`.

| phase | command | GPU time (rough) | main output |
|---|---|---|---|
| L1-L3 planning study | `bash cluster/submit.sh planning` | ~28 + 36 + 9 runs | see below |
| L1 signal test | (part of `planning`) | 28 runs x ~15 min + ensemble ~1-2 h | `results/cube/l1/signal_report.md` |
| L2 sigma^2 planners | (part of `planning`) | 45 runs x ~30 min | `results/cube/summary/l2_panel.md` |
| L3 risk head | (part of `planning`) | 1 + 9 runs | `results/cube/l3/risk_report.md`, `results/cube/summary/planning_panel.md` |
| L4 representation training | `bash cluster/submit.sh l4` | 5 runs x several hours (self-resubmitting) | `$STABLEWM_HOME/checkpoints/cube/l4_*` |
| L5 variant evaluation | automatic when each L4 run finishes | ~20 runs per variant | `results/cube/l5/representation.md`, `results/cube/runs/l4_*` |
| L6 collected results | `bash cluster/submit.sh report` | CPU, minutes | `results/cube/RESULTS.md` |

`planning` and `l4` are independent and can be submitted together right after
the smoke test. Every L4 variant is warm-started from the pretrained LeWM-Cube
and trained for 20k steps by default (`L4_MAX_STEPS=... bash cluster/submit.sh l4`
to change it; the smoke test prints the training throughput to size it). If training is limited by reading
the dataset from the shared file system, give the jobs a node-local scratch
directory with ~100 GB free: `L4_STAGE_DIR=/path/to/local/disk bash cluster/submit.sh l4`
(each job copies the .h5 there first when it fits).

`planning` also trains the latent dynamics ensemble (`cluster/jobs/12_ensemble.sbatch`:
encode all frames with the frozen encoder, train 5 predictor heads), which gives
the second uncertainty signal for L1 and the `ens_penalty` planner in L2.

Watching progress: `squeue -u $USER`, `tail -f logs/<job>.out`. A failed array
task can be rerun alone, e.g. `sbatch --array=7 cluster/jobs/21_l2_panel.sbatch`;
dependent jobs that were cancelled because of it can be resubmitted with the
phase command (completed runs are skipped).

### What to send back

The small result files are meant to be committed (records, oracle replays and
checkpoints are git-ignored and stay on the cluster):

```bash
git add results/cube results/calibration && git commit -m "Cube results" && git push
```

## Troubleshooting

* **`RENDER CHECK FAILED` / EGL errors**: make sure the job runs on a GPU node.
  As a fallback use software rendering:
  `conda activate lewm && conda install -y -c conda-forge mesalib` and
  `export MUJOCO_GL=osmesa` before submitting.
* **`Requested time limit is invalid`**: lower `--time` on the command line,
  e.g. `sbatch --time=02:00:00 ...` (all jobs are restartable).
* **CUDA not available inside a GPU job**: `nvidia-smi` must work in the job;
  the PyPI torch 2.8 wheels need an NVIDIA driver >= 525.
* **Out of memory**: `--mem=64G` on the command line.
* Reruns are skipped when results exist; delete the corresponding files in
  `results/cube/` to recompute.
