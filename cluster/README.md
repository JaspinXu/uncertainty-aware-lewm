# Running the Cube experiments on the SoC compute cluster (Slurm)

Everything below is run from the repository root on the login node. Jobs write
their logs to `logs/` and their results to `results/cube/...`; large artifacts
(dataset, checkpoints) live in `$STABLEWM_HOME` (default `~/lewm_data`, see
`cluster/env.sh`). Nothing heavy runs on the login node: every step is a Slurm
job submitted by `cluster/submit.sh`.

## The cluster (NUS SoC, checked with `sinfo` / `sacctmgr`, Oct 2026)

| | |
|---|---|
| partitions | `normal` (default, **CPU only**, 3 h), `long` (CPU, 3 days), `gpu` (3 h), `gpu-long` (3 days, fewer nodes) |
| GPU types | `a100-40`, `a100-80`, `h100-47`, `h100-96`, `h200-141` (only in `gpu`), `nv` (older cards: Titan V, Titan RTX, T4). A request without a type (`--gpus=1`) only gets `nv` cards |
| per user | 16 running and 32 queued jobs (every array task counts); 8 GPUs at once, at most a100-40 4, a100-80 2, h100-47 2, h100-96 1, h200-141 1 |
| storage | home: shared all-flash NFS; `/mnt/scratch`: shared HDD NFS; `/tmp`: node-local |
| network | HTTPS works from login and compute nodes; SSH to GitHub (ports 22 and 443) is blocked |

`cluster/submit.sh` follows these rules: GPU jobs go to the `gpu` partition
(a job submitted without a partition lands on a CPU node) and each asks for one
GPU of an explicit type. Jobs that render the simulator (planning, oracle
replays) take turns over whole GPUs (`a100-40 a100-80 a100-40 a100-80 a100-40
a100-40 h100-96 h200-141`): the `h100-47` cards are MIG slices of an H100, which
cannot render (software rendering, ~160 ms per frame instead of ~25 ms on an
A100). Training-type jobs take turns over `h100-47 h100-47 a100-40 h100-96
a100-80`. Each phase is split into a few jobs that run several evaluation runs
one after another, so that a whole phase stays far below 32 queued jobs. Every GPU job checks first that it really has a usable GPU and
stops at once otherwise.

## 0. One-time setup

The repository is private and the cluster cannot reach GitHub over SSH, so
clone over HTTPS with a fine-grained personal access token limited to this
repository (GitHub -> Settings -> Developer settings -> Fine-grained tokens;
repository access: only `uncertainty-aware-lewm`; permission Contents: Read and
write, so that results can be pushed from the cluster):

```bash
git -c credential.helper= -c credential.helper="store --file $HOME/.git-credentials-lewm" \
    clone --recursive https://JaspinXu@github.com/JaspinXu/uncertainty-aware-lewm.git
cd uncertainty-aware-lewm
git config credential.helper ""
git config --add credential.helper "store --file $HOME/.git-credentials-lewm"
```

Git asks once for the password: paste the token. `-c credential.helper=` skips
any older GitHub credential stored on the cluster, and the last two lines make
`git pull` / `git push` in this repository use the token without asking again
(the file is readable only by you). The submodules are public.

Then:

```bash
bash cluster/submit.sh bootstrap   # setup env -> download data + checkpoint -> smoke test
```

Steps that are already done are skipped. Setup takes ~20 min (conda env
`lewm`), the download ~45 min (46 GB archive -> 50 GB `.h5` + LeWM-Cube weights;
resumable, needs ~100 GB free in `$STABLEWM_HOME` while extracting).

## 1. Smoke test

```bash
bash cluster/submit.sh smoke       # ~15 min on one GPU
```

`logs/lewm_smoke_<id>.out` must end with `smoke test finished OK`. Check:

* `[env] GPU ...` near the top (a GPU was allocated; `no GPU in this job` means
  it ran on a CPU node);
* `render OK: ... render X ms` -- a few ms with GPU rendering (~200 ms means
  software rendering);
* `[oracle] fidelity: {...}` -- `max_abs_qpos_diff` ~0 for both `first`
  (plans at an episode start) and `last` (plans in the middle of an episode);
* `timing: {...}` -- also written to `results/cube/smoke/timing.json`, which
  `submit.sh` reads to decide how many runs go into one job.

Then `bash cluster/submit.sh results` pushes the smoke-test outputs and logs.

## 2. Experiments

| phase | command | main output |
|---|---|---|
| L1-L3 planning study | `bash cluster/submit.sh planning` | see below |
| L1 signal test | (part of `planning`) | `results/cube/l1/signal_report.md` |
| L2 sigma^2 planners | (part of `planning`) | `results/cube/summary/l2_panel.md` |
| L3 risk head | (part of `planning`) | `results/cube/l3/risk_report.md`, `results/cube/summary/planning_panel.md` |
| L4 representation training | `bash cluster/submit.sh l4` | `$STABLEWM_HOME/checkpoints/cube/l4_*` |
| L5 variant evaluation | automatic when each L4 run finishes | `results/cube/l5/representation.md`, `results/cube/runs/l4_*` |
| L6 collected results | `bash cluster/submit.sh report` | `results/cube/RESULTS.md` |
| Check: live-rendered vs dataset frames (~10 min) | `bash cluster/submit.sh diag` | `results/cube/diag/domain_gap.md` |

`planning` (at most 22 jobs) and `l4` (6 jobs) can be submitted together right after
a clean smoke test. `planning` also trains the latent dynamics ensemble
(`cluster/jobs/12_ensemble.sbatch`), the second uncertainty signal for L1 and
the `ens_penalty` planner in L2. Every L4 variant is warm-started from the
pretrained LeWM-Cube and trained for 20k steps (`L4_MAX_STEPS=...` to change);
a training job saves its state ~12 min before the 3 h limit and resubmits itself.
If training is limited by reading the dataset (the smoke test prints the
training samples/s), give the L4 jobs node-local disk with ~100 GB free:
`L4_STAGE_DIR=/tmp bash cluster/submit.sh l4`.

Progress: `bash cluster/submit.sh status` (runs done per phase, queue usage),
`squeue -u $USER`, `tail -f logs/<job>.out`.

**Reruns.** Every phase command can be run again at any time: finished runs are
left out and only the missing ones are submitted. A job that runs out of time
stops before starting a run that would not fit (exit code 4), and its dependent
jobs are not started; submit the phase again. If a phase needs more queue
slots than are free, `submit.sh` says so and submits nothing.

**Overrides.** `LEWM_GPU_TYPES="a100-40 a100-80"` (rendering jobs) and
`LEWM_TRAIN_GPU_TYPES="h100-47"` (training jobs) change the GPU types the jobs
rotate through; `LEWM_GPU_PARTITION=gpu-long`
uses the 3-day partition; any other sbatch option goes through `SBATCH_EXTRA`.
`sbatch --test-only -p gpu --gres=gpu:h100-47:1 --wrap=hostname` shows when a
GPU type would start.

### What to send back

```bash
bash cluster/submit.sh results
```

commits the small result files (records, oracle replays and checkpoints are
git-ignored and stay on the cluster) together with the last 400 lines of every
job log of the past week (`results/cube/logs/`), and pushes them.

## Troubleshooting

* **`[env] no GPU in this job`**: the job was not on a GPU partition. Submit
  through `cluster/submit.sh`, or add `--partition=gpu` to a manual `sbatch`.
* **`Requested node configuration is not available`**: the request cannot be met
  by any node, e.g. an untyped `--gpus=1` together with excluding the `nv` nodes.
  Ask for a type: `--gres=gpu:h100-47:1`.
* **`RENDER CHECK FAILED` / EGL errors**: as a fallback use software rendering
  (slow): `conda activate lewm && conda install -y -c conda-forge mesalib` and
  `export MUJOCO_GL=osmesa` before submitting.
* **`... needs N job(s), but M of your 32 queued-job slots are taken`**: wait
  for running jobs to finish, then submit the phase again.
* **`Requested time limit is invalid`**: the `gpu` partition allows 3 h; with
  `LEWM_GPU_PARTITION=gpu-long` up to 3 days.
* **Out of memory**: `SBATCH_EXTRA="--mem=96G"`.
* Results are skipped when they exist; delete the corresponding files in
  `results/cube/` to recompute.
