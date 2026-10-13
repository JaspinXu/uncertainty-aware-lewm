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
mkdir -p logs
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

| step | command | what it produces |
|---|---|---|
| L1 signal test | `bash cluster/submit.sh l1` | `results/cube/l1/signal_report.md`, baseline panel |

(`cluster/submit.sh` submits the job array and the dependent analysis job;
later phases are added to it as they become available.)

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
