# Reproduction guide

## Environment

Recorded experiments used Linux, Python 3.10, PyTorch 2.8.0 with CUDA 12.8 and an H100 47 GB allocation. GPU memory requirements and runtime vary with planning batch size. A fresh installation has not been retested during packaging.

Use Conda, not uv. From this repository root:

```bash
conda create -n lewm-prior --file environment/conda-linux-64.lock
conda activate lewm-prior
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/pip-freeze-linux.txt
export LEWM_ROOT="$PWD/workspace"
export STABLEWM_HOME="$LEWM_ROOT"
export MUJOCO_GL=egl
mkdir -p "$LEWM_ROOT/setup" "$LEWM_ROOT/data" "$LEWM_ROOT/runs"
git clone https://github.com/lucas-maes/le-wm.git "$LEWM_ROOT/le-wm"
git -C "$LEWM_ROOT/le-wm" checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
git clone https://github.com/galilai-group/stable-worldmodel.git "$LEWM_ROOT/stable-worldmodel-paper"
git -C "$LEWM_ROOT/stable-worldmodel-paper" checkout abdced49809d5eae38e24b27dc7b635c502c4812
python -m pip install --no-deps -e "$LEWM_ROOT/stable-worldmodel-paper"
python scripts/prepare_workspace.py --root "$LEWM_ROOT"
```

The explicit Conda lock is Linux-64 only. CUDA drivers and EGL libraries are supplied by the host, not the lockfile. The editable dependency's build backend may install `uv_build` internally; this is not use of the uv environment manager. Do not substitute the latest stable-worldmodel: its API may differ. This is the dependency revision used here, not a claim that it is the paper's uniquely specified dependency pin.

## Resources

Official resources are pinned by revision in `provenance/resources.json`. Check disk capacity first: the Cube archive is about 46 GB and its extracted HDF5 about 102 GB; PushT's archive is about 13 GB and extracted HDF5 about 46 GB. Allow additional space for caches and temporary files. The downloader keeps archives and never overwrites an existing extracted dataset.

```bash
python scripts/download_resources.py --root "$LEWM_ROOT" --task cube
python scripts/download_resources.py --root "$LEWM_ROOT" --task pusht
```

Check upstream dataset/model terms before redistributing any weights or recorded data. The repository does not distribute full datasets or checkpoints.

## Train action priors

Run compute-intensive commands on a scheduled compute node, not a shared login node. These commands never train or fine-tune the world model.

```bash
python src/cube/test_planner.py
python src/pusht/test_planner.py
python src/cube/cache_latents.py --output "$LEWM_ROOT/data/pilot-latents.h5" --max-episodes 300 --seed 2026
python src/cube/train_prior.py --cache "$LEWM_ROOT/data/pilot-latents.h5" --output "$LEWM_ROOT/runs/prior-pilot" --steps 2000 --seed 42
python src/pusht/cache_latents.py --output "$LEWM_ROOT/data/pusht-latents-v1.h5" --seed 5478
python src/pusht/train_prior.py --cache "$LEWM_ROOT/data/pusht-latents-v1.h5" --output "$LEWM_ROOT/runs/prior-pusht-v1" --steps 2000 --seed 42
```

Cube uses 240 training and 30 validation trajectories from the 300-trajectory pilot split. PushT uses 240 training, 50 validation and 50 test trajectories; its cache script fixes selection to 340 trajectories. `--max-episodes` is retained in its inherited CLI but does not change that selection. Both priors select the validation-best checkpoint (step 600 in recorded runs). Latent normalization is train-only. Source dataset paths are embedded in caches/checkpoints, so recreate them for a different root rather than silently moving them.

## Reproduce the locked evaluation

The drivers use paths relative to their own working directory. Keep the published `results/` separate from newly generated `workspace/runs/`; do not overwrite historical evidence with new runs.

Cube:

```bash
cd src/cube
python ppt_completion.py prepare
python ppt_completion.py val
python ppt_completion.py test --index 0
python ppt_completion.py test --index 1
python ppt_completion.py test --index 2
python ppt_completion.py test --index 3
python ppt_completion.py test --index 4
python ppt_completion.py summary
cd ../..
```

PushT:

```bash
cd src/pusht
python pusht_regression.py
cd ../..
```

Drivers lock protocol/checkpoint hashes and skip already-completed reports. They deliberately fail on incompatible existing protocol files; use a separate workspace for a changed experiment. A fresh training run need not reproduce the identical checkpoint bytes or exact success percentages.

## Rebuild the figures

```bash
python -m pip install numpy pillow imageio imageio-ffmpeg
python scripts/make_demo.py --frames assets/frames --output assets --font /path/to/Arial.ttf
```

No training is needed to rebuild the demonstrations. The `.npz` files contain only the selected recorded image frames and goal images. Distance-to-goal trajectories were not saved in these exports, so no numeric distance overlay is manufactured. Early-termination holds and the expert's shorter duration are disclosed in the preview.

## Upload

Review `THIRD_PARTY_NOTICES.md` and choose a license for your own contributions before advertising an open-source license. Do not replace third-party notices. Create an empty repository in your GitHub account, then run the following locally after entering this directory:

```bash
git init
git add .
git status --short
git commit -m "Release action-prior-guided LeWM experiments"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_REPOSITORY_URL` with your repository URL. Upload is not performed automatically. If using a browser uploader, retain `.gitignore`; the ZIP is for transport, not a substitute for an expanded repository.
