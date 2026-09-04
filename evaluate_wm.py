"""World-model evaluation for the LeWM baseline.

Computes *reproducible* prediction metrics on the held-out validation split
(no simulator / MuJoCo required), so it can run on CPU/MPS/GPU:

  * teacher-forced 1-step next-embedding MSE / normalized-MSE / cosine sim
  * open-loop autoregressive rollout error as a function of horizon
  * a "persistence" (copy-last-embedding) reference to contextualise the numbers
  * the SIGReg Gaussianity statistic of the learned latent space
  * error analysis: per-horizon error growth + histogram of per-sample errors
    + the worst-predicted samples

Results (metrics.json, rollout_mse.csv, *.png plots) are written to
``results/wm_eval/<run>/`` and are meant to feed the baseline-milestone report.

Usage (reuses the training config so data/model stay consistent):

    # evaluate a trained checkpoint over a 10-step horizon on 8 batches
    python evaluate_wm.py +ckpt=lewm/weights_epoch_100.pt num_preds=10 \
        +eval_batch_size=16 +num_batches=8 +device=mps

    # no checkpoint -> evaluates the randomly-initialised model (pipeline smoke test)
    python evaluate_wm.py num_preds=10 +num_batches=2 +device=cpu

``num_preds`` doubles as the evaluation horizon (num_steps = history_size + num_preds).
"""

import csv
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

import stable_pretraining as spt
import stable_worldmodel as swm

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    _HAS_MPL = False


def pick_device(pref):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_val_loader(cfg):
    """Rebuild the exact same dataset/transform/split as train.py, val part only."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(cfg.get("eval_batch_size", 16)),
        num_workers=int(cfg.get("eval_workers", 0)),
        shuffle=False,
        drop_last=True,
    )
    return loader


def load_model(cfg, device):
    """Load a checkpoint if provided, else instantiate a random-init model."""
    ckpt = cfg.get("ckpt", None)
    if ckpt:
        model = swm.wm.utils.load_pretrained(ckpt)
        print(f"[eval] loaded checkpoint: {ckpt}")
    else:
        model = hydra.utils.instantiate(cfg.model)
        print("[eval] WARNING: no ckpt given -> evaluating RANDOMLY INITIALISED model.")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def rollout_batch(model, batch, history_size, horizon, device):
    """Open-loop autoregressive rollout with ground-truth actions.

    Returns per-step tensors of shape (B, horizon, D):
        pred_roll  - model predictions (steps >1 consume their own predictions)
        tgt_roll   - ground-truth embeddings
        persist    - copy-last-context-embedding reference
    """
    pixels = batch["pixels"].to(device).float()  # (B, T, C, H, W)
    actions = torch.nan_to_num(batch["action"].to(device), 0.0)  # (B, T, raw_act)

    T = pixels.size(1)
    assert T >= history_size + horizon, (
        f"need num_steps >= history_size+horizon ({history_size}+{horizon}); "
        f"got T={T}. Increase num_preds."
    )

    emb_gt = model.encode({"pixels": pixels})["emb"]  # (B, T, D)
    act_emb_all = model.action_encoder(actions)  # (B, T, D)

    hs = history_size
    emb = emb_gt[:, :hs].clone()  # start from ground-truth context
    for _ in range(horizon):
        length = emb.size(1)
        emb_trunc = emb[:, -hs:]
        act_trunc = act_emb_all[:, length - hs : length]
        pred = model.predict(emb_trunc, act_trunc)[:, -1:]  # (B, 1, D)
        emb = torch.cat([emb, pred], dim=1)

    pred_roll = emb[:, hs : hs + horizon]
    tgt_roll = emb_gt[:, hs : hs + horizon]
    persist = emb_gt[:, hs - 1 : hs].expand(-1, horizon, -1)
    return pred_roll, tgt_roll, persist, emb_gt


def save_plots(out_dir, steps, mse, nmse, cos, persist_mse, per_sample_err):
    if not _HAS_MPL:
        print("[eval] matplotlib not available; skipping plots.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(steps, mse, "o-", label="LeWM rollout")
    plt.plot(steps, persist_mse, "s--", label="persistence (copy-last)")
    plt.xlabel("rollout horizon (steps)")
    plt.ylabel("embedding MSE")
    plt.title("Open-loop rollout error vs horizon")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "rollout_error_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(steps, cos, "o-")
    plt.xlabel("rollout horizon (steps)")
    plt.ylabel("cosine similarity (pred, target)")
    plt.title("Prediction cosine similarity vs horizon")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(per_sample_err, bins=40)
    plt.xlabel("per-sample 1-step MSE")
    plt.ylabel("count")
    plt.title("Distribution of 1-step prediction error")
    plt.tight_layout()
    plt.savefig(out_dir / "error_histogram.png", dpi=150)
    plt.close()
    print(f"[eval] plots written to {out_dir}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    device = pick_device(cfg.get("device", None))
    horizon = int(cfg.num_preds)
    history_size = int(cfg.history_size)
    max_batches = int(cfg.get("num_batches", 8))
    print(f"[eval] device={device} history_size={history_size} horizon={horizon}")

    loader = build_val_loader(cfg)
    model = load_model(cfg, device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    # accumulators (per horizon step)
    sse = np.zeros(horizon)  # sum squared error
    sse_persist = np.zeros(horizon)
    tgt_sq = np.zeros(horizon)  # sum of target^2 (for normalized MSE)
    cos_sum = np.zeros(horizon)
    count = np.zeros(horizon)  # number of (sample*dim) elements per step
    n_samples = np.zeros(horizon)  # number of samples per step

    sigreg_vals = []
    per_sample_err = []  # 1-step per-sample MSE
    worst = []  # (error, batch_idx, in_batch_idx)

    n_batches = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        pred, tgt, persist, emb_gt = rollout_batch(
            model, batch, history_size, horizon, device
        )

        err = (pred - tgt).pow(2)  # (B, H, D)
        err_persist = (persist - tgt).pow(2)
        b, h, d = err.shape

        sse += err.sum(dim=(0, 2)).cpu().numpy()
        sse_persist += err_persist.sum(dim=(0, 2)).cpu().numpy()
        tgt_sq += tgt.pow(2).sum(dim=(0, 2)).cpu().numpy()
        cos_sum += F.cosine_similarity(pred, tgt, dim=-1).sum(dim=0).cpu().numpy()
        count += b * d  # MSE divides by elements
        n_samples += b  # cosine divides by samples

        # 1-step per-sample error for the histogram / worst-case analysis
        step0 = err[:, 0].mean(dim=-1).cpu().numpy()  # (B,)
        per_sample_err.extend(step0.tolist())
        for j, e in enumerate(step0.tolist()):
            worst.append((float(e), bi, j))

        sigreg_vals.append(float(sigreg(emb_gt.transpose(0, 1)).item()))
        n_batches += 1
        print(f"[eval] batch {bi}: 1-step MSE={sse[0] / count[0]:.5f}")

    if n_batches == 0:
        raise RuntimeError("No batches evaluated; check dataset / num_batches.")

    mse = sse / count
    persist_mse = sse_persist / count
    nmse = sse / np.maximum(tgt_sq, 1e-8)
    cos = cos_sum / np.maximum(n_samples, 1)
    steps = list(range(1, horizon + 1))

    worst.sort(key=lambda x: -x[0])
    summary = {
        "run": str(cfg.get("ckpt", "random_init")),
        "device": str(device),
        "history_size": history_size,
        "horizon": horizon,
        "num_batches": n_batches,
        "batch_size": int(cfg.get("eval_batch_size", 16)),
        "one_step": {
            "mse": float(mse[0]),
            "normalized_mse": float(nmse[0]),
            "cosine": float(cos[0]),
            "persistence_mse": float(persist_mse[0]),
        },
        "final_step": {
            "mse": float(mse[-1]),
            "normalized_mse": float(nmse[-1]),
            "cosine": float(cos[-1]),
            "persistence_mse": float(persist_mse[-1]),
        },
        "rollout_mse_per_step": mse.tolist(),
        "persistence_mse_per_step": persist_mse.tolist(),
        "cosine_per_step": cos.tolist(),
        "sigreg_gaussianity": float(np.mean(sigreg_vals)),
        "worst_1step_samples": worst[:10],
    }

    run_name = str(cfg.get("ckpt", "random_init")).replace("/", "_")
    out_dir = Path("results", "wm_eval", run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "rollout_mse.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "mse", "normalized_mse", "cosine", "persistence_mse"])
        for k in range(horizon):
            w.writerow([steps[k], mse[k], nmse[k], cos[k], persist_mse[k]])

    save_plots(out_dir, steps, mse, nmse, cos, persist_mse, per_sample_err)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary["one_step"], indent=2))
    print(f"SIGReg (Gaussianity, lower=better): {summary['sigreg_gaussianity']:.4f}")
    print(f"Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    run()
