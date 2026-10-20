"""Step 1 / ensemble sigma^2: predictor heads trained on frozen-encoder latents.

    python scripts/step1_ensemble.py                       # cache + train all members
    python scripts/step1_ensemble.py step1.ensemble.steps=2000 +only_member=0   # pilot

Episodes: every Step-1 episode plus random others (3000 for training) and 200
held-out episodes for validation (never seen by any member). Each member gets
its own random init and an episode-level bootstrap of the training set.
Outputs: results/cube/step1/ensemble/{latents.pt, member_<m>.pt, train_log.json}.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402
import time  # noqa: E402

import h5py  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn import preprocessing  # noqa: E402

from evaluate import get_dataset, img_transform, pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.common import out_dir  # noqa: E402
from ua.step1.ensemble import (  # noqa: E402
    ClipSampler,
    base_head,
    cache_latents,
    eval_loss,
    new_head,
    train_head,
)


def h5_path(cfg) -> Path:
    cache = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return cache / "datasets" / f"{cfg.eval.dataset_name}.h5"


def split_episodes(cfg, out: Path, n_episodes: int):
    ecfg = cfg.step1.ensemble
    step1_eps = set()
    for seed in cfg.step1.seeds:
        log = torch.load(out / "closed_loop" / f"seed_{seed}.pt", weights_only=False)
        step1_eps.update(int(e) for e in log["episodes"])
    rng = np.random.default_rng(int(ecfg.episode_seed))
    others = rng.permutation(np.array([e for e in range(n_episodes) if e not in step1_eps]))
    n_val, n_train = int(ecfg.val_episodes), int(ecfg.train_episodes)
    val = sorted(others[:n_val].tolist())
    train = sorted(step1_eps) + sorted(others[n_val : n_val + n_train - len(step1_eps)].tolist())
    return train, val, sorted(step1_eps)


@hydra.main(version_base=None, config_path="../config/eval", config_name="step1")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    out = out_dir(cfg, ROOT)
    ens_dir = out / "ensemble"
    ens_dir.mkdir(exist_ok=True)
    ecfg = cfg.step1.ensemble
    base = load_base(cfg.policy, device)

    # ---- latent cache ----
    cache_path = ens_dir / "latents.pt"
    path = h5_path(cfg)
    with h5py.File(path, "r") as f:
        n_episodes = len(f["ep_len"])
    train_eps, val_eps, step1_eps = split_episodes(cfg, out, n_episodes)
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
        assert cache["episodes"] == train_eps + val_eps, "cached episodes differ from the split"
    else:
        t0 = time.time()
        print(f"[ensemble] caching {len(train_eps)} train + {len(val_eps)} val episodes from {path}")
        cache = cache_latents(path, train_eps + val_eps, base, img_transform(cfg), device)
        cache["episodes"] = train_eps + val_eps
        torch.save(cache, cache_path)
        print(f"[ensemble] cached {len(cache['emb'])} frames in {time.time() - t0:.0f}s")

    # action normalization of the planner (dataset StandardScaler)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    values = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
    mean, std = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)

    n_train = len(train_eps)
    train_rows, val_rows = np.arange(n_train), np.arange(n_train, n_train + len(val_eps))
    val_sampler = ClipSampler(cache, val_rows, mean, std, device, seed=0)
    base_val = eval_loss(base_head(base), val_sampler)
    print(f"[ensemble] released predictor val loss {base_val:.4f} ({len(val_sampler)} val clips)")

    log_path = ens_dir / "train_log.json"
    logs = json.loads(log_path.read_text()) if log_path.exists() else {}
    logs["released_val_loss"] = base_val
    members = [int(cfg.only_member)] if cfg.get("only_member") is not None else range(int(ecfg.members))
    for m in members:
        ckpt = ens_dir / f"member_{m}.pt"
        if ckpt.exists() and cfg.get("only_member") is None:
            print(f"[ensemble] member {m} exists, skipping")
            continue
        rng = np.random.default_rng(1000 + m)
        rows = rng.choice(train_rows, size=n_train, replace=True)  # episode bootstrap
        sampler = ClipSampler(cache, rows, mean, std, device, seed=m)
        head = new_head(ROOT / "checkpoints" / cfg.policy / "config.json", seed=m).to(device)
        print(f"[ensemble] member {m}: {len(sampler)} clips, {int(ecfg.steps)} steps x {int(ecfg.batch)}")
        hist = train_head(head, sampler, val_sampler, int(ecfg.steps), int(ecfg.batch), float(ecfg.lr))
        torch.save({"state_dict": head.state_dict(), "seed": m, "steps": int(ecfg.steps)}, ckpt)
        logs[f"member_{m}"] = hist
        log_path.write_text(json.dumps(logs, indent=2))
    log_path.write_text(json.dumps(logs, indent=2))


if __name__ == "__main__":
    main()
