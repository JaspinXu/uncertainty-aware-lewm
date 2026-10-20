"""Train the learned components of the compared planners on frozen LeWM latents (config/eval/compare.yaml).

    python scripts/cube_train.py                          # every component that is missing
    python scripts/cube_train.py "+only=[sigma_prior]"    # one component
    python scripts/cube_train.py +force=1                 # retrain even if a checkpoint exists

Writes <compare.out_dir>/models/:
  sigma_prior.pt  Gaussian BC prior: mean and per-dimension std of the expert's next 25 actions,
                  goal 25 steps ahead, Gaussian NLL. Used by bc_only and prior_cem_* (CEM from
                  N(mu_p, sigma^2)); ``sigma_mean`` (mean predicted std on held-out data) is the
                  constant of prior_cem_sconst.
  prior_tc.pt     time-conditioned BC prior (MSE, goal 1-100 steps ahead, the offset is an input).
  khat.pt         steps-to-goal estimate k̂ (expectile 0.1, goals 0-150 steps ahead).
                  prior_tc + khat are bc_select_khat.
  split.json      the cached training / validation episodes. scripts/cube_compare.py draws its
                  evaluation start states only from episodes outside the cache.

The latent cache (compare.latents) holds every frame of compare.cache_episodes episodes encoded by the
frozen LeWM encoder; it is built here from the dataset if missing (episodes drawn with
compare.episode_seed), and not loaded at all when every component and split.json already exist. The last compare.val_episodes cached episodes are held out for early stopping.
Actions are normalized with the planner's own StandardScaler (fit on the dataset, as evaluate.build_world).
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
import hdf5plugin  # noqa: E402,F401
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from sklearn import preprocessing  # noqa: E402

from evaluate import get_dataset, img_transform, pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.ensemble import cache_latents  # noqa: E402
from ua.step1.prior import PriorData, eval_mse, train_prior  # noqa: E402
from ua.step1.steps_to_goal import evaluate as eval_khat  # noqa: E402
from ua.step1.steps_to_goal import train_steps_to_goal  # noqa: E402

COMPONENTS = ("sigma_prior", "prior_tc", "khat")


def h5_path(cfg) -> Path:
    return Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / "datasets" / f"{cfg.eval.dataset_name}.h5"


def latent_cache(cfg, device) -> dict:
    """The cached latents of the training episodes; encoded from the dataset if the file is missing."""
    ccfg = cfg.compare
    path = ROOT / ccfg.latents
    if path.exists():
        print(f"[train] latent cache {path}", flush=True)
        return torch.load(path, weights_only=False, map_location="cpu")
    with h5py.File(h5_path(cfg), "r") as f:
        n_episodes = len(f["ep_len"])
    pick = np.random.default_rng(int(ccfg.episode_seed)).permutation(n_episodes)[: int(ccfg.cache_episodes)]
    n_val = int(ccfg.val_episodes)
    episodes = sorted(pick[:-n_val].tolist()) + sorted(pick[-n_val:].tolist())  # train, then held-out
    print(f"[train] encoding {len(episodes)} episodes of {h5_path(cfg)} -> {path}", flush=True)
    t0 = time.time()
    cache = cache_latents(h5_path(cfg), episodes, load_base(cfg.policy, device), img_transform(cfg), device)
    cache["episodes"] = episodes
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"[train] cached {len(cache['emb'])} frames in {time.time() - t0:.0f}s", flush=True)
    return cache


def action_scaler(cfg) -> tuple[np.ndarray, np.ndarray]:
    """The planner's action normalization (evaluate.build_world fits the same scaler)."""
    values = get_dataset(cfg, cfg.eval.dataset_name).get_col_data("action")
    scaler = preprocessing.StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
    return scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


@hydra.main(version_base=None, config_path="../config/eval", config_name="compare")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device = pick_device()
    tcfg = cfg.compare.train
    mdir = ROOT / cfg.compare.out_dir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    only = [str(c) for c in cfg.get("only", COMPONENTS)]
    assert set(only) <= set(COMPONENTS), f"unknown component in {only}; choose from {COMPONENTS}"
    force = bool(cfg.get("force", False))
    todo = [name for name in only if force or not (mdir / f"{name}.pt").exists()]
    for name in only:
        if name not in todo:
            print(f"[train] {name}: {mdir / f'{name}.pt'} exists, skipping", flush=True)
    if not todo and (mdir / "split.json").exists():
        return  # nothing to train and the split is known: the latent cache is not needed

    cache = latent_cache(cfg, device)
    episodes = [int(e) for e in cache["episodes"]]
    n_val = int(cfg.compare.val_episodes)
    train_rows, val_rows = list(range(len(episodes) - n_val)), list(range(len(episodes) - n_val, len(episodes)))
    (mdir / "split.json").write_text(json.dumps({
        "latents": str(cfg.compare.latents),
        "train_episodes": episodes[:-n_val], "val_episodes": episodes[-n_val:],
    }))
    mean, scale = action_scaler(cfg)
    print(f"[train] {len(train_rows)} training / {len(val_rows)} held-out episodes; device {device}", flush=True)

    def data(offsets):
        return (PriorData(cache, train_rows, mean, scale, device, offsets),
                PriorData(cache, val_rows, mean, scale, device, offsets))

    meta = {"action_mean": mean, "action_scale": scale, "train_episodes": episodes[:-n_val]}
    for name in todo:
        ckpt = mdir / f"{name}.pt"
        c = tcfg[name]
        offsets = tuple(int(k) for k in c.offsets)
        train, val = data(offsets)
        print(f"[train] {name}: goal {offsets[0]}-{offsets[1]} steps ahead, {int(c.steps)} steps", flush=True)
        if name == "khat":
            expectile = float(c.expectile)
            model, log = train_steps_to_goal(train, val, steps=int(c.steps), expectile=expectile)
            ev = eval_khat(model, val, expectile=expectile)
            print(f"[train] khat: val MAE {ev['mae']:.1f} steps, Spearman {ev['spearman']:.3f}; mean k̂ by true "
                  "distance " + ", ".join(f"{b} {v:.0f}" for b, v in ev["mean_pred_by_k"].items()), flush=True)
            state = {"state_dict": model.state_dict(), "offsets": offsets, "expectile": expectile}
        else:
            k_max, gaussian = int(c.get("k_max", 0)), name == "sigma_prior"
            model, log = train_prior(train, val, steps=int(c.steps), k_max=k_max, gaussian=gaussian)
            ev = eval_mse(model, val)
            extra = f", NLL {ev['nll']:.3f}, mean sigma_p {ev['sigma_mean']:.3f}" if gaussian else ""
            print(f"[train] {name}: val MSE {ev['mse']:.3f}{extra}", flush=True)
            state = {"state_dict": model.state_dict(), "offsets": offsets, "k_max": k_max, "gaussian": gaussian,
                     "sigma_mean": ev.get("sigma_mean")}
        torch.save({**state, **meta, "train_log": log, "val": ev}, ckpt)
        print(f"[train] wrote {ckpt}", flush=True)


if __name__ == "__main__":
    main()
