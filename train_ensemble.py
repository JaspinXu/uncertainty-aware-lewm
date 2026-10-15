"""Train a K-member latent dynamics ensemble on the frozen LeWM-Cube encoder.

    python scripts/cube/encode_dataset.py --policy cube/lewm     # once (latents)
    python train_ensemble.py --members 5 --steps 20000

Each member is the LeWM predictor architecture (AdaLN transformer, history 3,
action embedder, BatchNorm projection) trained from a different random
initialisation on a bootstrap subset (80%) of the training episodes, on clips
of 6 frames (frameskip 5) with the planner's own objective structure:
teacher-forced next-latent loss + 5-step open-loop rollout loss. Targets are
frozen latents, so no anti-collapse term is needed.

Validation (held-out episodes, same split as train_cube.py) reports every
member, the ensemble mean and -- for reference -- the pretrained predictor on
the same latents. Saves $STABLEWM_HOME/checkpoints/cube/ens_lewm/ensemble.pt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cube.clips import dataset_path, latents_path, split_episodes  # noqa: E402
from ua.ensemble import LatentEnsemble, build_member, member_config, save_ensemble  # noqa: E402

T, FS, HS, K_AR = 6, 5, 3, 5


def clip_starts(lengths: np.ndarray, offsets: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    span = (T - 1) * FS + FS  # frames r..r+25 and the 5 actions of the last block
    starts = [offsets[e] + np.arange(0, lengths[e] - span + 1) for e in episodes if lengths[e] >= span]
    return np.concatenate(starts).astype(np.int64) if starts else np.zeros(0, np.int64)


def batch(lat: torch.Tensor, act: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents (B, T, D) and per-block actions (B, T, FS*A) for the given clip starts."""
    frames = starts[:, None] + FS * torch.arange(T, device=starts.device)
    z = lat[frames].float()
    steps = starts[:, None] + torch.arange(T * FS, device=starts.device)
    a = act[steps].reshape(len(starts), T, -1)
    return z, a


def losses(model, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    act_emb = model.action_encoder(a)
    pred = model.predict(z[:, :HS], act_emb[:, :HS])
    tf = (pred.float() - z[:, 1 : HS + 1]).pow(2).mean()
    embs, ar = [z[:, 0]], []
    for t in range(K_AR):
        lo = max(0, 1 + t - HS)
        nxt = model.predict(torch.stack(embs[lo:], 1), act_emb[:, lo : 1 + t])[:, -1]
        embs.append(nxt)
        ar.append((nxt.float() - z[:, t + 1]).pow(2).mean())
    return tf, torch.stack(ar).mean()


@torch.no_grad()
def rollout_err(model, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Per-step open-loop squared error (B, K_AR) from one context frame."""
    act_emb = model.action_encoder(a)
    embs, errs = [z[:, 0]], []
    for t in range(K_AR):
        lo = max(0, 1 + t - HS)
        nxt = model.predict(torch.stack(embs[lo:], 1), act_emb[:, lo : 1 + t])[:, -1]
        embs.append(nxt)
        errs.append((nxt.float() - z[:, t + 1]).pow(2).sum(-1))
    return torch.stack(errs, 1), torch.stack(embs[1:], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="cube/lewm", help="frozen encoder whose latents are used")
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ar-weight", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    out = Path(args.out) if args.out else home / "checkpoints/cube/ens_lewm/ensemble.pt"
    if out.exists():
        print(f"[ens] {out} exists -- delete it to retrain")
        return

    lat_np = np.load(latents_path(args.policy), mmap_mode="r")
    with h5py.File(dataset_path(args.dataset), "r") as f:
        lengths = f["ep_len"][:].astype(np.int64)
        offsets = f["ep_offset"][:].astype(np.int64)
        act_np = f["action"][:].astype(np.float32)
    ok = ~np.isnan(act_np).any(1)
    a_mean, a_std = act_np[ok].mean(0), act_np[ok].std(0, ddof=1)
    act_np = np.nan_to_num((act_np - a_mean) / a_std, nan=0.0)
    lat = torch.from_numpy(np.ascontiguousarray(lat_np)).to(dev)
    act = torch.from_numpy(act_np).to(dev)
    train_eps, val_eps = split_episodes(len(lengths))
    val_starts = torch.from_numpy(clip_starts(lengths, offsets, val_eps)[::5]).to(dev)
    print(f"[ens] latents {tuple(lat.shape)}, episodes train={len(train_eps)} val={len(val_eps)}, val clips={len(val_starts)}")

    cfg = member_config(dim=lat.shape[1])
    members, report = [], {"members": []}
    t0 = time.time()
    for m in range(args.members):
        rng = np.random.default_rng(args.seed + 1000 * m)
        boot = np.sort(rng.choice(train_eps, size=max(1, int(args.bootstrap * len(train_eps))), replace=False))
        starts = torch.from_numpy(clip_starts(lengths, offsets, boot)).to(dev)
        torch.manual_seed(args.seed + m)
        model = build_member(cfg).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        gen = torch.Generator(device=dev).manual_seed(args.seed + m)
        for step in range(1, args.steps + 1):
            idx = starts[torch.randint(len(starts), (args.batch_size,), device=dev, generator=gen)]
            z, a = batch(lat, act, idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                tf, ar = losses(model, z, a)
                loss = tf + args.ar_weight * ar
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % max(1, min(1000, args.steps // 5)) == 0 or step == 1:
                print(f"[ens] member {m} step {step} tf={tf.item():.4f} ar={ar.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        members.append(model)
        print(f"[ens] member {m} done ({time.time() - t0:.0f}s)", flush=True)

    ens = LatentEnsemble(members).eval()
    # validation: members, ensemble mean, pretrained predictor on the same frozen latents
    from planners import load_base

    base = load_base(args.policy, dev)
    errs = {f"member_{i}": [] for i in range(len(members))}
    errs.update({"ensemble_mean": [], "pretrained": [], "sigma2_ens": []})
    with torch.no_grad():
        for s in range(0, len(val_starts), 2048):
            z, a = batch(lat, act, val_starts[s : s + 2048])
            preds = []
            for i, mdl in enumerate(members):
                e, p = rollout_err(mdl, z, a)
                errs[f"member_{i}"].append(e.cpu())
                preds.append(p)
            P = torch.stack(preds)  # (K, B, K_AR, D)
            errs["ensemble_mean"].append((P.mean(0).float() - z[:, 1:]).pow(2).sum(-1).cpu())
            errs["sigma2_ens"].append(P.float().var(0, unbiased=True).sum(-1).cpu())
            e, _ = rollout_err(base, z, a)
            errs["pretrained"].append(e.cpu())
    val = {k: torch.cat(v).mean(0).tolist() for k, v in errs.items()}
    sig = torch.cat(errs["sigma2_ens"]).numpy()
    em = torch.cat(errs["ensemble_mean"]).numpy()
    from cube.metrics import spearman

    report.update({
        "val_rollout_sq_err_per_step": val,
        "val_spearman_sigma2_vs_err_per_step": [spearman(sig[:, k], em[:, k]) for k in range(K_AR)],
        "args": vars(args),
        "minutes": (time.time() - t0) / 60,
    })
    save_ensemble(ens.cpu(), cfg, out, {"policy": args.policy, "dataset": args.dataset, "report": report})
    (out.parent / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
