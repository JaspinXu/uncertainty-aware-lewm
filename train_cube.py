"""Train LeWM and the L4 representation variants on OGBench-Cube.

    python train_cube.py --variant lewm --max-steps 60000
    python train_cube.py --variant prop_ms --max-steps 60000 --time-limit-min 170   # resumable

Variants (all share the LeWM recipe: ViT-tiny/14 encoder, 6-layer AdaLN
predictor with history 3, batch 128, AdamW lr 5e-5 / wd 1e-3, bf16,
grad-clip 1, SIGReg weight 0.09, frameskip 5):

  lewm        control: LeWM retrained with our budget (1-step teacher forcing)
              (saved as cube/l4_lewm; the pretrained model stays cube/lewm)
  ms          + 5-step autoregressive loss (the rollout the planner uses)
  prop        + proprio block z=[z_vis; z_prop] + de-redundancy penalty
  prop_aux    + training-only cube-pose head on z_vis
  prop_ms     prop_aux + 5-step autoregressive loss (everything)

Checkpoints: $STABLEWM_HOME/checkpoints/cube/l4_<variant>/{weights.pt,config.json}
(loadable with ``policy=cube/l4_<variant>`` in evaluate.py), resumable training state
in .../<run>/state/train_state.pt, metrics in .../<run>/log.csv.
Writes ``.../<run>/DONE`` when max-steps is reached.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cube.clips import CubeClips, dataset_path, normalize_pixels, proprio_columns, split_episodes  # noqa: E402
from module import SIGReg  # noqa: E402

VARIANTS = {
    "lewm": dict(ar_steps=0, prop=False, aux=False),
    "ms": dict(ar_steps=5, prop=False, aux=False),
    "prop": dict(ar_steps=0, prop=True, aux=False),
    "prop_aux": dict(ar_steps=0, prop=True, aux=True),
    "prop_ms": dict(ar_steps=5, prop=True, aux=True),
}
POS_COL, YAW_COL = "privileged_block_0_pos", "privileged_block_0_yaw"

_MLP = {
    "_target_": "stable_worldmodel.wm.lewm.module.MLP",
    "hidden_dim": 2048,
    "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
}


def model_config(prop_keys: list[str], prop_dim: int, args, spec: dict) -> dict:
    vis = 192
    lat = vis + (args.prop_latent if spec["prop"] else 0)
    return {
        "_target_": "cube.variants.LeWMVariant",
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": "tiny",
            "patch_size": 14,
            "image_size": 224,
            "pretrained": False,
            "use_mask_token": False,
        },
        "predictor": {
            "_target_": "stable_worldmodel.wm.lewm.module.Predictor",
            "num_frames": 3,
            "input_dim": lat,
            "hidden_dim": 192,
            "output_dim": lat,
            "depth": 6,
            "heads": 16,
            "mlp_dim": 2048,
            "dim_head": 64,
            "dropout": 0.1,
            "emb_dropout": 0.0,
        },
        # the predictor's AdaLN conditioning must have the predictor input width
        "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder", "input_dim": 25, "emb_dim": lat},
        "projector": {**copy.deepcopy(_MLP), "input_dim": 192, "output_dim": vis},
        "pred_proj": {**copy.deepcopy(_MLP), "input_dim": lat, "output_dim": lat},
        "prop_keys": prop_keys if spec["prop"] else [],
        "prop_dim": prop_dim if spec["prop"] else 0,
        "prop_latent": args.prop_latent if spec["prop"] else 0,
        "aux_dim": 5 if spec["aux"] else 0,
        "vis_dim": vis,
        "w_prop": 1.0,
        "history_size": 3,
    }


def r2_redundancy(z_vis: torch.Tensor, z_prop: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Fraction of z_vis variance linearly explained by (stop-grad) z_prop over the batch."""
    v = z_vis.float() - z_vis.float().mean(0, keepdim=True)
    p = z_prop.detach().float() - z_prop.detach().float().mean(0, keepdim=True)
    gram = p.T @ p + eps * len(p) * torch.eye(p.shape[1], device=p.device)
    beta = torch.linalg.solve(gram, p.T @ v)
    return (p @ beta).pow(2).sum() / v.pow(2).sum().clamp_min(1e-8)


def aux_targets(batch: dict, pos_mean: torch.Tensor, pos_std: torch.Tensor) -> torch.Tensor:
    pos = (batch[POS_COL] - pos_mean) / pos_std  # (B, T, 3)
    yaw = batch[YAW_COL][..., :1]
    return torch.cat([pos, torch.cos(4 * yaw), torch.sin(4 * yaw)], dim=-1)


def lr_at(step: int, max_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def compute_losses(model, sigreg, batch, spec, args, stats) -> dict:
    dev = args.device
    pixels = normalize_pixels(batch["pixels"].to(dev, non_blocking=True))
    info = {"pixels": pixels, "action": batch["action"].to(dev, non_blocking=True)}
    if spec["prop"]:
        info["proprio"] = torch.cat([batch[k].to(dev, non_blocking=True) for k in model.prop_keys], dim=-1)
    out = model.encode(info)
    emb, act_emb = out["emb"], out["act_emb"]  # (B, T, D), (B, T, A)
    HS = 3
    losses = {}
    pred = model.predict(emb[:, :HS], act_emb[:, :HS])
    losses["pred"] = (pred.float() - emb[:, 1 : HS + 1].float()).pow(2).mean()
    losses["sigreg"] = sigreg(emb.transpose(0, 1).float())
    total = losses["pred"] + args.sigreg_weight * losses["sigreg"]
    if spec["ar_steps"]:
        K = spec["ar_steps"]
        embs, ar = [emb[:, 0]], []
        for t in range(K):
            lo = max(0, 1 + t - HS)
            nxt = model.predict(torch.stack(embs[lo:], 1), act_emb[:, lo : 1 + t])[:, -1]
            embs.append(nxt)
            ar.append((nxt.float() - emb[:, t + 1].float()).pow(2).mean())
        losses["ar"] = torch.stack(ar).mean()
        total = total + args.ar_weight * losses["ar"]
    if spec["prop"]:
        z_vis = out["emb_vis"].flatten(0, 1)
        z_prop = out["emb_prop"].flatten(0, 1)
        losses["redund"] = r2_redundancy(z_vis, z_prop)
        total = total + args.redund_weight * losses["redund"]
    if spec["aux"]:
        tgt = aux_targets({k: batch[k].to(dev) for k in (POS_COL, YAW_COL)}, *stats["pos"])
        losses["aux"] = F.mse_loss(model.aux_head(out["emb_vis"].float()), tgt)
        total = total + args.aux_weight * losses["aux"]
    losses["loss"] = total
    return losses


@torch.no_grad()
def validate(model, sigreg, loader, spec, args, stats, max_batches: int = 50) -> dict:
    model.eval()
    acc: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            losses = compute_losses(model, sigreg, batch, spec, args, stats)
        for k, v in losses.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {f"val_{k}": v / max(1, n) for k, v in acc.items()}


def save_inference(model, cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "weights.pt.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, run_dir / "weights.pt")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--run-name", default=None, help="checkpoint folder under checkpoints/cube/ (default: variant)")
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--sigreg-weight", type=float, default=0.09)
    ap.add_argument("--ar-weight", type=float, default=1.0)
    ap.add_argument("--redund-weight", type=float, default=0.5)
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--prop-latent", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--time-limit-min", type=float, default=0, help="save + exit before this many minutes (0 = off)")
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    spec = VARIANTS[args.variant]
    t_start = time.time()

    import stable_worldmodel as swm
    from hydra.utils import instantiate

    run = args.run_name or f"l4_{args.variant}"  # never clash with the pretrained cube/lewm
    run_dir = swm.data.utils.get_cache_dir(sub_folder="checkpoints") / "cube" / run
    state_path = run_dir / "state" / "train_state.pt"
    if (run_dir / "DONE").exists():
        print(f"[train] {run} already finished ({run_dir / 'DONE'})")
        return
    (run_dir / "state").mkdir(parents=True, exist_ok=True)

    path = dataset_path(args.dataset)
    prop_keys = proprio_columns(path)
    columns = (prop_keys if spec["prop"] else []) + ([POS_COL, YAW_COL] if spec["aux"] else [])
    T = max(3 + 1, 1 + spec["ar_steps"])  # history 3 + 1-step target; 1 + K frames for the K-step rollout
    with __import__("h5py").File(path, "r") as f:
        n_eps = len(f["ep_len"])
    train_eps, val_eps = split_episodes(n_eps)
    train_ds = CubeClips(path, T, 5, train_eps, columns)
    val_ds = CubeClips(path, T, 5, val_eps, columns, action_stats=(train_ds.action_mean, train_ds.action_std), stride=5)
    print(f"[train] {run}: variant={args.variant} T={T} clips train={len(train_ds)} val={len(val_ds)} "
          f"episodes={n_eps} prop_keys={prop_keys if spec['prop'] else []}")

    stats = {}
    prop_dim = int(sum(train_ds.cache[k].shape[1] for k in prop_keys)) if spec["prop"] else 0
    cfg = model_config(prop_keys, prop_dim, args, spec)
    torch.manual_seed(args.seed)
    model = instantiate(copy.deepcopy(cfg)).to(args.device)
    if spec["prop"]:
        prop_all = np.concatenate([train_ds.cache[k] for k in prop_keys], axis=1)
        model.prop_mean.copy_(torch.from_numpy(prop_all.mean(0)))
        model.prop_std.copy_(torch.from_numpy(prop_all.std(0, ddof=1).clip(1e-6)))
    if spec["aux"]:
        pos = train_ds.cache[POS_COL]
        stats["pos"] = (torch.tensor(pos.mean(0), device=args.device), torch.tensor(pos.std(0, ddof=1).clip(1e-6), device=args.device))
    sigreg = SIGReg(knots=17, num_proj=1024).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step = int(st["step"])
        torch.set_rng_state(st["torch_rng"])
        print(f"[train] resumed from step {step}")
    (run_dir / "train_args.json").write_text(json.dumps({**vars(args), "T": T, "columns": columns}, indent=1))

    g = torch.Generator().manual_seed(args.seed + step)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(4, args.workers), pin_memory=True
    )

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    logf = log_path.open("a", newline="")
    writer = None
    model.train()
    use_amp = args.device.startswith("cuda")
    t_log, n_seen = time.time(), 0
    done = False
    while not done:
        for batch in loader:
            for group in opt.param_groups:
                group["lr"] = lr_at(step, args.max_steps, args.lr, args.warmup_steps)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                losses = compute_losses(model, sigreg, batch, spec, args, stats)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            n_seen += args.batch_size
            if step % args.log_every == 0 or step == 1:
                row = {"step": step, "lr": opt.param_groups[0]["lr"], "grad_norm": float(gn),
                       "samples_per_s": n_seen / max(1e-6, time.time() - t_log),
                       **{k: float(v) for k, v in losses.items()}}
                if step % args.val_every == 0:
                    row.update(validate(model, sigreg, val_loader, spec, args, stats))
                if writer is None:
                    writer = csv.DictWriter(logf, fieldnames=sorted(set(row) | {"val_loss", "val_pred", "val_sigreg", "val_ar", "val_redund", "val_aux", "ar", "redund", "aux"}))
                    if new_log:
                        writer.writeheader()
                writer.writerow(row)
                logf.flush()
                print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
                t_log, n_seen = time.time(), 0
            out_of_time = args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min
            if step % args.ckpt_every == 0 or step >= args.max_steps or out_of_time:
                tmp = state_path.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "torch_rng": torch.get_rng_state()}, tmp)
                os.replace(tmp, state_path)
                save_inference(model, cfg, run_dir)
            if step >= args.max_steps:
                done = True
                break
            if out_of_time:
                print(f"[train] time limit reached at step {step}; state saved, resubmit to continue")
                logf.close()
                sys.exit(3)
    logf.close()
    save_inference(model, cfg, run_dir)
    (run_dir / "DONE").write_text(f"{step}\n")
    print(f"[train] finished {run} at step {step} ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
