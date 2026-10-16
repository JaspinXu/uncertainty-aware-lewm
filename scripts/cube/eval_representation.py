"""L5: what do the Cube latents encode, and how well do they predict ahead?

    python scripts/cube/eval_representation.py --policies cube/lewm cube/l4_lewm cube/l4_prop_ms

For every checkpoint, on episodes held out from L4 training (the same
episode-level split train_cube.py uses):

probes   linear (ridge) and MLP probes from the latent of a frame to the cube
         position, cube yaw (cos 4θ, sin 4θ: the cube is 90° symmetric), arm joint
         positions / velocities, end-effector position and gripper opening;
         fitted on 80% of the held-out episodes, scored on the other 20%
         (Pearson r averaged over target dims, as LeWM Table 4, and R²).
k-step   open-loop rollouts of the planner (1 context frame, 5 action blocks):
         latent error ||ẑ_k - z_k||² relative to the latent variance, against the
         "nothing moves" persistence baseline, and the cube-position error in cm
         read out of ẑ_k with the linear cube probe (comparable across models).

Writes results/cube/l5/repr_<checkpoint>.json per checkpoint and the merged
representation.json / representation.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cube.clips import CubeClips, dataset_path, normalize_pixels, proprio_columns, split_episodes  # noqa: E402
from cube.rollout import rollout_from_latents  # noqa: E402

TARGETS = {
    "cube_pos": ["privileged_block_0_pos"],
    "cube_yaw4": ["privileged_block_0_yaw"],
    "joint_pos": ["proprio_joint_pos"],
    "joint_vel": ["proprio_joint_vel"],
    "effector_pos": ["proprio_effector_pos"],
    "gripper": ["proprio_gripper_opening"],
}


def target_matrix(batch_cols: dict, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for t in names:
        col = TARGETS[t][0]
        v = batch_cols[col]
        if t == "cube_yaw4":
            v = np.concatenate([np.cos(4 * v[:, :1]), np.sin(4 * v[:, :1])], axis=1)
        out[t] = v
    return out


def pearson_mean(pred: np.ndarray, y: np.ndarray) -> float:
    rs = []
    for j in range(y.shape[1]):
        if np.std(y[:, j]) < 1e-9 or np.std(pred[:, j]) < 1e-9:
            continue
        rs.append(np.corrcoef(pred[:, j], y[:, j])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((pred - y) ** 2).sum() / max(ss, 1e-12))


class Ridge:
    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha

    def fit(self, X, Y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        A = Xs.T @ Xs + self.alpha * len(Xs) * np.eye(Xs.shape[1])
        self.W = np.linalg.solve(A, Xs.T @ (Y - self.my))
        return self

    def predict(self, X):
        return ((X - self.mx) / self.sx) @ self.W + self.my


def mlp_probe(Xtr, Ytr, Xte, seed=0, epochs=200, device="cpu"):
    torch.manual_seed(seed)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
    my, sy = Ytr.mean(0), Ytr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mx) / sx, dtype=torch.float32, device=device)
    yt = torch.tensor((Ytr - my) / sy, dtype=torch.float32, device=device)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, yt.shape[1])
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for s in range(0, len(xt), 512):
            idx = perm[s : s + 512]
            loss = torch.nn.functional.mse_loss(net(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = net(torch.tensor((Xte - mx) / sx, dtype=torch.float32, device=device)).cpu().numpy()
    return p * sy + my


@torch.no_grad()
def encode_clips(model, ds: CubeClips, device, batch: int = 64, max_clips: int | None = None, seed: int = 0):
    """Latents of every frame of (a subsample of) the clips: (N, T, D) + columns (N, T, ...)."""
    idx = np.arange(len(ds))
    if max_clips and len(idx) > max_clips:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_clips, replace=False))
    Z, ZV, A, cols = [], [], [], {c: [] for c in ds.columns}
    for s in range(0, len(idx), batch):
        items = [ds[i] for i in idx[s : s + batch]]
        px = normalize_pixels(torch.stack([it["pixels"] for it in items]).to(device))
        info = {"pixels": px}
        if getattr(model, "prop_latent", 0):
            info["proprio"] = torch.cat([torch.stack([it[k] for it in items]) for k in model.prop_keys], -1).to(device)
        out = model.encode(info)
        Z.append(out["emb"].float().cpu())
        ZV.append(out.get("emb_vis", out["emb"]).float().cpu())
        A.append(torch.stack([it["action"] for it in items]))
        for c in ds.columns:
            cols[c].append(torch.stack([it[c] for it in items]))
    return (
        torch.cat(Z).numpy(),
        torch.cat(ZV).numpy(),
        torch.cat(A),
        {c: torch.cat(v).numpy() for c, v in cols.items()},
        idx,
    )


def evaluate_policy(policy: str, args, device) -> dict:
    from planners import load_base

    model = load_base(policy, device)
    path = dataset_path(args.dataset)
    with h5py.File(path, "r") as f:
        n_eps = len(f["ep_len"])
    _, val_eps = split_episodes(n_eps, val_frac=args.val_frac)  # same split as train_cube.py
    rng = np.random.default_rng(0)
    perm = rng.permutation(val_eps)
    n_te = max(1, len(perm) // 5)
    probe_tr, probe_te = np.sort(perm[n_te:]), np.sort(perm[:n_te])
    cols = sorted({TARGETS[t][0] for t in TARGETS} | set(getattr(model, "prop_keys", []) or []) | set(proprio_columns(path)))
    T = 6  # 1 context frame + 5 predicted blocks (the planner's rollout)
    train_ds = CubeClips(path, T, 5, probe_tr, cols, stride=args.stride)
    test_ds = CubeClips(path, T, 5, probe_te, cols, action_stats=(train_ds.action_mean, train_ds.action_std), stride=args.stride)
    Ztr, ZVtr, _, Ctr, _ = encode_clips(model, train_ds, device, max_clips=args.max_clips)
    Zte, ZVte, Ate, Cte, _ = encode_clips(model, test_ds, device, max_clips=args.max_clips // 4)
    res = {"policy": policy, "n_probe_train_frames": int(Ztr.shape[0] * T), "n_probe_test_frames": int(Zte.shape[0] * T),
           "latent_dim": int(Ztr.shape[-1]), "probes": {}}

    flat = lambda a: a.reshape(-1, a.shape[-1])  # noqa: E731
    Ytr = target_matrix({c: flat(v) for c, v in Ctr.items()}, list(TARGETS))
    Yte = target_matrix({c: flat(v) for c, v in Cte.items()}, list(TARGETS))
    feats = {"z": (flat(Ztr), flat(Zte))}
    if ZVtr.shape[-1] != Ztr.shape[-1]:
        feats["z_vis"] = (flat(ZVtr), flat(ZVte))
    cube_probe = None
    for fname, (Xtr, Xte) in feats.items():
        for t in TARGETS:
            lin = Ridge().fit(Xtr, Ytr[t])
            pl = lin.predict(Xte)
            pm = mlp_probe(Xtr, Ytr[t], Xte, epochs=args.mlp_epochs, device=device)
            res["probes"][f"{fname}/{t}"] = {
                "linear_r": pearson_mean(pl, Yte[t]), "linear_r2": r2(pl, Yte[t]),
                "mlp_r": pearson_mean(pm, Yte[t]), "mlp_r2": r2(pm, Yte[t]),
            }
            if fname == "z" and t == "cube_pos":
                cube_probe = lin

    # k-step open-loop prediction from one context frame (exactly the planner's rollout)
    z_true = torch.from_numpy(Zte)  # (N, T, D)
    z_hist = z_true[:, :1].to(device)
    acts = Ate[:, None].to(device)  # (N, 1, T, A) -> one candidate per clip
    preds = []
    for s in range(0, len(z_hist), 256):
        preds.append(rollout_from_latents(model, z_hist[s : s + 256], acts[s : s + 256, :, : T - 1]).float().cpu())
    pred = torch.cat(preds)[:, 0].numpy()  # (N, T-1, D)
    target = Zte[:, 1:]
    var = Zte.reshape(-1, Zte.shape[-1]).var(0).sum()
    cube_true = Cte["privileged_block_0_pos"][:, 1:]
    res["kstep"] = []
    for k in range(T - 1):
        err = ((pred[:, k] - target[:, k]) ** 2).sum(-1) / var
        pers = ((Zte[:, 0] - target[:, k]) ** 2).sum(-1) / var
        cube_cm = 100 * np.linalg.norm(cube_probe.predict(pred[:, k]) - cube_true[:, k], axis=-1)
        cube_cm_true = 100 * np.linalg.norm(cube_probe.predict(target[:, k]) - cube_true[:, k], axis=-1)
        res["kstep"].append({
            "k": k + 1, "rel_latent_err": float(err.mean()), "persistence_rel_err": float(pers.mean()),
            "cube_pos_err_cm": float(np.median(cube_cm)), "cube_pos_err_cm_true_latent": float(np.median(cube_cm_true)),
        })
    return res


def write_md(all_res: list[dict], path: Path) -> None:
    L = ["# L5 representation evaluation (held-out episodes)\n", "Generated by `scripts/cube/eval_representation.py`.\n",
         "## Probes: Pearson r (linear / MLP), latent z\n"]
    tnames = list(TARGETS)
    L.append("| checkpoint | dim | " + " | ".join(tnames) + " |")
    L.append("|---|---:|" + "---:|" * len(tnames))
    for r in all_res:
        cells = []
        for t in tnames:
            p = r["probes"].get(f"z/{t}")
            cells.append(f"{p['linear_r']:.3f} / {p['mlp_r']:.3f}" if p else "n/a")
        L.append(f"| {r['policy']} | {r['latent_dim']} | " + " | ".join(cells) + " |")
    vis = [r for r in all_res if any(k.startswith("z_vis/") for k in r["probes"])]
    if vis:
        L.append("\nVisual block z_vis only (factorised models):\n")
        L.append("| checkpoint | " + " | ".join(tnames) + " |")
        L.append("|---|" + "---:|" * len(tnames))
        for r in vis:
            L.append(f"| {r['policy']} | " + " | ".join(
                f"{r['probes'][f'z_vis/{t}']['linear_r']:.3f} / {r['probes'][f'z_vis/{t}']['mlp_r']:.3f}" for t in tnames) + " |")
    L.append("\n## Open-loop prediction (planner rollout: 1 context frame, k action blocks)\n")
    L.append("Relative latent error ||ẑ_k − z_k||² / total latent variance (persistence baseline in brackets); "
             "median cube-position error [cm] decoded from ẑ_k by the linear cube probe (from the true z_k in brackets).\n")
    ks = [e["k"] for e in all_res[0]["kstep"]] if all_res else []
    L.append("| checkpoint | " + " | ".join(f"k={k}" for k in ks) + " | " + " | ".join(f"cube cm k={k}" for k in ks) + " |")
    L.append("|---|" + "---:|" * (2 * len(ks)))
    for r in all_res:
        a = [f"{e['rel_latent_err']:.3f} ({e['persistence_rel_err']:.3f})" for e in r["kstep"]]
        b = [f"{e['cube_pos_err_cm']:.2f} ({e['cube_pos_err_cm_true_latent']:.2f})" for e in r["kstep"]]
        L.append(f"| {r['policy']} | " + " | ".join(a) + " | " + " | ".join(b) + " |")
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--max-clips", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=10, help="clip start stride inside an episode")
    ap.add_argument("--mlp-epochs", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.05, help="must match train_cube.py's split")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/l5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.policies:
        print(f"[repr] {p}")
        res = evaluate_policy(p, args, args.device)
        (args.out / f"repr_{p.replace('/', '__')}.json").write_text(json.dumps(res, indent=1))
    # one file per checkpoint (jobs for different variants run concurrently); merge all
    results = [json.loads(f.read_text()) for f in sorted(args.out.glob("repr_*.json"))]
    order = {"cube/lewm": 0}
    results.sort(key=lambda r: (order.get(r["policy"], 1), r["policy"]))
    (args.out / "representation.json").write_text(json.dumps({r["policy"]: r for r in results}, indent=1))
    write_md(results, args.out / "representation.md")
    print((args.out / "representation.md").read_text())


if __name__ == "__main__":
    main()
    os._exit(0)
