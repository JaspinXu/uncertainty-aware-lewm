"""Fast latent-space behavioral cloning; frozen world model is not trained."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from prior import ActionPrior


def valid_starts(offset, length, actions, horizon=25):
    starts = np.arange(offset, offset + max(0, length - horizon))
    bad = ~np.isfinite(actions).all(axis=1)
    prefix = np.r_[0, np.cumsum(bad)]
    return starts[(prefix[starts+horizon] - prefix[starts]) == 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.cache, "r") as f:
        assert f.attrs["complete"]
        manifest = json.loads(f.attrs["manifest"])
        z, a = f["z"][:], f["action"][:]
        starts = {"train": [], "val": []}
        train_rows = []
        for ep, off, length in zip(f["episode_ids"][:], f["offsets"][:], f["lengths"][:]):
            for split in starts:
                if int(ep) in manifest["splits"][split]:
                    starts[split].append(valid_starts(int(off), int(length), a))
                    if split == "train":
                        train_rows.append(np.arange(off, off+length))
        action_mean, action_std = f["action_mean"][:], f["action_std"][:]
    train_rows = np.concatenate(train_rows)
    assert np.isfinite(z).all()
    model = ActionPrior(action_dim=a.shape[1], dim=z.shape[1]).cuda()
    model.z_mean.copy_(torch.tensor(z[train_rows].mean(0), device="cuda"))
    model.z_std.copy_(torch.tensor(z[train_rows].std(0).clip(.001), device="cuda"))
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed)
    best, first = float("inf"), 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        assert ck["manifest"] == manifest
        model.load_state_dict(ck["state"])
        optim.load_state_dict(ck["optimizer"])
        rng.set_state(ck["rng"])
        first, best = ck["step"], ck["best_val"]
    z, a = torch.tensor(z, device="cuda"), torch.tensor(a, device="cuda")
    starts = {k: torch.tensor(np.concatenate(v), dtype=torch.long) for k, v in starts.items()}
    assert all(len(v) for v in starts.values())
    val_rng = torch.Generator().manual_seed(123)
    val = starts["val"][torch.randperm(len(starts["val"]), generator=val_rng)[:4096]]
    delta = torch.arange(25, device="cuda")

    def loss_for(rows):
        rows = rows.to("cuda")
        target = a[rows[:, None] + delta].reshape(-1, 5, 5*a.shape[1])
        return (model(z[rows], z[rows+25]) - target).square().mean()

    started = time.perf_counter()
    for step in range(first+1, args.steps+1):
        model.train()
        rows = starts["train"][torch.randint(len(starts["train"]), (args.batch_size,), generator=rng)]
        loss = loss_for(rows)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        if step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                vloss = sum(loss_for(part).item()*len(part) for part in val.split(256)) / len(val)
            improved = vloss < best
            best = min(best, vloss)
            ck = {"state": model.state_dict(), "spec": model.spec, "optimizer": optim.state_dict(),
                  "step": step, "best_val": best, "rng": rng.get_state(), "manifest": manifest,
                  "action_mean": torch.tensor(action_mean), "action_std": torch.tensor(action_std)}
            torch.save(ck, out / "last.pt")
            if improved:
                torch.save(ck, out / "best.pt")
            record = {"step": step, "train_mse": loss.item(), "val_mse": vloss,
                      "best_val_mse": best, "elapsed_s": time.perf_counter()-started}
            with (out / "training.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
