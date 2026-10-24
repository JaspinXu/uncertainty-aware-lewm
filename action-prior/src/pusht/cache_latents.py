"""Cache frozen LeWM features for an explicitly recorded trajectory split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import h5py
import hdf5plugin  # noqa
import numpy as np
import torch
import stable_worldmodel as swm


def image_tensor(images, device):
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    from torchvision.transforms.v2 import Resize
    return Resize(224)(((x - mean) / std))[:, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get("LEWM_ROOT", "workspace"))
    p.add_argument("--output", required=True)
    p.add_argument("--max-episodes", type=int, default=300, help="0 = all; pilot uses 300")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    os.environ["STABLEWM_HOME"] = str(root)
    model = swm.wm.utils.load_pretrained("pusht/lewm").cuda().eval().requires_grad_(False)
    source = root / "datasets/pusht_expert_train.h5"
    start_time = time.perf_counter()
    with h5py.File(source, "r") as src:
        lengths, offsets = src["ep_len"][:].astype(int), src["ep_offset"][:].astype(int)
        perm = np.random.default_rng(args.seed).permutation(np.flatnonzero(lengths > 26))
        perm = perm[:min(340,len(perm))]
        assert len(perm) >= 150, 'Need enough independent trajectories for 50 test and 50 validation'
        n_train, n_val = len(perm)-100, 50
        splits = {"train": sorted(perm[:n_train].tolist()),
                  "val": sorted(perm[n_train:n_train+n_val].tolist()),
                  "test": sorted(perm[n_train+n_val:].tolist())}
        ids = sorted(perm.tolist())
        total = int(lengths[ids].sum())
        # This fixed scaler matches the *original* evaluator/pretrained dynamics.
        # It is not a new learned proposal feature statistic.
        actions = src["action"][:]
        finite = np.isfinite(actions).all(axis=1)
        action_mean = actions[finite].mean(axis=0, dtype=np.float64)
        action_std = actions[finite].std(axis=0, dtype=np.float64)
        action_std[action_std < 1e-12] = 1
        manifest = {"seed": args.seed, "splits": splits, "source": str(source),
                    "world_model": "pusht/lewm", "goal_offset": 25,
                    "archive_sha256": "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318",
                    "paper_commit": (root / "setup/paper-code.commit").read_text().strip(),
                    "dependency_commit": (root / "setup/paper-dependency.commit").read_text().strip(),
                    "checkpoint_sha256": hashlib.sha256((root / "checkpoints/pusht/lewm/weights.pt").read_bytes()).hexdigest(),
                    "scaler": "Original full-dataset fixed action scaler; latent stats train-only",
                    "note": "Held out from proposal training, NOT necessarily from pretrained LeWM"}
        with h5py.File(str(output) + ".partial", "w") as dst:
            dst.attrs["manifest"] = json.dumps(manifest)
            dst.create_dataset("episode_ids", data=ids)
            dst.create_dataset("lengths", data=lengths[ids])
            dst.create_dataset("source_offsets", data=offsets[ids])
            dst.create_dataset("offsets", data=np.r_[0, np.cumsum(lengths[ids])[:-1]])
            dst.create_dataset("action_mean", data=action_mean)
            dst.create_dataset("action_std", data=action_std)
            z = dst.create_dataset("z", (total, 192), dtype="f4")
            a = dst.create_dataset("action", (total, actions.shape[1]), dtype="f4")
            cursor = 0
            for i, ep in enumerate(ids):
                n, off = int(lengths[ep]), int(offsets[ep])
                a[cursor:cursor+n] = (actions[off:off+n] - action_mean) / action_std
                for t in range(0, n, args.batch_size):
                    end = min(n, t + args.batch_size)
                    pixels = image_tensor(src["pixels"][off+t:off+end], "cuda")
                    with torch.inference_mode():
                        emb = model.encode({"pixels": pixels})["emb"][:, 0]
                    z[cursor+t:cursor+end] = emb.cpu().numpy()
                cursor += n
                if (i+1) % 10 == 0 or i+1 == len(ids):
                    print(json.dumps({"episodes": i+1, "total_episodes": len(ids),
                                      "frames": cursor, "elapsed_s": time.perf_counter()-start_time}), flush=True)
            dst.attrs["complete"] = True
    os.replace(str(output) + ".partial", output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print("CACHE READY", output, flush=True)


if __name__ == "__main__":
    main()
