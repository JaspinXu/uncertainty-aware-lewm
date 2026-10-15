"""Encode every dataset frame with a frozen LeWM encoder (for the latent ensemble).

    python scripts/cube/encode_dataset.py --policy cube/lewm

Writes $STABLEWM_HOME/latents/<policy with / -> __>.npy, a float16 memmap of
shape (rows, D) aligned with the HDF5 rows, plus a .json with metadata. Frames
are normalised exactly like the planner's observations (ImageNet statistics)
and encoded in fp32 with the projector in eval mode, so a stored latent equals
the latent the planner computes for the same frame.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

from cube.clips import dataset_path, latents_path, normalize_pixels  # noqa: E402


class Chunks(torch.utils.data.Dataset):
    def __init__(self, path: Path, n_rows: int, chunk: int):
        self.path, self.n, self.chunk = str(path), n_rows, chunk
        self._f = None

    def __len__(self):
        return (self.n + self.chunk - 1) // self.chunk

    def __getitem__(self, i):
        if self._f is None:
            self._f = h5py.File(self.path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
        s = i * self.chunk
        px = self._f["pixels"][s : min(self.n, s + self.chunk)]
        return i, torch.from_numpy(np.ascontiguousarray(px)).permute(0, 3, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="cube/lewm")
    ap.add_argument("--dataset", default="ogbench/cube_single_expert")
    ap.add_argument("--chunk", type=int, default=512, help="rows read per HDF5 access")
    ap.add_argument("--workers", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from planners import load_base

    model = load_base(args.policy, args.device)
    src = dataset_path(args.dataset)
    with h5py.File(src, "r") as f:
        n_rows = int(f["pixels"].shape[0])
    out = latents_path(args.policy)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix(".json")
    if meta_path.exists() and json.loads(meta_path.read_text()).get("complete"):
        print(f"[encode] {out} already complete")
        return
    with torch.no_grad():
        dim = int(model.encode({"pixels": torch.zeros(1, 1, 3, 224, 224, device=args.device)})["emb"].shape[-1])
    lat = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n_rows, dim))
    loader = torch.utils.data.DataLoader(
        Chunks(src, n_rows, args.chunk), batch_size=None, shuffle=False, num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        multiprocessing_context="spawn" if args.workers else None,  # lance (imported by stable_worldmodel) is not fork-safe
    )
    t0 = time.time()
    with torch.no_grad():
        for i, px in loader:
            x = normalize_pixels(px.to(args.device, non_blocking=True))
            z = model.encode({"pixels": x[:, None]})["emb"][:, 0]
            s = int(i) * args.chunk
            lat[s : s + len(z)] = z.float().cpu().numpy().astype(np.float16)
            if int(i) % 200 == 0:
                done = s + len(z)
                print(f"[encode] {done}/{n_rows} rows ({done / max(1e-6, time.time() - t0):.0f} frames/s)", flush=True)
    lat.flush()
    meta = {"policy": args.policy, "dataset": args.dataset, "rows": n_rows, "dim": dim, "dtype": "float16",
            "seconds": time.time() - t0, "complete": True}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[encode] wrote {out} {meta}")


if __name__ == "__main__":
    main()
