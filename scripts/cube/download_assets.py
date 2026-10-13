"""Fetch the OGBench-Cube dataset and the pretrained LeWM-Cube checkpoint.

    python scripts/cube/download_assets.py               # both
    python scripts/cube/download_assets.py --only model  # checkpoint only (72 MB)
    python scripts/cube/download_assets.py --only data   # dataset only (46 GB archive)

Layout written under $STABLEWM_HOME (see cluster/env.sh):

    checkpoints/cube/lewm/{config.json,weights.pt}    <- HF model  quentinll/lewm-cube
    datasets/ogbench/cube_single_expert.h5            <- HF dataset quentinll/lewm-cube
                                                         (cube_single_expert.tar.zst)

The archive download is resumable (re-run the script after an interruption).
Extraction streams the .tar.zst (no system zstd needed) into a temporary
directory next to the target, then moves the .h5 into place. Use
``--keep-archive`` to keep the 46 GB archive after a successful extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

MODEL_REPO = "quentinll/lewm-cube"
DATA_REPO = "quentinll/lewm-cube"
ARCHIVE = "cube_single_expert.tar.zst"
DATASET_NAME = "ogbench/cube_single_expert"


def stablewm_home() -> Path:
    home = os.environ.get("STABLEWM_HOME")
    if not home:
        sys.exit("STABLEWM_HOME is not set (source cluster/env.sh first)")
    return Path(home)


def fetch_model(root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    out = root / "checkpoints" / "cube" / "lewm"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "weights.pt"):
        p = hf_hub_download(MODEL_REPO, name, repo_type="model", local_dir=out)
        print(f"[model] {p} ({Path(p).stat().st_size / 1e6:.1f} MB)")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg.get("_target_") == "stable_worldmodel.wm.lewm.LeWM", cfg.get("_target_")
    assert cfg["action_encoder"]["input_dim"] == 25, cfg["action_encoder"]
    # stray HF cache metadata would make load_pretrained see extra files: keep it out
    shutil.rmtree(out / ".cache", ignore_errors=True)
    return out


def fetch_archive(root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = Path(hf_hub_download(DATA_REPO, ARCHIVE, repo_type="dataset", local_dir=raw))
    print(f"[data] archive {p} ({p.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    return p


def extract(archive: Path, root: Path) -> Path:
    import zstandard

    target = root / "datasets" / f"{DATASET_NAME}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / "_extract_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with archive.open("rb") as fh, dctx.stream_reader(fh, read_size=1 << 24) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"unsafe path in archive: {member.name}")
                dest = tmp / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                with dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 24)
                print(f"[data] extracted {member.name} ({member.size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    h5s = sorted(tmp.rglob("*.h5"))
    if len(h5s) != 1:
        raise RuntimeError(f"expected exactly one .h5 in the archive, found {h5s}")
    shutil.move(str(h5s[0]), target)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[data] dataset at {target} ({target.stat().st_size / 1e9:.2f} GB)")
    return target


def check_dataset(path: Path) -> None:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as f:
        n_ep = len(f["ep_len"])
        print(f"[data] {path.name}: {n_ep} episodes, {int(f['ep_len'][:].sum())} rows")
        for k in sorted(f.keys()):
            obj = f[k]
            shape = getattr(obj, "shape", None)
            print(f"         {k:32s} {shape} {getattr(obj, 'dtype', '')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("model", "data"), default=None)
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()
    root = stablewm_home()
    print(f"[assets] STABLEWM_HOME={root}")
    if args.only in (None, "model"):
        fetch_model(root)
    if args.only in (None, "data"):
        target = root / "datasets" / f"{DATASET_NAME}.h5"
        if target.exists():
            print(f"[data] {target} already present, skipping download")
        else:
            archive = fetch_archive(root)
            extract(archive, root)
            if not args.keep_archive:
                archive.unlink()
                print(f"[data] removed archive {archive}")
        check_dataset(target)
    print("[assets] done")


if __name__ == "__main__":
    main()
