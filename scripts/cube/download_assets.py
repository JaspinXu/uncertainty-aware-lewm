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


def http_download(url: str, dest: Path, chunk: int = 1 << 24) -> None:
    """Plain HTTPS download with resume (fallback when the HF client fails)."""
    import requests

    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(20):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                with part.open(mode) as f:
                    for i, block in enumerate(r.iter_content(chunk_size=chunk)):
                        f.write(block)
                        if i % 64 == 0:
                            print(f"[data]   {part.stat().st_size / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
            break
        except Exception as e:  # network hiccup: resume
            print(f"[data] download interrupted ({type(e).__name__}: {e}); retry {attempt + 1}/20", flush=True)
            time.sleep(10)
    os.replace(part, dest)


def fetch_archive(root: Path, use_hf: bool = True) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dest = raw / ARCHIVE
    if use_hf:
        try:
            from huggingface_hub import hf_hub_download

            dest = Path(hf_hub_download(DATA_REPO, ARCHIVE, repo_type="dataset", local_dir=raw))
        except Exception as e:
            print(f"[data] huggingface_hub download failed ({type(e).__name__}: {e}); falling back to plain HTTPS")
            use_hf = False
    if not use_hf:
        http_download(f"https://huggingface.co/datasets/{DATA_REPO}/resolve/main/{ARCHIVE}", dest)
    print(f"[data] archive {dest} ({dest.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    return dest


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
    ap.add_argument("--plain-http", action="store_true", help="skip the HF client, download with resumable HTTPS")
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
            archive = fetch_archive(root, use_hf=not args.plain_http)
            extract(archive, root)
            if not args.keep_archive:
                archive.unlink()
                print(f"[data] removed archive {archive}")
        check_dataset(target)
    print("[assets] done")


if __name__ == "__main__":
    main()
