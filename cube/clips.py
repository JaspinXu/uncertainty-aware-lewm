"""Fast fixed-length clip dataset over the Cube HDF5 file (training / probes / k-step eval).

Same clips as ``stable_worldmodel.data.HDF5Dataset(frameskip, num_steps)``:
a clip starting at row ``r`` of an episode holds frames ``r, r+fs, ...,
r+(T-1)*fs`` and, for frame ``k``, the ``fs`` raw actions ``r+k*fs ...
r+k*fs+fs-1`` concatenated (``T x fs*A``). Differences, all for speed:

* only the T frames used are read (row by row: an h5py strided selection is
  much slower on the chunked, compressed file), not all T*fs rows of the span;
* small columns are cached in RAM once and indexed directly;
* train/validation splits are made by *episode* (no clip of a held-out
  episode is ever trained on).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers compression filters)
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dataset_path(name: str = "ogbench/cube_single_expert", root: str | Path | None = None) -> Path:
    import os

    root = Path(root or os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    return root / "datasets" / f"{name}.h5"


def latents_path(policy: str, root: str | Path | None = None) -> Path:
    """Where scripts/cube/encode_dataset.py stores the frozen-encoder latents."""
    import os

    root = Path(root or os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    return root / "latents" / f"{policy.replace('/', '__')}.npy"


def proprio_columns(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("proprio_"))


def split_episodes(n_episodes: int, val_frac: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic episode-level split (validation = random 5% of episodes)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(val_frac * n_episodes)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


class CubeClips(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str | Path,
        num_steps: int,
        frameskip: int = 5,
        episodes: np.ndarray | None = None,
        columns: list[str] | None = None,
        action_stats: tuple[np.ndarray, np.ndarray] | None = None,
        stride: int = 1,
    ):
        self.path = str(path)
        self.T = int(num_steps)
        self.fs = int(frameskip)
        self.span = self.T * self.fs
        self.columns = list(columns or [])
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            lengths = f["ep_len"][:].astype(np.int64)
            offsets = f["ep_offset"][:].astype(np.int64)
            self.action = f["action"][:].astype(np.float32)
            self.cache = {c: f[c][:].astype(np.float32).reshape(len(self.action), -1) for c in self.columns}
        if action_stats is None:
            ok = ~np.isnan(self.action).any(axis=1)
            action_stats = (self.action[ok].mean(0), self.action[ok].std(0, ddof=1))
        self.action_mean, self.action_std = (np.asarray(s, np.float32) for s in action_stats)
        self.lengths, self.offsets = lengths, offsets
        eps = np.arange(len(lengths)) if episodes is None else np.asarray(episodes)
        starts = [offsets[e] + np.arange(0, lengths[e] - self.span + 1, stride) for e in eps if lengths[e] >= self.span]
        self.starts = np.concatenate(starts).astype(np.int64) if starts else np.zeros(0, np.int64)
        self.episodes = eps

    def __len__(self) -> int:
        return len(self.starts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            # room for a few decompressed pixel chunks (100 frames = 15 MB each)
            self._h5 = h5py.File(self.path, "r", swmr=True, rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=10007)
        return self._h5

    def __getitem__(self, i: int) -> dict:
        r = int(self.starts[i])
        px = self._file()["pixels"]
        # one row at a time: an h5py strided selection (r : r + span : fs) is
        # ~9x slower on the chunked, compressed pixels (13 s vs 1.5 s for 128 clips)
        pixels = np.stack([px[r + k * self.fs] for k in range(self.T)])  # (T, H, W, 3) uint8
        act = self.action[r : r + self.span]
        act = np.nan_to_num((act - self.action_mean) / self.action_std, nan=0.0)
        out = {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels)).permute(0, 3, 1, 2),  # (T, 3, H, W)
            "action": torch.from_numpy(act.reshape(self.T, -1)),
            "row": torch.tensor(r),
        }
        rows = r + self.fs * np.arange(self.T)
        for c in self.columns:
            out[c] = torch.from_numpy(self.cache[c][rows])
        return out


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 3, H, W) on any device -> ImageNet-normalised float (as LeWM training)."""
    x = pixels.float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x - mean) / std
