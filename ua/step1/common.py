"""Small helpers shared by the Step-1 scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import tv_tensors

ACTION_DIM = 5  # OGBench cube: (dx, dy, dz, dyaw, dgripper)


def out_dir(cfg, root: Path) -> Path:
    path = root / cfg.step1.out_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_env_actions(blocks: torch.Tensor | np.ndarray, action_scaler) -> np.ndarray:
    """Normalized plan blocks (..., T, 25) -> env actions (..., T*5, 5).

    Same path as ``WorldModelPolicy.get_action``: float32 blocks reshaped
    time-major, then the dataset ``StandardScaler`` inverse transform.
    """
    x = blocks.float().cpu().numpy() if torch.is_tensor(blocks) else np.asarray(blocks, np.float32)
    lead = x.shape[:-2]
    steps = x.reshape(*lead, -1, ACTION_DIM)
    flat = action_scaler.inverse_transform(steps.reshape(-1, ACTION_DIM))
    return flat.reshape(*lead, -1, ACTION_DIM)


@torch.no_grad()
def encode_frames(model, frames: np.ndarray, transform, device, batch: int = 256) -> torch.Tensor:
    """(N, 224, 224, 3) uint8 frames -> (N, D) embeddings, exactly like the
    policy's ``_prepare_info`` (HWC->CHW, tv Image, eval transform) + encode."""
    dtype = next(model.parameters()).dtype
    out = []
    for i in range(0, len(frames), batch):
        chw = torch.from_numpy(np.ascontiguousarray(np.transpose(frames[i : i + batch], (0, 3, 1, 2))))
        x = torch.stack([transform(tv_tensors.Image(img)) for img in chw])
        out.append(model.encode({"pixels": x[:, None].to(device, dtype)})["emb"][:, 0].float().cpu())
    return torch.cat(out) if out else torch.empty(0)
