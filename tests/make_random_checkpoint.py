"""Write a randomly initialised LeWM checkpoint with the official Cube architecture.

Same ``config.json`` as https://huggingface.co/quentinll/lewm-cube (action input
25 = 5 dims x frameskip 5); only for tests / sandbox integration runs:

    python tests/make_random_checkpoint.py --out /tmp/stablewm   # -> checkpoints/cube/lewm/
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

_MLP = {
    "_target_": "stable_worldmodel.wm.lewm.module.MLP",
    "input_dim": 192,
    "output_dim": 192,
    "hidden_dim": 2048,
    "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
}

CUBE_CONFIG = {
    "_target_": "stable_worldmodel.wm.lewm.LeWM",
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
        "input_dim": 192,
        "hidden_dim": 192,
        "output_dim": 192,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.0,
    },
    "action_encoder": {
        "_target_": "stable_worldmodel.wm.lewm.module.Embedder",
        "input_dim": 25,
        "emb_dim": 192,
    },
    "projector": copy.deepcopy(_MLP),
    "pred_proj": copy.deepcopy(_MLP),
}


def make_checkpoint(root: Path, name: str = "cube/lewm", seed: int = 0) -> Path:
    from hydra.utils import instantiate

    torch.manual_seed(seed)
    model = instantiate(copy.deepcopy(CUBE_CONFIG))
    # AdaLN-zero gates are initialised at 0, which makes a fresh predictor ignore
    # dropout entirely; perturb all weights so MC-dropout behaves like a trained model.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    out = Path(root) / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(CUBE_CONFIG, indent=2))
    torch.save(model.state_dict(), out / "weights.pt")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cube/lewm")
    args = ap.parse_args()
    print(make_checkpoint(Path(args.out), args.name))
