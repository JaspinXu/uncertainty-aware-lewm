"""LeWM representation variants for Cube (L4).

``LeWMVariant`` is LeWM (ViT-tiny encoder -> CLS -> MLP projector, causal
AdaLN predictor, action embedder) with three optional additions:

* proprio block -- z = [z_vis ; z_prop], z_prop = MLP(normalised proprio).
  The predictor models both blocks jointly (``FactorizedPredictor``: the LeWM
  predictor core plus linear maps from/to the factorised latent).
* aux head -- training-only MLP from z_vis to the cube pose
  (position, cos 4yaw, sin 4yaw); unused at planning time.
* block-weighted planning cost -- ||z_vis - g_vis||^2 + w_prop ||z_prop - g_prop||^2.

Training losses live in train_cube.py; this module only defines the model and
the planning interface (``get_cost``) used by the CEM solver. Checkpoints are
saved in the stable-worldmodel format (weights.pt + config.json with
``_target_: cube.variants.LeWMVariant``) so ``load_pretrained`` works unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from cube.rollout import rollout_from_latents


class FactorizedPredictor(nn.Module):
    """LeWM predictor core at the visual width, with linear maps from/to z = [z_vis; z_prop].

    ``core`` is the standard predictor (input = hidden = output = 192), so it
    loads the pretrained LeWM weights unchanged. ``in_proj`` / ``out_proj``
    start as [I | 0] and [I ; 0]: a warm-started model initially predicts the
    visual block exactly like the pretrained predictor and learns how to use
    and predict the proprio block.
    """

    def __init__(self, core: nn.Module, lat_dim: int, vis_dim: int = 192):
        super().__init__()
        self.core = core
        self.num_frames = core.num_frames
        self.in_proj = nn.Linear(lat_dim, vis_dim)
        self.out_proj = nn.Linear(vis_dim, lat_dim)
        with torch.no_grad():
            for lin in (self.in_proj, self.out_proj):
                lin.weight.zero_()
                lin.bias.zero_()
            self.in_proj.weight[:, :vis_dim] = torch.eye(vis_dim)
            self.out_proj.weight[:vis_dim] = torch.eye(vis_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.core(self.in_proj(x), c))


class LeWMVariant(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module,
        pred_proj: nn.Module,
        prop_keys: list[str] | None = None,
        prop_dim: int = 0,
        prop_latent: int = 0,
        prop_hidden: int = 128,
        aux_dim: int = 0,
        vis_dim: int = 192,
        w_prop: float = 1.0,
        history_size: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj
        self.vis_dim = int(vis_dim)
        self.prop_keys = list(prop_keys or [])
        self.prop_latent = int(prop_latent)
        self.w_prop = float(w_prop)
        self.history_size = int(history_size)
        if self.prop_latent > 0:
            self.prop_encoder = nn.Sequential(
                nn.Linear(prop_dim, prop_hidden),
                nn.LayerNorm(prop_hidden),
                nn.GELU(),
                nn.Linear(prop_hidden, self.prop_latent),
            )
            self.register_buffer("prop_mean", torch.zeros(prop_dim))
            self.register_buffer("prop_std", torch.ones(prop_dim))
        self.aux_head = (
            nn.Sequential(nn.Linear(self.vis_dim, 256), nn.GELU(), nn.Linear(256, aux_dim)) if aux_dim > 0 else None
        )

    # ------------------------------------------------------------------ #
    @property
    def latent_dim(self) -> int:
        return self.vis_dim + self.prop_latent

    def gather_proprio(self, info: dict, prefix: str = "") -> torch.Tensor:
        """Raw proprio vector from ``info['proprio']`` or the per-key columns."""
        if prefix == "" and "proprio" in info:
            return info["proprio"]
        return torch.cat([info[prefix + k].float() for k in self.prop_keys], dim=-1)

    def encode_prop(self, proprio: torch.Tensor) -> torch.Tensor:
        x = (proprio.float() - self.prop_mean) / self.prop_std
        return self.prop_encoder(x)

    def encode(self, info: dict, prefix: str = "") -> dict:
        """info['pixels']: (B, T, C, H, W) normalised; optional proprio (B, T, P)."""
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        b, t = pixels.shape[:2]
        out = self.encoder(pixels.flatten(0, 1), interpolate_pos_encoding=True)
        z_vis = self.projector(out.last_hidden_state[:, 0]).reshape(b, t, -1)
        info["emb_vis"] = z_vis
        if self.prop_latent > 0:
            z_prop = self.encode_prop(self.gather_proprio(info, prefix)).to(z_vis.dtype)
            info["emb_prop"] = z_prop
            info["emb"] = torch.cat([z_vis, z_prop], dim=-1)
        else:
            info["emb"] = z_vis
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        b = emb.size(0)
        return self.pred_proj(preds.flatten(0, 1)).reshape(b, -1, preds.size(-1))

    # ------------------------------------------------------------------ #
    # planning interface
    # ------------------------------------------------------------------ #
    def block_cost(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        d = (pred - goal) ** 2
        cost = d[..., : self.vis_dim].sum(-1)
        if self.prop_latent > 0:
            cost = cost + self.w_prop * d[..., self.vis_dim :].sum(-1)
        return cost

    def _goal_info(self, info_dict: dict) -> dict:
        goal = {"pixels": info_dict["goal"][:, 0]}
        if self.prop_latent > 0:
            goal["proprio"] = self.gather_proprio({k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}, "goal_")
        return goal

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        dev = next(self.parameters()).device
        if "_z_hist" not in info_dict:
            cur = {"pixels": info_dict["pixels"][:, 0].to(dev)}
            if self.prop_latent > 0:
                cur["proprio"] = self.gather_proprio(
                    {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
                ).to(dev)
            info_dict["_z_hist"] = self.encode(cur)["emb"]
            goal = {k: v.to(dev) for k, v in self._goal_info(info_dict).items()}
            info_dict["_z_goal"] = self.encode(goal)["emb"][:, -1]
        pred = rollout_from_latents(self, info_dict["_z_hist"], action_candidates.to(dev), self.history_size)
        return self.block_cost(pred[..., -1, :], info_dict["_z_goal"][:, None])
