"""Goal-conditioned action proposals, in the original planner's action units."""
import torch
from torch import nn


class ActionPrior(nn.Module):
    def __init__(self, dim=192, hidden=256, horizon=5, block=5, action_dim=5):
        super().__init__()
        self.spec = dict(dim=dim, hidden=hidden, horizon=horizon, block=block, action_dim=action_dim)
        self.register_buffer("z_mean", torch.zeros(dim))
        self.register_buffer("z_std", torch.ones(dim))
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * block * action_dim),
        )

    def forward(self, current, goal):
        current = (current - self.z_mean) / self.z_std
        goal = (goal - self.z_mean) / self.z_std
        x = torch.cat([current, goal, goal - current], dim=-1)
        return self.net(x).reshape(-1, self.spec["horizon"],
                                   self.spec["block"] * self.spec["action_dim"])


def load_prior(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ActionPrior(**ckpt["spec"]).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval().requires_grad_(False), ckpt


class Proposal:
    def __init__(self, world_model, prior):
        self.world_model, self.prior = world_model, prior

    @torch.inference_mode()
    def __call__(self, info):
        device = next(self.prior.parameters()).device
        current = self.world_model.encode({"pixels": info["pixels"].to(device)})["emb"][:, -1]
        goal = self.world_model.encode({"pixels": info["goal"].to(device)})["emb"][:, -1]
        return self.prior(current, goal)
