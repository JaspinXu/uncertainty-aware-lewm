"""First-iteration mixture proposal; later iterations follow official CEM.

Adapted from stable-worldmodel CEMSolver at abdced49809d5eae38e24b27dc7b635c502c4812.
rho=0 dispatches directly to the original solver, including its RNG behavior.
"""
import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver
from stable_worldmodel.solver.utils import prepare_init_action


class GuidedCEM(CEMSolver):
    def __init__(self, *args, proposal, rho=0.5, guided_std=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= rho <= 1 or guided_std < 0:
            raise ValueError("Require 0 <= rho <= 1 and guided_std >= 0")
        if self.callbacks:
            raise ValueError("This minimal extension does not support CEM callbacks")
        self.proposal, self.rho, self.guided_std = proposal, rho, guided_std

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.rho == 0:
            return super().solve(info_dict, init_action)
        total = len(next(iter(info_dict.values())))
        ref = self.proposal(info_dict).to(self.device, dtype=self.dtype)
        if ref.shape != (total, self.horizon, self.action_dim) or not torch.isfinite(ref).all():
            raise ValueError(f"Invalid proposal shape/values: {ref.shape}")
        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total, action_dim=self.action_dim)
        mean, std = self.init_action_distrib(total, init_action)
        mean, std = mean.to(self.device), std.to(self.device)
        costs_out = []
        guided_count = int(round(self.rho * self.num_samples))
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            bs = end - start
            mu, sigma = mean[start:end], std[start:end]
            infos = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    part = v[start:end].to(self.device, dtype=self.dtype if v.is_floating_point() else None)
                    infos[k] = part[:, None].expand(bs, self.num_samples, *part.shape[1:])
                elif isinstance(v, np.ndarray):
                    infos[k] = np.repeat(v[start:end, None], self.num_samples, axis=1)
            for step in range(self.n_steps):
                noise = torch.randn(bs, self.num_samples, self.horizon, self.action_dim,
                                    generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = noise * sigma[:, None] + mu[:, None]
                # Original solver always includes its mean as candidate zero.
                candidates[:, 0] = mu
                if step == 0 and guided_count:
                    # Fixed-count mixture: unmodified original candidates plus guided ones.
                    lo = self.num_samples - guided_count
                    candidates[:, lo:] = ref[start:end, None] + self.guided_std * noise[:, lo:]
                    candidates[:, -1] = ref[start:end]
                costs = self.model.get_cost(infos, candidates)
                if costs.shape != (bs, self.num_samples) or not torch.isfinite(costs).all():
                    raise ValueError("Non-finite or incorrectly shaped planning costs")
                vals, idx = torch.topk(costs, self.topk, dim=1, largest=False)
                elite = candidates[torch.arange(bs, device=self.device)[:, None], idx]
                mu, sigma = elite.mean(1), elite.std(1)
            mean[start:end], std[start:end] = mu, sigma
            costs_out.extend(vals.mean(1).cpu().tolist())
        return {"actions": mean.cpu(), "mean": [mean.cpu()], "var": [std.cpu()], "costs": costs_out}


class DirectPriorSolver(CEMSolver):
    """Execute BC through the same WorldModelPolicy action buffer and scaler."""
    def __init__(self, *args, proposal, **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal = proposal

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        return {"actions": self.proposal(info_dict).cpu()}
