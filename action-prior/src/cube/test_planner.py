import unittest
import os
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel import PlanConfig
from stable_worldmodel.solver import CEMSolver
from planner import GuidedCEM
from prior import ActionPrior


class Cost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.candidates = []

    def get_cost(self, info, actions):
        self.candidates.append(actions.clone())
        return (actions - 2).square().sum((-1, -2))


def make(cls, model, **extra):
    solver = cls(model=model, num_samples=10, topk=3, n_steps=3, seed=7, **extra)
    solver.configure(action_space=gym.spaces.Box(-1, 1, (2, 5), dtype=np.float32),
                     n_envs=2, config=PlanConfig(horizon=5, receding_horizon=5, action_block=5))
    return solver


class Tests(unittest.TestCase):
    def test_cache_transform_matches_official(self):
        from cache_latents import image_tensor
        from omegaconf import OmegaConf
        sys.path.insert(0, str(Path(os.environ["STABLEWM_HOME"]) / "le-wm"))
        from eval import img_transform
        data = np.random.default_rng(1).integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
        transform = img_transform(OmegaConf.create({"eval": {"img_size": 224}}))
        expected = torch.stack([transform(torch.from_numpy(x).permute(2, 0, 1)) for x in data])
        torch.testing.assert_close(image_tensor(data, "cpu")[:, 0], expected)

    def test_windows_never_cross_episode_or_nan(self):
        from train_prior import valid_starts
        actions = np.zeros((100, 5))
        actions[40] = np.nan
        starts = valid_starts(30, 30, actions)
        self.assertEqual(len(starts), 0)
        starts = valid_starts(60, 30, actions)
        np.testing.assert_array_equal(starts, np.arange(60, 65))

    def test_rho_zero_matches_original_with_warm_start(self):
        a, b = Cost(), Cost()
        info = {"x": torch.zeros(2, 1)}
        initial = torch.ones(2, 2, 25)
        original = make(CEMSolver, a).solve(info, initial)
        guided = make(GuidedCEM, b, proposal=lambda _: None, rho=0).solve(info, initial)
        torch.testing.assert_close(original["actions"], guided["actions"], rtol=0, atol=0)
        self.assertEqual(len(a.candidates), len(b.candidates))
        for x, y in zip(a.candidates, b.candidates):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_copied_update_matches_original_without_guided_samples(self):
        a, b = Cost(), Cost()
        info = {"x": torch.zeros(2, 1)}
        initial = torch.ones(2, 2, 25)
        original = make(CEMSolver, a).solve(info, initial)
        # Positive rho exercises the new loop but rounds to zero guided candidates.
        guided = make(GuidedCEM, b, proposal=lambda _: torch.zeros(2, 5, 25),
                      rho=.01).solve(info, initial)
        torch.testing.assert_close(original["actions"], guided["actions"], rtol=0, atol=0)
        for x, y in zip(a.candidates, b.candidates):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_first_iteration_mixture_only(self):
        a, b = Cost(), Cost()
        info = {"x": torch.zeros(2, 1)}
        make(CEMSolver, a).solve(info)
        result = make(GuidedCEM, b, proposal=lambda _: torch.full((2, 5, 25), 2.),
                      rho=.5, guided_std=0).solve(info)
        self.assertEqual(result["actions"].shape, (2, 5, 25))
        self.assertEqual(len(b.candidates), 6)
        for batch in (0, 3):
            torch.testing.assert_close(a.candidates[batch][:, :5], b.candidates[batch][:, :5])
            torch.testing.assert_close(b.candidates[batch][:, 5:], torch.full((1, 5, 5, 25), 2.))

    def test_network_output_and_gradients(self):
        net = ActionPrior()
        output = net(torch.randn(8, 192), torch.randn(8, 192))
        self.assertEqual(output.shape, (8, 5, 25))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in net.parameters()))


if __name__ == "__main__":
    unittest.main()
