"""Unit tests for the Cube experiment helpers (CPU only, no dataset download).

    PYTHONPATH=.:stable-worldmodel:stable-pretraining pytest -q tests/
"""

from __future__ import annotations

import copy
import types

import numpy as np
import pytest
import torch

from cube import data as cube_data
from cube.metrics import auroc, mcnemar_exact, spearman, wilson_ci
from cube.oracle import normalized_to_env_actions
from cube.rollout import deterministic_rollout, goal_cost, mc_rollouts, per_step_variance


# --------------------------------------------------------------------------- #
# dataset helpers
# --------------------------------------------------------------------------- #
class FakeDataset:
    def __init__(self, lengths, seed=0):
        rng = np.random.default_rng(seed)
        self.lengths = np.asarray(lengths)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.cols = {
            "ep_idx": np.concatenate([np.full(n, i) for i, n in enumerate(lengths)]),
            "step_idx": np.concatenate([np.arange(n) for n in lengths]),
        }
        self.cols["x"] = rng.normal(size=len(self.cols["ep_idx"]))

    @property
    def column_names(self):
        return list(self.cols)

    def get_col_data(self, c):
        return self.cols[c]

    def get_row_data(self, rows):
        return {c: v[rows] for c, v in self.cols.items()}


def legacy_sample(dataset, num_eval, seed, goal_offset):
    """The original evaluate.py implementation (O(episodes x rows))."""
    col = "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)
    ep = dataset.get_col_data(col)
    st = dataset.get_col_data("step_idx")
    episode_len = np.array([np.max(st[ep == e]) + 1 for e in ep_indices])
    max_start = episode_len - goal_offset - 1
    by_ep = {e: max_start[i] for i, e in enumerate(ep_indices)}
    per_row = np.array([by_ep[e] for e in ep])
    valid = np.nonzero(st <= per_row)[0]
    g = np.random.default_rng(seed)
    picked = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    rows = dataset.get_row_data(picked)
    return rows[col].tolist(), rows["step_idx"].tolist()


@pytest.mark.parametrize("seed", [0, 42, 1011])
def test_start_state_sampling_matches_legacy(seed):
    ds = FakeDataset([60, 80, 31, 200, 26, 90])
    new = cube_data.rows_to_episode_starts(ds, cube_data.sample_start_rows(ds, 20, seed, 25))
    assert new == legacy_sample(ds, 20, seed, 25)
    eps, starts = new
    lengths = dict(enumerate(ds.lengths))
    assert all(s + 25 < lengths[e] for e, s in zip(eps, starts))


def test_episode_lengths():
    ds = FakeDataset([5, 7, 3])
    assert cube_data.episodes_length(ds, [2, 0]).tolist() == [3, 5]
    with pytest.raises(KeyError):
        cube_data.episodes_length(ds, [9])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # with ties
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(auroc([1, 2, 3], [1, 1, 1]))


def test_spearman_and_wilson():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    lo, hi = wilson_ci(37, 50)
    assert lo < 0.74 < hi


def test_mcnemar():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 0], bool)
    b = np.array([1, 1, 1, 1, 1, 1, 0, 0], bool)
    r = mcnemar_exact(a, b)
    assert (r["b_only"], r["a_only"]) == (3, 1)
    assert 0 < r["p_value"] <= 1


# --------------------------------------------------------------------------- #
# rollouts == stable_worldmodel LeWM.rollout
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_lewm():
    from hydra.utils import instantiate

    from tests.make_random_checkpoint import CUBE_CONFIG

    cfg = copy.deepcopy(CUBE_CONFIG)
    cfg["predictor"]["depth"] = 2
    cfg["predictor"]["mlp_dim"] = 64
    cfg["projector"]["hidden_dim"] = 64
    cfg["pred_proj"]["hidden_dim"] = 64
    torch.manual_seed(0)
    model = instantiate(cfg).eval()
    # AdaLN-zero gates start at 0 (blocks = identity, dropout has no effect):
    # perturb every weight so the test model behaves like a trained one.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model


def test_rollout_matches_lewm(tiny_lewm):
    model = tiny_lewm
    B, S, H, A = 2, 7, 5, 25
    torch.manual_seed(1)
    pixels = torch.randn(B, 1, 3, 224, 224)
    actions = torch.randn(B, S, H, A)
    with torch.no_grad():
        info = {"pixels": pixels[:, None].expand(B, S, 1, 3, 224, 224)}
        ref = model.rollout(info, actions)["predicted_emb"][:, :, 1:]  # drop the encoded frame
        z_hist = model.encode({"pixels": pixels})["emb"]
        ours = deterministic_rollout(model, z_hist, actions)
    assert ours.shape == (B, S, H, 192)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)


def test_mc_rollouts_vary_and_restore_mode(tiny_lewm):
    model = tiny_lewm
    z = torch.randn(1, 1, 192)
    acts = torch.randn(1, 4, 5, 25)
    mc = mc_rollouts(model, z, acts, n_samples=3)
    assert mc.shape == (3, 1, 4, 5, 192)
    assert not model.training
    var = per_step_variance(mc)
    assert var.shape == (1, 4, 5) and torch.all(var > 0)
    cost = goal_cost(mc[0][..., -1, :], torch.zeros(1, 1, 192))
    assert cost.shape == (1, 4)


# --------------------------------------------------------------------------- #
# action conversion == WorldModelPolicy
# --------------------------------------------------------------------------- #
def test_normalized_to_env_actions_matches_policy():
    rng = np.random.default_rng(0)
    mean, scale = rng.normal(size=5), rng.uniform(0.5, 2, size=5)
    plan = rng.normal(size=(1, 5, 25)).astype(np.float32)  # (n_envs, horizon, block*A)
    # policy: plan.reshape(n, receding*block, -1) then inverse_transform per step
    scaler = types.SimpleNamespace(inverse_transform=lambda a: a * scale + mean)
    per_step = torch.from_numpy(plan).reshape(1, 25, -1)
    expected = np.stack([scaler.inverse_transform(per_step[:, t].numpy()) for t in range(25)], axis=1)
    ours = normalized_to_env_actions(plan, 5, mean, scale)
    np.testing.assert_allclose(ours, expected, rtol=1e-6)


# --------------------------------------------------------------------------- #
# L4 helpers
# --------------------------------------------------------------------------- #
def test_r2_redundancy_extremes():
    from train_cube import r2_redundancy

    torch.manual_seed(0)
    p = torch.randn(2048, 8)
    assert r2_redundancy(p @ torch.randn(8, 16), p).item() > 0.99
    assert r2_redundancy(torch.randn(2048, 16), p).item() < 0.05


def test_lr_schedule():
    from train_cube import lr_at

    assert lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert lr_at(10, 100, 1.0, 10) == pytest.approx(1.0)
    assert lr_at(100, 100, 1.0, 10) == pytest.approx(0.0, abs=1e-9)


def test_variant_block_cost():
    from cube.variants import LeWMVariant

    m = LeWMVariant.__new__(LeWMVariant)
    torch.nn.Module.__init__(m)
    m.vis_dim, m.prop_latent, m.w_prop = 3, 2, 0.5
    pred = torch.tensor([[1.0, 0, 0, 2.0, 0]])
    goal = torch.zeros(1, 5)
    assert m.block_cost(pred, goal).item() == pytest.approx(1.0 + 0.5 * 4.0)
