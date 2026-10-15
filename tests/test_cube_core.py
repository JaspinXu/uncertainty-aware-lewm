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
    """Same float32 env actions as WorldModelPolicy, bit for bit."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.normal(size=(500, 5)) * rng.uniform(0.5, 2, size=5) + rng.normal(size=5))
    plan = torch.from_numpy(rng.normal(size=(3, 5, 25)).astype(np.float32))  # (n_envs, horizon, block*A)
    # policy: plan.reshape(n, receding*block, -1); per step: .float().numpy() -> inverse_transform
    per_step = plan.reshape(3, 25, -1)
    expected = np.stack([scaler.inverse_transform(per_step[:, t].float().numpy()) for t in range(25)], axis=1)
    ours = normalized_to_env_actions(plan.numpy(), 5, scaler.mean_.tolist(), scaler.scale_.tolist())
    assert ours.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(ours, expected)


# --------------------------------------------------------------------------- #
# L4 helpers
# --------------------------------------------------------------------------- #
def test_r2_redundancy_extremes():
    from train_cube import r2_redundancy

    torch.manual_seed(0)
    p = torch.randn(2048, 8)
    v = p @ torch.randn(8, 16)
    assert r2_redundancy(v, p).item() > 0.99
    assert r2_redundancy(torch.randn(2048, 16), p).item() < 0.05
    # training runs under bf16 autocast (linalg.solve rejects mixed dtypes)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        r = r2_redundancy(v.bfloat16(), p.bfloat16())
    assert r.dtype == torch.float32 and r.item() > 0.99


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


# --------------------------------------------------------------------------- #
# latent ensemble
# --------------------------------------------------------------------------- #
def _tiny_member_cfg():
    from ua.ensemble import member_config

    cfg = member_config(dim=16, depth=1)
    cfg["predictor"].update(heads=2, dim_head=8, mlp_dim=32)
    cfg["pred_proj"].update(hidden_dim=32)
    return cfg


def test_ensemble_roundtrip_and_disagreement(tmp_path):
    from ua.ensemble import LatentEnsemble, build_member, load_ensemble, save_ensemble

    cfg = _tiny_member_cfg()
    members = []
    for s in range(3):
        torch.manual_seed(s)
        m = build_member(cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.1 * torch.randn_like(p))
        members.append(m.eval())
    ens = LatentEnsemble(members)
    z = torch.randn(2, 1, 16)
    acts = torch.randn(2, 4, 5, 25)
    out = ens.rollouts(z, acts)
    assert out.shape == (3, 2, 4, 5, 16)
    save_ensemble(ens, cfg, tmp_path / "e.pt", {"policy": "x"})
    ens2, meta = load_ensemble(tmp_path / "e.pt")
    torch.testing.assert_close(ens2.rollouts(z, acts), out)
    assert meta["policy"] == "x"
    from cube.rollout import per_step_variance

    assert torch.all(per_step_variance(out) > 0)


def test_train_ensemble_batch_layout():
    from train_ensemble import FS, T, batch, clip_starts

    lengths, offsets = np.array([40, 31, 20]), np.array([0, 40, 71])
    starts = clip_starts(lengths, offsets, np.array([0, 1, 2]))
    assert starts.max() + T * FS <= 71  # episode 1 clips never cross into episode 2
    assert 71 not in set(starts)  # episode 2 (20 rows) is too short for a 30-row clip
    lat = torch.arange(91, dtype=torch.float32)[:, None].repeat(1, 4)
    act = torch.arange(91, dtype=torch.float32)[:, None].repeat(1, 5)
    z, a = batch(lat, act, torch.tensor([3]))
    assert z[0, :, 0].tolist() == [3, 8, 13, 18, 23, 28]
    assert a.shape == (1, T, FS * 5) and a[0, 1, 0].item() == 8  # block 1 starts at row 3 + 5


def test_warm_start_reproduces_pretrained(tmp_path, monkeypatch):
    """A factorised variant warm-started from a LeWM checkpoint predicts its visual
    block exactly like the checkpoint (identity-initialised projections)."""
    import train_cube as tc
    from hydra.utils import instantiate

    from cube.rollout import rollout_from_latents
    from planners import load_base
    from tests.make_random_checkpoint import make_checkpoint

    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    make_checkpoint(tmp_path, "cube/lewm")
    base = load_base("cube/lewm", "cpu")
    args = types.SimpleNamespace(prop_latent=8)
    spec = tc.VARIANTS["prop_ms"]
    model = instantiate(copy.deepcopy(tc.model_config(["proprio_a"], 3, args, spec)))
    rep = tc.warm_start(model, "cube/lewm", factorized=True)
    assert not rep["skipped"] and rep["block_copied"] == 3
    model.eval()
    torch.manual_seed(0)
    px, acts = torch.randn(1, 1, 3, 224, 224), torch.randn(1, 3, 5, 25)
    with torch.no_grad():
        ref = rollout_from_latents(base, base.encode({"pixels": px})["emb"], acts)
        z = model.encode({"pixels": px, "proprio": torch.randn(1, 1, 3)})["emb"]
        out = rollout_from_latents(model, z, acts)
    torch.testing.assert_close(out[..., :192], ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# oracle replay of mid-episode plans (cube/simstate.py)
# --------------------------------------------------------------------------- #
def test_mid_episode_replay_is_exact():
    """Replaying from qpos/qvel + solver warm start + stale end-effector pose
    reproduces the live trajectory bit for bit, whatever the replay env did before."""
    pytest.importorskip("ogbench")
    from cube.oracle import make_cube_env, replay
    from cube.simstate import sim_state

    rng = np.random.default_rng(0)
    acts = np.clip(rng.normal(0, 0.6, size=(30, 5)), -1, 1).astype(np.float32)
    live = make_cube_env(seed=3)
    tid = live._cube_target_mocap_ids[0]
    goal, goal_q = live._data.mocap_pos[tid].copy(), live._data.mocap_quat[tid].copy()
    for a in acts[:15]:
        live._reset_next_step = False
        live.step(a)
    snap = {"qpos": live._data.qpos.copy(), "qvel": live._data.qvel.copy(), **sim_state(live)}
    for a in acts[15:]:
        live._reset_next_step = False
        live.step(a)

    other = make_cube_env(seed=4)
    for a in acts[:5]:  # unrelated history in the replay env
        other._reset_next_step = False
        other.step(-a)
    res = replay(other, snap["qpos"], snap["qvel"], goal, goal_q, acts[None, 15:], render_every=0,
                 sim=snap, mid_episode=True)
    np.testing.assert_array_equal(res.final_qpos[0], live._data.qpos)
