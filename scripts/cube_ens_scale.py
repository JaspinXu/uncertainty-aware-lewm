"""Outcome-free scale of the ensemble penalty of bc_select_khat_*_ens (config/eval/compare.yaml).

    python scripts/cube_ens_scale.py        # -> <compare.out_dir>/models/ens_scale.json

On held-out cached states (the last compare.val_episodes episodes of the latent cache: neither the
priors nor the Step-1 ensemble members trained on them; goals 10-50 steps ahead), the proposals of
bc_select_khat_sp_m4 (sigma_prior_tc: per k the mean plus 3 draws from N(mu_k, sigma_k^2)) are scored by
the world model + k̂ (time to goal T) and by the ensemble's latent sigma^2. The penalty unit is
median T / median sigma^2: at the medians, lambda = 1 unit charges as much as the time to goal itself.
No episode outcome is used. Also reported: whether sigma^2 separates the draws from the means, and how
often a draw would be chosen at lambda = 0, 1 and 3 units.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from evaluate import pick_device  # noqa: E402
from planners import load_base  # noqa: E402
from ua.step1.ensemble import load_members  # noqa: E402
from ua.step1.prior import PriorData, PriorSelectSolver, load_prior  # noqa: E402
from ua.step1.steps_to_goal import load_steps_to_goal  # noqa: E402

GOALS = (10, 50)  # goal offsets of the held-out states, as during an offset-50 episode
SAMPLES = 4       # bc_select_khat_sp_m4


@hydra.main(version_base=None, config_path="../config/eval", config_name="compare")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    device, ccfg = pick_device(), cfg.compare
    mdir = ROOT / ccfg.models_dir
    n_states = int(cfg.get("n_states", 800))
    ckpt = mdir / "sigma_prior_tc.pt"
    meta = torch.load(ckpt, weights_only=False, map_location="cpu")
    members = load_members(ROOT / ccfg.ensemble_dir, ROOT / "checkpoints" / cfg.policy / "config.json", device)
    solver = PriorSelectSolver(load_prior(ckpt, device), load_base(cfg.policy, device),
                               load_steps_to_goal(mdir / "khat.pt", device), samples=SAMPLES, init_std="prior",
                               seed=0, members=members)
    cache = torch.load(ROOT / ccfg.latents, weights_only=False, map_location="cpu")
    n_val = int(ccfg.val_episodes)
    val = list(range(len(cache["episodes"]) - n_val, len(cache["episodes"])))
    data = PriorData(cache, val, meta["action_mean"], meta["action_scale"], device, GOALS, n_eval=n_states)

    costs, variances = [], []
    for i in range(0, n_states, 100):
        z, g, _ = data.batch(data.eval_s[i : i + 100], data.eval_k[i : i + 100])
        _, cost, var = solver.score(z[:, None], g)
        costs.append(cost.cpu()), variances.append(var.cpu())
    T, v = torch.cat(costs), torch.cat(variances)
    unit = float(T.median() / v.median())
    is_draw = (torch.arange(T.shape[1]) % SAMPLES) > 0
    per_k = v.view(len(v), -1, SAMPLES)
    out = {
        "unit": unit, "median_time_to_goal": float(T.median()), "median_sigma2": float(v.median()),
        "n_states": n_states, "goal_offsets": list(GOALS), "samples": SAMPLES, "members": len(members),
        "median_sigma2_means": float(v[:, ~is_draw].median()), "median_sigma2_draws": float(v[:, is_draw].median()),
        "frac_draw_above_its_mean": float((per_k[:, :, 1:] > per_k[:, :, :1]).float().mean()),
        "draw_chosen": {str(m): float(is_draw[(T + m * unit * v).argmin(dim=1)].float().mean()) for m in (0, 1, 3)},
    }
    (mdir / "ens_scale.json").write_text(json.dumps(out, indent=1))
    print(f"[ens_scale] unit {unit:.1f} = median T {out['median_time_to_goal']:.1f} / median sigma^2 "
          f"{out['median_sigma2']:.3f}; sigma^2 means {out['median_sigma2_means']:.3f} vs draws "
          f"{out['median_sigma2_draws']:.3f}; draw chosen at 0/1/3 units: {out['draw_chosen']}", flush=True)


if __name__ == "__main__":
    main()
