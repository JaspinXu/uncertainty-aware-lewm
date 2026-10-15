"""Join planning records (cube/recording.py) with oracle replays (oracle_replay.py).

``load_run`` returns three pandas DataFrames:

candidates  one row per oracle-replayed candidate (executed plan, elites, randoms)
plans       one row per planning call
episodes    one row per evaluation episode (outcome of the real closed-loop run)

Column glossary (candidates):
  det_cost      ||z_hat_H - z_goal||^2 of the deterministic model (= planner cost)
  mc_cost       same with the mean of the MC-dropout rollouts
  mc_cost_std   std of the per-mask terminal costs
  sig2_k        MC-dropout variance (summed over latent dims) at predicted block k
  pred_err_k    ||z_hat_k - z_true_k||^2 against the encoded simulator frame
  true_cost     ||z_true_H - z_goal||^2
  sig2ens_k     latent-ensemble disagreement at block k (scripts/cube/score_ensemble.py)
  ens_cost      ensemble-mean terminal cost; ens_pred_err_H its true prediction error
  success       the candidate, executed open loop, solves the task in its segment
  final_dist    cube-target distance [m] at the end of the segment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

SUCCESS_RADIUS = 0.04


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_run(record_path: Path, oracle_path: Path | None = None, ensemble_path: Path | None = None) -> dict[str, pd.DataFrame]:
    rec = _load(record_path)
    meta, plans = rec["meta"], rec["plans"]
    oracle = _load(oracle_path)["results"] if oracle_path is not None and Path(oracle_path).exists() else {}
    ens = _load(ensemble_path)["results"] if ensemble_path is not None and Path(ensemble_path).exists() else {}
    run = meta["run_id"]
    seed = int(meta["seed"])
    n_elite = int(meta.get("n_elite", 30))
    ep_success = np.asarray(meta["episode_successes"]).astype(bool)

    cand_rows, plan_rows = [], []
    for i, p in enumerate(plans):
        env = int(p["env"])
        plan_id = f"{run}/{env}/{p['plan_idx']}"
        sig = p["sigma2"]  # (1+S, P)
        P = sig.shape[1]
        sub = p["subset"]
        o = oracle.get(i)
        en = ens.get(i)
        elite_costs = p["det_cost"][1:][p["solver_rank"][1:] < n_elite]
        plan_rows.append(
            {
                "run": run,
                "seed": seed,
                "env": env,
                "plan_idx": int(p["plan_idx"]),
                "t": int(p["t"]),
                "plan_id": plan_id,
                "episode_success": bool(ep_success[env]),
                "exec_det_cost": float(p["det_cost"][0]),
                "exec_mc_cost": float(p["mc_cost"][0]),
                "exec_mc_cost_std": float(p["mc_cost_std"][0]),
                "exec_sig2_H": float(sig[0, -1]),
                "exec_sig2_mean": float(sig[0].mean()),
                "pop_sig2_H_median": float(np.median(sig[1:, -1])),
                "pop_cost_median": float(np.median(p["det_cost"][1:])),
                "elite_cost_std": float(np.std(elite_costs)) if len(elite_costs) else np.nan,
                "elite_sig2_H_mean": float(sig[1:, -1][p["solver_rank"][1:] < n_elite].mean()),
                "start_dist": float(o["start_dist"]) if o is not None else np.nan,
                "has_oracle": o is not None,
                "exec_sig2ens_H": float(en["sig2"][0, -1]) if en is not None else np.nan,
            }
        )
        for k, pool_idx in enumerate(sub):
            row = {
                "run": run,
                "seed": seed,
                "env": env,
                "plan_idx": int(p["plan_idx"]),
                "plan_id": plan_id,
                "k": k,
                "pool_idx": int(pool_idx),
                "is_exec": bool(pool_idx == 0),
                "solver_rank": int(p["solver_rank"][pool_idx]),
                "is_elite": bool(pool_idx != 0 and p["solver_rank"][pool_idx] < n_elite),
                "episode_success": bool(ep_success[env]),
                "det_cost": float(p["det_cost"][pool_idx]),
                "mc_cost": float(p["mc_cost"][pool_idx]),
                "mc_cost_std": float(p["mc_cost_std"][pool_idx]),
            }
            for s in range(P):
                row[f"sig2_{s + 1}"] = float(sig[pool_idx, s])
            row["sig2_H"] = row[f"sig2_{P}"]
            if o is not None:
                row["success"] = bool(o["success_step"][k] > 0)
                row["success_step"] = int(o["success_step"][k])
                row["final_dist"] = float(o["cube_dist"][k, -1])
                row["min_dist"] = float(np.nanmin(o["cube_dist"][k]))
                row["start_dist"] = float(o["start_dist"])
                row["true_cost"] = float(o["true_cost"][k])
                for s in range(o["pred_err"].shape[1]):
                    row[f"pred_err_{s + 1}"] = float(o["pred_err"][k, s])
                row["pred_err_H"] = float(o["pred_err"][k, -1])
            if en is not None:
                for s_ in range(en["sig2"].shape[1]):
                    row[f"sig2ens_{s_ + 1}"] = float(en["sig2"][k, s_])
                row["sig2ens_H"] = float(en["sig2"][k, -1])
                row["ens_cost"] = float(en["cost"][k])
                if "pred_err" in en:
                    row["ens_pred_err_H"] = float(en["pred_err"][k, -1])
            cand_rows.append(row)

    episodes = pd.DataFrame(
        {
            "run": run,
            "seed": seed,
            "env": np.arange(len(ep_success)),
            "success": ep_success,
            "episode": meta["episodes"],
            "start_step": meta["start_steps"],
        }
    )
    plans_df = pd.DataFrame(plan_rows)
    first = plans_df[plans_df.plan_idx == 0].set_index("env")
    for col in ("exec_sig2_H", "exec_sig2ens_H", "exec_det_cost", "exec_mc_cost", "exec_mc_cost_std", "elite_cost_std", "start_dist"):
        episodes[f"first_{col}"] = episodes["env"].map(first[col]) if len(first) else np.nan
    return {"candidates": pd.DataFrame(cand_rows), "plans": plans_df, "episodes": episodes, "meta": meta}


def load_runs(record_dir: Path, oracle_dir: Path, runs: list[str] | None = None, ensemble_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    record_dir, oracle_dir = Path(record_dir), Path(oracle_dir)
    paths = sorted(record_dir.glob("*.pt"))
    if runs is not None:
        wanted = set(runs)
        paths = [p for p in paths if p.stem in wanted]
    out = {"candidates": [], "plans": [], "episodes": []}
    metas = []
    for p in paths:
        r = load_run(p, oracle_dir / p.name, (Path(ensemble_dir) / p.name) if ensemble_dir else None)
        for k in out:
            out[k].append(r[k])
        metas.append(r["meta"])
    res = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
    res["metas"] = metas
    return res
