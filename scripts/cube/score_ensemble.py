"""Score the recorded oracle candidates with the latent ensemble (post hoc).

    python scripts/cube/score_ensemble.py results/cube/records/*.pt

For each recorded planning call, every oracle-replayed candidate (executed
plan, elites, randoms) is rolled out by each ensemble member from the recorded
current latent. Writes results/cube/ensemble/<run>.pt with, aligned with the
record's ``subset``:

  sig2      (K, P)  variance across members, summed over latent dims
  cost      (K,)    ensemble-mean terminal cost ||mean z_H - z_goal||^2
  pred_err  (K, P)  ensemble-mean prediction error vs the oracle's true latents
                     (if the oracle replay exists)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cube.rollout import per_step_variance  # noqa: E402
from ua.ensemble import load_ensemble  # noqa: E402


def default_ensemble() -> Path:
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    return home / "checkpoints/cube/ens_lewm/ensemble.pt"


@torch.no_grad()
def score_file(rec_path: Path, oracle_dir: Path, out_dir: Path, ens, device: str) -> Path:
    rec = torch.load(rec_path, map_location="cpu", weights_only=False)
    orc_path = oracle_dir / rec_path.name
    orc = torch.load(orc_path, map_location="cpu", weights_only=False)["results"] if orc_path.exists() else {}
    results = {}
    for i, p in enumerate(rec["plans"]):
        z_hist = torch.from_numpy(p["z_hist"]).float().to(device)[None]  # (1, T0, D)
        acts = torch.from_numpy(p["subset_actions"]).float().to(device)[None]  # (1, K, H, A)
        preds = ens.rollouts(z_hist, acts)[:, 0].float()  # (M, K, P, D)
        sig2 = per_step_variance(preds)  # (K, P)
        mean = preds.mean(0)
        z_goal = torch.from_numpy(p["z_goal"]).float().to(device)
        entry = {
            "sig2": sig2.cpu().numpy().astype(np.float32),
            "cost": ((mean[:, -1] - z_goal) ** 2).sum(-1).cpu().numpy().astype(np.float32),
        }
        o = orc.get(i)
        if o is not None:
            true = torch.from_numpy(o["true_lat"].astype(np.float32)).to(device)  # (K, B, D)
            nb = min(true.shape[1], mean.shape[1])
            entry["pred_err"] = ((mean[:, :nb] - true[:, :nb]) ** 2).sum(-1).cpu().numpy().astype(np.float32)
        results[i] = entry
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / rec_path.name
    torch.save({"results": results, "n_members": len(ens)}, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--ensemble", type=Path, default=None)
    ap.add_argument("--oracle", type=Path, default=ROOT / "results/cube/oracle")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/ensemble")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ens, meta = load_ensemble(args.ensemble or default_ensemble(), args.device)
    print(f"[ens-score] ensemble of {len(ens)} members trained on {meta.get('policy')} latents")
    for p in args.records:
        print(f"[ens-score] {score_file(p, args.oracle, args.out, ens, args.device)}")


if __name__ == "__main__":
    main()
