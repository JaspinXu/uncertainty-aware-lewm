"""L3: train the learned risk head on oracle-labelled CEM candidates.

    python scripts/cube/train_risk_head.py            # seeds from cluster/seeds.sh defaults

Data: recorded baseline runs + oracle replays (cluster/jobs/10_record_oracle.sbatch)
  train  = TRAIN seeds (only used here)
  val    = CAL seeds   (early stopping, choice of mode/beta)
  test   = TEST seeds  (offline re-ranking report only; closed-loop eval is L3 panel)

Two heads are trained: ``full`` (features include the MC-dropout sigma^2
profile) and ``det`` (deterministic features only) -- the ablation tells
whether sigma^2 carries information a learned model can use.

Outputs (results/cube/l3/): risk_head_full.pt, risk_head_det.pt,
risk_selection.json (head/mode/beta chosen on val for the closed-loop panel),
risk_report.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cube.metrics import auroc, mcnemar_exact, spearman  # noqa: E402
from ua.risk import RiskHead, risk_features, save_head  # noqa: E402

TRAIN_SEEDS = list(range(100, 116))
CAL_SEEDS = [910, 911, 912]
TEST_SEEDS = [42, 43, 44, 45, 1011, 2022, 3033, 4044, 5055]
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def load_split(records: Path, oracle: Path, seeds: list[int], prefix: str = "baseline") -> dict:
    """Stack all oracle-replayed candidates of the given seeds."""
    keys = ("z0", "zg", "pred", "det_cost", "sig2", "mc_std", "err", "true_cost", "success", "plan", "start_dist", "is_exec")
    acc = {k: [] for k in keys}
    plan_counter = 0
    for seed in seeds:
        rp, op = records / f"{prefix}_s{seed}.pt", oracle / f"{prefix}_s{seed}.pt"
        if not (rp.exists() and op.exists()):
            print(f"[risk] missing {rp.name} or its oracle replay -- skipped")
            continue
        rec = torch.load(rp, map_location="cpu", weights_only=False)
        orc = torch.load(op, map_location="cpu", weights_only=False)["results"]
        for i, p in enumerate(rec["plans"]):
            o = orc.get(i)
            if o is None:
                continue
            sub = p["subset"]
            K = len(sub)
            acc["z0"].append(np.repeat(p["z_hist"][-1][None], K, 0))
            acc["zg"].append(np.repeat(p["z_goal"][None], K, 0))
            acc["pred"].append(p["subset_pred"].astype(np.float32))
            acc["det_cost"].append(p["det_cost"][sub])
            acc["sig2"].append(p["sigma2"][sub])
            acc["mc_std"].append(p["mc_cost_std"][sub])
            acc["err"].append(o["pred_err"][:, -1])
            acc["true_cost"].append(o["true_cost"])
            acc["success"].append(o["success_step"] > 0)
            acc["plan"].append(np.full(K, plan_counter))
            acc["start_dist"].append(np.full(K, o["start_dist"]))
            acc["is_exec"].append(sub == 0)
            plan_counter += 1
    if not acc["z0"]:
        return {}
    out = {k: torch.from_numpy(np.concatenate(v)) for k, v in acc.items()}
    out["n_plans"] = plan_counter
    return out


def head_path_str(path: Path) -> str:
    """Path as stored in risk_selection.json (relative to the repo when possible)."""
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def features(d: dict, use_sigma: bool) -> torch.Tensor:
    return risk_features(
        d["z0"].float(),
        d["zg"].float(),
        d["pred"].float(),
        d["det_cost"].float(),
        d["sig2"].float() if use_sigma else None,
        d["mc_std"].float() if use_sigma else None,
    )


def targets(d: dict) -> torch.Tensor:
    return torch.stack(
        [torch.log1p(d["err"].float()), torch.log1p(d["true_cost"].float()), d["success"].float()], dim=-1
    )


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        F.smooth_l1_loss(out[:, 0], y[:, 0])
        + F.smooth_l1_loss(out[:, 1], y[:, 1])
        + F.binary_cross_entropy_with_logits(out[:, 2], y[:, 2])
    )


def train_head(train: dict, val: dict, use_sigma: bool, device: str, epochs: int, seed: int) -> tuple[RiskHead, dict]:
    torch.manual_seed(seed)
    Xtr, Ytr = features(train, use_sigma), targets(train)
    Xva, Yva = features(val, use_sigma), targets(val)
    head = RiskHead(Xtr.shape[1]).to(device)
    head.feat_mean.copy_(Xtr.mean(0))
    head.feat_std.copy_(Xtr.std(0).clamp_min(1e-4))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = (t.to(device) for t in (Xtr, Ytr, Xva, Yva))
    best, best_state, patience, history = float("inf"), None, 0, []
    bs = 512
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for s in range(0, len(Xtr), bs):
            idx = perm[s : s + bs]
            loss = loss_fn(head(Xtr[idx]), Ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            vl = float(loss_fn(head(Xva), Yva))
        history.append((tot / len(Xtr), vl))
        if vl < best - 1e-4:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_val_loss": best, "epochs_run": len(history), "history": history}


@torch.no_grad()
def predict(head: RiskHead, d: dict, use_sigma: bool, device: str) -> np.ndarray:
    return head(features(d, use_sigma).to(device)).cpu().numpy()


def rerank(d: dict, score: np.ndarray, nontrivial: bool = True) -> np.ndarray:
    """Per-plan success of the argmin-score candidate (optionally non-trivial plans only)."""
    plan = d["plan"].numpy()
    succ = d["success"].numpy()
    sd = d["start_dist"].numpy()
    out = []
    for p in np.unique(plan):
        idx = np.nonzero(plan == p)[0]
        if nontrivial and sd[idx[0]] <= 0.04:
            continue
        out.append(bool(succ[idx[np.argmin(score[idx])]]))
    return np.asarray(out, dtype=bool)


def scores_for(d: dict, out: np.ndarray, mode: str, beta: float) -> np.ndarray:
    det = d["det_cost"].numpy()
    if mode == "det":
        return det
    if mode == "err":
        return det + beta * np.clip(np.expm1(out[:, 0]), 0, None)
    if mode == "true_cost":
        return np.expm1(out[:, 1])
    raise ValueError(mode)


def offline_eval(d: dict, out: np.ndarray) -> dict:
    """Prediction quality on candidates of a split."""
    nt = d["start_dist"].numpy() > 0.04
    err, tc, succ = d["err"].numpy(), d["true_cost"].numpy(), d["success"].numpy()
    return {
        "spearman_err": spearman(out[nt, 0], err[nt]),
        "spearman_true_cost": spearman(out[nt, 1], tc[nt]),
        "auroc_failure_from_success_logit": auroc(-out[nt, 2], ~succ[nt]),
        "auroc_failure_from_det_cost": auroc(d["det_cost"].numpy()[nt], ~succ[nt]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=ROOT / "results/cube/records")
    ap.add_argument("--oracle", type=Path, default=ROOT / "results/cube/oracle")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cube/l3")
    ap.add_argument("--train-seeds", type=int, nargs="*", default=TRAIN_SEEDS)
    ap.add_argument("--cal-seeds", type=int, nargs="*", default=CAL_SEEDS)
    ap.add_argument("--test-seeds", type=int, nargs="*", default=TEST_SEEDS)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    train = load_split(args.records, args.oracle, args.train_seeds)
    val = load_split(args.records, args.oracle, args.cal_seeds)
    test = load_split(args.records, args.oracle, args.test_seeds)
    if not train or not val:
        sys.exit("need recorded+replayed runs for the train and calibration seeds")
    sizes = {k: (int(len(v["det_cost"])) if v else 0, int(v["n_plans"]) if v else 0) for k, v in (("train", train), ("val", val), ("test", test))}
    print(f"[risk] candidates/plans: {sizes}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"sizes": sizes, "heads": {}}
    options = []  # (val_success, preference, choice)
    base_val = rerank(val, val["det_cost"].numpy())
    for name, use_sigma in (("full", True), ("det", False)):
        head, info = train_head(train, val, use_sigma, args.device, args.epochs, args.seed)
        meta = {"in_dim": int(head.feat_mean.numel()), "use_sigma": use_sigma, "hidden": 256, "dropout": 0.1,
                "train_seeds": args.train_seeds, "best_val_loss": info["best_val_loss"]}
        save_head(head.cpu(), args.out / f"risk_head_{name}.pt", meta)
        head.to(args.device)
        o_val = predict(head, val, use_sigma, args.device)
        entry = {"train": info["epochs_run"], "best_val_loss": info["best_val_loss"], "val_quality": offline_eval(val, o_val), "rerank_val": {}}
        for mode, betas in (("err", BETAS), ("true_cost", [None])):
            for beta in betas:
                s = rerank(val, scores_for(val, o_val, mode, beta or 0.0))
                key = f"{mode}" + (f"@{beta}" if beta is not None else "")
                entry["rerank_val"][key] = float(s.mean()) if len(s) else None
                if len(s):
                    # ties: prefer the sigma^2 head, the error penalty and beta closest to 1
                    pref = (name != "full", mode != "err", abs(np.log2(beta)) if beta else 0.0)
                    options.append((float(s.mean()), pref, {
                        "head": name, "mode": mode, "beta": float(beta or 1.0), "val_success": float(s.mean()),
                        "head_path": head_path_str(args.out / f"risk_head_{name}.pt")}))
        if test:
            o_test = predict(head, test, use_sigma, args.device)
            entry["test_quality"] = offline_eval(test, o_test)
        report["heads"][name] = entry
        print(f"[risk] head={name}: {json.dumps({k: entry[k] for k in ('best_val_loss', 'val_quality')})}")

    report["val_det_cost_success"] = float(base_val.mean()) if len(base_val) else None
    best = max(o[0] for o in options)
    choice = sorted([o for o in options if o[0] == best], key=lambda o: o[1])[0][2]
    report["selection"] = choice
    if test:
        from ua.risk import load_head

        head, meta = load_head(args.out / f"risk_head_{choice['head']}.pt", args.device)
        o_test = predict(head, test, meta["use_sigma"], args.device)
        s0 = rerank(test, test["det_cost"].numpy())
        s1 = rerank(test, scores_for(test, o_test, choice["mode"], choice["beta"]))
        ex = rerank(test, np.where(test["is_exec"].numpy(), 0.0, 1.0))  # executed plan
        report["test_rerank"] = {
            "n_plans": int(len(s0)),
            "exec_success": float(ex.mean()),
            "det_cost_success": float(s0.mean()),
            "risk_success": float(s1.mean()),
            "mcnemar": mcnemar_exact(s0, s1),
        }
    (args.out / "risk_selection.json").write_text(json.dumps(choice, indent=1))
    (args.out / "risk_report.json").write_text(json.dumps(report, indent=1, default=float))
    write_md(report, args.out / "risk_report.md")
    print(f"[risk] selection {choice}; done in {time.time() - t0:.0f}s")


def write_md(r: dict, path: Path) -> None:
    def f(v, pct=False):
        return "n/a" if v is None else (f"{100 * v:.1f}%" if pct else f"{v:.3f}")

    L = ["# L3 learned risk head (offline)\n", "Generated by `scripts/cube/train_risk_head.py`.\n",
         f"Candidates / plans: {r['sizes']}\n", "## Prediction quality on calibration seeds (non-trivial plans)\n",
         "| head | Spearman err | Spearman true cost | AUROC failure (head) | AUROC failure (LeWM cost) |", "|---|---:|---:|---:|---:|"]
    for name, e in r["heads"].items():
        q = e["val_quality"]
        L.append(f"| {name} | {f(q['spearman_err'])} | {f(q['spearman_true_cost'])} | {f(q['auroc_failure_from_success_logit'])} | {f(q['auroc_failure_from_det_cost'])} |")
    L.append("\n## Re-ranking replayed candidates (calibration seeds)\n")
    L.append(f"LeWM cost alone: {f(r['val_det_cost_success'], True)}\n")
    keys = list(next(iter(r["heads"].values()))["rerank_val"])
    L.append("| head | " + " | ".join(keys) + " |")
    L.append("|---|" + "---:|" * len(keys))
    for name, e in r["heads"].items():
        L.append(f"| {name} | " + " | ".join(f(e["rerank_val"][k], True) for k in keys) + " |")
    L.append(f"\nSelected for the closed-loop panel: `{r['selection']}`\n")
    if "test_rerank" in r:
        t = r["test_rerank"]
        m = t["mcnemar"]
        L.append("## Test seeds (offline re-ranking, non-trivial plans)\n")
        L.append(f"{t['n_plans']} plans: executed plan {f(t['exec_success'], True)}, LeWM-cost argmin {f(t['det_cost_success'], True)}, "
                 f"risk-aware argmin {f(t['risk_success'], True)} (McNemar p = {m['p_value']:.3g}, {m['b_only']} gained / {m['a_only']} lost)")
        for name, e in r["heads"].items():
            if "test_quality" in e:
                q = e["test_quality"]
                L.append(f"\n{name} head on test: Spearman err {f(q['spearman_err'])}, AUROC failure {f(q['auroc_failure_from_success_logit'])}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    os._exit(0)
