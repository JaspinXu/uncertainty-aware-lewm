"""The pre-registered Step-1 criteria (docs/step1_preregistration.md).

C1  within-state Spearman(sigma^2, real prediction error) over CEM candidates
C2  share of the oracle re-ranking headroom captured by a sigma^2 penalty
C3  per-replan AUROC(sigma^2 -> segment failure) and its gain over cost-only
G   share of failures where the model was exploited (category a)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# thresholds, fixed in the pre-registration
RHO_MEDIAN_MIN = 0.30
RHO_POSITIVE_MIN = 0.70
HEADROOM_SHARE_MIN = 0.20
AUROC_MIN = 0.65
CATEGORY_A_MIN = 0.20
LAMBDA_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
N_BOOT = 2000


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Paired binary outcomes: episodes b gains over a, episodes it loses, exact two-sided McNemar p."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    gain, loss = int((b & ~a).sum()), int((a & ~b).sum())
    return gain, loss, (1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue))


def spearman(x, y) -> float:
    rho = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return 0.0 if not np.isfinite(rho) else float(rho)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


# --------------------------------------------------------------------------- #
# C1
# --------------------------------------------------------------------------- #
@dataclass
class C1Result:
    rho_state: np.ndarray  # per-state mean of first/last pools
    rho_first: np.ndarray
    rho_last: np.ndarray
    median: float
    frac_positive: float
    median_ci: tuple
    passed: bool


def criterion_c1(sig_first, err_first, sig_last, err_last, seed=0) -> C1Result:
    """Inputs: lists over states of per-candidate arrays."""
    rf = np.array([spearman(s, e) for s, e in zip(sig_first, err_first)])
    rl = np.array([spearman(s, e) for s, e in zip(sig_last, err_last)])
    rs = 0.5 * (rf + rl)
    rng = np.random.default_rng(seed)
    boot = [np.median(rs[rng.integers(0, len(rs), len(rs))]) for _ in range(N_BOOT)]
    med, frac = float(np.median(rs)), float(np.mean(rs > 0))
    return C1Result(
        rs, rf, rl, med, frac, tuple(np.percentile(boot, [2.5, 97.5])),
        passed=med >= RHO_MEDIAN_MIN and frac >= RHO_POSITIVE_MIN,
    )


# --------------------------------------------------------------------------- #
# C2
# --------------------------------------------------------------------------- #
@dataclass
class C2Result:
    share: float
    ci: tuple
    lam_counts: dict
    headroom_mean: float  # mean o(base) - o(oracle), m
    captured_latent_oracle: float  # share captured by argmin d_real (reference)
    decision_change_rate: float
    passed: bool
    extra: dict = field(default_factory=dict)


def _penalty_outcomes(o, d, s, lam_grid):
    """(S, L) outcome of the penalty pick for every state and lambda."""
    out = np.empty((len(o), len(lam_grid)))
    change = np.empty((len(o), len(lam_grid)), dtype=bool)
    for i, (oi, di, si) in enumerate(zip(o, d, s)):
        zd, zs = _zscore(di), _zscore(si)
        base = int(np.argmin(di))
        for j, lam in enumerate(lam_grid):
            pick = int(np.argmin(zd + lam * zs))
            out[i, j] = oi[pick]
            change[i, j] = pick != base
    return out, change


def _crossfit_share(cap, head, idx, rng):
    """2-fold cross-fitted captured share on the (multi)set of states ``idx``."""
    perm = rng.permutation(idx)
    a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
    total_head = head[idx].sum()
    if total_head <= 0:
        return np.nan, None, None
    la = int(np.argmax(cap[a].sum(axis=0)))  # first max = smaller lambda on ties
    lb = int(np.argmax(cap[b].sum(axis=0)))
    captured = cap[b, la].sum() + cap[a, lb].sum()
    return captured / total_head, la, lb


def criterion_c2(o, d, s, d_real, lam_grid=LAMBDA_GRID, seed=0) -> C2Result:
    """o, d, s, d_real: lists over states of per-candidate arrays (last pool)."""
    o_base = np.array([oi[int(np.argmin(di))] for oi, di in zip(o, d)])
    o_orc = np.array([oi.min() for oi in o])
    o_lat = np.array([oi[int(np.argmin(ri))] for oi, ri in zip(o, d_real)])
    head = o_base - o_orc
    o_pen, change = _penalty_outcomes(o, d, s, lam_grid)
    cap = o_base[:, None] - o_pen  # (S, L)

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(o))
    shares, lam_counts = [], {lam: 0 for lam in lam_grid}
    for _ in range(100):  # repeated cross-fitting for the point estimate
        sh, la, lb = _crossfit_share(cap, head, all_idx, rng)
        shares.append(sh)
        if la is not None:
            lam_counts[lam_grid[la]] += 1
            lam_counts[lam_grid[lb]] += 1
    share = float(np.nanmean(shares))
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(o), len(o))
        boot.append(_crossfit_share(cap, head, idx, rng)[0])
    ci = tuple(np.nanpercentile(boot, [2.5, 97.5]))
    best = max(lam_counts, key=lam_counts.get)
    total = head.sum()
    return C2Result(
        share=share,
        ci=ci,
        lam_counts=lam_counts,
        headroom_mean=float(head.mean()),
        captured_latent_oracle=float((o_base - o_lat).sum() / total) if total > 0 else np.nan,
        decision_change_rate=float(change[:, list(lam_grid).index(best)].mean()),
        passed=share >= HEADROOM_SHARE_MIN and ci[0] > 0,
        extra={
            "states_with_headroom": int((head > 0).sum()),
            "base_success_rate": float(np.mean(o_base == 0)),
            "oracle_success_rate": float(np.mean(o_orc == 0)),
            "latent_oracle_success_rate": float(np.mean(o_lat == 0)),
            "penalty_success_rate_by_lambda": {
                lam: float(np.mean(o_pen[:, j] == 0)) for j, lam in enumerate(lam_grid)
            },
        },
    )


# --------------------------------------------------------------------------- #
# C3
# --------------------------------------------------------------------------- #
def oof_logistic(X, y, groups, n_splits=5) -> np.ndarray:
    p = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def _cluster_boot_idx(groups, rng):
    uniq = np.unique(groups)
    members = {g: np.nonzero(groups == g)[0] for g in uniq}
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([members[g] for g in pick])


@dataclass
class C3Result:
    n: int
    positives: int
    auroc: float
    auroc_ci: tuple
    auroc_cost: float
    auroc_full: float
    delta: float
    delta_ci: tuple
    passed: bool


def criterion_c3(y, sigma2, d_pred, d0, groups, seed=0) -> C3Result:
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    Xc = np.column_stack([np.log(d_pred), np.log(d0)])
    Xf = np.column_stack([Xc, np.log(sigma2)])
    pc = oof_logistic(Xc, y, groups)
    pf = oof_logistic(Xf, y, groups)
    a = roc_auc_score(y, sigma2)
    ac, af = roc_auc_score(y, pc), roc_auc_score(y, pf)
    rng = np.random.default_rng(seed)
    boot_a, boot_d = [], []
    for _ in range(N_BOOT):
        idx = _cluster_boot_idx(groups, rng)
        if y[idx].min() == y[idx].max():
            continue
        boot_a.append(roc_auc_score(y[idx], sigma2[idx]))
        boot_d.append(roc_auc_score(y[idx], pf[idx]) - roc_auc_score(y[idx], pc[idx]))
    d_ci = tuple(np.percentile(boot_d, [2.5, 97.5]))
    return C3Result(
        n=len(y), positives=int(y.sum()), auroc=float(a),
        auroc_ci=tuple(np.percentile(boot_a, [2.5, 97.5])),
        auroc_cost=float(ac), auroc_full=float(af), delta=float(af - ac), delta_ci=d_ci,
        passed=a >= AUROC_MIN and d_ci[0] > 0,
    )


def episode_auroc(y_ep, score_ep, seed=0):
    """Secondary: AUROC with a plain bootstrap over episodes."""
    y_ep, score_ep = np.asarray(y_ep).astype(int), np.asarray(score_ep)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y_ep), len(y_ep))
        if y_ep[idx].min() != y_ep[idx].max():
            boot.append(roc_auc_score(y_ep[idx], score_ep[idx]))
    return float(roc_auc_score(y_ep, score_ep)), tuple(np.percentile(boot, [2.5, 97.5]))


# --------------------------------------------------------------------------- #
# gate G
# --------------------------------------------------------------------------- #
@dataclass
class TaxonomyResult:
    tau: float
    counts: dict
    shares: dict
    labels: list
    passed: bool


def failure_taxonomy(failed_episodes, d_real_at_goal) -> TaxonomyResult:
    """failed_episodes: dicts with ``d_pred``/``d_real`` lists over replans and
    ``init_dist``/``final_dist`` (m). ``d_real_at_goal``: d_real of every replay
    that ended with the cube within the success radius."""
    tau = float(np.percentile(d_real_at_goal, 90))
    labels = []
    for ep in failed_episodes:
        dp, dr = np.asarray(ep["d_pred"]), np.asarray(ep["d_real"])
        if np.any((dp <= tau) & (dr > tau)):
            labels.append("a")
        elif np.any(dr <= tau):
            labels.append("b")
        elif ep["final_dist"] <= 0.5 * ep["init_dist"]:
            labels.append("d")
        else:
            labels.append("c")
    counts = {k: labels.count(k) for k in "abcd"}
    n = max(1, len(labels))
    shares = {k: v / n for k, v in counts.items()}
    return TaxonomyResult(tau, counts, shares, labels, passed=shares["a"] >= CATEGORY_A_MIN)


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #
def decide(mc_pass: bool, ens_pass: bool | None, gate_pass: bool) -> tuple[str, str]:
    """(branch, reason) following the pre-registered rule."""
    if not gate_pass:
        return "B", "gate G failed: model-exploitation failures are < 20% of failures"
    if mc_pass:
        return "A-mc", "MC-dropout sigma^2 passes C1-C3"
    if ens_pass:
        return "A-ensemble", "only the ensemble sigma^2 passes C1-C3"
    return "B", "neither MC-dropout nor ensemble sigma^2 passes C1-C3"
