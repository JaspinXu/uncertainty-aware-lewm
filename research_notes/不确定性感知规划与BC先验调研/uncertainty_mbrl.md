# Uncertainty-aware planning and offline model-based RL: how model uncertainty is estimated and injected, and whether it tracks true model error

Source note: every arXiv paper below was read in full-text HTML through ar5iv (`https://ar5iv.labs.arxiv.org/html/<id>`, with formulas taken from the LaTeX alt-text). Links point to the canonical arXiv abstract pages. "Not verified" marks claims I could not check against full text. WebSearch/WebFetch did not work in this environment, so I found papers through DuckDuckGo result pages. Two sources (Osband 2016 BDL workshop paper; RA-MPPI) are known only from search snippets and are flagged where used.

## 1. PETS (Chua et al. 2018): what probabilistic ensembles + trajectory sampling show about aleatoric vs epistemic modeling

### Takeaway
PETS's ablations rank model classes PE > P > DE > D, and find that model choice (capturing uncertainty while training the model) matters much more than how uncertainty is propagated at planning time. At planning time, PETS uses uncertainty only as a risk-neutral expectation over sampled particles: there is no penalty and no gate. Its probabilistic heads also output arbitrary variances off-distribution unless soft-bounded.

### Cited Findings
**Model taxonomy and losses**
- PETS separates aleatoric from epistemic uncertainty into four model classes: D (deterministic NN: neither), P (probabilistic NN: aleatoric only), DE (deterministic ensemble: epistemic only), PE (probabilistic ensemble: both). The GP baseline has homoscedastic aleatoric plus epistemic uncertainty — [Chua et al. 2018, PETS](https://arxiv.org/abs/1805.12114)
- P networks output a Gaussian `N(μθ(s,a), Σθ(s,a))` trained with Gaussian NLL `Σ_n [μθ(s_n,a_n) − s_{n+1}]ᵀ Σθ⁻¹ [μθ(s_n,a_n) − s_{n+1}] + log det Σθ(s_n,a_n)`. The paper notes this "does not model epistemic uncertainty, which cannot be captured with purely discriminative training" — [PETS](https://arxiv.org/abs/1805.12114)
- Ensembles: B bootstrap models, each trained on a resample (with replacement, size N) of the dataset. The predictive distribution is `f̃ = (1/B) Σ_b f̃_θb`. "We found B=5 sufficient" — [PETS](https://arxiv.org/abs/1805.12114)

**Planning objective and propagation**
- MPC picks `argmax_{a_t:t+T} Σ_τ E_f̃[r(s_τ,a_τ)]` with CEM. CEM learned faster than random shooting — [PETS](https://arxiv.org/abs/1805.12114)
- Trajectory sampling (TS): P particles, `s^p_{t+1} ~ f̃_θ_b(p,t)(s^p_t, a_t)`. TS1 resamples a bootstrap per particle at every step. TS∞ fixes each particle's bootstrap for the whole trial, which makes aleatoric and epistemic variance separable: aleatoric = average variance of same-bootstrap particles; epistemic = variance of the per-bootstrap particle means. Settings: P=20, B=5 — [PETS](https://arxiv.org/abs/1805.12114)
- Baseline propagators:
  - Expectation (E): `s_{t+1} = E[f̃(s_t,a_t)]`.
  - Moment matching (MM): `s^p_{t+1} iid~ N(E_{p,b}[s^{p,b}_{t+1}], V_{p,b}[s^{p,b}_{t+1}])`.
  - Distribution sampling (DS): moment-match over bootstraps only, `s^p_{t+1} ~ N(E_b[·], V_b[·])`.
  - Source: [PETS](https://arxiv.org/abs/1805.12114)

**Ablation results**
- Figure 4: "The model choice seems to be more important than the technique used to propagate the state/action space. Among the models the ranking in terms of performance is: PE > P > DE > D." PE-XX is best on all tasks except cartpole; P and DE are close seconds; D is worst — [PETS](https://arxiv.org/abs/1805.12114)
- "the quality of the model and the use of uncertainty at learning time significantly affect the performance … while the use of more advanced uncertainty propagation techniques seem to offer only minor improvements." MM is competitive in low dimensions but "not a reliable MBRL choice in higher dimensions, e.g. the half cheetah" — [PETS](https://arxiv.org/abs/1805.12114)
- PETS reaches PPO's asymptotic performance in under 100 trials on all four tasks. Its asymptotic performance "substantially exceeds" the deterministic MBMF of Nagabandi et al., which the authors equate with their D-E variant: "This result highlights the value of uncertainty estimation." It needs 8× fewer samples than SAC on half-cheetah. GP-MM slightly beats PETS on cartpole — [PETS](https://arxiv.org/abs/1805.12114)

**Failure modes and caveats**
- Probabilistic-head variance is "undefined outside the trained distribution … in practice we noticed how it occasionally collapse to zero, or explode toward infinity". Fix: soft-clamp log-variance between learned bounds (`logvar = max_logvar − softplus(max_logvar − logvar); logvar = min_logvar + softplus(logvar − min_logvar)`) with small penalties on `max_logvar` and `−min_logvar` (Appendix A.1). The main text: their "variance has arbitrary values for out-of-distribution inputs, which can disrupt planning" — [PETS](https://arxiv.org/abs/1805.12114)
- Horizon: probabilistic propagation is "robust to horizons set 'too long'" because particle separation reduces dependence on far-future expected cost ("the action selection procedure then effectively ignores the unpredictable"). Deterministic propagation degrades when the horizon is too long (Appendix A.7, Fig. A.8) — [PETS](https://arxiv.org/abs/1805.12114)
- Epistemic-only uncertainty for directed exploration is "left for future work"; PETS does not use it for exploration — [PETS](https://arxiv.org/abs/1805.12114)

### Inferences
- PETS's gains come mostly from (a) training-time regularization against overfitting (bootstrapped PE) and (b) averaging cost over diverging particles, which implicitly discounts unpredictable far-future steps. A frozen world model cannot get (a) at all. (b) only helps if the particle spread actually grows where the model is wrong.
- PETS-style "mean over particles" is risk-neutral. It does not penalize disagreement, so CEM can still pick action sequences where all particles agree on a wrong, optimistic outcome. An MC-dropout analog (mean over 5 masks) inherits this, and PETS's own ablation says propagation details matter little next to model quality.
- The OOD-variance pathology argues against using a raw learned variance or σ² head as a planning signal without bounding or calibration.

### Gaps
- Per-task numbers for the PE/P/DE/D × TS1/TS∞/MM/DS grid exist only as plots (Fig. 4). I could not extract exact values from text.
- PETS gives no direct test of whether its uncertainty correlates with prediction error.

## 2. MBPO, STEVE, M2AC (plus MOReL truncation, MACURA, Infoprop, CBOP): how uncertainty controls the effective horizon

### Takeaway
MBPO controls horizon with a fixed short-rollout schedule, not with uncertainty. STEVE and CBOP weight h-step targets by inverse ensemble variance. M2AC masks the most uncertain fraction of model transitions by rank. MOReL, MACURA and Infoprop truncate rollouts when an ensemble-disagreement or information-loss measure crosses a threshold calibrated on in-distribution quantiles. All of these gates sit in front of a value function or a terminal penalty. In the offline setting, inverse-variance weighting without pessimism (STEVE) diverged on 11/18 D4RL tasks.

### Cited Findings
**MBPO**
- Bound: `η[π] ≥ η̂[π] − [2γ r_max(ε_m + 2ε_π)/(1−γ)² + 4 r_max ε_π/(1−γ)]`, where `ε_m` is model TV error on the data policy's distribution and `ε_π` bounds policy shift — [Janner et al. 2019, MBPO](https://arxiv.org/abs/1906.08253)
- The branched-rollout bound has "one [term] exponentially decreasing in k and another scaling linearly with k", but "taken literally, this lower bound is maximized when k=0". Empirically, model error grows with KL(π‖π_D), more slowly when more data is available (Fig. 1). This motivates `ε̂_m'(ε_π) ≈ ε_m + ε_π·dε_m'/dε_π` — [MBPO](https://arxiv.org/abs/1906.08253)
- Practice: ensemble size B=7 probabilistic networks; 400 model rollouts per environment step; 20–40 policy updates per step. Horizon k = 1 on some tasks, otherwise a thresholded linear schedule `f(e) = min(max(x + (e−a)/(b−a)·(y−x), x), y)` growing from 1 to 15–25 over epochs — [MBPO, App. C](https://arxiv.org/abs/1906.08253). Only the top-5 "Elite" models (by validation) are used — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135). The propagating member is sampled randomly per step — [Frauenknecht et al. 2024, MACURA](https://arxiv.org/abs/2405.19014)
- "single-step rollouts provide a baseline that is difficult to beat". On Hopper, model-predicted and real returns are "highly correlated", i.e. the policy "may not exploit the model at all if the rollouts are sufficiently short"; "model returns tend to underestimate real environment returns". The authors claim "a combination of model ensembles with short model rollouts is sufficient to prevent model exploitation" — [MBPO](https://arxiv.org/abs/1906.08253)

**STEVE (uncertainty-weighted horizon)**
- MVE target: `T^MVE_H = r + Σ_{i=1..H} D^i γ^i r̂(s'_{i−1}, a'_{i−1}, s'_i) + D^{H+1} γ^{H+1} Q̂(s'_H, a'_H)`, with `D^i` the accumulated predicted non-termination. STEVE keeps ensembles of L Q-functions, N reward models and M dynamics models, which gives M·N·L samples of each candidate target `T_i` (i = 0..H) — [Buckman et al. 2018, STEVE](https://arxiv.org/abs/1807.01675)
- `T^STEVE_H = Σ_i (w̃_i / Σ_j w̃_j) T^μ_i`, with `w̃_i^{-1} = T^{σ²}_i` (inverse-variance weighting). This is derived by minimizing `Bias² + Σ w_i² Var` but with the bias terms dropped ("we could not devise a reliable estimator for the bias terms, and this is a limitation") and cross-covariances omitted — [STEVE](https://arxiv.org/abs/1807.01675)
- Dropout or BNN sampling could replace the ensemble, "however, we found that these methods degraded the accuracy of the base models" — [STEVE](https://arxiv.org/abs/1807.01675)
- Results: STEVE is "the only algorithm to show significant learning within 5M frames" on Humanoid-v1, RoboschoolHumanoid(Flagrun) and BipedalWalkerHardcore. Every ablation (Ensemble-MVE, uniform-weight Mean-MVE, TD(λ=0.25/0.75), COV-STEVE) shows "fast initial gains, which quickly taper off and are overtaken by the baseline". Each ablation was run only twice. Model usage sits near 50% for most environments, but STEVE "chooses to ignore the model almost immediately" on Humanoid — [STEVE](https://arxiv.org/abs/1807.01675)

**M2AC (masking uncertain rollouts)**
- Masked rollout bound: `Q^π(s,a) ≥ Q̃^π_mask(s,a) − αε Σ_t γ^t w(t;s,a)`, with `α = 2r_max/(1−γ)`, ε = the maximum TV error over masked-in pairs, and `w(t;s,a)` the probability that step t is still model-based. Equivalent per-step form: `E[Σ_{t<H} γ^t (r̃ − αε) + γ^H Q^π(s_H,a_H)]` — [Pan et al. 2020, M2AC](https://arxiv.org/abs/2010.04893)
- Rank-based mask: in each batch of B model transitions, "only keep the first wB samples with small uncertainty". Because it is rank-based, "we only need to find a predictor u(s,a) that is positively correlated to D_TV". Kept transitions get reward `r̃ − α u` — [M2AC](https://arxiv.org/abs/2010.04893)
- One-vs-Rest uncertainty: `u_k = KL(N(μ_k, σ²_k) ‖ N(μ_−k, σ²_−k))`, with `μ_−k = mean_{i≠k} μ_i` and `σ²_−k = (1/(K−1)) Σ_{i≠k}(σ_i² + μ_i²) − μ²_−k` — [M2AC](https://arxiv.org/abs/2010.04893)
- Settings: w = 0.5 when H_max = 1; `w_h = (H_max − h)/(2(H_max + 1))` otherwise (mean 0.25); α = 10⁻³ — [M2AC](https://arxiv.org/abs/2010.04893)
- Results:
  - At H_max = 1, M2AC ≈ MBPO. For H_max > 1, "the performance of MBPO drops rapidly" while M2AC "can even benefit from longer model rollout". STEVE beats MBPO at long rollouts, "most likely because of its reweighting mechanism".
  - Noisy-action environments (σ = 0.05/0.1/0.2; 25k/50k steps): M2AC is "significantly" better than MBPO, though Walker2d returns still drop at longer rollouts.
  - Ablations: not very sensitive to w (w = 0.9 hurts at long horizons). α = 0.001 is best among {0.01, 0.001, 0}: "too small an α encourages model exploitation". OvR beats full-ensemble (OvA) disagreement and negative likelihood.
  - Source: [M2AC](https://arxiv.org/abs/2010.04893)

**Uncertainty-gated truncation successors**
- MOReL (practical): "if the discrepancy … exceeds thresh, the rollout is truncated at this timestep and is assigned a large negative reward". `thresh = μ_d + β·σ_d` over dataset disagreements, with β tuned in [0, (m_d − μ_d)/σ_d] in steps of 5 — [Kidambi et al. 2020, MOReL](https://arxiv.org/abs/2005.05951)
- MACURA:
  - Trusted set `E_PE = {s | u_PE(s,a) < κ, a ~ π}`.
  - `u_GJS = (2/(E(E−1))) Σ_{e} Σ_{f<e} D_GJS(N_e ‖ N_f)`, a closed-form geometric Jensen–Shannon divergence between Gaussian members.
  - `κ = (ξ/K) Σ_k û_GJS,k`, where û is the ζ-quantile of first-step uncertainties at rollout round k.
  - "wherever the model is uncertain, rollouts are terminated quickly, while longer rollouts can be generated where it is certain."
  - Source: [MACURA](https://arxiv.org/abs/2405.19014)
- Infoprop:
  - Removes the epistemic part of the predictive distribution (TS "introduces additional stochasticity that leads to an overestimated total variance") and tracks information-theoretic "data corruption".
  - Two termination thresholds per state dimension: `λ1^k` = ζ1-quantile of single-step Infoprop entropies over D_env (restricts usage to the accurate set), and `λ2^k = ξ ×` ζ2-quantile (caps accumulated corruption). Values ζ1 = 0.99, ζ2 = 0.01, ξ = 100 for all experiments. A rollout stops as soon as any dimension overshoots.
  - Consistent rollouts up to 100 steps, versus MBPO's 11-step widely spread rollouts and MACURA's 10-step cap. Infoprop-Dyna is "on par with or better than MACURA" and substantially better than MBPO.
  - Limitations: nonstationary buffers and primacy bias.
  - Source: [Frauenknecht et al. 2025, Infoprop](https://arxiv.org/abs/2501.16918)
- CBOP (offline):
  - Treats each h-step return `R̂_h ~ N(μ_h, σ_h²)` (from ensembles) as a noisy observation of Q̂^π with an improper prior. The posterior has `ρ = Σ_h ρ_h` and `μ = Σ_h (ρ_h/Σρ_h) μ_h` — STEVE's weights — and CBOP uses the lower confidence bound (LCB) as the target.
  - The expected horizon `E[h] = Σ w_h h` grows as the dynamics model improves.
  - CBOP outperforms MOPO, MOReL and COMBO by 116.4%, 23.2% and 23.7% on average. STEVE (MAP, no LCB) "fails to learn useful policies for 11 out of 18 tasks".
  - Source: [Jeong et al. 2023, CBOP](https://arxiv.org/abs/2210.03802)

### Inferences
- For a gated-horizon planner, the literature supports three design rules:
  - Gates need only monotone ranking (M2AC) and should be calibrated against quantiles of in-distribution one-step uncertainty (MOReL's μ+βσ, MACURA's ζ-quantile, Infoprop's ζ1/ζ2), not a raw σ² scale.
  - Truncation must be paired with something at the cut point: a −κ absorbing reward (MOReL), a bootstrapped value (STEVE, CBOP, M2AC's model-free Q), or a learned terminal value.
  - Offline, the combination should be pessimistic (LCB), not MAP (CBOP vs STEVE).
- Key difference between Dyna-style use and MPC scoring. In Dyna, truncation just produces fewer model samples. In CEM/MPPI, truncating a candidate's cost sum at high σ² changes its score relative to other candidates. If cost is a goal distance accumulated over steps and a truncated candidate gets no terminal charge, early-uncertain candidates can look better. That opens a new exploitation channel unless a terminal penalty or value is added (MOReL's HALT −κ is the canonical fix).
- STEVE's own results show inverse-variance horizon weighting will "ignore the model" when variances are large everywhere (Humanoid). With a poorly discriminative σ², weighting collapses toward short horizons rather than selectively distrusting bad candidates.

### Gaps
- I did not capture MBPO's Theorem 4.2/4.3 exact expressions (text only describes their k-dependence), or the per-environment k mapping in the flattened MBPO table.
- MACURA's chosen ζ and ξ values, and the numeric learning curves for M2AC, STEVE, MACURA and Infoprop, are only in figures and were not extracted.

## 3. MOPO, MOReL, COMBO, RAMBO: reward penalty vs pessimistic MDP vs conservative regularization without explicit uncertainty

### Takeaway
MOPO subtracts λ·(max ensemble predicted std) from reward. MOReL sends "unknown" (disagreement > threshold) state-actions to an absorbing HALT state with reward −κ. COMBO and RAMBO drop explicit uncertainty and regularize the value function (COMBO) or the model itself (RAMBO). COMBO argues that heuristic uncertainty "struggles to predict the true model error" on tasks that require out-of-distribution generalization, but it offers only a qualitative regression plot as evidence. All of these methods are sensitive to their pessimism hyperparameter, and an intermediate level works best.

### Cited Findings
**MOPO (reward penalty)**
- `r̃(s,a) = r(s,a) − λ u(s,a)`, with `λ := γc`. Given an admissible estimator: `η_M(π) ≥ E_{ρ̂}[r − γ|G(s,a)|] ≥ E_{ρ̂}[r − λu] = η_M̃(π)`. Theorem 4.4: `η_M(π̂) ≥ sup_π {η_M(π) − 2λ ε_u(π)}` — [Yu et al. 2020, MOPO](https://arxiv.org/abs/2005.13239)
- Practical estimator: `u(s,a) = max_{i=1..N} ‖Σ^i_φ(s,a)‖_F` ("the maximum standard deviation of the learned models in the ensemble"). It "lacks theoretical guarantees". λ is a user hyperparameter: "larger if our heuristic u(s,a) underestimates the true error and smaller if u substantially overestimates" — [MOPO](https://arxiv.org/abs/2005.13239). MOPO swept λ ∈ {1, 5} with h ≤ 5 — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)
- Ablation (Table 3, average return; columns are halfcheetah-mixed / walker2d-mixed / halfcheetah-jump / ant-angle) — [MOPO, App. D](https://arxiv.org/abs/2005.13239):

  | Variant | halfcheetah-mixed | walker2d-mixed | halfcheetah-jump | ant-angle |
  |---|---|---|---|---|
  | MOPO | 6405.8 | 1916.4 | 4016.6 | 2530.9 |
  | ensemble-disagreement penalty | 6448.7 | 1923.6 | 3577.3 | 2256.0 |
  | no penalty | 6409.1 | 1421.2 | −980.8 | 18.6 |
  | MBPO | 5598.4 | 1021.8 | 2971.4 | 13.6 |
  | MBPO, no ensemble | 2247.2 | 500.3 | −68.7 | −720.1 |
  | oracle (true-error) penalty | 6984.0 | N/A | 3818.6 | N/A |

  On halfcheetah-jump, mean-variance and max-variance penalties gave 4166.3 vs 4140.6.
- On D4RL, MOPO is weak on narrow data: hopper-medium 28.0, and hopper-medium-expert 23.7 against BC's 111.9. The authors' hypothesis: "lack of action diversity in the medium datasets make it more difficult to learn a model that generalizes well" — [MOPO](https://arxiv.org/abs/2005.13239)

**MOReL (pessimistic MDP with HALT)**
- α-USAD flags (s,a) as known only if `D_TV(P̂, P) ≤ α` "can be guaranteed". The P-MDP sends unknown (s,a) to an absorbing HALT state with reward −κ. Bound: `J(π, M̂_p) ≥ J(π, M) − 2R_max/(1−γ)·D_TV(ρ0, ρ̂0) − 2γR_max/(1−γ)²·α − 2R_max/(1−γ)·E[γ^{T_U^π}]`, where T_U is the hitting time of unknown states — [MOReL](https://arxiv.org/abs/2005.05951)
- Practical: `disc(s,a) = max_{i,j} ‖f_φi(s,a) − f_φj(s,a)‖₂` over 4 MLP models. HALT penalty `r_min(D) −` {30, 50, 100, 200}, depending on environment — [MOReL](https://arxiv.org/abs/2005.05951)
- Threshold ablation, Hopper (Table 7; penalty 0 plus termination; values are P-MDP value / true value): 0.1×disc_D 1315/2082; 0.2× 2480/3244; 0.5× 3075/3360; 1.0× 3543/3596; 5.0× 3246/3028; naive MBRL 3656/2810. Best real performance comes at intermediate pessimism. High pessimism makes P-MDP optimization "slow or highly noisy" because of "non-smoothness … due to abrupt changes involving early episode terminations" — [MOReL](https://arxiv.org/abs/2005.05951)

**COMBO (conservatism without explicit uncertainty) and its critique of uncertainty**
- Prior methods "rely on some sort of strong assumption about uncertainty estimation, typically assuming access to a model error oracle … such methods use more heuristic uncertainty estimation methods, which can be difficult or unreliable for complex datasets or deep network models"; COMBO eschews "direct uncertainty estimation, which we argue is not necessary" — [Yu et al. 2021, COMBO](https://arxiv.org/abs/2102.08363)
- Mechanism (structure paraphrased from their Eq. 2): push Q down on `ρ(s,a) = d^π_M̂(s)π(a|s)` (model-rollout state-actions under the current policy), push Q up on dataset (s,a), plus a Bellman error under `d_f = f·d + (1−f)·d^μ_M̂`. The learned Q lower-bounds the true Q "without the need for explicit uncertainty estimation" — [COMBO](https://arxiv.org/abs/2102.08363)
- Evidence (§5.1.1, Fig. 2): on halfcheetah-jump and ant-angle, both model error and Max Var (`max_i ‖Σ^i‖_F`) were normalized to [0, 1] and fitted with a linear regression. "Max Var is unable to accurately predict the true model error". No correlation coefficient is reported in the text — [COMBO](https://arxiv.org/abs/2102.08363)
- Results:
  - halfcheetah-jump: COMBO 5308.7, MOPO 4016.6, MOReL 3228.7, CQL 741.1.
  - ant-angle: 2776.9 / 2530.9 / 2660.3 / 2473.4.
  - sawyer-door-close: 98.3% / 65.8% / 42.9% / 36.7%.
  - Best on 9/12 D4RL settings.
  - Source: [COMBO](https://arxiv.org/abs/2102.08363)

**RAMBO (adversarial model, no explicit uncertainty)**
- Model loss `L_φ = λ V^π_φ − E_D[log T̂_φ(s', r | s, a)]`: the MLE term dominates on the dataset, and "the adversarial term dominates outside of the dataset". Unlike MOPO/MOReL, which "only modify the reward function by applying reward penalties", RAMBO targets "worst-case transition dynamics" — [Rigter et al. 2022, RAMBO](https://arxiv.org/abs/2204.12581)

**1R2R (risk measure over the ensemble posterior)**
- Rollout transitions are reweighted adversarially within a coherent-risk envelope (CVaR or Wang): `V(s) = R + γ min_{ξ∈B_ρ} Σ_s' T ξ V`. Ablations: "Ablate risk" (= standard MBRL) leads to training that "diverges … or performance degrades" on many D4RL datasets. "Ablate ensemble" (aleatoric-only risk) "also leads to unstable training for some datasets" — [Rigter et al. 2023, 1R2R](https://arxiv.org/abs/2212.00124)

### Inferences
- The three injection styles need different properties of the uncertainty signal:
  - A soft penalty (MOPO) needs a correctly scaled estimate and a tuned λ.
  - A hard HALT or truncation gate (MOReL) needs a correctly ordered estimate plus a calibrated threshold.
  - COMBO/RAMBO need a learned critic or an adversarially trainable model, which a frozen world model with a pure planner does not have.
- The closest planner analog of COMBO is a learned "trust" or terminal critic trained to score dataset latent states above model-imagined, planner-selected latent states. This sidesteps σ² but needs extra training.
- MOPO's own table is the strongest evidence on when penalties matter. On in-distribution D4RL "mixed" data, no penalty ≈ MOPO (6409 vs 6406). On OOD-demanding tasks, removing the penalty is catastrophic (−981 vs 4017). The oracle penalty is only marginally better, and on halfcheetah-jump it is worse (3819 vs 4017). Once a penalty exists, its fidelity to the true error is not the main bottleneck.

### Gaps
- I did not extract RAMBO's numeric results or its λ-sensitivity analysis.
- COMBO's Fig. 2 regression slope or R² are not in the text.

## 4. Do uncertainty heuristics track true model error or OOD-ness? Lu, Ball et al. (ICLR 2022) and other measurement studies

### Takeaway
On D4RL, ensemble-mean-based heuristics (mixture std/var, max pairwise difference, max aleatoric) reach Spearman ρ ≈ 0.6–0.8 with true next-state MSE. Distribution-based heuristics (LOO-KL as in M2AC, log-likelihood variance as in LOMPO) are near zero (ρ ≈ −0.02–0.5). Correlations drop on trajectories from exploitative policies (the setting that matters). Detecting distributional OOD-ness is clearly weaker than detecting dynamics error, and precision on rare large errors is low. MC dropout is specifically documented to produce data-independent variance.

### Cited Findings
**Lu, Ball et al. 2022: penalty definitions**
- Max Aleatoric `max_i ‖Σ^i‖_F` (MOPO); Max Pairwise Diff `max_{i,j} ‖μ^i − μ^j‖₂` (MOReL); LL Var `Var_i(log P̂^i(s'|s,a))`, with s' drawn from one member (LOMPO); LOO KL `KL(P̂_i ‖ P̂_−i)` (M2AC); Ensemble Std/Var from the mixture `Σ* = (1/N) Σ_i (Σ_i² + μ_i²) − μ*²` — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**Correlation with true MSE (Table 1; Spearman ρ / Pearson r; averaged over random→expert datasets, 12 seeds)**
- "Transfer" means models trained on one D4RL dataset and evaluated on others. "True Model-Based" means trajectories from the 5 most exploitative unpenalized policies, replayed in MuJoCo — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

  | Penalty | Transfer HC | Transfer Hopper | True-MB HC | True-MB Hopper |
  |---|---|---|---|---|
  | Max Aleatoric | 0.78/0.55 | 0.71/0.41 | 0.58/0.42 | 0.73/0.48 |
  | Max Pairwise Diff | 0.79/0.62 | 0.77/0.57 | 0.58/0.52 | 0.75/0.55 |
  | Ens. Std | 0.82/0.64 | 0.79/0.56 | 0.61/0.52 | 0.79/0.55 |
  | Ens. Var | 0.82/0.67 | 0.79/0.59 | 0.60/0.49 | 0.77/0.55 |
  | LL Var | 0.13/0.14 | 0.36/0.12 | 0.04/0.07 | 0.50/0.16 |
  | LOO KL | 0.03/0.11 | 0.11/0.08 | −0.02/0.06 | 0.22/0.10 |

- Authors' reading: there is "a significant performance loss between the Transfer and True Model-Based HalfCheetah settings, with the latter being relatively poor", and "further work is needed to develop penalties that can successfully detect the type of dynamics discrepancies that actually arise in offline MBRL". Max Aleatoric has values "bunched" at its extreme (low kurtosis) — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**OOD-event detection (Table 2; AUC / AP; positives = errors above the 90th/95th/99th percentile)**
- "Dynamics" = model-error spikes; "Distribution" = 2-norm distance to the nearest offline point — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135):
  - 90th percentile, ensemble-based penalties: dynamics AUC 0.89–0.90 / AP 0.50–0.56; distribution AUC 0.76–0.79 / AP 0.34–0.38.
  - 99th percentile: dynamics AUC 0.92–0.93 but AP only 0.20–0.27; distribution AUC 0.89–0.90 / AP 0.15–0.18.
  - LOO KL: AUC 0.59–0.72, AP 0.04–0.24. LL Var: AUC 0.66–0.81.
- "the penalties are powerful at identifying dynamics discrepancy, but not as accurate at identifying when the world-model data is out-of-distribution". The authors attribute this to feature collapse, "where latent representations of points far away in the input space get mapped close together" — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**Sensitivity and tuning**
- Max-based penalties change distribution shape with ensemble size; Ensemble Std/Var "change the least". The optimal number of models is "highly dependent on environment, the behavior data, and penalty type". Model error "does not necessarily increase with drift"; instead "we observe spikes", from which rollouts can recover — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)
- Fixed h=20 with Ensemble Std and automatic λ tuning (constraint Λ=1) averages 49.0 on D4RL locomotion, +43% over grid-searched MOPO — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)
- Per-environment Bayesian-optimized hyperparameters average 65.2 ± 5.4, against MOPO 34.2 ± 9.8 (their codebase), MOPO 36.7 (reported), MOReL 64.4 (D4RL v2 data) and COMBO 64.1. The discovered settings vary widely: N 6–13, λ 0.21–39.08, h 5–47, and the best penalty type varies across environments — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**SUMO (AAAI 2025): KNN distance vs ensembles**
- Estimator: `u = log(‖(ŝ⊕â⊕ŝ') − kNN_D(ŝ⊕â⊕ŝ')‖₂ + 1)`, computed with FAISS. It is derived from a particle-based cross-entropy between model and dataset dynamics — [Qiao et al., SUMO](https://arxiv.org/abs/2408.12970)
- Protocol: 100 starts × horizon-100 rollouts of an unpenalized MOPO policy, replayed in the true environment (10,000 transitions per dataset).
  - Ensemble heuristics: Max Aleatoric ρ 0.62–0.79, Max Pairwise ρ 0.58–0.85. LOO KL: ρ 0.08–0.22. SUMO: ρ 0.73–0.88.
  - Example, walker2d-medium-replay (ρ/r): Max Aleatoric 0.66/0.54, Max Pairwise 0.65/0.59, LOO KL 0.09/0.06, SUMO 0.88/0.86.
  - SUMO is not best everywhere, e.g. hopper-random: Max Pairwise 0.85 vs SUMO 0.82.
  - The comparison omits the Ensemble Std/Var penalties that Lu et al. found best, and the text says 9 datasets while the table lists 15.
  - Source: [SUMO](https://arxiv.org/abs/2408.12970)

**Generic and dropout-specific evidence**
- Uncertainty quality "consistently degrades with increasing dataset shift regardless of method". Better i.i.d. calibration "does not usually translate to better calibration under dataset shift". Deep ensembles "perform the best … and be more robust to dataset shift"; M=5 may suffice — [Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)
- Dropout: "the dropout rate does not depend on the data". Lemma 1: the dropout distribution is invariant to duplicating the dataset, so an agent "can[not] tell the difference between observing a set of data once and observing it N≫1 times". Dropout performed poorly in their deep-exploration chain tests — [Osband et al. 2018, Randomized Prior Functions](https://arxiv.org/abs/1806.03335)
- Osband's 2016 workshop paper warns of "the dangers of a naive applications dropout … as a proxy for uncertainty" (search snippet only; full text not read) — [Osband 2016](http://bayesiandeeplearning.org/2016/papers/BDL_4.pdf)
- MC dropout "exhibits a more constant variance across the entire input range without pronounced increase in interpolation or extrapolation intervals", unlike GP/BNN. Combining MC-dropout runs across random seeds (effectively an ensemble) recovers higher uncertainty in data-scarce regions — [arXiv 2512.14851 (2025)](https://arxiv.org/abs/2512.14851)
- Robotics counterpoint: "Although dropout has been seen to be overconfident on novel observations … the combination of bootstrapping and dropout reliably detects novel scenarios" — [Lütjens et al. 2019](https://arxiv.org/abs/1810.08700)
- Calibration matters for planning: "it is much better to be calibrated than sharp". In a discrete MDP, a calibrated model gives exact policy values (Theorem 1). Their quantile-recalibrated agent does "at least as good, and often much better" on MuJoCo control (Fig. 3; I did not confirm that the base planner is PETS) — [Malik et al. 2019](https://arxiv.org/abs/1906.08312)

### Inferences
- Benchmarks for a σ² failure-discrimination test: in state-space MuJoCo, ensemble-based penalties reach AUC ≈ 0.90 for flagging top-decile per-transition error and ≈ 0.76–0.79 for distributional OOD. Distribution-shaped heuristics sit at AUC ≈ 0.59–0.74. An AUROC of ~0.66 is therefore in the "weak heuristic" band. Episode-level failure discrimination is a harder target than per-transition error, so this is an orientation point, not a like-for-like comparison.
- High AUC can coexist with low precision: AP 0.20–0.27 at 1% prevalence. Any hard gate or truncation built on σ² will mostly fire on non-failures unless calibrated to a target false-positive rate (see UNISafe's conformal approach in §5).
- The literature consistently favours mixture-variance or mean-disagreement from independently trained members over likelihood- or KL-shaped scores, and over MC dropout, whose variance is not data-dependent by construction. KNN distance to dataset embeddings (SUMO) is a cheap, training-free competitor that can be computed on frozen latents.
- The right validation protocol is Lu's "True Model-Based" or SUMO's replay protocol: score σ² on the planner-selected, exploitative candidates, not on held-out dataset transitions, because correlations degrade exactly there.

### Gaps
- I found no study that measures the σ²–error correlation for MC dropout versus ensembles inside a frozen JEPA-style latent world model, or for CEM-selected (rather than policy-generated) action sequences.
- Lu et al.'s per-dataset correlation scatter plots (their App. A.3) were not extracted.

## 5. LOMPO and other latent/pixel-based uncertainty-aware model-based methods

### Takeaway
LOMPO applies a MOPO-style penalty in a learned latent space, using log-likelihood variance across latent forward models. That penalty is decisive against an unpenalized latent MBPO on diverse data, yet it fails on narrow expert data, where BC wins. Offline DV2 shows the same pattern with ensemble mean-disagreement. Later work (Plan2Explore, UNISafe) uses post-hoc ensembles of one-step latent predictors, and UNISafe adds a warning: OOD detection that looks only at predicted latents is overconfident.

### Cited Findings
**LOMPO**
- Uncertainty-penalized latent MDP `r̃ = r − λu`. K latent forward models are trained inside one variational model (a random model per step in the ELBO, with a shared inference network q). Rollouts pick a random model each step. Reward: `r̃_t = (1/K) Σ_i r_θ(s^(i)_t, a_t) − λu(s_t,a_t)`, with `u = Var_i(log T̂_θi(s_t | s_{t−1}, a_{t−1}))`, chosen because it "estimates disagreement across the means of the forward models, as well as the variances" — [Rafailov et al. 2021, LOMPO](https://arxiv.org/abs/2012.11547)
- Results (Table 1, LOMPO vs unpenalized latent MBPO "LMBRL" vs BC) — [LOMPO](https://arxiv.org/abs/2012.11547):
  - Walker Walk medium-replay 74.9 vs 44.7.
  - Adroit Pen medium-replay 82.8 vs 5.2; expert 96.1 vs 0.2.
  - Door Open medium-expert 95.7 vs 0.0.
  - D'Claw Screw medium-replay 71.8 vs 72.4 (no gain).
  - Door Open expert: LOMPO 0.0, LMBRL 0.0, BC 97.4 — the authors call this "a well-known phenomenon when learning dynamics models from narrow expert data".
- In state-space MuJoCo, the LL Var penalty correlates weakly with MSE (ρ 0.04–0.50); "it was designed for use with a KL-regularized latent state space model which has well-behaved dynamics" — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**COMBO in LOMPO's latent space**
- walker_walk medium-replay: COMBO 69.2, LOMPO 66.9, LMBPO 59.8. sawyer-door expert: COMBO 96.7%, LOMPO 0.0%, others 0% — [COMBO](https://arxiv.org/abs/2102.08363)

**Offline DV2 (V-D4RL)**
- DreamerV2 with a bootstrap latent-dynamics ensemble and penalty `r̃ = r − λ Σ_k (μ^k(s,a) − μ*(s,a))²` — [Lu, Ball, Rudner et al. 2022, V-D4RL](https://arxiv.org/abs/2206.04779)
- walker-walk: random 28.7 (DrQ+BC 5.5); mixed 56.5 (28.7); medium 34.1 (DrQ+BC 46.8); medexp 43.9 (DrQ+BC 86.4); expert 4.8 (BC 91.5).
- cheetah-run: medium 17.2 vs BC 51.6; expert 10.9 vs BC 67.4. Humanoid ≈ 0.
- Authors: "Our model-based method does best on the diverse low-reward datasets, model-free on the diverse high-reward datasets, and behavioral cloning on the narrow expert data."
- Source: [V-D4RL](https://arxiv.org/abs/2206.04779)

**Plan2Explore (latent disagreement estimator)**
- K one-step predictors map model state and action to the next image embedding h_{t+1}, trained with MSE (unit-variance Gaussian). Disagreement `D(s,a) = (1/(K−1)) Σ_k (μ_k − μ̄)²`, which approximates expected information gain under fixed variance. Implementation: K=5 two-hidden-layer MLPs — [Sekar et al. 2020, Plan2Explore](https://arxiv.org/abs/2005.05960)

**MPUR (MC-dropout penalty through a latent stochastic model)**
- The policy is trained by backpropagating `C + λU` through an unrolled model, where U is the trace of the output covariance across K dropout masks (differentiable). Covariance splits into aleatoric (latent z) and epistemic (dropout) parts — [Henaff et al. 2019, MPUR](https://arxiv.org/abs/1901.02705)
- Results (success rate / mean distance): MPUR stochastic + z-dropout 74.8% / 171.2; MPUR deterministic 69.1%; VG deterministic (= MPUR without the penalty) 0.0% / 11.9; SVG 0.0%; 1-step IL 7.3%. "Including this penalty is essential" — [MPUR](https://arxiv.org/abs/1901.02705)

**UNISafe (post-hoc ensemble on a pretrained latent world model)**
- K Gaussian next-latent predictors are trained on latent transitions encoded by a frozen, pretrained world model, "regressing the pretrained latent dynamics". Epistemic uncertainty `D(z,a)` is the Jensen–Rényi divergence of the members. Plain ensemble variance "conflates aleatoric … with epistemic" — [UNISafe, arXiv 2505.00779](https://arxiv.org/abs/2505.00779)
- "OOD detection methods that rely solely on the predicted latent state z are overconfident, as predicted latents from OOD scenarios are projected into in-distribution representations". The world model "hallucinates as soon as the scenario goes out-of-distribution" (e.g. teleporting out of a failure region) — [UNISafe](https://arxiv.org/abs/2505.00779)
- Threshold calibration uses class-conditioned, trajectory-level conformal prediction on held-out in-distribution trajectories:
  - Nonconformity score = the (1−α_trans)-quantile of u along a trajectory.
  - ε̂ = the (1−α_cal)-quantile over calibration trajectories, which guarantees `P(Q_τ ≤ ε̂) ≥ 1 − α_cal` for in-distribution trajectories (bounded false-positive rate).
  - The augmented state (z, u) with OOD failure set {u > ε̂} is fed to reachability analysis.
  - Validated on vision-based Franka tasks.
  - Source: [UNISafe](https://arxiv.org/abs/2505.00779)

**2026 diagnoses of latent world-model planning (recent preprints, not peer reviewed)**
- Across 4 model families × 4 tasks (256 starts, 300 candidates per start), "center error dominates MSE in 14/16 model-task cells", yet in 6 cells correcting only action-relative responses gives better physical rank correlation and top-30 elite quality "while leaving more latent MSE". For LeWorldModel (LeWM), encoding realized endpoints raises physical Spearman from 0.464 to 0.975 (Reacher) and from 0.193 to 0.631 (PushT), "while Cube's encoded-goal cost remains uninformative" — [Min et al., arXiv 2610.05550 (Oct 2026)](https://arxiv.org/abs/2610.05550)
- ATM: "true-transition action-identifiability tracks downstream planning substantially better than standard prediction loss" across TwoRoom, PushT and OGBench-Cube. Predicted transitions can be "highly self-decodable yet cross-domain-inconsistent". The diagnostic gives 98.81% pairwise accuracy at screening models — [ATM, arXiv 2606.09028](https://arxiv.org/abs/2606.09028)

### Inferences
- The recipe best matched to a frozen JEPA world model is the UNISafe/Plan2Explore pattern: a separate small ensemble of next-latent predictors on frozen latents, an epistemic-only divergence over (z, a) transitions rather than over z alone, and a conformal threshold. This replaces MC dropout on the frozen predictor, which by §4 has no data-dependent variance.
- Narrow (expert-like) data is where latent-space penalties fail and BC wins (LOMPO Door-expert, V-D4RL expert). This matches the independent finding that a BC prior is the main lever on Cube.
- If the LeWM latent goal cost on Cube is itself uninformative about physical outcome (2610.05550), no dynamics-uncertainty term can repair candidate ranking. The failure is in the cost or representation, not in dynamics variance. This would also explain a σ² that cannot discriminate failures. The preprint is unreviewed and its protocol may differ from the thesis setup.

### Gaps
- LOMPO's own penalty-vs-no-penalty ablation inside LOMPO (beyond the LMBRL baseline), and its λ, were not extracted.
- UNISafe's quantitative results (Table 1) were not captured.
- I found no paper that evaluates uncertainty-penalized CEM on a JEPA latent world model on OGBench Cube.

## 6. Risk-sensitive and safe planning with learned models (Robust CEM, CVaR-style objectives, mean+variance costs)

### Takeaway
Risk-aware planners inject ensemble or dropout uncertainty in three ways: (a) a worst case over members for constraints, with feasibility-first elite selection (RCE); (b) a coherent risk measure (CVaR, Wang) over the epistemic posterior (1R2R); (c) mean + λ·variance costs (Kahn; Lütjens). Reported gains are mostly fewer violations or collisions at slightly lower reward, plus training stability, rather than higher task success. All of them depend on the ensemble's spread actually covering the bad outcome.

### Cited Findings
**Robust CEM (RCE)**
- Ensemble of B MSE-trained networks with epistemic variance `Σ̃ = Σ_b (f_b − μ̃)²/B`. Candidate scores: reward = `Σ_t γ^t (1/B) Σ_b r(s^b_{t+1})` (mean over members); cost = `Σ_t β^t max_b c(s^b_{t+1})` (worst case over members). RCE first keeps the feasible set, sorts it by reward and takes the top-k elites; if no candidate is feasible, it takes the k lowest-cost ones — [Liu et al. 2020, RCE](https://arxiv.org/abs/2010.07968)
- Safety Gym (Point/Car Goal; 3 seeds): MPC-RCE "achieves the lowest constraint violation rate, though its reward is slightly lower" than MPC-CEM or MPC-random. Model-based methods converge after 5e3 steps versus 4e6 for TRPO-Lagrangian ("40 times more sample efficient") — [RCE](https://arxiv.org/abs/2010.07968)

**1R2R (CVaR/Wang over the ensemble posterior)**
- Dynamic Markov coherent risk is applied to successor states sampled from the ensemble, so a single risk measure covers both epistemic and aleatoric uncertainty. Stochastic domains are evaluated with static CVaR₀.₁. Ablations: without risk reweighting, training diverges or degrades on D4RL; a single model (aleatoric risk only) is unstable on some datasets. "risk-aversion to epistemic uncertainty (represented by the ensemble) is crucial" — [1R2R](https://arxiv.org/abs/2212.00124)

**Mean + variance risk costs**
- Kahn et al.: a collision predictor with bootstrapping + dropout uncertainty and "a speed-dependent collision cost" that "naturally chooses to move cautiously when uncertainty is high"; the robot experiences "substantially fewer dangerous collisions during training" — [Kahn et al. 2017](https://arxiv.org/abs/1702.01182)
- Lütjens et al.: MPC over motion primitives `u* = argmin λ_v Var_N(P_coll) + λ_c E_N(P_coll) + λ_g t_goal`, using N = n_d·n_b forward passes (MC dropout, keep probability 0.7, × bootstrapped LSTMs). The policy is "more robust to novel observations and take[s] safer actions than an uncertainty-unaware baseline" — [Lütjens et al. 2019](https://arxiv.org/abs/1810.08700)

**Trajectory pruning and behavior priors as offline risk control**
- MOPP:
  - `U_{n,t} = max_{i,j} ‖f^i(s_t^n, a_t^n) − f^j(s_t^n, a_t^n)‖²`; keep `T_p = {T_n | U_{n,t} < L ∀t}`. If `|T_p| < N_m = 0.2N`, add the remaining trajectories with the lowest cumulative uncertainty.
  - Actions are sampled from a behavior-cloned prior with std boosted by σ_M, then max-Q filtered.
  - Ablations: L = 0.1 (too strict) and L = 10 (too loose) both underperform L = 2.0. Pruning helps only when σ_M is moderate; at σ_M = 1.0 too few trajectories pass.
  - Results: walker2d-medium 80.7 vs MBOP 41.0; hopper-medium-expert 95.4 vs 55.1.
  - Source: [Zhan et al., MOPP](https://arxiv.org/abs/2105.07351)
- MBOP: BC prior plus a truncated value function plus an ensemble. Each trajectory keeps a consistent ensemble member; returns are averaged over the ensemble, with no explicit uncertainty penalty. halfcheetah-medium-expert: 105.9 vs BC prior 90.8 vs MOPO 57.9 (as reported by MBOP) — [Argenson & Dulac-Arnold 2021, MBOP](https://arxiv.org/abs/2008.05556)

**CVaR in MPPI**
- RA-MPPI uses CVaR inside MPPI for "arbitrary forms of uncertainty". Search snippet only; I did not verify whether it uses learned dynamics — [Yin et al. 2022, RA-MPPI](https://arxiv.org/abs/2209.12842)

### Inferences
- A Goal-CVaR soft penalty is the planning analog of 1R2R/RCE. It only changes elite selection if the per-candidate distribution (over members or masks) has a lower tail that reaches failure outcomes. With MC-dropout masks on one frozen network, the spread is narrow and roughly uniform (§4), so CVaR over 5 samples is close to the mean minus a near-constant margin, which barely reorders candidates.
- The evidence shows risk terms trading reward for fewer violations (RCE), not raising success. For a success-rate metric on Cube, risk penalties are unlikely to beat a BC-prior plus pruning design (MOPP > MBOP), in which risk is controlled by sampling near the data and then removing the high-disagreement trajectories.

### Gaps
- I did not extract numeric violation counts for RCE or numeric collision rates for Kahn 2017.
- I found no paper reporting CVaR-over-ensemble CEM on a learned latent visual world model with success-rate gains.

## 7. Plan2Explore and related: uncertainty for exploration and data collection rather than scoring

### Takeaway
The most robust use of epistemic uncertainty in this literature is choosing which data to collect next (Plan2Explore, CEE-US with iCEM), combined with retraining on the data the controller itself induces. Ross & Bagnell give guarantees for that aggregation that come from on-distribution data, not from the precision of the uncertainty estimate.

### Cited Findings
- Plan2Explore: the exploration policy is trained in imagination on intrinsic reward = latent ensemble disagreement (Eq. 5; information-gain interpretation). Zero-shot performance "is competitive to Dreamer, even outperforming it in the hopper hop task", and beats Curiosity, MAX and random exploration. After 1M exploration steps it adapts with few episodes — [Plan2Explore](https://arxiv.org/abs/2005.05960)
- CEE-US:
  - Ensemble of GNN world models; intrinsic reward `R_I = tr(Cov_m(f_m(s_t, a_t)))`.
  - iCEM MPC plans to maximize cumulative epistemic uncertainty during "free play", re-planning every step, with 20 rollouts × 100 steps per iteration. "After only 30 iterations of free play, the robot starts lifting objects."
  - The same model solves downstream tasks zero-shot.
  - Source: [Sancaktar et al. 2022, CEE-US](https://arxiv.org/abs/2206.11403)
- Agnostic system identification: batch model learning suffers from train/test distribution mismatch (uniform exploration is often the only safe option). The iterative alternative alternates (1) "collecting new data … by executing a good policy under the current model, as well as by sampling from a given exploration distribution" and (2) refitting. Guarantees reduce to no-regret online learning and "do not scale with the size of the MDP when given a good exploration distribution" — [Ross & Bagnell 2012](https://arxiv.org/abs/1203.1007)
- More data from the data-collecting policy lowers model error and its growth with policy shift (MBPO Fig. 1) — [MBPO](https://arxiv.org/abs/1906.08253). USAD effectiveness depends on "having sufficient data points 'close' to the query" — [MOReL](https://arxiv.org/abs/2005.05951)
- Kahn et al. aggregate samples gathered by the uncertainty-aware controller "to further improve the collision prediction model" — [Kahn et al. 2017](https://arxiv.org/abs/1702.01182)

### Inferences
- Using σ² to select planner-induced (exploited) candidates for real execution, labeling and model fine-tuning is the DAgger-for-models pattern plus active exploration. It is more tolerant of a mediocre σ² (AUROC ~0.66) than scoring is. Even a weak ranking concentrates labeling budget on the region the planner is exploiting, and the correction itself comes from the planner-distribution data (Ross & Bagnell).
- For a frozen-encoder setting, fine-tuning the predictor (or a residual head) on planner-induced latent transitions while keeping the encoder fixed preserves the latent geometry. This is an engineering inference, not something these papers test.

### Gaps
- I found no study that measures how much planner-induced fine-tuning reduces CEM exploitation in a JEPA-style latent world model.
- Plan2Explore's and CEE-US's numeric per-task results were not extracted.

## 8. Across these methods: when uncertainty-aware scoring helps planning and when it does not

### Takeaway
Uncertainty terms pay off when the optimizer is pushed off-distribution (OOD-demanding tasks, long horizons, low or noisy data) and the estimator is ensemble-based, correctly scaled or rank-calibrated, and matched to a tuned pessimism level. They add little when candidates stay in-distribution (short rollouts, BC-prior sampling). They fail when the data is narrow (BC wins), when the estimator is distribution-shaped or dropout-based, or when the error that matters for ranking is not the error the estimator measures (feature collapse; uninformative latent cost).

### Cited Findings
**Mechanism catalog**

| Mechanism | Estimator | Injection rule | Key evidence |
|---|---|---|---|
| Expectation over particles | PE + TS1/TS∞ | `argmax Σ E[r]` | Model class > propagation; PE > P > DE > D — [PETS](https://arxiv.org/abs/1805.12114) |
| Reward penalty | max_i ‖Σ_i‖_F | `r − λu` | Essential on OOD tasks, ≈0 effect in-distribution; oracle only marginally better — [MOPO](https://arxiv.org/abs/2005.13239) |
| Absorbing HALT / truncation | max pairwise ‖Δμ‖ | to HALT with −κ if disc > μ_d + βσ_d | Intermediate threshold best (Hopper true 3596 vs naive 2810) — [MOReL](https://arxiv.org/abs/2005.05951) |
| Horizon weighting | ensemble variance of h-step targets | inverse-variance weights | Online gains — [STEVE](https://arxiv.org/abs/1807.01675); offline MAP fails 11/18, LCB fixes it — [CBOP](https://arxiv.org/abs/2210.03802) |
| Rollout masking | OvR KL | keep lowest-u fraction w + `r − αu` | Robust to long H and noise — [M2AC](https://arxiv.org/abs/2010.04893) |
| Adaptive termination | GJS pairwise; info-loss entropy | quantile-calibrated thresholds | Longer consistent rollouts — [MACURA](https://arxiv.org/abs/2405.19014), [Infoprop](https://arxiv.org/abs/2501.16918) |
| Trajectory pruning | max pairwise ‖Δf‖² | drop trajectories with any U > L (keep ≥ 0.2N) | L = 2 beats 0.1 and 10 — [MOPP](https://arxiv.org/abs/2105.07351) |
| Risk measure | ensemble posterior | CVaR/Wang reweighting; max over members | Stabilizes offline training — [1R2R](https://arxiv.org/abs/2212.00124); fewer violations — [RCE](https://arxiv.org/abs/2010.07968) |
| Exploration bonus | latent disagreement; tr Cov | intrinsic reward | Zero-shot competitive with Dreamer — [Plan2Explore](https://arxiv.org/abs/2005.05960); free-play manipulation — [CEE-US](https://arxiv.org/abs/2206.11403) |
| Implicit conservatism | none | Q down on model rollouts / adversarial model | Better on OOD-generalization tasks — [COMBO](https://arxiv.org/abs/2102.08363), [RAMBO](https://arxiv.org/abs/2204.12581) |

**Conditions where uncertainty helps**
- OOD-demanding objectives: no penalty gives −980.8 (halfcheetah-jump) and 18.6 (ant-angle), against 4016.6 / 2530.9 with the penalty — [MOPO](https://arxiv.org/abs/2005.13239)
- Long model horizons and noisy dynamics (M2AC vs MBPO); complex dynamics with horizon weighting (STEVE on Humanoid) — [M2AC](https://arxiv.org/abs/2010.04893), [STEVE](https://arxiv.org/abs/1807.01675)
- Diverse offline image data: LOMPO vs unpenalized latent MBPO (Pen medium-expert 94.6 vs 0.0) — [LOMPO](https://arxiv.org/abs/2012.11547)
- Planning through a learned model in traffic: removing the dropout penalty takes success from 69–75% to 0% — [MPUR](https://arxiv.org/abs/1901.02705)

**Conditions where it adds little or hurts**
- In-distribution data: halfcheetah-mixed with no penalty 6409 ≈ MOPO 6406 — [MOPO](https://arxiv.org/abs/2005.13239)
- Short rollouts already prevent exploitation; k=1 is "difficult to beat" — [MBPO](https://arxiv.org/abs/1906.08253)
- Narrow or expert data: V-D4RL walker expert 4.8 vs BC 91.5; LOMPO Door expert 0.0 vs BC 97.4; MOPO hopper-medium-expert 23.7 vs BC 111.9 — [V-D4RL](https://arxiv.org/abs/2206.04779), [LOMPO](https://arxiv.org/abs/2012.11547), [MOPO](https://arxiv.org/abs/2005.13239)
- Too much pessimism: MOReL 0.1×disc_D gives 2082 true return vs 3596 at 1.0×; MOPP L = 0.1 is worse than L = 2; M2AC with large α discourages model use — [MOReL](https://arxiv.org/abs/2005.05951), [MOPP](https://arxiv.org/abs/2105.07351), [M2AC](https://arxiv.org/abs/2010.04893)
- Hyperparameters dominate: Bayesian-optimized settings average 65.2 vs 34.2, with λ ranging 0.21–39 and h 5–47 — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)

**Miscalibration far from data**
- Arbitrary variance off-distribution (collapse or explosion) — [PETS](https://arxiv.org/abs/1805.12114)
- Max Var fails to predict error on OOD tasks — [COMBO](https://arxiv.org/abs/2102.08363)
- Weaker detection of distributional OOD than of dynamics error (feature collapse), and lower correlation on exploitative trajectories — [Lu, Ball et al. 2022](https://arxiv.org/abs/2110.04135)
- Latent-only OOD detection is overconfident, because OOD inputs are projected to in-distribution latents — [UNISafe](https://arxiv.org/abs/2505.00779)
- Uncertainty quality degrades with shift for all methods — [Ovadia et al. 2019](https://arxiv.org/abs/1906.02530)
- Dropout variance does not depend on the data — [Osband et al. 2018](https://arxiv.org/abs/1806.03335), [arXiv 2512.14851](https://arxiv.org/abs/2512.14851)

### Inferences
These are design implications for a genuinely uncertainty-aware CEM/MPPI planner on a frozen latent world model, in priority order:
1. **Keep candidates in-distribution first**: a BC/behavior prior for sampling, as in MBOP/MOPP, with uncertainty pruning layered on top. This is the regime where the evidence shows real gains, and where a weak σ² still helps because it only has to reject the tail.
2. **Replace MC dropout with an independently initialized post-hoc latent ensemble** (5–10 next-latent predictors trained on frozen encoder latents; UNISafe/Plan2Explore pattern). Use mixture-variance or a JRD/GJS epistemic divergence over (z, a), not likelihood- or KL-shaped scores. Optionally add a KNN-in-latent distance (SUMO-style) as a training-free second opinion. The project context says an ensemble σ² also missed the failure test. The literature points to the details that separate working from failing ensembles: independent members (not dropout masks), conditioning on (z, a) rather than z alone, an epistemic-only divergence rather than total variance, and evaluation on planner-selected candidates. Item 7 is the alternative explanation: the cost itself may be the bottleneck.
3. **Calibrate gates by rank or quantile**, not by raw scale: M2AC's rank masking, MACURA/Infoprop quantile thresholds, MOReL's μ+βσ, or UNISafe's trajectory-level conformal threshold with a target false-positive rate. Validate on planner-selected candidates replayed in the simulator (Lu's True-Model-Based / SUMO protocol).
4. **If truncating horizons, add a terminal charge**: a HALT-style −κ or a pessimistic terminal value (LCB, CBOP). Otherwise the truncated cost sum can reward candidates that become uncertain early.
5. **Treat risk measures (CVaR) as useful only once the ensemble has real epistemic spread.** 1R2R's "ablate ensemble" result shows aleatoric-only risk is insufficient.
6. **Use σ² for data collection and fine-tuning on planner-induced data** (Plan2Explore, CEE-US, Ross & Bagnell). This is the use most tolerant of a mediocre estimator.
7. **Before investing more in dynamics uncertainty, check whether the latent goal cost ranks candidates by physical outcome at all.** A 2026 preprint reports that LeWM's encoded-goal cost on Cube "remains uninformative", and that action-identifiability tracks planning better than prediction loss on OGBench-Cube. If so, the bottleneck is the cost or representation rather than model variance (speculative; unreviewed preprints).

### Gaps
- I found no controlled study comparing uncertainty-penalized and BC-prior CEM on the same frozen visual world model with success-rate endpoints. The closest evidence is MBOP/MOPP (state-based) and LOMPO/V-D4RL (policy learning, not MPC).
- I found no primary evidence on how many planner-induced fine-tuning rounds are needed to remove CEM exploitation in latent world models.
- Several numeric results exist only in figures: PETS Fig. 4, M2AC/STEVE/MACURA/Infoprop curves, RCE violation counts, UNISafe tables.
