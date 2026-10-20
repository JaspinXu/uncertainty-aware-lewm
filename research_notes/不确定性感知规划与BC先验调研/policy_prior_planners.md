# Policy priors, value functions and model uncertainty in model-based planners: MBOP, MOPP, TD-MPC / TD-MPC2, POPLIN

Reading notes. Full texts were read from ar5iv / arXiv HTML, method and experiment sections plus appendices. TD-MPC and TD-MPC2 planner details were also checked against the official code. Numbers are copied from the papers' tables. Where a value appears only in a figure, I say so and list it under Gaps.

Source key (used inline below):
- MBOP = Argenson & Dulac-Arnold, "Model-Based Offline Planning", arXiv 2008.05556 (v1 Aug 2020, v2 Jan 2021, v3 Mar 2021; ICLR 2021). Full text: https://ar5iv.labs.arxiv.org/html/2008.05556
- MOPP = Zhan, Zhu, Xu, "Model-Based Offline Planning with Trajectory Pruning", IJCAI-ECAI 2022, arXiv 2105.07351. Full text: https://ar5iv.labs.arxiv.org/html/2105.07351
- TD-MPC = Hansen, Wang, Su, "Temporal Difference Learning for Model Predictive Control", arXiv 2203.04955. Full text: https://ar5iv.labs.arxiv.org/html/2203.04955. Code: https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py
- TD-MPC2 = Hansen, Su, Wang, "TD-MPC2: Scalable, Robust World Models for Continuous Control", arXiv 2310.16828. Full text: https://arxiv.org/html/2310.16828. Code: https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py and https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py
- POPLIN = Wang & Ba, "Exploring Model-based Planning with Policy Networks", ICLR 2020, arXiv 1906.08649 (v1 only). Full text: https://ar5iv.labs.arxiv.org/html/1906.08649. ICLR page: https://iclr.cc/virtual_2020/poster_H1exf64KwH.html. Code: https://github.com/WilsonWangTHU/POPLIN
- Follow-ups that analyze these planners: FOWM = Feng, Hansen, Xiong, Rajagopalan, Wang, "Finetuning Offline World Models in the Real World" (arXiv v1 Oct 2023; CoRL 2023 venue from memory, not re-verified), https://arxiv.org/html/2310.16029. TD-M(PC)² = Lin, Wang, Schneider, Shi, "TD-M(PC)²: Improving Temporal Difference MPC Through Policy Constraint" (arXiv v1, Feb 2025), https://arxiv.org/html/2502.03550

---

## Q1. MBOP: what it learns, how MBOP-Trajopt samples and scores, whether ensemble disagreement is used, results by dataset quality, and zero-shot changes of objective

### Takeaway
MBOP is PDDM-style MPPI shooting with two additions: a behavior-cloning (BC) action prior that generates the candidate trajectories, and a truncated value function that scores what lies beyond a short horizon. Each control step does one guided pass of N=100–1000 rollouts with small iid Gaussian noise (no inner refinement iterations) and softmax-weights the trajectories by return. The ensembles are used only to give each trajectory its own fixed ensemble member and to average reward and value predictions. Ensemble disagreement is never used as a penalty or a filter. MBOP beats its own BC prior and MOPO on D4RL medium and medium-expert data. It is weaker than MOPO/MBPO on random and mixed (medium-replay) data. When the BC prior is removed, it collapses on D4RL. It adapts zero-shot to new goals and constraints by adding an extra objective term to the MPPI exponent.

### Cited Findings
**What is learned (three bootstrap ensembles, all trained by plain supervised L2 regression)**
- f_m: S×A→S×ℝ, a one-step model that predicts both the next state and the reward, (r̂_t, ŝ_{t+1}) = f_m(s_t, a_t). It is used to roll out candidate action sequences — [MBOP §3.4](https://ar5iv.labs.arxiv.org/html/2008.05556)
- f_b: a behavior-cloned policy a_t = f_b(s_t, a_{t−1}). It conditions on the previous action and acts as the prior that guides trajectory sampling — [MBOP §3.4](https://ar5iv.labs.arxiv.org/html/2008.05556)
- f_R: a truncated value function R̂_H = f_R(s_t, a_{t−1}). It predicts the expected return over a fixed horizon under the data-generating behavior. It is regressed on dataset returns, not trained by Bellman backups — [MBOP §3.4](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Each function is a bootstrap ensemble of K feed-forward networks that differ only in weight initialization and share the same dataset. Settings: K=3 (15 for D4RL Walker), 2 FC layers of 500 units, lr 1e-3, batch 512, 40 epochs. The discount is fixed at γ=1, so only finite-horizon returns are considered — [MBOP §3.1, §3.4, App. 5.5](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Training is on 90% of the data, with 10% held out for validation. On RLU Quadruped and Walker, low-return episodes are filtered out before training f_b and f_R, but f_m is always trained on unfiltered data — [MBOP §4.1](https://ar5iv.labs.arxiv.org/html/2008.05556)

**How MBOP-Trajopt samples and scores (Alg. 2)**
- For each trajectory n=1..N, the method picks ensemble member l = n mod K. That same member is used for both the BC policy and the dynamics for the whole rollout (a design borrowed from PDDM). At each step, a_t = f_b^l(s_t, a_{t−1}) + ε with ε ~ N(0, σ²) iid. The authors found no benefit from PDDM's time-correlated noise — [MBOP §3.6, Alg. 2](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Beta-mixing with the previous plan: A_{n,t} = (1−β)·a_t + β·T_{min(t,H−1)}, where T is the trajectory buffer carried across MPC steps. The paper describes β as a learning rate for how fast the plan changes. This amortizes optimization over time instead of iterating within one step — [MBOP §3.5–3.6](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Scoring: the next state comes from member l. The reward at each step is the average over all K members of f_m. The terminal term is the average over all K members of f_R at (s_{H+1}, A_{n,H}). So R_n = Σ_t mean_i r̂^i + mean_i f_R^i — [MBOP Alg. 2, lines 12–15](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Update: T'_t = Σ_n e^{κR_n} A_{n,t+1} / Σ_n e^{κR_n}, which is MPPI-style exponentiated-return weighting. The executed action is T_0, and the buffer is kept for the next step — [MBOP Alg. 1–2](https://ar5iv.labs.arxiv.org/html/2008.05556)

**Where model uncertainty enters**
- It does not enter as a penalty or threshold. The ensemble only (a) diversifies rollouts through the consistent per-trajectory member and (b) averages reward and value. The authors' own appendix lists "leveraging uncertainty" as an open research direction — [MBOP §3.6, App. 5.1](https://ar5iv.labs.arxiv.org/html/2008.05556)

**Hyperparameters actually used (grid-searched per dataset; App. 5.2 tables)**
- D4RL HalfCheetah: H=2–4, N=100, κ=1–3, σ=0.2 (0.8 on random), β=0. Hopper: H=4–10, N=100, κ=0.3–10, σ=0.01–0.4, β=0. Walker2d: H=2–8, N=1000, κ=0.1–3, σ=0.02–0.4, β=0 — [MBOP Tables 6–8](https://ar5iv.labs.arxiv.org/html/2008.05556)
- RLU Quadruped: H=8, N=1000, κ=3.8, σ=0.8, β=0.2. RLU Cartpole: H=64, N=100, β=0.2. RLU Walker: H=4–16, N=100, σ=0.1–0.4, β=0.2. Adroit door: H=4–32, N=100–1000, σ=0.05–0.2, β=0 — [MBOP Tables 2–5](https://ar5iv.labs.arxiv.org/html/2008.05556)
- The no-value-function ablation (MBOP-NOVF) needed much longer horizons to work at all (H=15–150 on D4RL), against H=2–10 for full MBOP — [MBOP Tables 6–8](https://ar5iv.labs.arxiv.org/html/2008.05556)
- A sweep over κ and H on the three RLU environments shows consistent scores across wide ranges, degrading only near extreme values — [MBOP §4.4](https://ar5iv.labs.arxiv.org/html/2008.05556)

**Results by dataset quality (D4RL, scores normalized to converged SAC, 5 seeds × 20 episodes; "BC" is MBOP's own f_b prior)**
| Dataset | Env | BC (f_b) | MBOP | MOPO | MBPO |
|---|---|---|---|---|---|
| random | halfcheetah / hopper / walker2d | 0.0 / 9.0 / 0.1 | 6.3 / 10.8 / 8.1 | 31.9 / 13.3 / 13.0 | 30.7 / 4.5 / 8.6 |
| medium | hc / hop / walk | 35.0 / 48.1 / 15.4 | 44.6 / 48.8 / 41.0 | 40.2 / 26.5 / 14.0 | 28.3 / 4.9 / 12.7 |
| mixed (medium-replay) | hc / hop / walk | 0.0 / 9.5 / 11.5 | 42.3 / 12.4 / 9.7 | 54.0 / 92.5 / 42.7 | 47.3 / 49.8 / 22.2 |
| medium-expert | hc / hop / walk | 90.8 / 15 / 65.5 | 105.9 / 55.1 / 70.2 | 57.9 / 51.7 / 55.0 | 9.7 / 56.0 / 7.6 |
- Source for the table above — [MBOP Table 1](https://ar5iv.labs.arxiv.org/html/2008.05556). MOPO and MBPO numbers are copied from the MOPO paper.
- The authors' reading: MBOP beats MOPO, sometimes by a large margin, on medium and medium-expert data. It is less performant on higher-variance datasets (random, mixed), which they attribute to relying on a policy-conditioned (unimodal) prior. They propose multi-modal stochastic priors as future work — [MBOP §4.2](https://ar5iv.labs.arxiv.org/html/2008.05556)
- D4RL MuJoCo has no pure "expert" split in MBOP's Table 1; the expert data point is Adroit door-expert. With expert data, MBOP only matches or slightly beats cloning at small and medium data sizes. Raw returns, MBOP vs cloning: 2000 points 2944.8 vs 2889.5; 50k points 3028.2 vs 2984.5. At larger sizes it falls slightly below cloning: 200k points 2967.0 vs 3028.0; 1M points 2910.2 vs 3004.3. At 400 points it is far worse (22.7 vs 352.2) — [MBOP Table 5](https://ar5iv.labs.arxiv.org/html/2008.05556)
- RL Unplugged (RWRL Cartpole, Quadruped, Walker; data from a 70%-performant MPO policy, with smaller subsets subsampled). MBOP is near-optimal on Cartpole and Quadruped with about 5000 steps (5 episodes). Walker needs about 23 episodes for a reasonable policy and converges to about 900 — [MBOP §4.1–4.2](https://ar5iv.labs.arxiv.org/html/2008.05556)
  - Examples, MBOP vs cloning: Quadruped 5k points 974.1 vs 796.0; Cartpole 5k points 803.1 vs 229.2; Walker 50k points 775.0 vs 615.7; Walker 5M points 908.8 vs 759.9 — [MBOP Tables 2–4](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Versus model-free offline RL: MBOP's own paper compares only to model-based offline RL (MOPO, MBPO) and BC. The only MBOP-vs-BCQ/CQL numbers I found are in MOPP's Adroit table, which reports MBOP numbers alongside BC, BCQ and CQL. MBOP vs (BC / BCQ / CQL):
  - pen-human 53.4 vs (34.4 / 68.9 / 37.5)
  - door-expert 101.2 vs (34.9 / 99.0 / 101.5)
  - hammer-expert 107.6 vs (125.6 / 107.2 / 86.7)
  - relocate-expert 41.7 vs (101.3 / 41.6 / 95.0)
  — [MOPP Table 2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Data filtering for the prior matters more than the main text suggests. The text says filtering gives "only" a slight boost. But on RLU Quadruped with 25k points, training f_b/f_R on the top 10% of episodes gives 973 ± 49, against 115 ± 69 with no filtering. With 100k points: top 20% gives 989, no filtering gives 547. On RLU Walker the effect is small (50k points: top 80% 772 vs 100% 770) — [MBOP §4.1 vs Tables 10–11](https://ar5iv.labs.arxiv.org/html/2008.05556)

**Zero-shot adaptation by changing the planning objective (yes, demonstrated)**
- A user-supplied state objective R'_n = Σ_t f_obj(s_t) enters the MPPI weights as e^{κR_n + κ_obj R'_n}. For goal-conditioned control, κ=0 (the original reward is ignored); for constrained control, both terms are kept — [MBOP §4.3](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Tasks:
  - Position-constrained Cartpole: stays on one side of the track while keeping return above 750.
  - Heading-conditioned Quadruped (forward, backward, left, right): the goal has clear influence, but f_b and f_R bias the controller toward the forward direction seen in the data.
  - Height-constrained Walker: about 80% constraint satisfaction at reward about 730.
  — [MBOP §4.3, App. "Continued Analysis of Constrained Tasks"](https://ar5iv.labs.arxiv.org/html/2008.05556)
- The authors note that MBOP variants without the BC prior can also be used for goal/constraint control in high-data regimes. They list goal-conditioned f_b and f_R as future work — [MBOP §3.6](https://ar5iv.labs.arxiv.org/html/2008.05556)

**Ablations, speed, stated limitations**
- Ablations: MBOP-NOPP (BC prior replaced by a Gaussian), MBOP-NOVF (no f_R), and PDDM (neither). The full combination is generally best. PDDM is generally below both partial ablations. The BC prior alone does well on some environments but stagnates at behavior-policy level on others — [MBOP §4.4](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Without the prior, D4RL performance collapses. Raw returns:
  - HalfCheetah-medium: MBOP 5406.5, NOVF 4959.8, cloning 4242.4, NOPP −427.0, PDDM −331.9.
  - Walker2d-medium: MBOP 1881.9, cloning 706.8, NOVF 341.7, NOPP −9.9.
  - Door: NOPP and PDDM score about −53 at every data size.
  - Exception: on Walker2d-random, NOPP (484.5) and PDDM (498.9) beat MBOP (371.1).
  — [MBOP Tables 5–8](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Speed: BC runs at 362 Hz; MBOP at 106 Hz for H=4, 71 Hz for H=8 and 40 Hz for H=16 (Table 9). The main text instead pairs 40 Hz with "h=40", which is an internal inconsistency. The authors suggest distilling into a policy when high-frequency control is needed — [MBOP §4.4, Table 9](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Stated limitations: performance degrades with multi-modal or random data. Offline model selection and evaluation (OPE/OHS) is out of scope. Future work: goal-conditioned priors and values, and multi-modal stochastic models — [MBOP §3.3, §5](https://ar5iv.labs.arxiv.org/html/2008.05556)

### Inferences
- MBOP's tables are the cleanest evidence in this set that, offline, the BC prior is what keeps the planner on the data manifold. Replacing it with a Gaussian takes D4RL returns from thousands to negative, while the same model and value function are used. This is the offline "CEM exploits the model" failure, and MBOP's fix is to sample narrowly around the prior (σ as small as 0.01–0.2), not to penalize uncertainty.
- On expert data (door-expert), planning around a strong prior adds at most a few percent and can slightly hurt at large data sizes. This matches a regime where the prior is near the ceiling and the planner can mostly only lose.
- The truncated value f_R is MC-regressed under the behavior policy, so it is a behavior value, not an optimal value. It biases plans toward what the data did, which is visible in the Quadruped heading bias. For goal-reaching, the analogue would be a goal-conditioned behavior value, which the authors list as future work.

### Gaps
- I did not cross-check the ICLR camera-ready on OpenReview; OpenReview showed a browser-verification page that I did not bypass. All numbers come from the ar5iv rendering of arXiv 2008.05556, and ar5iv does not state which version (v1–v3) it shows.
- MBOP's per-dataset "DATA policy" (behavior) returns appear only in figure box plots, so they are not reported numerically here.

---

## Q2. MOPP: how it changes MBOP (wider guided sampling, max-Q, uncertainty pruning), the uncertainty estimator, and what pruning contributes

### Takeaway
MOPP keeps MBOP's MPPI skeleton but deliberately samples more widely around the behavior policy. It replaces MBOP's deterministic BC with an ensemble of expressive autoregressive (ADM) behavior policies and dynamics models, and draws actions from a Gaussian shaped by the behavior std and scaled to σ_M. Among m sampled actions per step, it greedily keeps the one with the highest Q_b, where Q_b is an FQE estimate of the behavior policy's Q. It then prunes rollouts whose MOReL-style ensemble discrepancy exceeds a threshold L, with a floor that keeps at least 20% of rollouts. L is set to the 85th percentile of in-data uncertainty. Compared with MBOP, the gains are largest on medium-expert and mixed data. The ablations show three things. Pruning helps only when sampling is moderately aggressive. Max-Q contributes more than the terminal value. The threshold must be neither strict nor lax.

### Cited Findings
**What is learned**
- Ensembles of autoregressive dynamics models (ADM). The dynamics model (r_t, s_{t+1}) = f_m(s_t, a_t) and the behavior policy a_t = f_b(s_t) both predict a per-dimension mean and std autoregressively over output dimensions. The ensemble uses randomly permuted output orderings, which the authors say adds diversity and expressiveness. Unlike MBOP, f_b does not take the previous action — [MOPP §4.1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Q_b(s, a) is learned by fitted Q evaluation (FQE) of the behavior policy, with γ' < 1. V_b(s) = E_{a~π_b} Q_b(s, a) is estimated with K_Q=10 samples from f_b. The paper calls this a conservative estimate tied to the behavioral distribution — [MOPP §4.2, Alg. 1, App. A.2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Configuration: K1=K2=3. ADM uses a 500-unit embedding plus (200, 100) per-dimension heads; Q_b uses FC (500, 500); 5e5 training steps with Adam at lr 1e-3 — [MOPP App. A.2, D](https://ar5iv.labs.arxiv.org/html/2105.07351)

**Guided trajectory rollout (what changes relative to MBOP)**
- At each step, sample m actions a^i_t ~ N(μ_a(s_t), diag((σ_M / max σ_a(s_t)) · σ_a(s_t))²). This rescales the behavior std so its largest dimension equals σ_M while keeping the relative per-dimension spread. σ_M may exceed the behavior's own max std, which allows more aggressive sampling. MBOP, by contrast, adds small isotropic noise to a deterministic policy — [MOPP §4.3, Eq. 3](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Max-Q guidance: â_t = argmax over the m samples of Q_b(s_t, a). The result is β-mixed with the previous plan and rolled out with a randomly chosen dynamics member. Reward is averaged over all K1 members, and V_b(s_H) is added at the end — [MOPP Eq. 3, Alg. 1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Optimizer: the same exponentiated-return MPPI average as MBOP, but computed only over the pruned set T_f — [MOPP Eq. 5](https://ar5iv.labs.arxiv.org/html/2105.07351)

**Uncertainty estimator and pruning rule**
- The estimator is MOReL's discrepancy, disc(s, a) = max over member pairs (i, j) of ‖f_m^i(s, a) − f_m^j(s, a)‖². It is computed at every step of every rollout, giving U_{n,t} — [MOPP §4.3](https://ar5iv.labs.arxiv.org/html/2105.07351)
- The rule: keep T_p = {T_n : U_{n,t} < L for all t}. If fewer than N_m = 0.2⌊N⌋ rollouts survive, top up with the remaining rollouts that have the lowest cumulative uncertainty Σ_t U_{n,t}. The stated aim is to discard bad out-of-distribution (OOD) samples but keep OOD regions where the model generalizes — [MOPP Eq. 4](https://ar5iv.labs.arxiv.org/html/2105.07351)
- L is chosen as the 85th percentile of uncertainty measures computed on the offline dataset — [MOPP App. A.2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Constraints can be routed through pruning. A "rollout constraint" adds 100·max(v_x − 10, 0) to U_{n,t}. Because of the N_m floor, some violating rollouts can survive when most rollouts violate the constraint — [MOPP App. A.3.2](https://ar5iv.labs.arxiv.org/html/2105.07351)

**Hyperparameters**
- N=100 (1000 for Walker2d). H=2–10 (16 for relocate-expert). β=0 on every task. κ=0.03–10. σ_M=0.05–1.15. L=0.1–8 — [MOPP Table 3](https://ar5iv.labs.arxiv.org/html/2105.07351)

**Results (D4RL normalized; 5 seeds × 20 episodes)**
- MOPP vs MBOP (the method's own behavior prior in parentheses):
  - random: hc 9.4 (2.2) vs 6.3; hopper 13.7 (9.8) vs 10.8; walker 6.3 (2.6) vs 8.1.
  - medium: hc 44.7 (36.6) vs 44.6; hopper 31.8 (30.0) vs 48.8; walker 80.7 (15.6) vs 41.0.
  - mixed: hc 43.1 (32.7) vs 42.3; hopper 32.3 (28.2) vs 12.4; walker 18.5 (12.9) vs 9.7.
  - med-expert: hc 106.2 (37.6) vs 105.9; hopper 95.4 (44.3) vs 55.1; walker 92.9 (13.5) vs 70.2.
  — [MOPP Table 1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Versus model-based offline RL, MOPP is better on medium and medium-expert and worse than MBPO/MOPO on random and mixed. The authors attribute this to RL methods benefiting from high-diversity data — [MOPP §5.1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Discrepancy: the MOPO numbers in MOPP's table differ from those quoted in MBOP's table (e.g., random-halfcheetah 35.4 vs 31.9; mixed-hopper 67.5 vs 92.5) — [MOPP Table 1](https://ar5iv.labs.arxiv.org/html/2105.07351) vs [MBOP Table 1](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Adroit, MOPP vs MBOP:
  - pen-human 73.5 vs 53.4; door-human 11.9 vs 2.7; pen-expert 149.5 vs 105.5; hammer-expert 128.7 vs 107.6; door-expert 105.3 vs 101.2; relocate-expert 98.0 vs 41.7.
  - Exceptions to the text's claim that MOPP is consistently better: hammer-human 2.8 vs 14.8, and relocate-cloned −0.1 vs 0.1.
  - Against model-free baselines, MOPP often beats BCQ and CQL, e.g., pen-expert BCQ 114.9, CQL 107.0.
  — [MOPP Table 2, §5.1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Appendix B ablation: the more expressive ADM dynamics model matters more than the ADM behavior policy. Example on med-expert hopper with the ADM behavior policy: MBOP's FC dynamics model gives 51.6, the ADM dynamics model gives 95.4 — [MOPP App. B, Table 4](https://ar5iv.labs.arxiv.org/html/2105.07351)

**What pruning, max-Q and V_b contribute (walker2d-med-expert ablation; figures only)**
- Raising σ_M from 0.01 to 0.5 helps. σ_M=1.0 hurts because too many OOD samples are generated. At σ_M=1.0, pruning stops working: most rollouts are problematic and the N_m floor forces some to be kept. At σ_M=0.5, MOPP clearly beats MOPP-noP (no pruning), i.e., pruning controls planning variance — [MOPP §5.2, Fig. 1a](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Max-Q raises aggressiveness. It helps at moderate σ_M, but at σ_M=1.0 only the less aggressive MOPP-noMQ still reaches high returns — [MOPP §5.2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Max-Q contributes more than V_b (MOPP-noMQ is consistently below MOPP-noV). The authors' explanation is that max-Q acts at every step while V_b only enters at the end. Short horizons (H=2–4) suffice when both are used — [MOPP §5.2, Fig. 1b](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Threshold L: both strict (L=0.1) and tolerant (L=10) thresholds are worse than L=2.0, which gave the best result with low variance (with σ_M=0.5, H=2) — [MOPP §5.2, Fig. 1c](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Changing objectives: switching to a new reward improves over the old-objective planner on halfcheetah-jump. Re-fitting Q_b by FQE under the new reward helps further, because the offline Q_b and max-Q can hurt when the reward changes a lot. Re-fitting is cheap supervised learning: about 26 min for 1M samples on CPU — [MOPP §5.3, App. C Table 6](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Speed, CPU-only, Walker2d: 2.69 Hz at H=4, 2.13 Hz at H=8, 1.50 Hz at H=16 — [MOPP App. C Table 5](https://ar5iv.labs.arxiv.org/html/2105.07351)

### Inferences
- MOPP's biggest gains over its own prior come where the BC fits multimodal data poorly: med-expert walker goes from an ADM prior of 13.5 to 92.9 with planning. In those cases the improvement comes from Q_b selecting among prior samples, a value-based reranking, more than from the dynamics model's lookahead.
- Pruning is only as good as the discrepancy signal is calibrated. MOPP sets L from the in-data distribution of discrepancy (85th percentile), not from failure labels. Its ablation shows the benefit exists only in a middle band of sampling width. If ensemble disagreement does not separate good from bad rollouts, the method has no fallback, and the N_m floor forces it to keep the least-uncertain rollouts anyway.
- Widening plus pruning is a "trust region by uncertainty" around the prior. MBOP's tiny σ is the trust-region-by-construction alternative.

### Gaps
- The number m of actions sampled per step for the max-Q operation is not given in the hyperparameter table or text I read.
- Ablation magnitudes (MOPP vs noP / noMQ / noV, and the L sweep) appear only in Fig. 1, with no numbers in the text.
- It is not stated whether the discrepancy uses next-state predictions only or also reward, nor whether it uses means or samples of the probabilistic ADM outputs.

---

## Q3. TD-MPC and TD-MPC2: decoder-free latent model, policy prior, Q-ensemble, MPPI with policy samples, terminal value, horizon, pessimism, and whether epistemic uncertainty enters planning

### Takeaway
Both are online RL agents. They jointly learn a decoder-free latent dynamics model, reward, terminal Q and policy, using latent-consistency, reward and TD losses, and they plan with MPPI. Most of the population is Gaussian samples (512), plus a small set of policy-prior rollouts: about 25 in TD-MPC (5%), 24 of the 512 in TD-MPC2. Horizon is H=5 (TD-MPC) or H=3 (TD-MPC2), with a terminal Q. Pessimism lives only in the value function:
- TD-MPC uses min(Q1, Q2) both in TD targets and, per the code, inside planning.
- TD-MPC2 uses a 5-Q ensemble with 1% dropout. TD targets take the min of 2 randomly subsampled target Qs, but planning uses the average of 2 randomly subsampled Qs.

No dynamics-ensemble or epistemic term enters default planning. TD-MPC2 reports an optional test-time penalty on Q-ensemble std (taken from FOWM) in one offline multitask experiment: it helped at small strength and hurt at large strength.

### Cited Findings
**TD-MPC: what is learned**
- TOLD has five components: encoder h, latent dynamics d, reward R, value Q, and policy π. All are deterministic MLPs, with no RNN and no probabilistic model. There is no observation decoder; a latent-consistency loss predicts the target-encoder embedding h_θ−(s_{t+1}) instead — [TD-MPC §4](https://ar5iv.labs.arxiv.org/html/2203.04955)
- The loss is Σ_i λ^{i−t}[c1·reward MSE + c2·TD MSE + c3·latent-consistency MSE], unrolled over H=5 steps with gradients back-propagated through time. Coefficients: λ=0.5, c1=0.5, c2=0.1, c3=2. Replay uses prioritized experience replay (PER) — [TD-MPC Eqs. 7–10, Table 4](https://ar5iv.labs.arxiv.org/html/2203.04955)
- There are two Q heads (LayerNorm+tanh after the first layer). The latent dimension is 50 (100 for Humanoid/Dog), and the model has about 1.5M parameters on Walker Run — [TD-MPC App. F](https://ar5iv.labs.arxiv.org/html/2203.04955)
- π is a deterministic DDPG-style policy that maximizes Q. The authors find it inferior to planning on complex tasks but expressive enough for value learning — [TD-MPC §4, Eq. 11](https://ar5iv.labs.arxiv.org/html/2203.04955)

**TD-MPC: how planning samples and scores**
- Return estimate φ_Γ = E[γ^H Q(z_H, a_H) + Σ_{t<H} γ^t R(z_t, a_t)], with one particle. Each iteration takes the top-k samples and fits a return-weighted mean and std, with weights exp(τ·φ), temperature τ=0.5 — [TD-MPC §3, Eqs. 3–4, Table 4](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Settings: N=512 Gaussian samples plus 5% policy samples; top-64 elites; 6 iterations (8 for Dog and pixels, 12 for Humanoid); (μ0, σ0) = (0, 2); momentum 0.1 — [TD-MPC §5, Table 4](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Exploration and horizon schedules: the std floor ε decays linearly from 0.5 to 0.05 over 25k steps. The planning horizon is also ramped from 1 to 5 over the first 25k steps, because early in training the model is inaccurate and longer planning would be dominated by model bias — [TD-MPC §3, Eq. 5, Table 4](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Warm start: the mean is initialized from the previous step's 1-step-shifted solution, citing MBOP. The variance is always reset large to avoid local minima — [TD-MPC §3](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Policy-guided sampling: policy rollouts compete in the same top-k selection. A rollout scored poorly is excluded; one scored well gets influence proportional to its estimated return. Policy actions get linearly annealed Gaussian noise, as in DDPG — [TD-MPC §3](https://ar5iv.labs.arxiv.org/html/2203.04955)
- In the code:
  - num_pi_trajs = int(mixture_coef·num_samples), i.e., 25 policy rollouts, generated once per decision step with noise std cfg.min_std and concatenated to the 512 Gaussian samples on every iteration.
  - The terminal term is torch.min(Q1, Q2) evaluated at π(z_H), i.e., clipped double-Q pessimism inside planning.
  - The std is clamped to [annealed floor, 2].
  - The executed action is an elite trajectory sampled in proportion to its score, plus std noise during training.
  — [TD-MPC code, estimate_value / plan](https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py)
- TD targets also use the min of the two target Qs — [TD-MPC code, _td_target](https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py)

**TD-MPC: settings and headline results**
- 92 tasks:
  - 6 Humanoid/Dog tasks (|A| up to 38)
  - 15 state-based DMControl tasks
  - 6 DMControl-100k and 12 Dreamer-benchmark pixel tasks
  - 2 multimodal tasks
  - 50 goal-conditioned Meta-World v2 tasks plus MT10
  It is described as the first documented solution of the DMControl Dog tasks — [TD-MPC §5](https://ar5iv.labs.arxiv.org/html/2203.04955)
- DMControl-100k from pixels: Cartpole Swingup 770, Reacher Easy 628, Cup Catch 933, Finger Spin 943, Walker Walk 577, Cheetah Run 222 — [TD-MPC Table 1](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Wall-time on one RTX 3090:
  - Walker Walk solved in 0.47 h, against LOOP 7.72 h (16× faster), SAC 0.41 h and MPC with a simulator 0.91 h.
  - Humanoid Stand: 9.39 h vs SAC 9.31 h.
  - Default planning costs about 20 ms per step (50 Hz); the policy alone is about 6× faster.
  — [TD-MPC Table 2, App. H](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Planning vs policy:
  - More planning helps on complex tasks such as Quadruped Walk. Inference planning cost can be cut by 50% (fewer iterations) without loss. On Quadruped Run the horizon can drop from 5 to 1 without a significant reduction.
  - The policy alone is generally worse than planning, and the gap is larger for high-dimensional action spaces.
  — [TD-MPC §5, App. C, App. H](https://ar5iv.labs.arxiv.org/html/2203.04955)
- Stated limitation: lower sample efficiency than SAC/LOOP on the hard-exploration Finger Turn Hard; the authors suggest better exploration — [TD-MPC §5](https://ar5iv.labs.arxiv.org/html/2203.04955)

**TD-MPC2: what changes**
- Components: encoder h(s, e), latent dynamics d(z, a, e), reward R, terminal value Q, and a policy prior p(z, e), all conditioned on a learnable task embedding e with ‖e‖ ≤ 1. The model is decoder-free and implicit — [TD-MPC2 §3.1, Eq. 2](https://arxiv.org/html/2310.16828)
- Objective: joint-embedding prediction ‖z'_t − sg(h(s'_t))‖² plus cross-entropy reward and value losses, using discrete regression over 101 bins in log-transformed space. Coefficients 20 / 0.1 / 0.1, λ=0.5 — [TD-MPC2 §3.1, Eq. 3, Table 8](https://arxiv.org/html/2310.16828)
- The policy prior is a stochastic max-entropy policy, with the Q-term scaled by moving 5–95% percentiles and entropy coefficient 1e-4. The latent uses SimNorm (softmax over 8-dim simplices). All MLPs use LayerNorm and Mish — [TD-MPC2 §3.1, Table 8](https://arxiv.org/html/2310.16828)
- Value ensemble: 5 Q-functions with 1% dropout. TD targets are the min of two randomly subsampled EMA target Qs (REDQ-style). The default single-task model has 5M parameters; TD-MPC has about 1M and only 2 Qs with no dropout — [TD-MPC2 §3.1, App. A](https://arxiv.org/html/2310.16828)

**TD-MPC2: planner**
- MPPI settings: H=3; 6 iterations (+2 if |A| ≥ 20); population 512 including 24 policy-prior samples; 64 elites; std in [0.05, 2]; temperature 0.5; no momentum. Each step is warm-started from the previous solution shifted by one — [TD-MPC2 §3.2, Table 8, App. A](https://arxiv.org/html/2310.16828)
- Code details:
  - The 24 prior rollouts are sampled once per decision step from the stochastic prior and occupy fixed slots. Only the other 488 slots are resampled from the Gaussian each iteration.
  - The terminal value is Q(z_H, p(z_H)) with return_type='avg', i.e., the mean of 2 randomly chosen Qs out of 5. One random pair is drawn per call and shared by the whole batch.
  - TD targets use return_type='min' on the target Qs.
  - The executed action is a Gumbel-softmax sample over elite scores, plus std noise during training.
  — [TD-MPC2 code, _estimate_value / _plan / _td_target](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py); [world_model.Q](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)

**TD-MPC2: where uncertainty enters**
- Default planning has no epistemic term. The multitask datasets are offline, but they get no special offline treatment — [TD-MPC2 App. J](https://arxiv.org/html/2310.16828)
- Optional test-time regularizer (from FOWM): each step's reward and the terminal value are reduced by u_t = c · avg(q̂_1..N) · std(q̂_1..N). Scaling by the mean value makes c task-agnostic — [TD-MPC2 App. J, Eqs. 9–12](https://arxiv.org/html/2310.16828)
- Effect on a 19M-parameter agent trained offline on the 80-task data (normalized score): no regularization 56.54; c=0.001 58.14; c=0.01 62.01; c=0.1 44.13. The regularizer is not used in any other experiment — [TD-MPC2 Table 10](https://arxiv.org/html/2310.16828)

**TD-MPC2: settings and headline results**
- 104 tasks: DMControl (39, including 11 new), Meta-World (50), ManiSkill2 (5) and MyoSuite (10), with |A| up to 39. All use one hyperparameter set. TD-MPC2 outperforms SAC, DreamerV3 and TD-MPC in every domain. It reaches over 60% success on Pick YCB, where the other methods fail within budget. TD-MPC sometimes diverges from exploding gradients — [TD-MPC2 §4.1, App. D](https://arxiv.org/html/2310.16828)
- Scaling on 80 tasks (545M transitions from 240 single-task agents' replay buffers, random to expert behavior), normalized score by model size: 1M 16.0; 5M 49.5; 19M 57.1; 48M 68.0; 317M 70.6. The 317M model takes about 33 RTX 3090 GPU-days — [TD-MPC2 §4.1, Table 1](https://arxiv.org/html/2310.16828)
- Few-shot fine-tuning gives about 2× the score of training from scratch at 20k steps — [TD-MPC2 §4.1](https://arxiv.org/html/2310.16828)
- Stated limits and risks: extension to discrete actions is open; no current algorithm masters everything out of the box; reward misspecification; risks of unconstrained autonomy on physical robots; data cost — [TD-MPC2 §5, App. I](https://arxiv.org/html/2310.16828)

### Inferences
- In the TD-MPC family the policy prior is a minority proposal that must win on model-plus-Q scores. Nothing protects a good prior if the model or value mis-ranks it. This works online because new data corrects exploited errors; offline or with a frozen model, that safeguard is absent.
- The only pessimism in TD-MPC2 planning is averaging two random Qs, which is milder than TD-MPC's min. The paper's own offline result says a Q-ensemble-std penalty helps only in a narrow coefficient band.
- The planner's short H (3–5) and terminal value mean most of the decision rests on Q accuracy. TD-M(PC)² (Q5) shows Q is badly overestimated in high-dimensional tasks.

### Gaps
- TD-MPC2's Fig. 9 ablation of the choice of actor (planning vs policy prior only) is shown only as curves and bars; I extracted no numbers.
- I did not re-verify the venues (TD-MPC: ICML 2022; TD-MPC2: ICLR 2024) in the fetched texts.
- The value of cfg.min_std (the noise on TD-MPC's policy proposals) was not checked in the config file.

---

## Q4. POPLIN: policy network to initialize or parameterize CEM (POPLIN-A vs POPLIN-P), model and propagation, distillation, results vs PETS

### Takeaway
POPLIN is online model-based RL built on the PETS codebase. POPLIN-A uses a policy network to propose a reference action sequence and runs CEM over additive action noise. POPLIN-P runs CEM directly over perturbations of the policy network's weights. The reported experiments use PETS's probabilistic ensemble (PE) with expectation propagation (E), not trajectory sampling (TS). The planner is distilled back into the policy by BC, a GAN, or parameter averaging. POPLIN-P beats PETS by large margins on complex MuJoCo tasks, while POPLIN-A barely helps there. Larger populations help only POPLIN-P. Even a zero-weight policy network searched in parameter space works well, which points to search-space reparameterization, rather than the learned prior, as the main driver.

### Cited Findings
**Model and uncertainty**
- Candidate returns are expectations over P particles propagated through a deterministic or probabilistic ensemble, as in PETS — [POPLIN §4](https://ar5iv.labs.arxiv.org/html/1906.08649)
- All reported PETS and POPLIN experiments use model type PE with propagation E (expectation). The authors call other combinations prohibitively expensive (PE-DS takes about 68 h per seed for 200k steps) and note that PE-E is among the best in many environments — [POPLIN App. A.3.1](https://ar5iv.labs.arxiv.org/html/1906.08649). The repo's example commands likewise use `-ca model-type PE -ca prop-type E` — [POPLIN GitHub README](https://github.com/WilsonWangTHU/POPLIN)
- There is no uncertainty penalty or pruning. Model uncertainty enters only through the PETS ensemble expectation — [POPLIN §3.2, §4](https://ar5iv.labs.arxiv.org/html/1906.08649)

**POPLIN-A (action space)**
- The policy is rolled through the model to give a reference sequence â. CEM then optimizes the mean and covariance of additive noise δ, starting from μ0=0 and Σ0=σ0²I. Elite statistics are smoothed with coefficient α — [POPLIN §4.1, Eq. 2](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Two variants:
  - POPLIN-A-Init: the reference sequence stays fixed from the initial forward pass and noise is added to it.
  - POPLIN-A-Replan: each perturbed rollout re-queries π(s_t) + δ_t. This is more aggressive and, per the authors, raises the risk of divergent behavior.
  Init performed better than Replan, suggesting divergent or overconfident updates in Replan — [POPLIN §4.1, §5.4](https://ar5iv.labs.arxiv.org/html/1906.08649)

**POPLIN-P (parameter space)**
- CEM searches over noise ω added to the policy weights θ: POPLIN-P-Sep uses per-timestep noise, POPLIN-P-Uni uses the same noise for the whole horizon. Sep performs much better. The authors interpret this as a stochastic policy via the reparameterization trick — [POPLIN §4.2, §5.4](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Motivation: CEM in action space averages modes (e.g., going straight between left and right detours around an obstacle). PCA visualizations show the reward surface is much smoother in parameter space — [POPLIN §4.2, §5.3](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Population size: PETS and POPLIN-A do not improve with larger populations; POPLIN-P does — [POPLIN §5.3, Fig. 4](https://ar5iv.labs.arxiv.org/html/1906.08649)
- POPLIN-P with a fixed all-zero-weight network still searches very efficiently. The authors conclude that optimization efficiency, especially for POPLIN-P-AVG, is the key reason planning succeeds — [POPLIN App. A.6](https://ar5iv.labs.arxiv.org/html/1906.08649)
- The authors also report that deeper networks give better search efficiency and more multimodal action distributions — [POPLIN §5.3, App. A.7](https://ar5iv.labs.arxiv.org/html/1906.08649)

**Distillation back into the policy**
- BC: L2 regression on planned actions, with parameter noise set to zero. GAN: a discriminator, with noise drawn from the initial CEM distribution. AVG: θ ← θ + mean of the optimized ω, which keeps the network only as a search initialization. Training can use real or hallucinated data — [POPLIN §4.4, App. A.3.1](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Policy control without MPC lags MPC and is almost random on Ant and Walker2d. POPLIN-A, with its deterministic policy, is more prone to distillation collapse than POPLIN-P. POPLIN-P-BC and POPLIN-P-GAN are comparable. AVG gives good MPC performance but sacrifices policy control — [POPLIN §5.2](https://ar5iv.labs.arxiv.org/html/1906.08649)

**Settings and results (4 seeds, MuJoCo, end of training)**
- Hyperparameter grids searched:
  - PETS: population 100–2000, horizon 30/50/100, initial σ 0.01–0.5, CEM iterations 5–20, elites 50–200.
  - POPLIN: initial σ 0.001–0.1.
  — [POPLIN Tables 4–6](https://ar5iv.labs.arxiv.org/html/1906.08649)
- The README example for POPLIN-P uses policy_network_shape [32], init_var 0.1, POPLINP-SEP and AVG-R. The default policy network is [64, 64] — [POPLIN GitHub README](https://github.com/WilsonWangTHU/POPLIN)
- POPLIN-P vs PETS:
  - Cheetah (PETS version, 50k steps): 12227.9 ± 5652.8 vs 4204.5 ± 789.0 (POPLIN-A 4651.1; SAC 1745.9; TD3 218.9).
  - Ant (200k): 2330.1 vs 1165.5.
  - Hopper (200k): 2055.2 vs 114.9.
  - Gym HalfCheetah "Cheetah-v0" (200k): 4235.0 vs 2288.4 (SAC 3459.8).
  - Walker2d: 597.0 vs 282.5.
  - Swimmer: 334.4 vs 326.2 (POPLIN-A best at 344.9).
  - Pusher: −55.8 vs −52.7 (PETS slightly better).
  — [POPLIN Table 1, App. Tables 2–3](https://ar5iv.labs.arxiv.org/html/1906.08649)
- The abstract claims POPLIN is about 3× more sample-efficient than PETS, TD3 and SAC — [POPLIN GitHub README / abstract](https://github.com/WilsonWangTHU/POPLIN)
- POPLIN-A is best on simpler environments (Pendulum, Cartpole, Swimmer) but shows no clear gain over PETS on Ant, Cheetah and Hopper — [POPLIN §5.1](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Stated limitations: it cannot efficiently solve Walker or Humanoid, and model-free methods eventually reach better asymptotic performance — [POPLIN §5.1](https://ar5iv.labs.arxiv.org/html/1906.08649)

### Inferences
- POPLIN shows that "planning around a prior" splits into two separate effects: proposal quality (POPLIN-A) and search-space geometry (POPLIN-P). In complex tasks, most of the gain comes from geometry. A policy network acts as a smooth, low-dimensional, state-feedback parameterization of the trajectory, so perturbations stay locally coherent instead of producing jagged action sequences.
- POPLIN-A-Replan's divergence is the same model-exploitation pattern in another form: re-querying the prior along perturbed (off-data) rollouts compounds model error. The safer Init variant keeps the prior fixed.

### Gaps
- The per-environment hyperparameters finally chosen (horizon, population, iterations) are not reported in the text I read.
- The arXiv v1 numbers could differ from the ICLR camera-ready; OpenReview showed a browser-verification page that I did not bypass. The ICLR 2020 acceptance itself is confirmed by the ICLR virtual page.
- The question assumed PETS-style trajectory sampling (TS). The paper's experiments use E propagation instead, so I could not determine whether POPLIN gains or loses with TS.

---

## Q5. Across all four (plus follow-up analyses): when planning beats the prior, model exploitation, and narrow vs wide sampling around the prior

### Takeaway
Across these papers, planning beats the prior when:
- the prior is mediocre or averages over modes and a value function can pick out the better behavior (MOPP med-expert walker: prior 13.5, planner 92.9);
- the data are small but consistent (MBOP on RL Unplugged);
- online data can correct the errors the planner exploits (TD-MPC, POPLIN).

Planning gains little, or slightly hurts, when the prior is already expert-level (MBOP door-expert at large data). With random data, a unimodal prior can be worse than no prior.

Model exploitation is controlled in three ways: narrow sampling centered on the prior (MBOP); bounded widening plus disagreement-based pruning (MOPP); and pessimistic or ensemble value functions plus online self-correction (TD-MPC family, FOWM). None of the four papers validates dynamics-ensemble disagreement as a detector of failures. The one positive in-planning uncertainty result in this lineage uses the spread of a Q-ensemble (FOWM; TD-MPC2 App. J), and it is coefficient-sensitive.

### Cited Findings
**Side-by-side mechanics**
| | MBOP | MOPP | TD-MPC | TD-MPC2 | POPLIN |
|---|---|---|---|---|---|
| Setting | offline | offline | online | online (+ offline multitask pretraining) | online |
| Prior | BC f_b(s, a_{t−1}), deterministic + iid noise σ | ADM behavior policy, std-shaped Gaussian scaled to σ_M, max-Q over m samples | deterministic π (DDPG-style) + noise, 5% of samples | max-entropy stochastic prior, 24/512 samples | policy net as initializer (A) or search parameterization (P) |
| Value | MC truncated return f_R (behavior) | FQE Q_b / V_b (behavior) | TD Q (2 heads), min in planning | TD Q-ensemble (5), avg of 2 random in planning | none (reward over horizon) |
| Iterations per step | 1 pass + β-mixing | 1 pass + β-mixing (β=0 used) | 6–12 MPPI iterations | 6–8 MPPI iterations | CEM iterations (grid 5–20) |
| Horizon | 2–10 (D4RL), up to 64 (RLU) | 2–16 | 5 (ramped from 1) | 3 | grid 30/50/100 |
| Uncertainty in planning | none (consistent head + averaging) | prune if max pairwise discrepancy ≥ L | none (min of 2 Qs) | none by default; optional c·avg·std(Q) | PE ensemble expectation only |
- Sources for the table above: [MBOP](https://ar5iv.labs.arxiv.org/html/2008.05556); [MOPP](https://ar5iv.labs.arxiv.org/html/2105.07351); [TD-MPC](https://ar5iv.labs.arxiv.org/html/2203.04955) with [code](https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py); [TD-MPC2](https://arxiv.org/html/2310.16828) with [code](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py); [POPLIN](https://ar5iv.labs.arxiv.org/html/1906.08649)

**When planning beats the prior**
- MBOP vs its BC prior on D4RL:
  - Large gains: medium-walker 41.0 vs 15.4; med-expert hopper 55.1 vs 15; mixed-halfcheetah 42.3 vs 0.0.
  - Losses: mixed-walker 9.7 vs 11.5.
  - On RLU, MBOP is above cloning at all reported sizes.
  - With expert door data, MBOP is about equal to or below cloning at 200k–1M points.
  — [MBOP Tables 1, 2–5](https://ar5iv.labs.arxiv.org/html/2008.05556)
- MOPP's planner beats its ADM prior most on med-expert (hc 106.2 vs 37.6; walker 92.9 vs 13.5) and barely on medium-hopper (31.8 vs 30.0) — [MOPP Table 1](https://ar5iv.labs.arxiv.org/html/2105.07351)
- TD-MPC: planning beats its own policy, especially with high-dimensional actions — [TD-MPC §5, App. C](https://ar5iv.labs.arxiv.org/html/2203.04955). POPLIN: MPC beats policy-only control, and the distilled policy is almost random on Ant and Walker2d — [POPLIN §5.2](https://ar5iv.labs.arxiv.org/html/1906.08649)
- TD-M(PC)² gives a theoretical view, adopted from LOOP: an H-step planner reduces reliance on value accuracy by roughly a factor of γ^{H−1} relative to a greedy policy. This helps explain why TD-MPC2 works despite large value overestimation — [TD-M(PC)² §3.3, Thm 3.1](https://arxiv.org/html/2502.03550)

**Model exploitation: what each paper says and does**
- Offline data lacks counterfactuals that could contradict prediction errors, which MBOP calls fundamental to policy improvement. Its no-prior ablations collapse on D4RL: for example HalfCheetah-medium −427 (NOPP) vs 5406.5 (MBOP), and door at about −53 at every size — [MBOP §3.3, Tables 5–8](https://ar5iv.labs.arxiv.org/html/2008.05556)
- MOPP states that randomly generated actions cause large exploitation errors offline. It also shows that overly wide prior sampling (σ_M=1.0) degrades performance even with pruning — [MOPP §4.3, §5.2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- TD-MPC ramps the horizon from 1 to H because early planning would be dominated by model bias — [TD-MPC §3](https://ar5iv.labs.arxiv.org/html/2203.04955). POPLIN-A-Replan risks divergent behavior — [POPLIN §4.1](https://ar5iv.labs.arxiv.org/html/1906.08649)
- FOWM (TD-MPC backbone, offline-to-online) points out that even with a conservatively learned value (IQL-style expectile V, AWR policy), the planner queries dynamics, reward and value on unseen state-actions, causing severe extrapolation errors. Its fix:
  - Penalize planning returns: R̂ = γ^h(Q − λu_h) + Σ_t γ^t(R − λu_t), with u_t = std of a 5-Q ensemble. Value estimates use the min of 2 random Qs.
  - The authors chose a Q-ensemble over ensembling the whole world model as a cheap, sufficient proxy for epistemic uncertainty.
  — [FOWM §3.1–3.2, App. C](https://arxiv.org/html/2310.16029)
- FOWM's λ results: λ > 0 almost always beats λ = 0 both offline and online. Large λ (e.g., 20) can hurt online. The best λ differs by task and between the offline and online stages — [FOWM §4.1 Q3.3, §6](https://arxiv.org/html/2310.16029)
- FOWM on a real robot (success %, offline-to-online TD-MPC vs FOWM):
  - Reach: 50 vs 72 at 0 trials; 78 vs 89 at 20 trials.
  - Pick at 20 trials: 28 vs 50.
  - Kitchen at 20 trials: 61 vs 78.
  - Online-only TD-MPC scores 0 throughout.
  — [FOWM Table 1](https://arxiv.org/html/2310.16029)
- FOWM's data finding: exploratory data matters more than quantity. Neither method succeeded zero-shot on the BC-generated "Base" dataset. Adding 20 exploratory trajectories ("Diverse") gave 33% (70 trajectories) and 72% (120 trajectories) zero-shot success for FOWM — [FOWM Table 4](https://arxiv.org/html/2310.16029)
- TD-M(PC)² finds persistent value overestimation in TD-MPC2, which it attributes to a structural mismatch between the planner policy that generates the data and the learned prior used in TD backups.
  - Value-estimate error by task: Hopper-Stand 15%; Dog-Trot 231%; h1hand-run 2159%; h1hand-slide 746%.
  - The fix adds a β·log μ(a|s) behavior-likelihood term to the policy loss. It improves TD-MPC2 by more than 100% on average across HumanoidBench and slightly on DMControl.
  - A pure BC policy-update variant performs nearly on par; the authors attribute this to the dominant role of conservatism in high-dimensional tasks.
  — [TD-M(PC)² §3.3, §5, §6.2–6.3](https://arxiv.org/html/2502.03550)

**Narrow vs wide sampling around the prior**
- Narrow (MBOP): σ is often 0.01–0.2 on D4RL, the plan is mixed with the previous one (β=0.2 on RLU), and there is one pass per step. Narrowness is the main safety mechanism — [MBOP Tables 2–8](https://ar5iv.labs.arxiv.org/html/2008.05556)
- Moderately wide plus filtered (MOPP): σ_M=0.25–0.6 on most medium/mixed/med-expert tasks (up to 1.15 on random). Pruning is needed and works only in a middle band — [MOPP Table 3, §5.2](https://ar5iv.labs.arxiv.org/html/2105.07351)
- Wide with a few prior samples (TD-MPC family): Gaussian std up to 2, with the prior as about 5% of samples. This is safe only because the agents keep collecting on-policy data — [TD-MPC Table 4](https://ar5iv.labs.arxiv.org/html/2203.04955); [TD-MPC2 Table 8](https://arxiv.org/html/2310.16828)
- Reparameterized (POPLIN-P): searching policy weights rather than actions. Larger initial noise helps POPLIN-P-Avg but not PETS — [POPLIN §5.4](https://ar5iv.labs.arxiv.org/html/1906.08649)
- Random-data exception: on Walker2d-random, prior-free sampling (NOPP 484.5, PDDM 498.9) beat prior-guided MBOP (371.1). When the prior carries no skill, narrowing around it only costs coverage — [MBOP Table 8](https://ar5iv.labs.arxiv.org/html/2008.05556)

### Inferences
- These inferences relate the literature to the reader's setting: frozen JEPA latent model, CEM exploits model errors, BC prior at 98%, a disagreement penalty that had no effect.
- With a prior already at the ceiling, every paper here predicts that a planner can add little and can lose a lot. MBOP on door-expert matches or is slightly below cloning; TD-M(PC)²'s BC-like update is nearly as good. The transferable role for the planner is a conservative filter or reranker of prior proposals: narrow σ, few or no refinement iterations, keep the BC mean unless the evidence is strong. It is not a wide optimizer.
- The four planners do not use dynamics-ensemble disagreement as a penalty on prior proposals. MBOP ignores disagreement; MOPP thresholds it with a percentile-calibrated cut and a floor; TD-MPC2/FOWM penalize value-ensemble spread instead. Since an ensemble-disagreement penalty on BC-prior proposals had no effect for the reader, the closest untested alternatives are:
  - (a) MOPP-style hard pruning with an in-data-percentile threshold. This only helps if disagreement ranks bad rollouts, which the reader's σ² test suggests it does not.
  - (b) A value-side signal: a Q-ensemble or FQE-style behavior value on the frozen latents, scored as in FOWM, or max-Q reranking among BC samples as in MOPP. MOPP found max-Q to be the strongest single component.
- Short horizons with a learned terminal value are the norm: H=2–4 in MBOP/MOPP on D4RL, H=3 in TD-MPC2. With a frozen world model that has no reward or value head, the transferable piece is a goal-conditioned behavior value, playing the role of MBOP's f_R or MOPP's Q_b, learned on the dataset in latent space. Longer horizons without a value did worse in MBOP (NOVF).
- MBOP's per-trajectory consistent ensemble member, with averaged rewards and values, is a cheap way to inject model diversity into proposals without trusting disagreement as a score.
- Filtering the data used to train the prior and value (MBOP Table 10: 115 → 973 on RLU Quadruped at 25k points) is a large, cheap lever in the same family as the reader's BC-prior finding.

### Gaps
- No paper here reports how often the planner overrules a correct prior proposal, i.e., failure attribution comparable to the reader's d_pred vs d_real analysis.
- None of these planners were evaluated on sparse-success, goal-conditioned, long-horizon offline manipulation comparable to OGBench Cube. The closest are Adroit (MBOP/MOPP, offline) and Meta-World / ManiSkill2 (TD-MPC/TD-MPC2, online with rewards).
- FOWM's per-component ablation magnitudes and λ curves are in figures only.
- I did not extract quantitative ablations for TD-MPC2's planning-vs-policy comparison (Fig. 9 only).
- WebSearch was unavailable in this session (model error), so discovery of other follow-ups (e.g., LOOP's actor-regularized control, beyond its role as a TD-MPC baseline) was not done.
