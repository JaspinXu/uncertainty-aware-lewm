# Uncertainty-aware learned policies (BC / action-chunking / diffusion): gating, runtime monitoring, proposal verification

Scope note: these notes cover primary sources (arXiv abstracts, arXiv HTML / ar5iv full texts) read on 2026-10-19. Numbers in tables were transcribed from the papers' HTML full text. Papers dated 2026 are recent preprints unless a venue is stated. arXiv full-text search was rate-limited (HTTP 429) late in the session, so 2025–2026 coverage is broad but may be incomplete.

## Q1. Interactive imitation with uncertainty gating (SafeDAgger, EnsembleDAgger, HG-DAgger, UAIL, LazyDAgger, ThriftyDAgger, Diff-DAgger and successors): what uncertainty do they use, and how well does it predict when the policy is wrong?

### Takeaway
Robot-gated DAgger methods ask for help using one of four signals: (i) a learned classifier that predicts how far the robot's action is from the expert's (SafeDAgger, LazyDAgger); (ii) action variance across an ensemble or MC-dropout, called "doubt" or "novelty" (EnsembleDAgger, HG-DAgger, UAIL, ThriftyDAgger); (iii) a learned probability of success, called "risk" (ThriftyDAgger); (iv) the policy's own training loss (Diff-DAgger). Thresholds come from data quantiles or a target intervention budget. Later work measured these gates directly as episode-failure predictors on multimodal diffusion-policy tasks. The ensemble and novelty gates of Ensemble-DAgger and Thrifty-DAgger did badly there. Their true-negative rates were often 0.02–0.31, so they fired on most successful episodes. Final policies trained with these gates often did worse than plain BC. A gate built on the diffusion loss reached 0.73–0.85 accuracy.

### Cited Findings
- **SafeDAgger (Zhang & Cho, arXiv 2016).** Extends DAgger with a learned "safety policy" that decides when to query the reference policy. Evaluated in a car-racing simulator, it needs fewer reference queries and converges faster; the authors conjecture the speed-up comes from automatic curriculum learning — [arXiv 1605.06450](https://arxiv.org/abs/1605.06450). EnsembleDAgger describes SafeDAgger's rule as a discrepancy rule: the novice acts only if ‖ā_nov − a_exp‖² < τ — [EnsembleDAgger full text](https://ar5iv.labs.arxiv.org/html/1807.08364).
- **EnsembleDAgger (Menda, Driggs-Campbell, Kochenderfer; IROS 2019).** Approximates a GP with an ensemble of networks. The novice acts only if both rules pass: the discrepancy rule (‖ā_nov − a_exp‖² < τ) and the doubt rule (ensemble action variance σ² < χ). The stated assumption is that doubt measures epistemic unfamiliarity and therefore serves as "a model-free proxy for proximity to failure states". Tested on inverted pendulum and MuJoCo HalfCheetah (5-member ensemble); the combined rule is "strictly more conservative" but sometimes improves on either rule alone — [arXiv 1807.08364](https://arxiv.org/abs/1807.08364); [full text](https://ar5iv.labs.arxiv.org/html/1807.08364).
- **HG-DAgger (Kelly et al., arXiv 2018).** The human decides when to intervene, but the method also learns a safety threshold τ on ensemble doubt. Rollouts started inside the estimated permissible set {doubt ≤ τ} had collision rate 0.607×10⁻³/m and road-departure rate 0.607×10⁻³/m. Rollouts started outside it had 7.533×10⁻³/m and 12.092×10⁻³/m. On the test vehicle (first 5,000 samples), HG-DAgger had 0 collisions and 0 road departures; BC had 11 and 66; DAgger had 11 and 11 — [arXiv 1810.02890](https://arxiv.org/abs/1810.02890); [full text, Tables I–II](https://ar5iv.labs.arxiv.org/html/1810.02890).
- **UAIL (Cui et al., ICRA 2019).** Uses MC-dropout uncertainty on the control output to hand control to the expert before the policy makes mistakes. In simulated driving, the authors report the uncertainty "can be leveraged to reliably predict infractions" (no AUROC was extracted here) — [arXiv 1905.02780](https://arxiv.org/abs/1905.02780).
- **LazyDAgger (Hoque et al., CASE 2021).**
  - Mechanism: a SafeDAgger-style discrepancy classifier f(s) hands control to the supervisor when f(s) ≥ 0.5 (predicted discrepancy above β_H). Control returns to the robot only when the actual discrepancy L(a, a_H) < β_R, with β_R = β_H/10 (HalfCheetah, Walker2D) or β_H/2 (Ant). This hysteresis is combined with noise injection during interventions — [full text, Alg. 1 & Table II](https://ar5iv.labs.arxiv.org/html/2104.00053).
  - Results: on 3 MuJoCo tasks it has 60% fewer context switches than SafeDAgger at similar performance. On a physical YuMi fabric task, execution-time success was 8/10 for LazyDAgger vs 2/10 for SafeDAgger and 0/10 for BC, with 21 vs 53 context switches.
  - Failure attribution: both of LazyDAgger's remaining failures were mode "D", where the policy did not request an intervention despite high ground-truth action discrepancy. In other words, the residual failures were misses by the uncertainty gate — [arXiv 2104.00053](https://arxiv.org/abs/2104.00053); [Table I](https://ar5iv.labs.arxiv.org/html/2104.00053).
- **ThriftyDAgger (Hoque et al., CoRL 2021 oral).**
  - Signals: Novelty is the per-dimension variance of bootstrapped-ensemble policy outputs, averaged over action dimensions. Risk is 1 − Q̂_G(s, π_r(s)), where Q̂_G estimates the discounted probability of reaching the goal set. It is trained by TD on supervisor transitions from offline data plus interventions.
  - Switching: hand to the human if Novelty > δ_h OR Risk > β_h. Risk is only trusted on non-novel states. Hand back if ‖π_r − π_h‖² < δ_r AND Risk < β_r.
  - Thresholds from a user budget α_h (the desired fraction of intervention timesteps): β_h and δ_h are the (1−α_h)-quantiles of risk and novelty over states visited by π_r. δ_r is the mean action discrepancy on supervisor states, and β_r is the median risk. α_h = 0.01 was used — [full text §4.3–4.4, §5.5](https://ar5iv.labs.arxiv.org/html/2109.08273).
  - Peg insertion (sim), autonomous success: BC 24/100, SafeDAgger 24/100, LazyDAgger 48/100, HG-DAgger 57/100, ThriftyDAgger without novelty 49/100, without risk 49/100, full ThriftyDAgger 73/100. Intervention-aided success was 20/20 for both ThriftyDAgger and HG-DAgger — [Table 1](https://ar5iv.labs.arxiv.org/html/2109.08273).
  - Other results: 100% intervention-aided success at execution time in both sim and physical cable routing. In a 10-user fleet study, human performance was 58% higher and robot performance 80% higher than the next-best algorithm — [arXiv 2109.08273](https://arxiv.org/abs/2109.08273).
- **Fleet-DAgger (CoRL 2022)** formalizes interactive fleet learning. It is motivated by commercial fleets that query remote teleoperators when robots are "at risk or unable to make task progress". It reports up to 8.8× higher return on human effort than baselines — [arXiv 2206.14349](https://arxiv.org/abs/2206.14349). **Sirius (RSS 2023)** is human-gated with trust-weighted BC: +8% success in sim and +27% on hardware — [arXiv 2211.08416](https://arxiv.org/abs/2211.08416).
- **Model-based runtime monitoring (Liu, Dass, Martín-Martín, Zhu; arXiv 2023).** Learns a latent dynamics model (cVAE) jointly with the BC policy, plus a failure classifier trained on deployment data.
  - Rule: sample N = 200 future latent rollouts over horizon 10; flag failure if more than β = 0.6 of the futures are classified as failures; flag OOD with a nearest-neighbor threshold α = 0.06 — [full text, Table I](https://ar5iv.labs.arxiv.org/html/2310.17552).
  - Combined human+robot success in round 3 vs ThriftyDAgger: Nut Assembly 100.0 vs 88.1; Threading 98.7 vs 77.5; real Coffee Pod 92.5 vs 57.5; real Gear Assembly 82.9 vs 70.0.
  - Overlap of predicted with human-labeled error intervals (IOU): 0.279 vs 0.153 (Nut Assembly) and 0.317 vs 0.044 (Threading) — [Tables II–III](https://ar5iv.labs.arxiv.org/html/2310.17552).
  - Abstract: success rates 23% (sim) and 40% (hardware) higher than baselines — [arXiv 2310.17552](https://arxiv.org/abs/2310.17552).
- **Sirius-Fleet (CoRL 2024).** A visual world model predicts outcomes of future actions. Anomaly predictors flag likely anomalies and adapt their criteria as autonomy improves, which reduces human requests over time (no numbers extracted) — [arXiv 2410.22689](https://arxiv.org/abs/2410.22689).
- **Diff-DAgger (Lee, Kang, Kuo; ICRA 2025).**
  - Signal: the diffusion training loss of the generated action. A state is called OOD when this loss exceeds the α-quantile of expected losses on the training data. Expert takeover happens only if the threshold is violated for K consecutive steps, which "significantly reduc[es] the rate of false positives" — [full text §III-C1](https://arxiv.org/html/2410.14868).
  - Gate quality over 100 rollouts (query/no-query vs episode outcome):
    - Stacking: Diff-DAgger TPR 0.88, TNR 0.77, accuracy 0.84; Ensemble-DAgger 0.81/0.31/0.59; Thrifty-DAgger 0.95/0.21/0.65.
    - Pushing (1 expert): 0.83/0.92/0.85 vs Ensemble 1.00/0.11/0.73 vs Thrifty 1.00/0.02/0.72.
    - Pushing (2 experts): 1.00/0.52/0.82 vs Ensemble 0.29/0.90/0.42 vs Thrifty 0.58/0.66/0.62.
    - Plugging: 0.68/0.82/0.73 vs Ensemble 0.22/0.27/0.37 vs Thrifty 0.26/0.31/0.21.
    - [Table II](https://arxiv.org/html/2410.14868)
  - Final success, state/image observations: Plugging BC 0.75/0.49, Ensemble-DAgger 0.69/0.37, Thrifty-DAgger 0.65/0.47, Diff-DAgger 0.92/0.84. Stacking BC 0.89/0.85, Ensemble 0.72/0.83, Thrifty 0.79/0.86, Diff-DAgger 0.94/0.94 — [Table III](https://arxiv.org/html/2410.14868).
  - Abstract: failure prediction +39.0%, task completion +20.6%, wall-clock 7.8× faster — [arXiv 2410.14868](https://arxiv.org/abs/2410.14868).
  - The authors attribute the ensemble methods' high false-negative rates to action variance from diverging valid plans (clockwise vs counter-clockwise pushing). This makes ensemble policies "insensitiv[e] toward true OOD data", and they speculate it is why those methods "perform worse than behavior cloning in many tasks" — [§IV-E](https://arxiv.org/html/2410.14868).
- **UPS: "When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering" (Yuan, Wu, Bajcsy; RSS 2026).**
  - Method: a VLM verifier is placed over diffusion-policy samples. Conformal prediction calibrates the VLM-plus-base-policy composition so the system chooses between executing a high-confidence action, asking a clarification question, or requesting an action intervention when the policy is "incapable". Residual learning then uses the collected interventions — [arXiv 2602.22474](https://arxiv.org/abs/2602.22474).
  - Hardware cup pick-and-place (20 trials each), value in the table's first ("TP") column for straightforward/ambiguous instructions:
    - Base policy: 0.35 / 0.35
    - EnsembleDAgger + residual: 0.40 / 0.50
    - HG-DAgger + residual: 0.50 / 0.55
    - FOREWARN: 0.80 / 0.50
    - UPS + clarification + residual: 0.90 / 0.80
  - The caption ties these figures to success rates in their Fig. 6; how to read the "TP" column is my transcription and was not independently re-verified — [Table I](https://arxiv.org/html/2602.22474).

### Inferences
- The data-quantile and budget-quantile threshold recipes (ThriftyDAgger, Diff-DAgger) set how often the gate fires. They say nothing about whether it fires at the right times. Gate quality (TPR/TNR against outcome) was rarely reported until Diff-DAgger, and when measured on multimodal tasks, ensemble-variance gates were close to "always on".
- ThriftyDAgger's risk term is a goal-conditioned success-probability critic. It is the closest classical analogue of a task-progress or value signal, and the ablation shows it adds as much as novelty (73/100 with both vs 49/100 with either alone). For a goal-conditioned BC prior, a learned "probability of reaching the goal from here" critic is a better-supported gate than state novelty.
- In LazyDAgger the gate's misses accounted for all residual failures. For a strong prior, the failures left over are, almost by construction, the cases the uncertainty signal does not see.

### Gaps
- I did not extract AUROC-style numbers for EnsembleDAgger, UAIL or SafeDAgger. Their original papers mostly report safety/performance curves rather than failure-prediction AUROC.
- No study found here evaluates these gates for a goal-conditioned BC prior that proposes chunks to a world-model planner (as opposed to handing control to a human).

## Q2. Runtime failure detection and monitoring for generative / chunked policies: which signals work best, with what AUROC / accuracy / success gains?

### Takeaway
For single-task diffusion/flow policies, the best failure-data-free signals are:
- temporal consistency of overlapping action chunks, measured with a statistical distance (Sentinel's STAC, Rewind-IL's TIDE, ActProbe's TCE);
- density/OOD scores on the policy's own observation embedding (FAIL-Detect's logpZO, RND; FIPER's RND-OE);
- the policy's loss or entropy over its action-chunk samples (DDPM loss, FIPER's action-chunk entropy).

Thresholds are set by conformal prediction on successful rollouts. Simple output variance is weak in multimodal tasks. Pure embedding-novelty detectors raise false alarms on benign OOD successes (TNR can drop to 0). Accuracy and AUROC are moderate: about 0.7–0.85 balanced accuracy or AUROC in most multi-task or harder settings; ≥ 0.9 only in a few single-task domains. Policy-level signals miss "task-progression" failures where the policy is consistently wrong (Sentinel STAC TPR 44%), which needs a progress monitor (VLM or value).

### Cited Findings
- **Sentinel (Agia et al., CoRL 2024).**
  - Failure taxonomy: "erratic" failures, caught by STAC (statistical temporal action consistency), and "task-progression" failures, caught by a VLM over video. Combined, Sentinel "detects 18% more failures than using either of the two detectors alone" — [arXiv 2410.04640](https://arxiv.org/abs/2410.04640).
  - STAC computes an MMD or KL distance between action distributions predicted at t and t+k on their overlapping horizon, using samples. It accumulates this as η_t and alarms when η_t > γ. γ is the 1−δ quantile of cumulative scores on successful calibration rollouts, which gives FPR ≤ δ — [full text](https://arxiv.org/html/2410.04640).
  - Close Box, combined ID+OOD (policy success 67%):
    - STAC-MMD: TPR 0.99, TNR 0.94, accuracy 0.96.
    - DDPM-loss score: accuracy 0.94.
    - Diffusion output variance: TPR 0.26, accuracy 0.72.
    - Policy-encoder embedding: OOD TNR 0.00, combined accuracy 0.78. CLIP and ResNet embeddings: OOD TNR 0.00.
    - GPT-4o, single image: accuracy 0.29. GPT-4o, video QA: accuracy 0.87.
    - Sentinel: TPR 1.00, TNR 0.87, accuracy 0.91.
    - Detection time (ID): STAC-MMD 7.2 s vs VLM 21.2 s.
    - [Table 1](https://arxiv.org/html/2410.04640)
  - In multimodal PushT, STAC is "the only method to achieve a balanced accuracy of over 90%". The DP's output variance does poorly "because both successes and failures can exhibit high-variance outputs in multimodal domains". Non-statistical temporal distances (min-distance) do worse than baselines — [§6](https://arxiv.org/html/2410.04640).
  - Task-progression failures: STAC TPR only 44%. STAC + VLM reaches 93% TPR with a 7% FPR increase — [§6](https://arxiv.org/html/2410.04640).
  - Real Push Chair: calibrated on only 10 successful rollouts, STAC TPR 0.80 / TNR 0.90, GPT-4o 0.90 / 1.00, Sentinel 1.00 / 0.90 (95% accuracy). Diffusion output variance: 0.60 / 0.90 — [Table 2](https://arxiv.org/html/2410.04640).
  - Execution horizon matters: k = 2 dropped STAC TPR to 61%, vs 78% at k = 4 and 95% at k = 8 (TNR > 95% throughout) — [App. C.1](https://arxiv.org/html/2410.04640).
- **FAIL-Detect (Xu et al., RSS 2025).**
  - Framing: failure detection as sequential OOD detection, trained on successful data only. Candidate scalar scores:
    - learned: logpZO (a new flow-based density score), lopO, NatPN, DER, RND, CFM;
    - post-hoc: SPARC, STAC, PCA-kmeans.
  - Thresholds: time-varying conformal-prediction bands calibrated on successes — [arXiv 2503.08558](https://arxiv.org/abs/2503.08558); [Table I](https://arxiv.org/html/2503.08558).
  - Average best balanced accuracy is about 78% in simulation and about 72% on hardware.
  - logpZO was top-1 in 10/16 sim cases and RND in 5/16; STAC was the best post-hoc score in 3/16.
  - STAC's detection time "consistently exceeds practical limits", longer than the average successful trajectory, and it needs 256 action samples per step.
  - On hardware, logpZO was top-1 in 8/12 scenarios — [§VI](https://arxiv.org/html/2503.08558).
  - Calibration sizes (CP significance 0.05): about 224–296 successes for the band mean in sim with the FM policy, versus 4–7 for one hardware task — [Table III](https://arxiv.org/html/2503.08558).
- **FIPER (Römer, Kobras, Worbis, Schoellig; NeurIPS 2025).**
  - Two indicators: OOD observations, scored by RND in the policy's embedding space (RND-OE), and a new action-chunk entropy (ACE). Both are CP-calibrated on a small set of successful rollouts. An alarm fires only when both, aggregated over short windows, exceed their thresholds — [arXiv 2510.09459](https://arxiv.org/abs/2510.09459).
  - Accuracy against baselines, by task:

    | Task | FIPER | Best other |
    |---|---|---|
    | Pretzel | 0.85 | STAC 0.67, logpZO 0.65 |
    | PushT | 0.71 | tied with STAC and ACE |
    | Stacking | 0.73 | PCA-kmeans 0.75 |
    | Sorting | 0.66 | logpZO 0.67 |
    | PushChair (TWA metric) | 0.83 | logpZO 0.78 |

    [Table 1](https://arxiv.org/html/2510.09459)
  - AND vs OR (CP band): AND gives TPR 0.72, TNR 0.84, accuracy 0.78; OR gives TPR 0.98 but TNR 0.18, accuracy 0.58 — [Table 2](https://arxiv.org/html/2510.09459).
  - Action-based scores have a smaller fail–success gap than observation-based ones: "failures are harder to detect from the policy outputs than from the inputs" — [§6](https://arxiv.org/html/2510.09459).
- **SAFE (Gu et al., NeurIPS 2025).** A multitask failure detector for VLAs that probes internal features, trained on both successes and failures and tested on unseen tasks. Simulation ROC-AUC averaged over OpenVLA, π0-FAST, π0 and π0* (seen / unseen tasks):

  | Detector | Seen | Unseen |
  |---|---|---|
  | SAFE-LSTM | 82.26 | 77.04 |
  | SAFE-MLP | 81.43 | 78.00 |
  | Cosine k-NN on embeddings | 81.28 | 73.93 |
  | Mahalanobis | 80.28 | 67.45 |
  | logpZO | 79.67 | 70.97 |
  | RND | 74.56 | 65.95 |
  | Action total variation, 10 samples | 71.33 | 70.76 |
  | STAC | 63.45 | 65.14 |
  | STAC-Single | 74.21 | 73.65 |
  | PCA-KMeans | 64.36 | 58.44 |
  | Token max-probability (token VLAs only) | 55.79 | 61.64 |

  [Table 1](https://arxiv.org/html/2506.09937); [arXiv 2506.09937](https://arxiv.org/abs/2506.09937).
- **Bidirectional Decoding (BID; Liu et al., 2024).** A test-time search that samples several chunks and picks one by backward coherence (agreement with the previous decision) and forward contrast (samples with high likelihood for future plans; ablated in the paper as positive vs negative sample sets). It "boosts the performance of recent policies by more than 32%" relative, and by 46% relative when combined with EMA, over seven simulation tasks — [arXiv 2408.17355](https://arxiv.org/abs/2408.17355); [full text](https://arxiv.org/html/2408.17355).
  - VQ-BeT Push-T, closed loop: 48.9 → 54.4 (no noise), 38.3 → 45.3 (noise 1.0), 29.5 → 31.7 (noise 1.5).
  - Sample count: 1 sample gives 49.1% at 13.2 ms; 32 samples give 54.4% at 26.8 ms — [Tables 1, 2, 4](https://arxiv.org/html/2408.17355).
- **Rewind-IL (2026).** TIDE (temporal inter-chunk discrepancy between overlapping chunks) with split-conformal calibration, plus rewinding to VLM-verified checkpoint states. On RoboCasa: ACT 56.7% → 73.3% success; detection balanced accuracy TIDE 0.93 vs FAIL-Detect 0.83 vs RND 0.85; total failure-handling overhead 22.0 ± 7.2 (their Table IV; units not captured) — [arXiv 2604.16683](https://arxiv.org/abs/2604.16683); [Tables III–IV](https://arxiv.org/html/2604.16683).
- **ActProbe (2026).** Uses only two action-space features from one forward pass: temporal consistency error between consecutive chunks (TCE) and action-chunk magnitude (ACM), fed into a task-conditioned LSTM-MLP. Reports +12.7% average hypervolume on the F1-vs-timeliness Pareto front and +9.0% early-detection ROC-AUC on unseen tasks — [arXiv 2606.08508](https://arxiv.org/abs/2606.08508).
- **PATCH (2026)** is a world-model-style monitor conditioned on the active action chunk. It predicts latent patch evolution inside a projected "execution corridor", accumulates residuals not explained by the robot's own motion, and triggers pause → recovery → resume. Claims more stable, context-relevant triggers than competing monitors (qualitative abstract) — [arXiv 2606.16690](https://arxiv.org/abs/2606.16690).
- **TUD (NeurIPS 2026)**, used as a signal-validation study (details in Q5):
  - Re-invocation criteria compared at matched call budgets on VLA-Arena L0:

    | Signal | Success, ~17–18 calls/episode | Success, ~8–10 calls/episode |
    |---|---|---|
    | Dispersion of chunk re-predictions (TUD) | 86.7% | 82.8% |
    | Diff-DAgger flow loss | 83.7 | 80.7 |
    | BayesDiff | 83.5 | 77.0 |
    | Visual-feature distance | 83.0 | 79.8 |
    | AAC entropy | 82.7 | 77.2 |
    | MC aleatoric variance | 80.2 | 73.5 |
    | Posterior-ensemble dispersion (GenUQ) | 74.7 | 73.8 |

  - Abstract: the dispersion signal "more consistently separates successful from failed rollouts" than the alternatives.
  - [Table 4](https://arxiv.org/html/2610.05025); [arXiv 2610.05025](https://arxiv.org/abs/2610.05025)
- **Conformal / PAC failure predictors.** A PAC-Bayes failure predictor from observation histories, with guaranteed false-positive/false-negative bounds that "closely match empirical error rates" (drone navigation, grasping) — [Farid et al., arXiv 2202.05894](https://arxiv.org/abs/2202.05894). ReconVLA applies conformal prediction to VLA action tokens and to robot state for failure anticipation — [arXiv 2604.16677](https://arxiv.org/abs/2604.16677). KnowNo uses conformal prediction for ask-for-help with LLM planners, with task-completion guarantees while minimizing help — [arXiv 2307.01928](https://arxiv.org/abs/2307.01928).

### Inferences
- Across Sentinel, FAIL-Detect, FIPER, SAFE and Rewind-IL, the successful failure-data-free signals are policy-internal: consistency between the policy's own successive plans, density in the policy's embedding, and the policy's loss or entropy. None of these rely on an external dynamics model's epistemic uncertainty on the executed actions.
- Performance depends strongly on the setting. STAC is 0.96 accuracy on single-task Close Box but 0.63–0.65 AUROC in SAFE's multi-task VLA benchmark, and FAIL-Detect found its detection time impractical. Results from one benchmark should not be expected to transfer.
- Realistic AUROC or balanced accuracy for good detectors is about 0.7–0.85. A project-specific detector at 0.21–0.57 is far below this, i.e., near chance or worse.

### Gaps
- Few papers report AUROC directly. Sentinel, FAIL-Detect and FIPER report TPR/TNR/accuracy at a CP threshold, which is not comparable to AUROC.
- No study found evaluates detectors for goal-conditioned BC policies operating in a frozen world model's latent space, or for OGBench-style long-horizon "move" tasks.

## Q3. Policy-as-proposal plus model/value-based verification with uncertainty: sampling several actions from a policy and selecting or rejecting them with a learned model, value or verifier

### Takeaway
"Sample from the prior, score with a model or critic, execute the best" is now a mature pattern. Lineage:
- offline MBRL: MBOP, MOPP, D-MPC;
- offline RL with generative behavior models: EMaQ, SfBC, IDQL, PA-RL;
- value or VLM steering of generalist policies: V-GPS, RoboMonkey, FOREWARN, UPS, DreamSteer, Pre-VLA, TeV, DiVeR.

Uncertainty usually enters as pruning or pessimism over model rollouts, as in MOPP's ensemble-disagreement filter, rather than as a trigger. Gains are large when the base policy is weak or OOD: +25–60 points, e.g., DreamSteer 23.75% → 66.25%. They are small when the base is strong, e.g., V-GPS on OpenVLA in SIMPLER 0.24 → 0.27. Scoring can also hurt some tasks. Pure search without the policy's proposals can fail completely: DreamSteer with motion primitives only scored 0%.

### Cited Findings
- **MBOP (Argenson & Dulac-Arnold, 2020).**
  - Planner: BC policy f_b, a learned ensemble dynamics model, and a truncated value f_R.
  - Each of the N rollouts uses one ensemble head throughout. Actions are a_t = f_b(s_t, a_{t−1}) + ε with ε ~ N(0, σ²), β-mixed with the previous plan. The output is a return-weighted (κ) average trajectory — [full text, Alg. 2](https://ar5iv.labs.arxiv.org/html/2008.05556).
  - Ablations: replacing the BC prior with a Gaussian prior (MBOP-NOPP) or removing the value function hurts. "The BC prior when used alone can perform well on certain environments, but on others it stagnates at behavior policy's performance."
  - Speed: 106 Hz (h = 4) to 40 Hz (h = 40), vs 362 Hz for BC alone — [§4.4](https://ar5iv.labs.arxiv.org/html/2008.05556).
- **MOPP (Zhan, Zhu, Xu; IJCAI-ECAI 2022).**
  - Rollouts are guided by the behavior policy: sample from N(μ_a, (σ_M/max σ_a · σ_a)²) and pick argmax Q_b among m samples per step.
  - Rollouts are then pruned by ensemble disagreement disc(s, a) = max_{i,j} ‖f^i(s, a) − f^j(s, a)‖². Any trajectory with a step above threshold L is removed, and at least N_m = 0.2N lowest-cumulative-uncertainty trajectories are kept — [full text §4.3](https://ar5iv.labs.arxiv.org/html/2105.07351).
  - Ablation: raising σ_M from 0.01 to 0.5 helps, but σ_M = 1.0 is "detrimental", and pruning "is no longer effective" when most rollouts are problematic — [§5.2](https://ar5iv.labs.arxiv.org/html/2105.07351).
  - D4RL comparisons with MBOP are mixed: walker2d-medium 80.7 vs 41.0, but hopper-medium 31.8 vs 48.8 — [Table 1](https://ar5iv.labs.arxiv.org/html/2105.07351).
- **D-MPC (TMLR)** learns a diffusion multi-step action proposal and a diffusion dynamics model, combined in online MPC. It reports D4RL performance significantly better than MBOP — [arXiv 2410.05364](https://arxiv.org/abs/2410.05364).
- **Generative behavior models plus a critic.**
  - EMaQ restricts the extracted policy "exactly within the support of a given behavior policy" — [arXiv 2007.11091](https://arxiv.org/abs/2007.11091).
  - SfBC decouples an expressive generative behavior model from an action-evaluation model and uses in-sample planning — [arXiv 2209.14548](https://arxiv.org/abs/2209.14548).
  - IDQL samples from a diffusion behavior policy and resamples with critic-derived weights — [arXiv 2304.10573](https://arxiv.org/abs/2304.10573).
  - PA-RL applies a supervised loss on "optimized" actions to any policy class — [arXiv 2412.06685](https://arxiv.org/abs/2412.06685).
- **V-GPS (Nakamoto et al., CoRL 2024).**
  - Re-ranks K ∈ {10, 50} sampled actions from a generalist policy with an offline-RL Q-function (Cal-QL or IQL), using softmax temperature β ∈ {0, 0.1, 1.0} — [Table 4](https://arxiv.org/html/2410.13816).
  - Real-world with Octo-small-1.5: average 0.24 → 0.44 (+82.8%).
  - SIMPLER totals: Octo-s 0.27 → 0.34; Octo-b 0.17 → 0.18; Octo-s-1.5 0.02 → 0.18; RT-1-X 0.12 → 0.14; OpenVLA 0.24 → 0.27.
  - Some tasks get worse, e.g., Octo-s spoon-on-towel 0.52 → 0.46 — [Tables 1–2](https://arxiv.org/html/2410.13816); [arXiv 2410.13816](https://arxiv.org/abs/2410.13816).
- **GPC (Qi et al., RA-L).** A frozen diffusion policy plus an action-conditioned world model trained on demonstrations and random exploration rollouts. It ranks (RANK) and/or gradient-refines (OPT) the policy's proposals — [arXiv 2502.00622](https://arxiv.org/abs/2502.00622). Vision Push-T IoU and time per decision:

  | Variant | IoU | Time per decision |
  |---|---|---|
  | BC | 0.642 | 0.46 s |
  | RANK, K = 50 | 0.698 | 5.8 s |
  | RANK, K = 100 | 0.739 | 11.7 s |
  | OPT, M = 25 | 0.791 | 39 s |
  | RANK + OPT (K = 10, M = 25) | 0.882 | 374 s |

  [Table III](https://arxiv.org/html/2502.00622)
- **FOREWARN (Wu, Tian, Swamy, Bajcsy; 2025).** A latent world model "imagines" outcomes of diverse policy plans; a VLM aligned to the latent states narrates and filters them.
  - Success (20 rollouts), Cup/Bag/Fork: base 0.30/0.20/0.10 → 0.80/0.70/0.70. With novel task descriptions: 0.50/0.40/0.30 → 0.80/0.70/0.60.
  - Total time 3.7 s per decision, of which the world model takes 0.1 s.
  - [Tables II–III](https://arxiv.org/html/2502.01828); [arXiv 2502.01828](https://arxiv.org/abs/2502.01828)
- **RoboMonkey (Kwok et al., 2025).** Samples a few VLA actions, applies Gaussian perturbation and majority voting to form a proposal distribution, then selects with a 7B VLM-based action verifier trained on synthetic data. It finds an exponentiated power law between action error and number of samples — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811).
  - LIBERO-Long: OpenVLA 49.8% → 56.5%.
  - Real OOD WidowX tasks: 60% vs 35% (OpenVLA) and 30% (V-GPS) — [Table 1, §5.3](https://arxiv.org/html/2506.17811).
  - Verifier latency: 0.35 s at batch size 16 — [Table 3](https://arxiv.org/html/2506.17811).
- **DreamSteer (2026).** Candidate chunks come from the VLA and from predefined motion primitives; an action-conditioned latent world model imagines outcomes; a language-conditioned value model ranks them — [arXiv 2607.02865](https://arxiv.org/abs/2607.02865).
  - OOD objects (80 trials):

    | Configuration | Success |
    |---|---|
    | π0 alone | 23.75% |
    | π0 + DreamSteer | 42.50% |
    | π0 + primitives + DreamSteer | 66.25% |
    | Primitives + DreamSteer, no π0 proposals | 0% |
    | π0 + primitives with random selection | 0% |

  - Instruction following: 38.75% → 56.25% — [Tables 2–3](https://arxiv.org/html/2607.02865).
- **Pre-VLA (2026).** A verifier predicts a safety confidence and a critic-derived advantage for candidate action chunks before execution or world-model imagination. A "preemptive resampling scheduler" under a compute budget triggers resampling. LIBERO success 30.79% → 37.62% over RynnVLA-002, at 183.9 ms of verification per chunk — [arXiv 2605.22446](https://arxiv.org/abs/2605.22446).
- **Other 2025–2026 verifier work.**
  - Latent Policy Steering uses an optical-flow-pretrained world model plus a value function to evaluate the base policy's action candidates — [arXiv 2507.13340](https://arxiv.org/abs/2507.13340).
  - TeV trains an energy-based verifier on observation-action history so chunks are judged as trajectory continuations, and also guides flow samples — [arXiv 2609.39038](https://arxiv.org/abs/2609.39038).
  - DiVeR estimates "decision criticality" from the dispersion of sampled action representations and reweights verifier training toward those states — [arXiv 2610.04933](https://arxiv.org/abs/2610.04933).
  - EVE uses zero-shot VLM verifiers with classifier-guidance fusion — [arXiv 2512.21430](https://arxiv.org/abs/2512.21430).
  - "Flow-Support-Weighted Proposals" identifies proposal contraction: fine-tuned flow VLAs produce near-duplicate samples. Their training-time fix raises oracle best-of-8 success on RoboCasa from 57.6% to 63.0% and improves fixed verifiers by 2.4–4.9 points — [arXiv 2610.14588](https://arxiv.org/abs/2610.14588).
- **Arbitration between a prior controller and a learned policy using both systems' uncertainty.** Bayesian Controller Fusion fuses uncertainty-aware distributional outputs of a hand-crafted controller and an RL policy to arbitrate control (IJRR 2023) — [arXiv 2107.09822](https://arxiv.org/abs/2107.09822).

### Inferences
- When dynamics-model uncertainty appears in this lineage (MOPP, MBOP's per-head rollouts, MOReL-style discrepancy), it is a support/OOD filter that stops the planner from exploiting model errors under aggressive sampling. It is not a predictor of task failure. If proposals come from a BC prior and stay in-support, the filter should seldom fire. Tiny world-model disagreement on prior proposals is therefore the expected outcome, not evidence that the model is reliable on long-horizon outcomes.
- Gains scale inversely with base-policy quality:
  - large where the base is weak or OOD (V-GPS Octo-s-1.5 0.02 → 0.18; RoboMonkey OOD +25 points; FOREWARN +30–60 points; DreamSteer +42.5 points);
  - small near the ceiling (V-GPS OpenVLA SIMPLER +3 points; AAC on π0.5 +0.9 points, see Q4).
  - A 94–98% prior leaves 2–6 points of headroom, so verification can only claim small absolute gains and could lose points through verifier errors.
- DreamSteer's 0% for primitives-only search, MOPP's σ_M = 1.0 collapse and MBOP's NOPP ablation all point the same way. Search should be anchored on the prior's proposals; the planner adds value by re-ranking or locally refining them, not by replacing them.

### Gaps
- None of the verification papers found report whether the verifier's uncertainty (ensemble spread of Q or world-model disagreement over the K proposals) predicts when the selected action will fail. They report success deltas only.
- I could not confirm from primary text how TD-MPC2 mixes policy-prior samples into its MPPI; the abstract only describes local trajectory optimization in the latent space of an implicit world model — [arXiv 2310.16828](https://arxiv.org/abs/2310.16828). Treat any "TD-MPC2 uses a policy prior" claim as unverified here.
- I found no paper named "uncertainty-triggered verification". Its closest instantiations are TUD, RAIN, Pre-VLA's preemptive resampling, and UPS's act/ask/learn selection.

## Q4. Aleatoric (multimodality) vs epistemic uncertainty for BC policies: does policy variance or multimodality predict failures, especially at contact transitions?

### Takeaway
There is consistent evidence that raw action variance, sample dispersion or ensemble disagreement mixes aleatoric multimodality (several valid ways to act) with epistemic error. In multimodal tasks it fires on successes as often as on failures:
- Sentinel: diffusion output variance accuracy 0.72 on Close Box; poor on PushT;
- Diff-DAgger: Ensemble-DAgger and Thrifty-DAgger TNR 0.02–0.31;
- SAFE: sample-consistency AUROC ≈ 0.71;
- TUD: MC aleatoric variance and posterior-ensemble dispersion are the two weakest triggers.

Signals that work instead do one of three things: compare the policy's distributions over time with a statistical distance (STAC, TIDE, TUD dispersion of re-predictions), score the action under the policy's own loss or entropy (Diff-DAgger, FIPER-ACE), or AND-combine action uncertainty with an observation-OOD score (FIPER). For contact transitions specifically, the evidence is qualitative only.

### Cited Findings
- Diff-DAgger: ensemble methods "often misinterpret policy disagreements as uncertainty at multi-modal decision points". Expressive ensembles "may disagree due to the presence of multiple viable strategies where each policy is certain", and for action sequences ensemble uncertainty "may explode due to divergence over time" — [arXiv 2410.14868](https://arxiv.org/abs/2410.14868); [§II-B](https://arxiv.org/html/2410.14868).
- Sentinel: "both successes and failures can exhibit high-variance outputs in multimodal domains". Diffusion output variance had TPR 0.26 on Close Box. Temporal consistency with non-statistical distances "omits action multimodality" and underperforms — [§6, Table 1](https://arxiv.org/html/2410.04640). Sentinel's hypothesis is that a predictive policy works like an internal world model; if its prediction of s_{t+k} made at time t is wrong, the action distributions at t and t+k diverge — [App. A.1](https://arxiv.org/html/2410.04640).
- SAFE (VLAs): sample-consistency signals over 10 samples reach ROC-AUC 71.33/70.76 (action total variation) and 68.71/69.44 (cluster entropy). Embedding-distance and feature-probe signals reach about 80–82. Token-probability uncertainty is about 48–62 — [Table 1](https://arxiv.org/html/2506.09937).
- FIPER: failures are "associated with successive OOD observations and high-uncertainty conditional action distributions". Action-chunk entropy alone separates "Success OOD" from "Fail ID" better than RND-A or STAC, but action-based scores have a smaller fail–success gap than observation-based ones. AND-gating raises TNR from 0.18 (OR) to 0.84 — [§6–7, Table 2](https://arxiv.org/html/2510.09459).
- TUD at matched budgets: intra-step Monte-Carlo aleatoric variance (80.2 / 73.5% success) and posterior-ensemble "generative uncertainty" (74.7 / 73.8%) did worse than the cross-step re-prediction dispersion (86.7 / 82.8%) and also worse than the flow-loss signal (83.7 / 80.7%) — [Table 4](https://arxiv.org/html/2610.05025).
- Action chunking trades consistency for reactivity. BID shows that longer action horizons capture expert temporal dependencies in deterministic settings, but horizon 1 is best under high stochasticity (total-variation distances in their Table 3). Their test-time search exploits this to choose among multimodal samples — [§3, §5](https://arxiv.org/html/2408.17355).
- AAC (CVPR 2026) uses action entropy to adapt chunk size at inference. Gains are modest:
  - GR00T on RoboCasa: 59.7% (fixed h = 16) → 62.0%; a fixed h = 8 reaches 61.2%.
  - GR00T on LIBERO: 94.1% → 95.0%.
  - π0.5 on LIBERO: 97.0% → 97.9%.
  - [arXiv 2604.04161](https://arxiv.org/abs/2604.04161); [Tables 1–2](https://arxiv.org/html/2604.04161)
- "Decision-critical" states are sparse. DiVeR argues that at most states plausible sampled actions are similar, and only a sparse set of states "admits meaningfully different actions that can substantially affect downstream outcomes". It identifies them through dispersion of sampled action representations — [arXiv 2610.04933](https://arxiv.org/abs/2610.04933).
- Fine-tuning shrinks the aleatoric spread. Fine-tuned flow VLAs show "proposal contraction" (repeated samples differ marginally), so sample-based uncertainty and best-of-K selection both lose signal — [arXiv 2610.14588](https://arxiv.org/abs/2610.14588).
- Familiar but risky states are missed by novelty. ThriftyDAgger motivates its risk critic with "bottleneck" regions that have low novelty but low tolerance for error. Removing either signal drops autonomous success from 73/100 to 49/100 — [§4.2, Table 1](https://ar5iv.labs.arxiv.org/html/2109.08273).
- Qualitative contact-transition examples:
  - Diff-DAgger asks for help "once the robots start to lose grip of the charger" while the other methods miss it — [§IV-E](https://arxiv.org/html/2410.14868).
  - FAIL-Detect's logpZO flags failure "when the square slips from the gripper" and when both arms drop the hammer — [§VI, Fig. 7](https://arxiv.org/html/2503.08558).
  - RoboMonkey's verifier mainly fixes "imprecise grasping, task progression failures, and collisions" — [§5.3](https://arxiv.org/html/2506.17811).

### Inferences
- The aleatoric-vs-epistemic confusion cuts in both directions for a goal-conditioned BC prior. At multimodal decision points (which cube to approach, which side to grasp), sample variance is high on successes. On confidently wrong long-horizon plans it is low, which is Sentinel's "task-progression" category. Neither case is helped by world-model dynamics uncertainty on in-support proposals.
- Signals that respect multimodality are distribution-level comparisons across time (MMD between overlapping chunks, dispersion of re-predictions of the same future slot) and likelihood or loss of the executed chunk under the policy. These are the natural candidates to replace world-model ensemble disagreement as a trigger.

### Gaps
- I found no quantitative study showing that BC action variance specifically at contact transitions (grasp onset, release, push contact) predicts episode failure; only qualitative examples. This needs dedicated analysis, e.g., aligning the signal to contact events in Cube rollouts.
- No primary source found decomposes aleatoric vs epistemic uncertainty for a goal-conditioned BC policy on latent world-model features.

## Q5. Concrete recipes for an "uncertainty-aware" controller built from a BC prior plus a frozen world model: when to invoke search, how far to search from the prior, how to set thresholds without test-set tuning

### Takeaway
The best-supported 2024–2026 recipe treats uncertainty as a compute-allocation trigger:
1. Run the cheap prior by default.
2. Invoke search, refinement or a slow system when a policy-side signal fires. Candidates: dispersion of chunk re-predictions, inter-chunk inconsistency, action-chunk entropy, a predicted error ensemble, or a stalled-progress or value signal.
3. Calibrate the threshold on successful rollouts of the deployed system with a quantile or CP rule, add a persistence or hysteresis requirement, and cap the time between calls.
4. Search locally around the prior's proposals, by ranking K samples plus small perturbations or a few refinement steps, rather than globally.
5. Report success against compute or calls at matched budgets.

TUD and RAIN show that this framing yields clear wins: 75% fewer slow-system calls with higher success, and 26.6% fewer refiner calls at equal success. Theory and evidence suggest the benefit is bounded by the trigger's discrimination: the gain over random deferral scales with the gate's AUROC.

### Cited Findings
- **When to invoke search (uncertainty-triggered compute).**
  - TUD (NeurIPS 2026):
    - Signal: u_t, the mean per-dimension std of the fast specialist's re-predictions of the same upcoming chunk slot over a sliding buffer of E re-predictions (default 5).
    - Rule: call the slow generalist when the running mean of u_t exceeds a percentile threshold θ_p, or when Δ_max steps have passed since the last call. A Lipschitz bound ties the skip error to u_t plus an observation-drift term — [§3.2–3.3](https://arxiv.org/html/2610.05025).
    - Real robot (30 rollouts per method):

      | Method | Success | Calls per episode |
      |---|---|---|
      | Single-system π0 | 0.433 | 49.0 |
      | Fixed interval Δ = 9 | 0.467 | 48.5 |
      | TUD p50 | 0.767 | 24.5 |
      | TUD p90 | 0.833 | 12.1 |

      [Table 5](https://arxiv.org/html/2610.05025)
    - Abstract: generalist calls cut by 75% versus the strongest fixed-interval baseline — [arXiv 2610.05025](https://arxiv.org/abs/2610.05025).
  - RAIN (2026): an ensemble predicts, before any refinement, the error of a lightweight policy's chunk at several refinement levels, then decides whether and how much to refine. Calibration "balances uncertainty, computation, and temporal consistency".
    - Simulation: 66.6% success, approaching the best diffusion policy with 5.71× lower latency.
    - Real robot: ACT 55.0% → 72.5%.
    - Matches the strongest fixed-refinement success with 26.6% fewer refiner evaluations.
    - [arXiv 2610.19765](https://arxiv.org/abs/2610.19765)
  - "When to Plan" (Labiosa & Hanna, RLC 2026): a meta-policy learned by RL chooses between reactive, short, medium and long planning. It conditions on the reactive policy's uncertainty (the reactive policy is an M = 4 network ensemble trained by BC on planner trajectories), distance to goal, and the previous meta-action — [§5, §6.2](https://arxiv.org/html/2607.16421).
    - It beats every fixed strategy in all 5 domains, e.g., Box Push return −24.3 vs −44.7 for the best fixed plan length, about 2× better in Box Push and 2.5× in Fetch.
    - In Fetch it chooses reactive control about 77% of the time — [Tables 1–2](https://arxiv.org/html/2607.16421).
    - It "shift[s] toward fully reactive control as the reactive policy improves" — [arXiv 2607.16421](https://arxiv.org/abs/2607.16421).
  - "System Switch" (2026, Doom with a reasoning VLM): offline, deferring the least-confident 30% of decisions to a reasoning model gains over random deferral "in proportion to the actor's AUROC" (rank correlation 0.87). Accuracy, calibration and sensitivity are distinct properties of the actor — [arXiv 2610.09683](https://arxiv.org/abs/2610.09683).
  - AAC, the entropy-driven adaptive chunk length, gives small but consistent gains on strong VLAs (+0.9 to +2.3 points) — [Tables 1–2](https://arxiv.org/html/2604.04161).
- **How to set thresholds without test-set tuning.**
  - Budget quantiles (ThriftyDAgger): the switch-in threshold is the (1−α_h)-quantile of risk or novelty over states the robot policy has visited, given a target intervention fraction α_h (0.01). The switch-back thresholds are the mean action discrepancy and the median risk — [§4.4](https://ar5iv.labs.arxiv.org/html/2109.08273).
  - Training-loss quantiles plus persistence (Diff-DAgger): α-quantile of the policy's training-set losses, and K consecutive violations — [§III-C1](https://arxiv.org/html/2410.14868).
  - Conformal calibration on successful policy rollouts (Sentinel): γ is the 1−δ quantile of cumulative scores (leave-one-trajectory-out for embedding baselines), which gives FPR ≤ δ. Two cautions from the paper:
    - Calibrating on demonstration data "led to a significant increase in the detector's FPR" because of covariate shift.
    - Calibration rollouts that succeed in undesired ways must be excluded — [App. B.3–B.4](https://arxiv.org/html/2410.04640).
  - Time-varying CP bands (FAIL-Detect, significance 0.05): typical simulation calibration used a few hundred successful rollouts. STAC's single time-invariant threshold was a stated weakness — [Table III, §VI](https://arxiv.org/html/2503.08558).
  - Small-sample CP with AND-combination (FIPER): a "small set of successful rollouts"; window aggregation; alarm only when the observation-OOD and action-entropy scores both exceed threshold (AND TNR 0.84 vs OR 0.18) — [arXiv 2510.09459](https://arxiv.org/abs/2510.09459); [Table 2](https://arxiv.org/html/2510.09459).
  - Hysteresis (LazyDAgger): switch back only below a much lower threshold (β_R = β_H/10 or β_H/2), cutting context switches by about 60% — [Table II](https://ar5iv.labs.arxiv.org/html/2104.00053).
  - Percentile operating curves (TUD): sweeping one percentile p traces the whole cost-success curve, and p70 is used as a representative default — [App. G](https://arxiv.org/html/2610.05025).
  - Conformal selection among actions (UPS): conformal prediction calibrates when to execute, clarify, or request an intervention — [arXiv 2602.22474](https://arxiv.org/abs/2602.22474).
- **How far to search from the prior.**
  - MBOP: BC-prior sampling with small Gaussian noise and β-mixing with the previous plan. A Gaussian prior (NOPP) or no prior (PDDM) is worse — [Alg. 2, §4.4](https://ar5iv.labs.arxiv.org/html/2008.05556).
  - MOPP: moderately widening the prior's std (σ_M = 0.5) helps; σ_M = 1.0 is harmful; ensemble-disagreement pruning keeps at least 20% of rollouts — [§4.3, §5.2](https://ar5iv.labs.arxiv.org/html/2105.07351).
  - GPC: ranking 50–100 prior samples gives +0.06 to +0.10 IoU on Push-T; adding gradient refinement gives +0.24 IoU but costs about 800× more time per decision — [Table III](https://arxiv.org/html/2502.00622).
  - RoboMonkey: a small sample set plus Gaussian perturbation plus majority voting forms the proposal set for a verifier — [arXiv 2506.17811](https://arxiv.org/abs/2506.17811).
  - V-GPS: K = 10–50 samples with softmax or greedy selection by Q — [Table 4](https://arxiv.org/html/2410.13816).
  - DreamSteer: policy proposals are necessary; primitive-only candidates give 0% — [Table 2](https://arxiv.org/html/2607.02865).
  - BID: 8–32 samples with coherence and contrast criteria; about 2× inference time — [Table 2](https://arxiv.org/html/2408.17355).
- **World-model-side monitoring options for a frozen world model.**
  - Simulate many futures from the policy's proposals and act on the predicted failure fraction (model-based runtime monitoring: 200 futures, horizon 10, β = 0.6) — [Table I](https://ar5iv.labs.arxiv.org/html/2310.17552).
  - Monitor world-model "innovation", meaning residuals between predicted and observed latents not explained by the robot's own chunk, and pause or recover — [PATCH, arXiv 2606.16690](https://arxiv.org/abs/2606.16690).
  - Use a VLM or value function over world-model-imagined outcomes for progress or semantic checks — [FOREWARN](https://arxiv.org/abs/2502.01828); [DreamSteer](https://arxiv.org/abs/2607.02865).

### Inferences
These connect the literature to the project context: a 94–98% BC prior; world-model ensemble disagreement AUROC 0.21–0.57; remaining failures are long-horizon "move" tasks.

- **Viability.** An uncertainty-aware story is still viable, but its claim should change.
  - The evidence does not support "world-model epistemic uncertainty on prior proposals predicts failure". Expected in-support behavior (MOPP/MBOP) and the AUROC-proportional gain result (System Switch) both suggest that a gate near AUROC 0.5 can do no better than random triggering.
  - It does support claims of the form "match full-search success at a fraction of planner calls", tested with matched-budget curves and random- and fixed-interval-trigger controls (TUD Table 4/5, RAIN). Selective prediction with CP guarantees is the second option (Sentinel, FAIL-Detect, FIPER).
- **Better triggers to test** in place of world-model disagreement, all computable from the BC prior and the frozen world model:
  - TUD-style dispersion of re-predicted chunks for the same future slot;
  - STAC/TIDE-style MMD between overlapping chunks;
  - FIPER-style action-chunk entropy over K prior samples, AND-ed with an RND or density score on the prior's latent input;
  - Diff-DAgger-style prior loss or likelihood of the chosen chunk;
  - for the long-horizon "move" failures, a progress-stall signal, e.g., the trend of predicted latent goal-distance, or a ThriftyDAgger-style goal-reaching critic. Sentinel found consistency signals catch only 44% of such task-progression failures.
- **Search radius.** Keep search anchored on the prior: rank K prior samples (BID/V-GPS/GPC-RANK); add MOPP/MBOP-style small perturbations; use world-model disagreement only as a pruning filter against model exploitation, not as a trigger. Aggressive or global search (σ_M = 1.0; primitive-only candidates) is consistently harmful in the sources.
- **Threshold protocol** that respects a locked, pre-registered workflow:
  - Calibrate on a held-out set of successful prior rollouts from the same deployment distribution, not on demonstrations.
  - Fix the target false-alarm or trigger rate δ or α in advance (e.g., 5–10%, or a planner-call budget).
  - Require K-step persistence or windowed aggregation.
  - Report the full percentile sweep as an operating curve plus one pre-registered operating point, with paired tests against random triggering at the same call rate.
- **Statistical-power caution.** At 94–98% base success, a few hundred episodes contain only about 6–30 failures. AUROC estimates and success deltas will have wide confidence intervals, so paired, same-seed comparisons and pooled non-trivial episodes are needed. This is my reasoning, not a sourced claim.

### Gaps
- No primary source found implements exactly "frozen latent world model + goal-conditioned BC chunk prior + uncertainty-triggered search" with a failure-prediction evaluation. TUD, RAIN, GPC, FOREWARN and DreamSteer are the closest analogues, each missing one ingredient.
- Most 2026 sources (Rewind-IL, ActProbe, PATCH, RAIN, Pre-VLA, DreamSteer, TeV, DiVeR, ADS, System Switch) are recent preprints without confirmed peer review. TUD (NeurIPS 2026), UPS (RSS 2026), When-to-Plan (RLC 2026) and AAC (CVPR 2026) list acceptances on arXiv.
- I found no direct evidence on how far from a near-ceiling (>95%) prior a world-model planner can search before model exploitation outweighs gains. Reported sweeps (MOPP σ_M) are on D4RL locomotion, not goal-conditioned manipulation.
