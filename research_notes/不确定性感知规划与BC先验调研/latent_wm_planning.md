# Latent world-model planning from pixels (DINO-WM, V-JEPA 2-AC, PLDM, LeWM and 2026 follow-ups): failure modes, planning vs. goal-conditioned policies, uncertainty-aware and value-guided variants

*Research date 2026-10-19. Sources: arXiv abstract and HTML full-text pages, the arXiv API, and the CLEAR-LeWM project page and GitHub. Most 2026 items are arXiv preprints with no stated venue. A venue is given only where the arXiv comments or a citing paper state one. Numbers were taken from the arXiv HTML tables. Where the LaTeX-to-HTML rendering duplicated or padded digits, I say so and decoded them to match the paper's own text. Spot-check against the PDFs before quoting them in the thesis. WebSearch/WebFetch were unavailable (model error), and arXiv search and Semantic Scholar were intermittently rate-limited (HTTP 429). So coverage of OpenReview-only or workshop-only papers is thin.*

## Q1. How DINO-WM, V-JEPA 2-AC, PLDM and LeWM plan, and the failure modes reported for CEM/MPPI with latent-distance costs

### Takeaway
All four methods use the same recipe. A frozen action-conditioned latent predictor is optimized with a sampling-based planner: CEM, or MPPI for PLDM. Each candidate action sequence is scored by its distance to the encoded goal image. The cost is terminal MSE for DINO-WM and LeWM, L1 for V-JEPA 2-AC, and for PLDM a per-step distance summed over the horizon plus an optional ensemble-variance penalty. Horizons are short (LeWM/DINO-WM: 5 action blocks × frameskip 5 = 25 environment steps; V-JEPA 2-AC uses horizon 1 in its reported comparison), and all run under receding-horizon MPC. The 2026 literature on these planners reports four recurring failures:
- Myopic or non-monotonic latent distance at long goal offsets.
- Poor ranking of candidate actions. On OGBench-Cube, the encoded-goal cost is reported as uninformative.
- The optimizer exploiting model errors: gradient planners, raw-action sampling, and unconstrained high-level search.
- Strong sensitivity of the headline numbers to the evaluation protocol.

### Cited Findings

#### DINO-WM (Zhou, Pan, LeCun, Pinto; arXiv 2411.04983, v1 Nov 2024; LeWM's reference list gives the venue as ICML 2025)
- The predictor forecasts future DINOv2 patch features from a frozen encoder. Planning treats the goal's features as the prediction target. The paper reports zero-shot solutions on six environments "without expert demonstrations, reward modeling, or pre-learned inverse models." — [DINO-WM abs](https://arxiv.org/abs/2411.04983)
- Planner: MPC with CEM. The cost is the MSE between the predicted latent at the final timestep T and the goal latent. CEM samples N sequences from a Gaussian, keeps the top-K, refits mean and covariance, then executes the first k actions and replans. — [DINO-WM HTML, App. A.5.1](https://arxiv.org/html/2411.04983)
- Gradient descent vs. CEM: the paper reports that "CEM outperforms GD" and hypothesizes that "regularizations during training and in the planning objectives could further improve performance." Table 8 success rates for PointMaze / Push-T / Wall:
  - open-loop CEM: 0.80 / 0.86 / 0.74
  - open-loop GD: 0.22 / 0.28 / NA
  - MPC (CEM with replanning): 0.98 / 0.90 / 0.96

  — [DINO-WM HTML, §planning + A.5.3](https://arxiv.org/html/2411.04983)
- Table 1, reward-free offline world models, success rate on Maze / Wall / Reach / PushT:
  - DINO-WM: 0.98 / 0.96 / 0.92 / 0.90
  - DreamerV3: 1.00 / 1.00 / 0.64 / 0.30
  - IRIS: 0.74 / 0.04 / 0.18 / 0.32
  - TD-MPC2: 0.00 on all four

  — [DINO-WM HTML](https://arxiv.org/html/2411.04983)
- Generalization to unseen configurations (Table 3): WallRandom 0.82, PushObj 0.34, GranularRandom chamfer distance 0.63. All were best among the baselines, but PushObj stays low. — [DINO-WM HTML](https://arxiv.org/html/2411.04983)
- Planning depends heavily on data. PushT success rises from 0.08 with 200 trajectories to 0.48 (1k), 0.72 (5k), 0.88 (10k) and 0.92 (18.5k). — [DINO-WM HTML, Table 5](https://arxiv.org/html/2411.04983)

#### V-JEPA 2-AC (Assran et al.; arXiv 2506.09985, Jun 2025; 48 pp; venue not stated)
- A 300M-parameter block-causal transformer is post-trained on frozen V-JEPA 2 features with under 62 hours of unlabeled Droid robot video. It was deployed zero-shot on Franka arms in two labs. — [V-JEPA 2 abs](https://arxiv.org/abs/2506.09985)
- Planner:
  - At each step it minimizes a goal-conditioned energy: the L1 distance between the imagined representation T steps ahead and the goal representation.
  - The energy is minimized with CEM.
  - Only the first action is executed before replanning.

  — [V-JEPA 2 HTML §3.2](https://arxiv.org/html/2506.09985)
- Table 3 (Lab 2): V-JEPA 2-AC uses 800 samples, 10 CEM iterations and horizon 1, at 16 s per action. The Cosmos video-generation world model uses 80 samples, 10 iterations and horizon 1, at 4 minutes per action. Cosmos scores Reach 80% and Grasp/Pick-&-Place 0–20%. — [V-JEPA 2 HTML](https://arxiv.org/html/2506.09985)
- Table 2 (planning vs. a BC policy), averaged over two labs, 10 trials each, reported as V-JEPA 2-AC vs. Octo (a goal-image-conditioned BC/VLA model fine-tuned on Droid with hindsight relabeling):

  | Task | V-JEPA 2-AC | Octo |
  |---|---|---|
  | Reach | 100% | 100% |
  | Grasp cup / box | 65% / 25% | 15% / 0% |
  | Reach-with-object cup / box | 75% / 75% | 15% / 70% |
  | Pick-&-Place cup / box | 80% / 65% | 15% / 10% |

  Pick-&-Place needed two sub-goal images in addition to the final goal. — [V-JEPA 2 HTML](https://arxiv.org/html/2506.09985)
- Stated limitations:
  - Sensitivity to camera position: the model must infer the action coordinate axis from a monocular camera, and the authors "manually tried different camera positions."
  - Long-horizon planning: autoregressive error accumulation plus a search space that grows exponentially with horizon. "Long-horizon planning is necessary for solving non-greedy prediction tasks, e.g., pick-and-place without image sub-goals."
  - Reliance on image goals.

  — [V-JEPA 2 HTML §4.3, App. B.4](https://arxiv.org/html/2506.09985)
- For a reaching task, the energy landscape is "relatively smooth and locally convex," and the end-effector gets within 4 cm of the goal. — [V-JEPA 2 HTML](https://arxiv.org/html/2506.09985)

#### PLDM (Sobal, Zhang, Cho, Balestriero, Rudner, LeCun; arXiv 2502.14819, v1 Feb 2025 → v4 Oct 2025; LeWM cites an earlier version, "Stress-testing offline reward-free reinforcement learning…", at the 7th Robot Learning Workshop)
- Architecture: an encoder plus K predictors. The goal cost is C_goal = (1/K) Σ_k Σ_t ‖h(s_g) − f_k(ẑ_t, a_t)‖, summed over every step rather than only the terminal one. An uncertainty term C_unc = Σ_t γ^t Σ_j Var_k(f_k(·)_j) is added with weight β.
- Optimizer and replanning: MPPI, replanning after every interaction (i = 1) by default, which makes PLDM about 4× slower than the model-free baselines.

  — [PLDM HTML §3.3](https://arxiv.org/html/2502.14819)
- Anti-collapse uses a VICReg-style objective (variance, covariance, temporal-smoothness and inverse-dynamics terms). It is brittle, as the ablation shows. Success on Two-Rooms / Diverse Maze:
  - full objective: 98.0 / 98.7
  - without the variance term: 13.4 / 11.4
  - without covariance: 29.2 / 7.8
  - without time-smoothness: 71.0 / 95.6
  - without IDM: 98.0 / 75.5

  — [PLDM HTML, App. D.1.2](https://arxiv.org/html/2502.14819)

#### LeWM / LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero; arXiv 2603.19312, v1 Mar 2026, v3 Jun 2026; no venue stated on arXiv)
- An end-to-end JEPA from pixels trained with two losses: next-embedding prediction and SIGReg, which pushes the embeddings toward an isotropic Gaussian. It has about 15M parameters and plans "up to 48× faster than foundation-model-based world models." — [LeWM abs](https://arxiv.org/abs/2603.19312)
- Architecture: ViT-Tiny encoder (patch 14, 224×224, 192-d CLS token) and a ViT-S predictor. Frameskip is 5. History length is 3 for PushT and Cube and 1 for TwoRoom. — [LeWM HTML App. D](https://arxiv.org/html/2603.19312)
- Planner:
  - Cost: terminal ‖ẑ_H − z_g‖².
  - CEM with 300 samples; 30 iterations on PushT, 10 elsewhere; top-30 elites; initial variance 1.
  - Horizon 5 blocks, which is 25 environment steps.
  - "Receding-horizon MPC with a horizon of 5, meaning that the entire optimized action sequence is executed before replanning." This setup "follows the setup used in [DINO-WM]."

  — [LeWM HTML §3.2, App. D](https://arxiv.org/html/2603.19312)
- Evaluation: the start state is sampled from a dataset trajectory, and the goal is the state 25 steps later in the same trajectory. The budget is 50 steps in every environment. — [LeWM HTML App. F.1](https://arxiv.org/html/2603.19312)
- Published success rates for TwoRoom / Reacher / PushT / Cube. LeWM's Fig. 6 is an image; these values are as transcribed by follow-up papers:
  - LeWM: 87 / 86 / 96 / 74
  - DINO-WM without proprioception: 100 / 79 / 74 / 86
  - PLDM: 97 / 78 / 78 / 65

  — [Fast-LeWM Table 1](https://arxiv.org/abs/2606.26217); same values with ± std in [Qantara Table 1](https://arxiv.org/abs/2607.04978)
- Push-T planner ablation (Table 10):
  - LeWM: CEM 96.0 ± 2.8; Adam 84 ± 7.1; RMSProp 67.3; SGD 26.
  - PLDM: CEM 78; Adam 80; RMSProp 49.3; SGD 4.7.

  — [LeWM HTML](https://arxiv.org/html/2603.19312)
- Predictor dropout during training matters. Push-T success is 78 at p = 0, 96 at p = 0.1, 85.3 at p = 0.2 and 66.7 at p = 0.5. — [LeWM HTML Table 9](https://arxiv.org/html/2603.19312)
- Stated limitations: "Planning remains restricted to short horizons"; the method "relies on offline datasets with sufficient coverage"; and low data diversity weakens SIGReg (the explanation offered for TwoRoom). — [LeWM HTML §6](https://arxiv.org/html/2603.19312)
- Cube probing (Table 4):
  - All three models struggle to recover block orientation.
  - DINO-WM is better on dynamic and rotational quantities.
  - LeWM's linear probe for joint velocity has r = 0.095.
  - In violation-of-expectation tests on Cube, DINO-WM's surprise does not rise significantly and PLDM's rise is weak.

  — [LeWM HTML](https://arxiv.org/html/2603.19312)

#### Failure modes reported in 2026 diagnostics of these planners
- **Myopic, non-monotonic latent distance.** In a LeWM reproduction on TwoRoom:
  - Squared latent distance correlates with true distance at r = 0.426. It saturates around 80 arena units and decreases beyond 120, "so moving away from the goal can lower the cost."
  - The predictor is not the limit: 75 environment steps ahead it is still only 0.189 as wrong as a "frozen world" baseline, while the planner never imagines beyond 25 steps.
  - Replacing only the objective, with nothing retrained, raises success at goal offset 100 from 26.0% to 98.0%.

  — [The Objective Is the Bottleneck, 2608.12959](https://arxiv.org/abs/2608.12959)
- Even with exact dynamics and globally optimal short-horizon search, scoring by distance to the final goal "rejects the first steps of a route that initially moves away from the goal." — [Aim Short to Reach Far, 2609.30036](https://arxiv.org/abs/2609.30036)
- The original LeWM controller reaches 7.0% on 100 high-distance TwoRoom episodes. — [TRM, 2605.22164](https://arxiv.org/abs/2605.22164)
- LeWM's own conclusion says planning is short-horizon only. — [LeWM](https://arxiv.org/html/2603.19312)
- **Ranking and decision-metric misalignment.**
  - For LeWM, scoring the encodings of the endpoints actually reached raises physical Spearman correlation from 0.193 to 0.631 on PushT and from 0.464 to 0.975 on Reacher, "while Cube's encoded-goal cost remains uninformative." In 14 of 16 model-task cells, error in the candidate-pool center dominates MSE. — [When Low Prediction Error Misleads Planning, 2610.05550](https://arxiv.org/abs/2610.05550)
  - A "Plan-Real Spearman" diagnostic identifies encoder distortion, terminal rollout error and candidate margins as the controlling quantities. — [DA-LeWM, 2608.18746](https://arxiv.org/abs/2608.18746)
  - A model can have low factual prediction error yet poorly distinguish candidate actions. On Cube, an action-discriminative model raises hard-start success from 3.7% to 52.0%, and "CEM-aligned elite regret tracks success more closely" than prediction error. — [AD-WM, 2609.30264](https://arxiv.org/abs/2609.30264)
  - True-transition action-identifiability tracks planning success on TwoRoom, PushT and Cube better than prediction loss does. — [ATM, 2606.09028](https://arxiv.org/abs/2606.09028)
- **Optimizer exploiting model errors.**
  - "Sampling raw actions when planning with WAMs lets MPC exploit dynamics-model inaccuracies; planning in the noise space of the policy head instead improves the closed-loop performance." — [LeWAM, 2610.12407](https://arxiv.org/abs/2610.12407)
  - In a hierarchical LeWM, "unconstrained search can select latent macro-actions that appear favorable under the learned model but produce poor control targets." Constraining the search to macro-actions encoded from training trajectories gives +11.3 pp at medium horizons and +14.7 pp at the longest PushT horizon. — [Hi-LeWM, 2607.12547](https://arxiv.org/abs/2607.12547)
  - On Cube, the fraction of episodes that "fail with a sharp rise in model-predicted cost after execution" falls from 23.3% (LeWM) to 2.0% (SALT). — [SALT, 2609.33595](https://arxiv.org/abs/2609.33595)
  - "The candidate whose predicted endpoint is closest to the goal may not perform best when executed." — [Traj-LeWM, 2608.14125](https://arxiv.org/abs/2608.14125)
  - Gradient planners lose to CEM (the DINO-WM and LeWM ablations above).
- **Error accumulation and search size.**
  - Autoregressive rollout "exposes the predicted trajectory to accumulated latent errors as the horizon grows." — [Fast-LeWM](https://arxiv.org/abs/2606.26217)
  - "A fixed candidate budget must search an exponentially larger action space." At goal offset 150, vanilla LeWM gets 4.7% on PushT and 20.7% on Cube. — [SAGE, 2607.17973](https://arxiv.org/abs/2607.17973)
- **Representation blind spots.** On the official LeWM checkpoints for PushT, Reacher, Cube and TwoRoom, every linear velocity probe sits at or below chance, while position probes reach R² ≈ 0.95. — [Frozen in a Frame, 2610.04585](https://arxiv.org/abs/2610.04585)
- **Protocol sensitivity.**
  - On TwoRoom, the paper's appendix and the repository config imply different goal offsets and budgets; on the authors' checkpoint they give 14.0% vs. 84.0%. Changing only how the goal is constructed moves that checkpoint from 84.0% to 8.0%. — [Independent reproduction, 2608.10145](https://arxiv.org/abs/2608.10145)
  - The CLEAR-LeWM evaluation suite finds that the historical LeWM evaluator keeps already-solved start-goal pairs (detail in Q3). — [CLEAR-LeWM](https://davidsunok.github.io/CLEAR-LeWM/)

### Inferences
- Under fixed compute, all four systems spend their budget on a sampling optimizer over a terminal latent-distance score. Several follow-ups locate failures in the score and the search (ranking, myopia, exploitation) rather than in one-step prediction. This matches the user's Cube observation that CEM exploits model errors.
- Gradient-based planning underperforms CEM in both DINO-WM and LeWM (Adam 84 vs. CEM 96; SGD 26). This is consistent with gradient optimizers finding adversarial inputs to the differentiable predictor. Zero-order sampling hides this partially but not fully (SALT's cost-rise failures, the LeWAM statement).
- On Cube specifically, several independent signals suggest the latent-distance cost itself is a weak decision signal: it is "uninformative" after endpoint encoding, orientation and velocity are poorly encoded, and DINO-WM's surprise does not react on Cube. That points to representation and cost limits, which epistemic penalties on the dynamics would not fix.

### Gaps
- LeWM's per-task Fig. 6 values, including the GCIVL and GCIQL baselines, exist only as an image. The values above are transcriptions by Fast-LeWM, Qantara and INTACT, and I could not read GCIVL/GCIQL anywhere.
- I saw V-JEPA 2-AC's planning horizon (1) only in the Table 3 comparison and did not verify it for every experiment.
- DINO-WM's own per-environment CEM sample, elite and iteration counts were not extracted. I rely on LeWM's statement that its configuration follows DINO-WM.

## Q2. PLDM's study of data quality, diversity and generalization, and other evidence on when planning beats goal-conditioned BC / RL policies

### Takeaway
PLDM's navigation study concludes that "model-free RL benefits most from large amounts of high-quality data, whereas model-based planning generalizes better to unseen layouts and is more data-efficient." Planning was the only method that transferred to unseen maze layouts and to a new avoidance task. HILP and GCIQL, however, stitched across missing coverage far better: on the no-door data, PLDM scored 34%. In manipulation under LeWM-style evaluation the balance flips. Direct goal-conditioned policies on the same latents beat CEM on Cube (GCBC 84, GC-IDM 98.7–99.3, INTACT 100 vs. CEM 67–74). Planning still wins on PushT and on several OGBench tasks, and it is needed at long goal offsets and out of distribution.

### Cited Findings

#### PLDM (navigation: Two-Rooms 64×64 pixels, Diverse PointMaze, Ant-U-Maze)
- Six methods were compared (HILP, HIQL, GCIQL, CRL, GCBC, PLDM) on 23 datasets. — [PLDM Fig. 1](https://arxiv.org/html/2502.14819)
- Table 1 star summary: PLDM is the only method rated ★★★ on "transfer to new environment layouts" and on "transfer to a new task". GCBC is ★✩✩ on layouts, new task, random-policy data and stitching, and ★★✩ on best-case performance. — [PLDM HTML Table 1](https://arxiv.org/html/2502.14819)
- With abundant high-quality data (3M transitions, about 30k trajectories), "HIQL, GCIQL, HILP, and PLDM [are] nearing 100% success rate." — [PLDM §4.2](https://arxiv.org/html/2502.14819)
- Table 2, success on good-quality data vs. data with no door-passing trajectories. The HTML renders alignment phantom digits; these values are decoded to match the text, which says "HILP and GCIQL achieve perfect performance, while PLDM performance drops but is still higher than the rest":

  | Method | Good-quality data | No door-passing data |
  |---|---|---|
  | CRL | 89.3 | 14.7 |
  | GCBC | 86.0 | 18.4 |
  | GCIQL | 98.0 | 99.6 |
  | HILP | 100 | 100 |
  | HIQL | 96.4 | 26.3 |
  | PLDM | 97.8 | 34.4 |

  — [PLDM HTML Table 2, §4.4](https://arxiv.org/html/2502.14819)
- Data scarcity: "PLDM and the model-free GCIQL outperform other model-free methods when data is scarce"; HILP is "more data-hungry." — [PLDM §4.3](https://arxiv.org/html/2502.14819)
- Short episodes (lengths 91 / 64 / 32 / 16 at 3M transitions): goal-conditioned model-free methods other than GCIQL "struggle." HILP does well, and PLDM "can learn an accurate model from short trajectories and stitch together a plan." — [PLDM §4.4](https://arxiv.org/html/2502.14819)
- Random-policy data: HILP, GCIQL and PLDM outperform the rest. The authors explain that far goals become out of distribution for TD learning, whereas PLDM uses the data only to learn dynamics. — [PLDM §4.5](https://arxiv.org/html/2502.14819)
- New task (avoiding a chaser, done by flipping the sign of the cost): PLDM beats HILP, and goal-conditioned methods "cannot avoid specific states by design." — [PLDM §4.6](https://arxiv.org/html/2502.14819)
- Ant-U-Maze stitching (trajectory lengths 25–500; start and goal about 200 steps apart): "PLDM, HIQL, and HILP … achieving 100% success rates while other methods fail with shorter trajectories." — [PLDM §4.7](https://arxiv.org/html/2502.14819)
- Unseen layouts (training on 5, 10, 20 or 40 layouts): "PLDM generalizes best – even when trained on just 5 maps – while other methods fail." On a single fixed layout every method is near 100% (Table 4: PLDM 0.990, CRL 0.980, GCBC 0.970, GCIQL, HIQL and HILP 1.000). HILP's learned distances fail to generalize to out-of-distribution layouts (App. I). — [PLDM §4.8, Table 4](https://arxiv.org/html/2502.14819)
- Compute (Table 5): replanning every step costs 16.0 s per episode. Replanning every 4 steps costs 4.8 s at 0.95 normalized success, every 16 steps 0.90, and every 32 steps 0.62. For comparison, GCIQL takes 3.6 s and HIQL 4.0 s. — [PLDM App. F](https://arxiv.org/html/2502.14819)
- Reconstruction-based latents plan poorly (Table 6, decoded): DreamerV3 24.0, a reconstruction model 26.2, TD-MPC2 0 (collapse) and TD-MPC2 + IDM 35.0, against PLDM 97.4. — [PLDM App. G](https://arxiv.org/html/2502.14819)
- The authors' limitation: "All our experiments were conducted in navigation environments, excluding robot manipulation or partially observable settings." — [PLDM App. L](https://arxiv.org/html/2502.14819)

#### Planning vs. policies on manipulation and LeWM-suite tasks
- LeWM-suite GCBC, as run by the LeWM authors on DINOv2 patch embeddings: PushT 75, TwoRoom 100, Cube 84. LeWM-CEM scores 96, 87 and 74 on the same tasks. Qantara labels this a cross-pipeline reference. — [Qantara Table 2](https://arxiv.org/abs/2607.04978); LeWM values via [Fast-LeWM](https://arxiv.org/abs/2606.26217)
- GC-IDM (an MLP mapping current latent, goal latent and remaining horizon to the next action) vs. CEM, both on the frozen LeWM checkpoint with the same solver configuration:

  | Task | GC-IDM n = 50 | CEM n = 50 | GC-IDM n = 200 | CEM n = 200 |
  |---|---|---|---|---|
  | Cube | 99.3 ± 1.2 | 73.3 ± 9.0 | 98.7 ± 0.6 | 67.0 ± 2.1 |
  | Reacher | 100.0 | 68.0 | 99.7 | 70.3 |
  | TwoRoom | 100.0 | 82.0 | 100.0 | 84.0 |
  | PushT | 84.7 | 89.3 | 84.2 | 82.5 |

  GC-IDM is 100–130× cheaper per decision. — [GC-IDM HTML Table 1](https://arxiv.org/html/2605.08732)
- GC-IDM degrades with goal offset on PushT (94.0 at offset 5 down to 70.0 at offset 50) but stays at 93–99 on Cube. — [GC-IDM Table 2](https://arxiv.org/html/2605.08732)
- INTACT, a search-free direct policy, vs. pure CEM 300×30 on the same INTACT model, for PushT / Cube / Reacher / TwoRoom:
  - Direct: 85.78 / 100.00 / 97.67 / 97.89
  - Pure CEM: 88.44 / 68.44 / 83.67 / 82.89

  — [INTACT HTML Table 2](https://arxiv.org/html/2607.26056)
- PRISM (MPPI with J = 30, H = 5, action block 5; frozen LeWM; 3 seeds × 50 episodes):
  - BC-only Gaussian head (1.0M parameters): PushT 31 ± 5, Cube 66 ± 4.
  - Diffusion Policy (19.3M parameters): PushT 41 ± 10, Cube 77 ± 5.
  - Vanilla MPPI on LeWM, K = 32/64/128: PushT 59 / 61 / 57, Cube 46 / 44 / 44.

  So here BC beats planning on Cube but loses on PushT. — [PRISM HTML Table 1](https://arxiv.org/html/2606.07974)
- Qantara, from one checkpoint: CEM reaches Cube 93.7 and PushT 90.1. The goal-blind BC path reaches Cube 70.8 and PushT 82.1 at 15–65× lower cost, and falls to "near-noise" on Reacher, "whose released training corpus comes from a random policy." — [Qantara abs + Table 2](https://arxiv.org/abs/2607.04978)
- DTRC's six additional visual OGBench tasks, LeWM-CEM vs. GCBC: PointMaze 13.3 vs. 28.7; Cube-Double 50.7 vs. 44.7; Scene 56.7 vs. 48.7; AntMaze-Medium 32.0 vs. 28.0; HumanoidMaze-Medium 82.0 vs. 83.3; Puzzle-3x3 24.7 vs. 35.3. GC-IDM gets 38.7 on Cube-Double, and DTRC gets the best Cube-Double score, 54.7. — [DTRC HTML Table 1](https://arxiv.org/html/2610.08960)
- OGBench with its own long-horizon test goals (five fixed tasks per environment), for GCBC / GCIVL / GCIQL / QRL / CRL / HIQL. The HTML doubles digits; decoded values:
  - cube-single-play: 6 / 53 / 68 / 5 / 19 / 15
  - visual-cube-single-play: 5 / 60 / 30 / 41 / 31 / 89
  - visual-cube-double-play: 1 / 10 / 1 / 5 / 2 / 39

  — [OGBench HTML Table 2](https://arxiv.org/html/2410.20092); OGBench is ICLR 2025 per [arXiv](https://arxiv.org/abs/2410.20092)
- In a robot-arm transfer setting, V-JEPA 2-AC planning beats the goal-conditioned BC/VLA baseline Octo on grasping and pick-and-place in unseen labs (e.g., Pick-&-Place cup 80% vs. 15%). — [V-JEPA 2 Table 2](https://arxiv.org/html/2506.09985)
- A JEPA world-action model's closed-loop policy mode "match[es] a regular flow-matching policy trained on the same encoder at matched size," and planning works better in the policy's noise space than in raw-action space. — [LeWAM, 2610.12407](https://arxiv.org/abs/2610.12407)

### Inferences
- Policies win when evaluation goals stay close to the training distribution. The LeWM protocol samples goals 25 steps ahead in the same trajectory, which favors them, and on Cube single, policies on frozen LeWM latents reach about 98–100%. This externally corroborates the user's bc_only 98% result.
- Planning wins:
  - under distribution shift (unseen layouts, new tasks or costs, new labs);
  - with random or low-quality data;
  - on contact-rich local tasks where the model is good (PushT);
  - on some harder OGBench tasks (Cube-Double, Scene).
- The OGBench numbers show the LeWM-protocol advantage of BC does not carry over to long-horizon task goals: GCBC gets 5–6% on cube-single while HIQL gets 89% on visual-cube-single. Value-based or hierarchical methods are the relevant competitors there, not CEM over latent distance.
- If the thesis needs a regime where a world model adds value over a BC prior, the literature points to longer goal offsets, out-of-distribution goals or start states, or new costs. The in-distribution 25-step Cube protocol is close to BC's ceiling, which the user already found.

### Gaps
- PLDM has no manipulation or Cube experiments, and its ensemble-penalty ablation (App. E) is shown only as figures, so I have no numbers for it.
- LeWM's GCIVL and GCIQL numbers on Cube could not be extracted.
- The cross-paper numbers above use different pipelines, data, epochs and episode counts, and are not paired comparisons. INTACT notes that published values "are not paired controls."

## Q3. 2026 LeWM follow-ups: what each does, whether it uses uncertainty, and reported numbers

### Takeaway
The named follow-ups fall into four groups:
- **Search-free policies on LeWM-style latents:** GC-IDM, INTACT, and Qantara's BC path. They match or beat CEM at roughly 100× lower cost and dominate on Cube.
- **Planner fixes:** PRISM fuses a BC prior into MPPI; Fast-LeWM predicts action prefixes.
- **Representation changes:** C-JEPA uses object-centric masking.
- **Evaluation:** CLEAR-LeWM removes pre-solved pairs. With them removed, official LeWM-CEM scores 50.33% on Cube, against a random-policy floor of 15.67%.

Explicit uncertainty appears only in PRISM, which weights the BC prior by its predicted per-state σ; in INTACT, whose distributional action law leaves uncertainty-triggered verification as future work; and in a non-named follow-up, EpicWorldModel, which uses flow variance for exploration. None of the named follow-ups penalizes CEM with epistemic model uncertainty.

### Cited Findings

#### Named follow-ups

**GC-IDM**, "Latent Geometry Beyond Search: Amortizing Planning in World Models" (arXiv 2605.08732, May/Jun 2026, arXiv-only).
- A 1.5M-parameter MLP maps (z_t, z_g, remaining horizon) to a_t on the frozen LeWM.
- It "matches or exceeds CEM in seven of eight environment-protocol settings while reducing per-decision cost by 100–130×."
- A sweep over CEM, MPPI, iCEM and gradient planners shows the result "is not specific to a particular optimizer."

— [abs](https://arxiv.org/abs/2605.08732)

Further details:
- A first attempt used pairwise IDM decoding plus linear latent interpolation. It achieved R² = 0.993 action reconstruction "but poor planning."
- Training takes about 20 minutes per environment on one GPU.
- Numbers are in Q2. Trajectory quality: action jerk is 15–36× lower than CEM's.

— [HTML](https://arxiv.org/html/2605.08732)

No uncertainty component found.

**PRISM**, "PRior-guided Imagination Sampling in world Models" (arXiv 2606.07974, Jun 2026, arXiv-only).
- A lightweight MLP on the frozen JEPA encoder predicts a state-conditioned Gaussian action prior (μ_p, σ_p).
- The prior is fused into the planner's sampling distribution by a precision-weighted product of Gaussians, "making the prior confident where it is and ceding control where it is not."
- The abstract reports +35 pp on Cube and +32 pp on PushT over vanilla world-model MPC.

— [abs](https://arxiv.org/abs/2606.07974)

Table 2 (LeWM encoder, K = 32/64/128):

| Variant | Cube | PushT |
|---|---|---|
| Vanilla MPPI | 46 / 44 / 44 | 59 / 61 / 57 |
| Warm start with μ_p only | 51 / 57 / 55 | 75 / 69 / 66 |
| PRISM-MPPI (μ_p + σ_p) | 79 / 78 / 79 | 82 / 86 / 89 |

PRISM-CEM refits σ from the elites and scores Cube 67 / 81 / 85 and PushT 43 / 87 / 91. Time per plan is about 211 ms at K = 128 on an RTX 5090. — [HTML](https://arxiv.org/html/2606.07974)

Uncertainty: yes, but it is the policy prior's predicted per-state σ, not world-model epistemic uncertainty.

**Fast-LeWM** (arXiv 2606.26217, Jun → Sep 2026, arXiv-only). Replaces repeated one-step rollout with action-prefix prediction, so the futures for all prefixes are predicted in parallel. — [abs](https://arxiv.org/abs/2606.26217)

Table 1, TwoRoom / Reacher / PushT / Cube:

| Method | TwoRoom | Reacher | PushT | Cube | Avg |
|---|---|---|---|---|---|
| LeWM | 87 | 86 | 96 | 74 | 85.8 |
| Fast-LeWM | 98 | 88 | 96 | 80 | 90.5 |
| Fast-LeWM + self-consistency scoring | 98 | 90 | 98 | 82 | 92.0 |

Ablations: "Long-Action LeWM" scores 76 / 70 / 80 / 58. Open-loop latent loss grows more slowly with horizon. — [abs](https://arxiv.org/abs/2606.26217)

Uncertainty: none; the self-consistency term is a scoring-consistency check.

**Qantara**, "Bridge-Flow Training for Multi-Paradigm JEPA Inference" (arXiv 2607.04978, Jul → Oct 2026, arXiv-only).
- A 21M-parameter JEPA trained with a Brownian bridge on the state axis and flow matching on the action axis.
- One checkpoint serves three modes: latent planning, BC action sampling, and video-inverse.
- At a matched 10-epoch budget: average 91.2 over the four tasks; Cube 93.7, which is +7.7 over DINO-WM and +17.0 over LeWM; PushT 90.1; TwoRoom 100.
- Its own LeWM reproduction scores PushT 92.3, TwoRoom 89.9, Cube 76.7, Reacher 64.3.
- Replacing its edge-of-square noise sampler with uniform sampling drops PushT planning from 90.1 to 53.3.

— [abs](https://arxiv.org/abs/2607.04978), [HTML Tables 1–2](https://arxiv.org/html/2607.04978)

No uncertainty-aware planning found.

**C-JEPA**, "Causal-JEPA: Learning World Models through Object-Level Latent Masking" (arXiv 2602.11389; ICML 2026 accepted, per the arXiv comments).
- Masks object-level latents during training.
- About +20% absolute on CLEVRER counterfactual QA.
- Push-T planning: 88.67% with 6 tokens × 128 dimensions, vs. DINO-WM 91.33% with 196 × 384. The abstract describes this as "only 1% of the total latent input features." OC-DINO-WM gets 60.67 and OC-JEPA 76.00.

— [abs](https://arxiv.org/abs/2602.11389), [HTML Table 6](https://arxiv.org/html/2602.11389)

No Cube, Reacher or TwoRoom planning numbers found (INTACT lists only PushT for it). No uncertainty.

**INTACT**, "Isomorphic Intent-to-Action Learning for Search-Free World Models" (arXiv 2607.26056, 28 Jul 2026, arXiv-only; authors Junhan Sun, Hao Zhao, Guofeng Zhang).
- An end-to-end JEPA whose "physical intent" z_{t+1} − z_t and "deployment intent" sg(z_g) − z_t share one predictor grammar.
- The conditional mean of a distributional action law serves as a search-free policy.
- One-epoch, zero-search success on PushT / Cube / Reacher / TwoRoom: 85.78 / 100.00 / 97.67 / 97.89.
- Optional local CEM centred on the direct plan reaches a 96.86 macro average with 384 instead of 9,000 candidates, a 23.44× sampling reduction and +16.00 points over pure CEM.
- A shared four-task encoder reaches an 89.39 macro average. Direct inference takes 2.9–5.5 ms.

— [abs](https://arxiv.org/abs/2607.26056)

Table 2 matched rows on the same model (PushT / Cube / Reacher / TwoRoom):
- Pure CEM: 88.44 / 68.44 / 83.67 / 82.89
- Actor-on CEM: 93.56 / 96.89 / 86.67 / 98.00
- Guarded 128×3: 92.22 / 99.78 / 97.44 / 98.00

INTACT's matched-E1 reproductions with a one-epoch head on a frozen E1 encoder:
- GC-IDM: 19.33 / 69.33 / 84.33 / 99.33
- PRISM: 18.67 / 86.67 / 66.33 / 88.33

The published GC-IDM row used a 50-epoch head. — [HTML Table 2](https://arxiv.org/html/2607.26056)

Uncertainty: the deployed action is the mean of a diagonal Gaussian, which "at junctions or contact transitions … can lie between valid modes." The authors list "uncertainty-triggered verification" and mixture actors as natural extensions, not as evaluated components. — [HTML](https://arxiv.org/html/2607.26056)

**CLEAR-LeWM** (GitHub DavidSunok/CLEAR-LeWM, project page v0.8; Junhan Sun, Guofeng Zhang, Hao Zhao, at Zhejiang University and Tsinghua). The README calls it "an independent community evaluation project, not an official LeWM release." — [GitHub](https://github.com/DavidSunok/CLEAR-LeWM) (via search snippet), [project page](https://davidsunok.github.io/CLEAR-LeWM/)

Findings:
- The historical evaluator "retains initially solved pairs," can terminate Reacher inside dm-control, and has a TwoRoom collision rewrite that admits physically invalid transitions.
- Random-policy success, historical → v0.8 Moderate: PushT 7 → 4%, Cube 47 → 16%, Reacher 16 → 7%, TwoRoom 30 → 7%.
- Moderate Cube contract: cube position ≤ 4 cm at first hit; orientation and robot pose unscored. Strict mode adds 24-fold cube orientation.

Official LeWM under v0.8 (seeds 0, 1, 42; 100 episodes each; pure CEM 300×30):

| Task | Moderate | Strict |
|---|---|---|
| PushT | 86.67 | 70.67 |
| Cube | 50.33 (random 15.67) | 21.67 |
| Reacher | 79.67 | 87.00 |
| TwoRoom | 83.00 | 51.33 |

Two community models were submitted (self-reported, CI-checked). Under Moderate, PushT / Cube / TwoRoom, the project's "canonical model SR" chart (seeds 0, 1, 42) gives:
- Official LeWM: 86.7 / 50.3 / 83
- "DINOv2 No-Proprio LeWM": 7 / 44.7 / 43.7
- "GCBC Joint LeWM": 5.3 / 17.7 / 16

The per-bundle seed-42 entries are 8 / 43 / 55 and 9 / 16 / 15 respectively. Either way, the community GCBC sits at about the Cube random floor (about 15–16).

— [project page](https://davidsunok.github.io/CLEAR-LeWM/)

No uncertainty component.

#### Other 2026 LeWM follow-ups with Cube or long-horizon numbers (all arXiv preprints unless noted)
- **SpecWM** (Molin, Peng, Bagatella, Balestriero; 2610.15429): a temporal spectral-alignment kernel. CEM success goes from LeWM 52% to 68% on OGBench Cube, 49% to 77% on Scene, and 29% to 51% on CALVIN. Its Cube protocol: 300 candidates, 30 iterations, 30 elites, horizon 5 blocks, 15 steps executed before replanning, goal offset 25, budget 60. — [abs](https://arxiv.org/abs/2610.15429), [HTML Table 2](https://arxiv.org/html/2610.15429)
- **FIRM-WM** (2609.22816): a state-factorized recurrent state that separates a goal-comparable configuration from a 128-dimensional "dynamic fiber" excluded from the goal cost. On TwoRoom / Reacher / Cube it scores 99.0 / 92.7 / 88.0, against 89.0 / 88.0 / 70.0 for LeWM. — [abs](https://arxiv.org/abs/2609.22816)
- **IDM + state alignment JEPA** (2609.03565; IROS 2026 PWMS workshop): TwoRoom 100%, PushT 98%, Cube 87%; comparable to LeWM on Reacher. — [abs](https://arxiv.org/abs/2609.03565)
- **Traj-LeWM** (2608.14125): a latent trajectory cost added to endpoint distance. Gains of +3 / +14 / +7 / +7 pp on PushT / Cube / Reacher / TwoRoom. — [abs](https://arxiv.org/abs/2608.14125)
- **LEAP** (2609.03294): quasi-Newton planning through the frozen LeWM with a goal-conditioned proposal, a terminal-window state energy and action projection. Mean success over four domains goes from 77.5% (LeWM + CEM) to 94.8%. — [abs](https://arxiv.org/abs/2609.03294)
- **SAGE** (2607.17973): a goal-conditioned subgoal generator plus prior-conditioned action generation. At offset 150, PushT goes from 4.7% to 64.7% and Cube from 20.7% to 67.3%. On LIBERO, Scene2 goes from 0% to 48.7% and Caddy reaches 58%. — [abs](https://arxiv.org/abs/2607.17973)
- **Hi-LeWM** (2607.12547; WM@Booth 2026): hierarchy helps only when high-level search is constrained to be compatible with the low-level controller (+11.3 and +14.7 pp, as in Q1). — [abs](https://arxiv.org/abs/2607.12547)
- **Dual-WM** (2609.37644): at offsets 50 and 100, mean success goes from 75.9% to 84.4% and from 61.4% to 69.5%; +30.8 pp over LeWM at offset 100. — [abs](https://arxiv.org/abs/2609.37644)
- **EpicWorldModel** (2610.05996; NeurIPS 2026 accepted, per the arXiv comments): a stochastic JEPA trained with flow matching. Up to 22% improvement over LeWM in partially observable tasks (details in Q4). — [abs](https://arxiv.org/abs/2610.05996)
- **AD-WM** (2609.30264): Cube hard-start success 3.7% → 52.0%. With a frozen V-JEPA 2 encoder on a Franka setup, basic pick-and-place goes from 42.2% to 71.1%. — [abs](https://arxiv.org/abs/2609.30264)
- **Smaller or less specific gains:**
  - CGS: up to +20 pp over LeWM at 128 candidates. — [2609.35603](https://arxiv.org/abs/2609.35603)
  - ProWorld: +9.67 average. — [2608.01926](https://arxiv.org/abs/2608.01926)
  - VLWM: +13% average on long-horizon tasks. — [2606.21775](https://arxiv.org/abs/2606.21775)
  - ATLAS: improves PushT, TwoRoom and Cube. — [2609.36333](https://arxiv.org/abs/2609.36333)
  - AnisoWM: improves over LeWM in all four environments. — [2609.37441](https://arxiv.org/abs/2609.37441)
  - PPWM: more than 3× CEM speedup. — [2610.08627](https://arxiv.org/abs/2610.08627)

### Inferences
- The user's frozen-LeWM Cube CEM result (51.5% on non-trivial episodes) agrees with two independent references: CLEAR-LeWM's official LeWM at 50.33% on Cube under v0.8 Moderate (pre-solved pairs removed), and SpecWM's LeWM baseline of 52%.
- The user's BC result (98%) agrees with GC-IDM (98.7–99.3) and INTACT direct (100). So neither half of the user's result looks anomalous.
- Training budget and implementation matter a great deal for the policy side. CLEAR's "GCBC Joint LeWM" scores about random on Cube, and INTACT's one-epoch GC-IDM reproduction gets 69%.
- On Cube, the planner variants that succeed change the proposal distribution using a policy: PRISM PoG fusion, INTACT actor-on or guarded CEM, LeWAM noise-space planning. Variants that only change the cost or add search budget help less, unless they also change the representation. This favors "planning as a verifier or local refiner around a BC prior" over "uncertainty-penalized CEM from scratch."
- Historical-evaluator numbers (LeWM 74, DINO-WM 86, GCBC 84, GC-IDM 99) include pre-solved pairs; a random policy reaches 47% on Cube under that evaluator. Only paired, non-trivial-subset comparisons such as CLEAR Moderate or the user's protocol are informative.

### Gaps
- Whether INTACT's "official SR" manifests (seeds 0, 1, 42; 100 episodes each) use CLEAR v0.8 Moderate (pre-solved pairs removed) or the historical evaluator was not verified.
- Qantara's Reacher number was not extracted; only the four-task average of 91.2 is known.
- EpicWorldModel's per-task tables were not read.
- The CLEAR "GCBC Joint" method card was not read, so why it fails is unknown.
- No venue information for GC-IDM, PRISM, Fast-LeWM, Qantara or INTACT; all appear to be arXiv-only.

## Q4. Uncertainty estimation in latent / JEPA world models for planning (2024–2026), and whether it improved closed-loop success

### Takeaway
Uncertainty in latent and JEPA world models has been used mainly for four things:
1. Keeping the planner in-distribution: PLDM's ensemble-variance penalty, tested on navigation only.
2. Optimism and exploration under partial observability: EpicWorldModel, up to +22% over LeWM.
3. Representing multimodal futures: MDN-VJEPA, controlled tasks only.
4. Calibrated or evidential confidence: C3 for video world models, LucidWM, and conformal safety filters.

I found no 2024–2026 paper showing that ensemble, MC-dropout, energy-based or conformal epistemic penalties improve the closed-loop success of a frozen LeWM or DINO-WM CEM planner on Cube. Two 2026 papers explicitly report that prediction-derived or ensemble-disagreement uncertainty can stay small or overconfident exactly when the model is wrong. The uncertainty signal with the largest measured Cube benefit is the BC prior's own predicted σ in PRISM.

### Cited Findings
- **PLDM ensemble penalty.** With K > 1 predictors, the planner minimizes C_goal + β·Σ_t γ^t Σ_j Var_k(f_k(·)_j). This "penalizes the model from choosing state-action transitions that deviate from the training distribution" and "resembles how GCIQL, HIQL, and HILP use expectile regression … to remain in-distribution." Ant-U-Maze uses an ensemble of 5. App. E ablates the penalty on Two-Rooms and AntMaze in figures only. — [PLDM HTML §3.3, D.1.5, App. E](https://arxiv.org/html/2502.14819)
- **EpicWorldModel** (NeurIPS 2026): "JEPA [world models] are deterministic by design" and break down when occlusion makes the future multimodal. A flow-matching predictor's "flow predictive variance," motivated as an upper bound on predictive entropy, is added to CEM to balance goal-reaching "with exploration of uncertain regions where occluded goals are most likely to be located." The paper reports "up to 22% empirical improvement in success rate over LeWorldModel." — [abs](https://arxiv.org/abs/2610.05996)
- **MDN-VJEPA** (2610.13956, 69 pp): a mixture-density predictor built on Variational JEPA. In controlled experiments, "retaining the complete multimodal belief rather than its conditional mean can lead to materially different and substantially better planning decisions under branching futures." No standard-benchmark closed-loop numbers appear in the abstract. — [abs](https://arxiv.org/abs/2610.13956)
- **LucidWM** (2609.37156):
  - "Existing uncertainty estimates derived from predictions can remain overconfident on unfamiliar state-action pairs."
  - Uses subjective-logic evidential doubt on categorical latent transitions, with trust accumulated multiplicatively along imagined trajectories.
  - Compared against 17 uncertainty readouts on four base world models.
  - In a controlled navigation case, steps to goal fall from 362 to 190.

  — [abs](https://arxiv.org/abs/2609.37156)
- **When World Models Lie** (2609.34300): "Existing latent safety filters often rely on auxiliary signals such as ensemble disagreement or value-target consistency residuals …, but these signals can remain small even when the world model's predictions deviate from observations." The paper calibrates on directly observed world-model error with Adaptive Conformal Inference, for Hamilton–Jacobi latent safety filters rather than goal-reaching. — [abs](https://arxiv.org/abs/2609.34300)
- **C3** (2512.05927, Dec 2025): calibrated uncertainty for action- or text-conditioned video world models. It trains with strictly proper scoring rules, estimates uncertainty in latent space, and gives subpatch-level confidence aimed at hallucination detection "for robot policy evaluation and planning." The abstract excerpt I read reports no closed-loop planning success numbers. — [abs](https://arxiv.org/abs/2512.05927)
- **EV-WM** (2606.13053): verifies candidate futures in pretrained-feature space using task-progress, semantic-consistency, physical-feasibility and uncertainty terms, and uses the verifier to gate candidate actions in sampling-based planning (navigation, deformable objects, LIBERO). — [abs](https://arxiv.org/abs/2606.13053)
- **PRISM:** the policy prior's predicted σ_p, kept through MPPI's fixed covariance, accounts for most of the gain. On Cube, warm-starting with μ_p alone gives 51–57, while μ_p + σ_p gives 78–79. "Unlike warm-starting (which discards σ_p) or CEM (which overwrites it after one iteration)," PRISM keeps the confidence throughout optimization. — [PRISM HTML](https://arxiv.org/html/2606.07974)
- **INTACT:** a distributional action law with sampling "available for diversity or optional verification." Uncertainty-triggered verification is proposed as future work. — [abs](https://arxiv.org/abs/2607.26056), [HTML](https://arxiv.org/html/2607.26056)
- **LeWM built-ins:**
  - The predictor is trained with dropout (best p = 0.1), so MC dropout is architecturally available. — [LeWM Table 9](https://arxiv.org/html/2603.19312)
  - LeWM's violation-of-expectation "surprise" reliably spikes on teleportation, but it is used for evaluation, not planning. DINO-WM's surprise does not rise significantly on Cube. — [LeWM HTML](https://arxiv.org/html/2603.19312)
- **Failure signatures that an uncertainty signal could target:** SALT's "sharp rise in model-predicted cost after execution" on Cube (23.3% of episodes for LeWM) — [SALT](https://arxiv.org/abs/2609.33595); Fast-LeWM's self-consistency scoring term (+1.5 average) — [Fast-LeWM](https://arxiv.org/abs/2606.26217).
- **Pre-2023 background:**
  - Probabilistic-ensemble MPC: PETS, "Deep RL in a Handful of Trials using Probabilistic Dynamics Models," NIPS 2018. — [arXiv 1805.12114](https://arxiv.org/abs/1805.12114)
  - Model-based offline RL: MOPO (NeurIPS 2020) — [2005.13239](https://arxiv.org/abs/2005.13239); MOReL (NeurIPS 2020) — [2005.05951](https://arxiv.org/abs/2005.05951)
  - Offline RL from images with latent-space models (LOMPO) — [2012.11547](https://arxiv.org/abs/2012.11547)

  Their uncertainty-penalty mechanisms are from prior knowledge and were not re-read here.

### Inferences
- The user's pre-registered failure of the σ² test (Branch B) is consistent with the 2026 reports that ensemble or prediction-derived uncertainty can be small or overconfident when wrong (When World Models Lie, LucidWM). It is also consistent with the Q1 evidence that Cube's failures stem from the cost and representation (uninformative goal distance, missing velocity and orientation), which epistemic dynamics uncertainty does not measure.
- The successful "uncertainty" uses in this ecosystem are:
  1. **Aleatoric or multimodal** uncertainty, used optimistically for exploration under occlusion (EpicWorldModel). This does not apply to fully observed Cube.
  2. **Confidence of an action prior**, used to shape the search distribution (PRISM). This is a BC-prior method.
  3. **Gating or verification** (EV-WM, conformal safety filters, INTACT's proposed uncertainty-triggered verification).

  A defensible pivot is to use uncertainty to decide when to trust the BC policy versus when to verify or refine with the world model. This matches the user's B-2 recommendation (ensemble σ²) only if σ² is validated as a gate, not as a cost penalty.
- MC dropout on LeWM's predictor is cheap to try. I found no published evaluation of it, so any positive or negative result would be novel, but the evidence above lowers the prior that it fixes CEM's Cube gap.

### Gaps
- No paper found evaluating MC dropout, deep ensembles, energy or density OOD scores, or conformal prediction as a CEM cost penalty on frozen LeWM, DINO-WM or PLDM on manipulation tasks. The arXiv search for these terms was partially rate-limited, so absence is not certain.
- The PLDM ensemble ablation numbers are in figures only.
- The base Variational JEPA (VJEPA) paper referenced by MDN-VJEPA was not located or read.
- C3's planning-related results were not read beyond the abstract.

## Q5. Learned cost-to-go for latent goal-reaching (temporal-distance / quasimetric) as planner costs, and evidence that it fixes myopic latent distance

### Takeaway
Learned temporal or reachability costs on frozen LeWM or PLDM latents fix long-offset navigation dramatically: TwoRoom goes from 26 to 98%, from 7.0 to 96.7%, and from 36.3 to 99.2% in three different papers. Gains on Cube are small: TEMPO adds 0.8–2.3 pp, and the learned temporal geometry on Cube is weak (DTRC Spearman 0.22 over offsets 1–100). Larger Cube gains come from two other routes:
- changing the representation: SpecWM +16, Temporal-Distance JEPA +14.2, FIRM-WM +18;
- using the temporal critic to train a direct policy: DTRC is +30 pp over LeWM.

Outside the LeWM ecosystem, quasimetric and temporal-distance methods (QRL, HIQL, contrastive successor features, multistep quasimetric learning) are mainly policy- or value-learning methods. HIQL is the strongest method on OGBench visual-cube-single at 89%.

### Cited Findings

#### Learned temporal costs plugged into frozen LeWM / PLDM planners
- **TEMPO** (2610.04988, "How Long, Not How Close"): learns a small map of the frozen latent whose distances reflect environment steps, and blends it into the cost, adding at most 0.3% compute. It needs "no rewards, policies or success labels." — [abs](https://arxiv.org/abs/2610.04988)

  Table 1, LeWM → LeWM + TEMPO:

  | Goal distance | TwoRoom | Reacher | PushT | Cube | Scene |
  |---|---|---|---|---|---|
  | One plan | 87.7 → 97.2 | 94.7 → 95.5 | 93.6 → 95.1 | 74.8 → 77.1 | 69.6 → 71.1 |
  | Two plans | 58.5 → 99.2 | – | 49.7 → 55.7 | 57.9 → 58.7 | – |
  | Three plans | 36.3 → 99.2 | – | 22.0 → 28.1 | 68.7 → 70.4 | – |

  PLDM at one plan: PushT 76.1 → 90.1, Cube 60.3 → 68.7. — [HTML Table 1](https://arxiv.org/html/2610.04988)
- **TRM** (2605.22164): a small temporal pairwise cost trained on logged trajectories, used to rank predicted endpoints. On 100 high-distance TwoRoom episodes LeWM goes from 7.0% to 96.7%; shuffled-label controls stay at 0.0%. PLDM goes from 32.7% to 84.0%. The gain comes from "reweighting latent directions carrying the action decision": the XY row space accounts for under 1% of terminal latent MSE but carries most of the control-relevant information. "Contact rich PushT expose[s] layout, dynamics, and recovery limits." — [abs](https://arxiv.org/abs/2605.22164)
- **Objective replacement** on TwoRoom (2608.12959): offset-100 success goes from 26.0% to 98.0% with nothing retrained. A head learned from frame separation predicts spatial distance worse than a position probe (r = 0.819 vs. 0.9897) "yet plans better," charging 24% more to cross the dividing wall where squared latent distance charges 4% less. "It has learned reachability, not proximity." — [abs](https://arxiv.org/abs/2608.12959)
- **Temporal-Distance JEPA** (2607.25337): a directed temporal cost mined from reward-free logs. Deployed as the cost, it reaches 100.0% on TwoRoom vs. LeWM's 97.4% under its locked evaluation. Used instead as a training signal while keeping Euclidean planning, it raises OGB-Cube by 14.2 points. — [abs](https://arxiv.org/abs/2607.25337)
- **DTRC** (2610.08960): a directed temporal quasimetric on frozen LeWM features, used as a temporal critic for direct goal-conditioned flow-policy learning rather than for planning. — [abs](https://arxiv.org/abs/2610.08960)
  - Relative to LeWM: matches PushT; +14.7 (TwoRoom), +24.0 (Reacher), +30.0 pp (Cube), where it is second-highest.
  - Temporal supervision improves the same policy by +9.3 / +5.3 / +32.0 / +6.0 on PushT / TwoRoom / Reacher / Cube.
  - Diagnostics, ρ over offsets 1–100: PushT 0.845, TwoRoom 0.800, Reacher 0.510, Cube 0.221.

  — [HTML](https://arxiv.org/html/2610.08960)
- **RC-aux** (2605.07278): multi-horizon prediction plus "budget-conditioned reachability supervision" with temporal hard negatives, and a reachability-aware planner. It "improves LeWM-style planning with modest additional cost." No Cube number in the abstract. — [abs](https://arxiv.org/abs/2605.07278)
- **Training-free alternatives:**
  - Anchored Planning retrieves a dataset segment toward the goal and aims at an observation shortly after its start. It "outperforms both additional final-goal search and the LeWM planner on long-range goals." — [2609.30036](https://arxiv.org/abs/2609.30036)
  - Hi-LeWM constrains subgoal search to training macro-actions. — [2607.12547](https://arxiv.org/abs/2607.12547)
  - SAGE uses learned subgoal and action generators. — [2607.17973](https://arxiv.org/abs/2607.17973)
- **Geometry-shaping representation methods:**
  - SpecWM: its temporal kernel is connected to the "behavioral successor measure"; Cube 52 → 68. — [2610.15429](https://arxiv.org/abs/2610.15429)
  - Commute-Time-Preserving World Models: show that isotropic anti-collapse regularizers degrade the eigenvalue scaling needed for commute-time distances, and match or beat LeWM with half the parameters. — [2610.01373](https://arxiv.org/abs/2610.01373)
  - ProWorld: hyperbolic progress order, +9.67 average over LeWM. — [2608.01926](https://arxiv.org/abs/2608.01926)
- **Cube-specific negative evidence:** "Cube's encoded-goal cost remains uninformative" even after encoding realized endpoints. — [2610.05550](https://arxiv.org/abs/2610.05550)

#### Quasimetric / temporal-distance GCRL (policy or value side)
- **QRL** (Wang, Torralba, Isola, Zhang; 2304.01203): quasimetric models of the optimal goal-reaching value with "strong theoretical recovery guarantees." It improves sample efficiency and performance on offline and online goal-reaching, with state and image observations. — [abs](https://arxiv.org/abs/2304.01203)
- **HIQL** (Park, Ghosh, Eysenbach, Levine; NeurIPS 2023; 2307.11949): one action-free value function, with a high-level policy that outputs a latent subgoal and a low-level policy. The hierarchy makes it "robust to noise in the estimated value function." — [abs](https://arxiv.org/abs/2307.11949)
- **Contrastive successor features as a temporal distance** (Myers, Zheng, Dragan, Levine; ICML 2024; 2406.17098): after a change of variables, contrastive successor features give a temporal distance that satisfies the triangle inequality even in stochastic settings, which enables stitching. — [abs](https://arxiv.org/abs/2406.17098)
- **Offline GCRL with Quasimetric Representations** (Myers, Zheng, Eysenbach, Levine; Sep 2025; 2509.20478): unifies contrastive and quasimetric representations to learn optimal goal-reaching distances "even with suboptimal data and in stochastic environments." I believe this is the paper the prompt calls "TMD," but that acronym does not appear in the abstract I read. — [abs](https://arxiv.org/abs/2509.20478)
- **Multistep Quasimetric Learning** (Zheng, Myers, Eysenbach, Levine; Nov 2025; 2511.07730): fits a quasimetric with multistep Monte Carlo returns. It handles tasks up to 4,000 steps with visual observations and enables real-world multistep stitching on Bridge. — [abs](https://arxiv.org/abs/2511.07730)
- **ProQ** (2506.18847): a learned asymmetric distance used both as a keypoint-spreading energy and as a directional cost toward proximal subgoals, with a Lagrangian OOD detector. — [abs](https://arxiv.org/abs/2506.18847)
- **Theory:** "intrinsic (least-action)" JEPA energies are quasimetrics, and "symmetric finite energies are structurally mismatched with one-way reachability." — [2602.12245](https://arxiv.org/abs/2602.12245)
- **OGBench cube benchmark** (decoded values, GCBC / GCIVL / GCIQL / QRL / CRL / HIQL):
  - visual-cube-single-play: 5 / 60 / 30 / 41 / 31 / 89
  - state cube-single-play: 6 / 53 / 68 / 5 / 19 / 15

  — [OGBench HTML Table 2](https://arxiv.org/html/2410.20092)

### Inferences
- The user's result that a learned steps-to-goal cost-to-go fixed long-goal selection matches TEMPO, TRM, the objective-replacement study and Temporal-Distance JEPA. All show that temporal or reachability costs repair candidate ranking at long offsets, most clearly in navigation.
- For Cube closed-loop success, the evidence predicts limited benefit from a better cost alone: TEMPO's Cube gains are within about 2 pp, and DTRC's Cube temporal geometry is weak. The large Cube gains come from changing the representation (which needs retraining, so it falls outside a frozen-LeWM thesis) or from distilling the critic into a direct policy (DTRC). Both lead back to policies or BC priors.
- A thesis-compatible synthesis:
  - Use the BC prior or GC-IDM-style policy as the default controller.
  - Use the learned cost-to-go to select goals, subgoals and plans for long offsets.
  - Reserve world-model search for local verification or refinement around the policy, as in INTACT's guarded 384-candidate CEM or PRISM's PoG-MPPI, optionally gated by an uncertainty signal that has been validated.

### Gaps
- No paper found that plugs QRL, HIQL or contrastive-successor distances directly into a JEPA CEM planner on Cube. TEMPO and TRM learn their own step-count heads.
- TEMPO's PLDM Cube rows beyond one plan, and its per-environment training details, were not extracted.
- DTRC's four-task absolute success numbers are in a figure; only the relative gains are cited.
- The "TMD" naming could not be verified.
