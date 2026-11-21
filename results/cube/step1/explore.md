# Step 1, exploratory follow-ups (not pre-registered)

These results cannot change the Step-1 decision. They inform the choice *within* Branch B.

## B-1 feasibility: can a learned model predict real outcomes?

- Segment failure rate by replan: t=0: 49.8% (n=400), t=25: 97.5% (n=199). A second plan almost never recovers, so only first plans are scored below.
- First plan fails within 25 steps: cost-only logistic AUROC **0.654**; gradient boosting on cost + σ²₁…₅ + CEM statistics AUROC **0.897** (Δ 95% CI [+0.186, +0.303]); raw s2_mc20: 0.571, raw s2_ens: 0.574.
- Per candidate, a learned re-ranker of the last CEM pool captures **+12.4%** of the oracle headroom (95% CI [-7.3%, +29.4%]); success of the picked candidate 38% (baseline) → 32% (re-ranked), oracle 57%.

## B-2 premise: is the model worse on planner-visited states?

- First-block prediction error per latent element (one context frame): executed plans median 0.0583 / mean 0.1582; held-out expert clips median 0.0036 / mean 0.0051 (×16 at the median).
- Across executed plans, Spearman(σ²_k, e_k) for k = 1..5 (s2_mc20): +0.51, +0.38, +0.24, +0.13, -0.02
- Across executed plans, Spearman(σ²_k, e_k) for k = 1..5 (s2_ens): +0.68, +0.63, +0.56, +0.43, +0.20
