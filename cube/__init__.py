"""OGBench-Cube experiments for the uncertainty-aware LeWM project (CS5478).

Modules
-------
data       dataset helpers (column names, fast start-state sampling, scalers)
rollout    latent rollouts of a frozen LeWM (deterministic and MC-dropout)
recording  CEM/policy subclasses that log every planning call (L1 signal test)
oracle     simulator replay of planned action sequences (ground-truth outcomes)
metrics    AUROC / rank correlations / bootstrap CIs / McNemar test
"""
