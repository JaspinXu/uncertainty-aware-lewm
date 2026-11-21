"""Step-1 diagnostics: does the planning-time uncertainty sigma^2 separate
failures on OGBench Cube?  (see docs/step1_preregistration.md)

- ``protocol``  : non-trivial start states (cube not already at the goal).
- ``recording`` : planning-time logging (MuJoCo snapshots, CEM populations,
                  frozen-encoder embeddings, executed plans).
- ``replay``    : execute logged action sequences from saved simulator states.
- ``rollout``   : latent rollouts from cached embeddings, MC-dropout and
                  ensemble sigma^2.
- ``ensemble``  : latent cache + predictor heads trained on a frozen encoder.
- ``stats``     : the pre-registered criteria C1-C3, gate G and the decision.
"""
