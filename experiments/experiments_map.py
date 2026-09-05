"""Experiment → sweep mapping (plan Section 7).

E1 sweeps alpha, E2 sweeps the adversary/robot ratio, E4 sweeps the model ablation
flags, E6 sweeps beta. E3/E5/E7 are *read-only* — they re-analyse E1 (and Override)
run directories rather than launching new runs. E8/E9 run in the Override env.

Only the new-run experiments that fit the ``method × alpha`` sweep are launchable
directly with ``python -m experiments.sweep --experiment <name>``; the others carry
their spec here for the runner/analysis to consume.
"""
from __future__ import annotations

_ABLATION = ["tenure", "tenure_off", "tenure_feature"]
_ALL = ["tenure", "tenure_off", "greedy", "defensive", "cbba"]

EXPERIMENTS: dict[str, dict] = {
    # headline: held value / retention vs contest speed
    "E1": {"methods": _ALL, "alphas": [0.0, 0.5, 1.0, 2.0, 4.0],
           "note": "headline alpha sweep; E5 (defense_fraction) and E7 (estimator error) read from these runs"},
    # model ablation: multiplicative vs feature vs off
    "E4": {"methods": _ABLATION, "alphas": [1.0],
           "note": "retention_mode causal ablation at alpha=1"},
    # adversary/robot ratio (swept via cfg_overrides, one sub-sweep per value)
    "E2": {"methods": ["tenure", "greedy", "defensive"], "alphas": [1.0],
           "override_axis": ("n_adversaries", [2, 4, 6, 8]),
           "note": "sweep n_adversaries; run one sweep per value with cfg_overrides"},
    # contest parameter beta (degenerate limits at 0 and large)
    "E6": {"methods": ["tenure", "greedy", "defensive"], "alphas": [1.0],
           "override_axis": ("beta", [0.0, 0.5, 1.0, 2.0]),
           "note": "sweep beta; suppress/none contest_mode reachable via cfg_overrides for the degenerate limits"},
}

READ_ONLY = {
    "E3": "evaluate E1/E4 checkpoints on held-out N, M, K, alpha (generalization)",
    "E5": "read defense_fraction from E1 run directories (no new runs)",
    "E7": "read retention estimator error (dense vs completion-event) from E1 + Override runs",
    "E8": "Override environment: shipped v1 policy retrained under contested.enabled",
    # Consultant's reframing: Override contains three alpha regimes *in space* at once.
    "E9": "Override: plot predicted retention by site class in a single match "
          "(R_hat ~ 1 on alliance goals, intermediate on neutral goals, low on toggle-dependent "
          "value) WITHOUT telling the head which is which -> it reads adversary geometry, not "
          "layout. Tool: offsim.sim.graph_env.retention_by_site_class",
    "E10": "cross-environment falsifiable prediction: estimate where Override sits on the "
           "canonical alpha axis from its dwell times + geometry, predict the TENURE-vs-baseline "
           "gap from the canonical E1 curve, then check the observed Override gap matches. "
           "Turns two environments into one theory with a confirmed prediction.",
}
