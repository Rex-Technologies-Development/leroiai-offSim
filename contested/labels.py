"""Retention label extraction (plan Section 3.9) — the subtlest part of the build.

For a pair ``(decision step d, task i)`` the label is the realized held fraction
over the remaining episode::

    R_label[d, i] = (1 / (T - t_d)) * sum_{u = t_d}^{T} c_i(u) * dt

computed by a **reverse scan** over the recorded per-decision held integrals.

Two label sources are produced, both needed:

1. **Dense** — every ``(d, i)`` with ``c_i = 1`` at decision ``d``. Cheap, plentiful.
2. **Completion events** — decisions during the service run that *led to* a
   ``0 -> 1`` completion (the task is still incomplete there). These are the only
   labels that teach the head about tasks that are **not yet complete**, which is
   exactly where the head is used (on ``ACQUIRE`` candidates). E7 reports estimator
   error separately for the two regimes.

Non-stationarity: labels are realized under the current behaviour policy, so the
label buffer must be **discarded at every PPO update**, never accumulated.

KNOWN LIMITATION — R is not a marginal quantity (consultant, Claim 4 review).
``R_label`` integrates the held fraction over the *whole* remaining episode, so it
**includes re-completions the policy performs later**. The value of completing task
``i`` *now* is really ``complete_now`` minus ``complete_later_anyway``; if the policy
would come back and re-grab the task, that marginal value is much smaller than
``w * R * (T - t)/T``. Under heavy churn (small ``tau_rev`` / high ``alpha``) this gap
is large and ``w * R`` **systematically over-values tasks**, which can by itself make
the multiplicative decoder worse than ``off`` — independent of any training issue.
The separation ``w*R ≈ marginal value`` holds only when **re-completion is rare
relative to the horizon** (low-to-moderate ``alpha``); treat sweeps in the churn zone
with this caveat, and see the *no-revisit counterfactual R* follow-up (define R as the
held fraction until the FIRST reversal, i.e. assuming no re-grab) for the rigorous fix.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import CanonicalConfig
from .core import CanonicalCore


def compute_retention_labels(
    decision_c: Tensor,        # (B, D, T) bool  — c at the START of each decision
    decision_held: Tensor,     # (B, D, T) float — int c_i dt over each decision
    decision_sigma: Tensor,    # (B, D, T) float — sigma at the START of each decision
    final_c: Tensor,           # (B, T)   bool   — c after the last decision
    final_sigma: Tensor,       # (B, T)   float  — sigma after the last decision
    t_decision: Tensor,        # (D,)     float  — clock at the START of each decision
    task_valid: Tensor,        # (B, T)   bool
    horizon_T: float,
    dt: float,
    no_revisit: bool = False,  # marginal R: hold-until-first-reversal (no re-grab)
) -> dict[str, Tensor]:
    """Pure label computation from recorded per-decision trajectories."""
    B, D, T = decision_c.shape
    tv = task_valid.unsqueeze(1)                                   # (B, 1, T)
    denom = (horizon_T - t_decision).clamp_min(dt)                # (D,)

    if no_revisit:
        # MARGINAL R: held integral only up to the FIRST reversal at/after each decision,
        # counterfactually assuming the policy does NOT come back to re-grab the task. This
        # removes the double-count where future re-completions inflate R under heavy churn.
        c_next_r = torch.cat([decision_c[:, 1:], final_c.unsqueeze(1)], dim=1)     # (B, D, T)
        revert = decision_c & ~c_next_r                                            # 1->0 during d
        fwd = torch.zeros(B, D + 1, T, dtype=decision_held.dtype, device=decision_held.device)
        fwd[:, 1:] = decision_held.cumsum(1)                                       # prefix sums
        nxt = torch.full((B, T), D, dtype=torch.long, device=decision_c.device)   # sentinel = "none"
        end1 = torch.empty(B, D, T, dtype=torch.long, device=decision_c.device)
        for d in range(D - 1, -1, -1):                                             # first revert >= d
            nxt = torch.where(revert[:, d], torch.full_like(nxt, d), nxt)
            end1[:, d] = nxt
        end1 = (end1 + 1).clamp(max=D)                                             # held-through endpoint (+1)
        held = torch.gather(fwd, 1, end1) - fwd[:, :D]                             # (B, D, T)
        R = (held / denom.view(1, D, 1)).clamp(0.0, 1.0)
    else:
        # realized held fraction over the WHOLE remaining episode (includes re-completions)
        remaining = decision_held.flip(1).cumsum(1).flip(1)           # (B, D, T)
        R = (remaining / denom.view(1, D, 1)).clamp(0.0, 1.0)

    # look-ahead views (append the post-episode state as the (D+1)-th column)
    c_next = torch.cat([decision_c[:, 1:], final_c.unsqueeze(1)], dim=1)        # (B, D, T)
    sigma_next = torch.cat([decision_sigma[:, 1:], final_sigma.unsqueeze(1)], dim=1)

    completing = (~decision_c) & c_next                          # 0->1 during decision d
    serviced = (sigma_next > decision_sigma + 1e-6) | completing  # service progressed

    # a decision is in the completing run if it is incomplete, serviced, and leads
    # (possibly through later serviced decisions) to a completion — backward pass
    in_run = torch.zeros_like(completing)
    nxt = torch.zeros(B, T, dtype=torch.bool, device=decision_c.device)
    for d in range(D - 1, -1, -1):
        here = (~decision_c[:, d]) & serviced[:, d] & (completing[:, d] | nxt)
        in_run[:, d] = here
        nxt = here

    dense_mask = decision_c & tv                                 # complete-task labels
    event_mask = in_run & tv                                     # incomplete acquisition-run labels
    return {"R": R, "dense_mask": dense_mask, "event_mask": event_mask, "t_decision": t_decision}


class RetentionLabelRecorder:
    """Records per-decision trajectories over a rollout, then emits labels.

    Wrap one episode of core rollout::

        rec = RetentionLabelRecorder()
        core.reset(seed)
        for d in range(cfg.n_decisions):
            rec.before_decision(core)
            core.step(actions_d)
            rec.after_decision(core)
        labels = rec.finalize(core)   # dict of (B, D, T) tensors + masks

    Requires the core's telemetry accumulator (``task_held_time``), which is created
    by :meth:`CanonicalCore.reset`.
    """

    def __init__(self) -> None:
        self._c: list[Tensor] = []
        self._sigma: list[Tensor] = []
        self._held: list[Tensor] = []
        self._t: list[float] = []
        self._held_before: Tensor | None = None

    def before_decision(self, core: CanonicalCore) -> None:
        s = core.state
        self._c.append(s.task_c.clone())
        self._sigma.append(s.sigma.clone())
        self._t.append(float(s.t.reshape(-1)[0]))
        assert core.telemetry is not None, "labels require the core telemetry accumulator"
        self._held_before = core.telemetry.task_held_time.clone()

    def after_decision(self, core: CanonicalCore) -> None:
        assert self._held_before is not None, "before_decision() must precede after_decision()"
        self._held.append(core.telemetry.task_held_time - self._held_before)

    def finalize(self, core: CanonicalCore) -> dict[str, Tensor]:
        cfg: CanonicalConfig = core.cfg
        decision_c = torch.stack(self._c, dim=1)
        decision_sigma = torch.stack(self._sigma, dim=1)
        decision_held = torch.stack(self._held, dim=1)
        t_decision = torch.tensor(self._t, device=decision_c.device, dtype=decision_held.dtype)
        return compute_retention_labels(
            decision_c, decision_held, decision_sigma,
            core.state.task_c, core.state.sigma, t_decision, core.state.task_valid,
            cfg.horizon_T, cfg.dt, no_revisit=cfg.retention_no_revisit,
        )
