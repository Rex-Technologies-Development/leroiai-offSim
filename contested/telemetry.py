"""Per-episode telemetry accumulation (plan Section 3.11).

An :class:`EpisodeTelemetry` is owned optionally by the core and updated through
light hooks in ``tick`` / ``step``. It never influences dynamics (the reference-
match path runs with no accumulator attached), so it cannot perturb the physics.

Environment-computable metrics live here. The model-dependent ones from Section
3.11 — ``mean_R_error`` and ``R_calibration_bins`` — need the retention head's
predictions and are computed in the training loop (E7), not the environment.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import CanonicalConfig
from .core import CanonicalState

_EPS = 1e-9


class EpisodeTelemetry:
    """Accumulates behavioural metrics over one batch of episodes."""

    def __init__(self, state: CanonicalState, cfg: CanonicalConfig, dtype: torch.dtype):
        B, T = state.task_valid.shape
        dev = state.device
        self.cfg = cfg
        self.dt = cfg.dt
        self.dtype = dtype
        z = lambda: torch.zeros(B, device=dev, dtype=dtype)
        self.task_held_time = torch.zeros(B, T, device=dev, dtype=dtype)  # int c_i dt
        self.ever_complete = state.task_c.clone()
        self.n_reversals = z()
        self.n_recompletions = z()
        self.defend_count = z()
        self.decision_robot_count = z()
        self.assign_changes = z()
        self.prev_assignment: Tensor | None = None
        self.n_decisions = 0
        self.n_ticks = 0
        self.standoff_task_ticks = z()  # PHASE 2: (task, tick) slots frozen in a matched-no-incumbent standoff

    def on_tick(self, state: CanonicalState, complete: Tensor, revert: Tensor,
                standoff: Tensor | None = None) -> None:
        d = self.dtype
        self.task_held_time += state.task_c.to(d) * self.dt
        self.n_reversals += revert.sum(-1).to(d)
        self.n_recompletions += (complete & self.ever_complete).sum(-1).to(d)
        self.ever_complete = self.ever_complete | complete
        self.n_ticks += 1
        if standoff is not None:
            self.standoff_task_ticks += standoff.sum(-1).to(d)

    def on_decision(self, state: CanonicalState, assignment: Tensor) -> None:
        d = self.dtype
        T = self.cfg.max_tasks
        rv = state.robot_valid
        is_defend = (assignment >= T) & (assignment < 2 * T) & rv
        self.defend_count += is_defend.sum(-1).to(d)
        self.decision_robot_count += rv.sum(-1).to(d)
        if self.prev_assignment is not None:
            self.assign_changes += ((assignment != self.prev_assignment) & rv).sum(-1).to(d)
        self.prev_assignment = assignment.clone()
        self.n_decisions += 1

    def summary(self, state: CanonicalState) -> dict[str, Tensor]:
        cfg, d = self.cfg, self.dtype
        Th = cfg.horizon_T
        tv = state.task_valid.to(d)
        total_w = (state.task_w * tv).sum(-1).clamp_min(_EPS)

        j_h = state.held_integral / (Th * total_w)                       # normalized blue held value
        j_h_red = state.held_integral_red / (Th * total_w)               # normalized red held value
        j_h_diff = j_h - j_h_red                                         # differential objective (Override score diff)
        j_t = (state.task_w * state.task_c.to(d) * tv).sum(-1) / total_w  # terminal held value
        n_valid = tv.sum(-1).clamp_min(1.0)
        standoff_fraction = self.standoff_task_ticks / (max(self.n_ticks, 1) * n_valid)  # 0 => no deadlock

        completed = (state.first_complete_t >= 0).to(d)                  # (B, T)
        remaining = ((Th - state.first_complete_t).clamp_min(0.0)) * completed
        held_completed = self.task_held_time * completed
        retention_rate = held_completed.sum(-1) / remaining.sum(-1).clamp_min(_EPS)

        defense_fraction = self.defend_count / self.decision_robot_count.clamp_min(_EPS)
        minutes = (state.t / 60.0).clamp_min(_EPS)
        n_rob = state.robot_valid.sum(-1).clamp_min(1).to(d)
        assignment_churn = self.assign_changes / n_rob / minutes

        fct = state.first_complete_t.clone()
        fct = torch.where(fct < 0, torch.full_like(fct, Th), fct)        # never completed -> capped at T
        time_to_first = fct.min(-1).values

        return {
            "J_H": j_h,
            "J_H_red": j_h_red,
            "J_H_diff": j_h_diff,
            "standoff_fraction": standoff_fraction,
            "J_T": j_t,
            "retention_rate": retention_rate,
            "defense_fraction": defense_fraction,
            "n_reversals": self.n_reversals.clone(),
            "n_recompletions": self.n_recompletions.clone(),
            "assignment_churn": assignment_churn,
            "time_to_first_completion": time_to_first,
            "per_task_held_fraction": self.task_held_time / Th,          # (B, T)
        }
