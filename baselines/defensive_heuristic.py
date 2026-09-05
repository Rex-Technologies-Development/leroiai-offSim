"""Defensive heuristic allocator (plan Section 6).

Prioritises DEFEND on *held tasks that are actually under threat* (an adversary is
near), then fills remaining robots with greedy acquisition. It has the defend
*action* but no learned retention estimate — a hand-coded stand-in for "defend
without retention". At ``alpha > 0`` it should retain more than pure greedy but can
over-commit defenders to low-value tasks, which is the behaviour retention fixes.
"""
from __future__ import annotations

import torch
from torch import Tensor

from contested.config import CanonicalConfig
from contested.core import CanonicalState
from .base import BaselinePolicy, adv_threat, blank_scores, robot_task_geometry

_DEFEND_PRIORITY = 100.0     # defending a threatened held task outranks acquisition


class DefensiveHeuristic(BaselinePolicy):
    name = "defensive"

    def scores(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        _, travel = robot_task_geometry(state, cfg)                     # (B, R, T)
        w = state.task_w                                               # (B, T)
        threat = adv_threat(state, cfg)                               # (B, T)
        complete = (state.task_c & state.task_valid).to(travel.dtype)
        incomplete = (~state.task_c & state.task_valid).to(travel.dtype)

        acquire = (incomplete * w).unsqueeze(1) / (travel + 0.1)
        defend = _DEFEND_PRIORITY * (complete * w * threat).unsqueeze(1) / (travel + 0.1)

        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = acquire
        s[:, :, cfg.max_tasks:2 * cfg.max_tasks] = defend             # 0 where unthreatened -> acquisition wins
        return s
