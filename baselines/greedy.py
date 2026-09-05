"""Greedy value/geometry allocator (plan Section 6).

The conventional allocator: send robots to the highest value-per-travel-time
*incomplete* task. It never defends, so at ``alpha > 0`` it bleeds held value to
reversal — which is exactly the gap retention estimation is meant to close. This is
the natural "no retention, no defense" reference point.
"""
from __future__ import annotations

import torch
from torch import Tensor

from contested.config import CanonicalConfig
from contested.core import CanonicalState
from .base import BaselinePolicy, blank_scores, robot_task_geometry


class GreedyAllocator(BaselinePolicy):
    name = "greedy"

    def scores(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        _, travel = robot_task_geometry(state, cfg)                     # (B, R, T)
        w = state.task_w.unsqueeze(1)                                   # (B, 1, T)
        incomplete = (~state.task_c & state.task_valid).unsqueeze(1).to(travel.dtype)
        acquire = incomplete * w / (travel + 0.1)                       # value per travel time

        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = acquire
        return s
