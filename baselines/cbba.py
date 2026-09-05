"""CBBA — Consensus-Based Bundle Algorithm (Choi, Brunet, How 2009), plan Section 6.

Each robot allocates one objective per decision, so the bundle length is 1 and CBBA
reduces to its single-item consensus auction: robots bid a time-discounted marginal
reward on candidate tasks and each task is awarded to the highest bidder, with
conflicts resolved consistently. The environment's greedy matcher implements exactly
that winner-take-all, conflict-free consensus, so CBBA here is the *scoring* (the
diminishing-marginal-gain, time-discounted DMG bid) plus that shared resolution.

Distinct from ``greedy`` (value / travel-time) through the ``gamma ** travel_time``
temporal discount and by bidding on threatened held tasks as well as acquisitions.
"""
from __future__ import annotations

import torch
from torch import Tensor

from contested.config import CanonicalConfig
from contested.core import CanonicalState
from .base import BaselinePolicy, adv_threat, blank_scores, robot_task_geometry


class CBBA(BaselinePolicy):
    name = "cbba"

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma       # temporal discount on reward-time-to-service (DMG scoring)

    def scores(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        _, travel = robot_task_geometry(state, cfg)                     # (B, R, T)
        discount = self.gamma ** travel                                # DMG time discount
        w = state.task_w.unsqueeze(1)                                  # (B, 1, T)
        threat = adv_threat(state, cfg).unsqueeze(1)                   # (B, 1, T)
        incomplete = (~state.task_c & state.task_valid).unsqueeze(1).to(travel.dtype)
        complete = (state.task_c & state.task_valid).unsqueeze(1).to(travel.dtype)

        acquire = incomplete * w * discount
        defend = complete * w * (threat > 0).to(travel.dtype) * discount

        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = acquire
        s[:, :, cfg.max_tasks:2 * cfg.max_tasks] = defend
        return s
