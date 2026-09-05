"""Component D: comparison baselines (plan Section 6).

Allocators evaluated against TENURE in the *same* canonical environment. Fair
adaptation (plan Section 6) is non-negotiable and enforced by the harness, not by
per-baseline scripts: every baseline sees adversary features (it reads the full
:class:`~contested.core.CanonicalState`), and learned baselines are retrained from
scratch in the contested environment with matched sample/wall-clock budgets.

Self-contained here: ``greedy``, ``defensive_heuristic``, ``cbba``. The learned and
external-code baselines (MAPPO/QMIX via BenchMARL, CapAM, RTAW, DC-MRTA) have typed
stubs with availability notes — RTAW/DC-MRTA code availability is the week-one
action item flagged in the plan.
"""
from __future__ import annotations

from .base import BaselinePolicy, adv_threat, robot_task_geometry
from .greedy import GreedyAllocator
from .defensive_heuristic import DefensiveHeuristic
from .cbba import CBBA

REGISTRY = {
    "greedy": GreedyAllocator,
    "defensive": DefensiveHeuristic,
    "cbba": CBBA,
}

__all__ = ["BaselinePolicy", "GreedyAllocator", "DefensiveHeuristic", "CBBA",
           "REGISTRY", "adv_threat", "robot_task_geometry"]
