"""Common baseline interface and shared geometry (plan Section 6).

Heuristic/optimization baselines act directly on the :class:`CanonicalState` (full
observability, like TENURE), resolving conflicts with the environment's own greedy
matcher so their assignments obey the same masking and multi-assignment rules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from contested.actions import action_masks, greedy_assign
from contested.config import CanonicalConfig
from contested.core import CanonicalState

_NEG = -1e30


class BaselinePolicy(ABC):
    """Maps a batched state to per-robot action indices ``(B, max_robots)``."""

    name: str = "baseline"

    @abstractmethod
    def scores(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        """Return ``(B, R, A)`` action scores; conflict resolution is shared."""

    def act(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        return greedy_assign(self.scores(state, cfg), action_masks(state, cfg), cfg)


def robot_task_geometry(state: CanonicalState, cfg: CanonicalConfig) -> tuple[Tensor, Tensor]:
    """Distance and travel time from every robot to every task -> both ``(B, R, T)``."""
    d = torch.linalg.vector_norm(state.robot_pos.unsqueeze(2) - state.task_pos.unsqueeze(1), dim=-1)
    return d, d / cfg.v_max


def adv_threat(state: CanonicalState, cfg: CanonicalConfig, radius_factor: float = 3.0) -> Tensor:
    """Count of adversaries within ``radius_factor * service_radius`` of each task -> ``(B, T)``."""
    d = torch.linalg.vector_norm(state.adv_pos.unsqueeze(2) - state.task_pos.unsqueeze(1), dim=-1)
    near = (d < radius_factor * cfg.service_radius) & state.adv_valid.unsqueeze(-1) & state.task_valid.unsqueeze(1)
    return near.sum(1).to(state.robot_pos.dtype)


def blank_scores(state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
    B, R = state.robot_pos.shape[0], cfg.max_robots
    s = torch.full((B, R, cfg.action_dim), _NEG, device=state.device, dtype=state.robot_pos.dtype)
    s[:, :, cfg.idle_action] = 1e-3       # a tiny IDLE floor so a robot idles rather than forcing a bad pick
    return s
