"""Randomized failure injection for robustness training.

Configurable per-env-instance. Probabilities are ramped up via
curriculum training (see training/curriculum.py).
"""

from __future__ import annotations
import dataclasses
import numpy as np


@dataclasses.dataclass
class FailureConfig:
    teammate_fail_rate: float = 0.15          # chance teammate's action does nothing this step
    action_delay_range: tuple = (1.0, 2.5)    # multiplier on movement speed (>1 = slower)
    object_stolen_rate: float = 0.10          # chance target object gets taken by opponent
    stuck_rate: float = 0.05                  # chance robot gets stuck for 2-3 seconds
    teammate_offline_rate: float = 0.02       # chance teammate goes permanently offline this episode

    @staticmethod
    def none() -> "FailureConfig":
        """No failures — for clean training or demos."""
        return FailureConfig(
            teammate_fail_rate=0.0,
            action_delay_range=(1.0, 1.0),
            object_stolen_rate=0.0,
            stuck_rate=0.0,
            teammate_offline_rate=0.0,
        )


class FailureInjector:
    """Applies randomized failures to the sim each tick."""

    def __init__(self, config: FailureConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.teammate_offline: bool = False
        self.stuck_ticks: dict[int, int] = {}          # robot_idx -> remaining stuck ticks
        self.speed_multipliers: dict[int, float] = {}  # robot_idx -> current speed mult

    def reset(self):
        self.teammate_offline = False
        self.stuck_ticks.clear()
        self.speed_multipliers.clear()

    def on_tick(self, num_robots: int = 2):
        """Roll failures for this tick. Call once per sim tick."""
        # Teammate offline (permanent, checked once per tick until it triggers)
        if not self.teammate_offline:
            # Scale rate to per-tick probability (original is per-episode)
            # ~2400 ticks per episode, so per-tick ≈ rate / 2400
            per_tick = 1.0 - (1.0 - self.config.teammate_offline_rate) ** (1.0 / 2400.0)
            if self.rng.random() < per_tick:
                self.teammate_offline = True

        # Stuck
        for idx in range(num_robots):
            if idx in self.stuck_ticks and self.stuck_ticks[idx] > 0:
                self.stuck_ticks[idx] -= 1
            elif self.rng.random() < self.config.stuck_rate * 0.01:
                # stuck_rate is per-decision, scale to per-tick
                stuck_secs = self.rng.uniform(2.0, 3.0)
                from sim.config import DT
                self.stuck_ticks[idx] = int(stuck_secs / DT)

        # Speed multipliers (action delay)
        lo, hi = self.config.action_delay_range
        for idx in range(num_robots):
            self.speed_multipliers[idx] = 1.0 / self.rng.uniform(lo, hi)

    def is_stuck(self, robot_idx: int) -> bool:
        return self.stuck_ticks.get(robot_idx, 0) > 0

    def get_speed_mult(self, robot_idx: int) -> float:
        return self.speed_multipliers.get(robot_idx, 1.0)

    def should_teammate_fail(self) -> bool:
        """Per-decision check: does the teammate's action fail this decision?"""
        return self.rng.random() < self.config.teammate_fail_rate

    def should_steal_object(self) -> bool:
        """Per-decision check: does an opponent steal the target object?"""
        return self.rng.random() < self.config.object_stolen_rate
