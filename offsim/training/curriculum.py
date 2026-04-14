"""Difficulty ramp schedule for curriculum training.

Stage 1 (0-500K steps):    random opponents, no failures
Stage 2 (500K-1M steps):   greedy opponents, 10% teammate failure
Stage 3 (1M-2M+ steps):    mixed opponents, 20% failure, occasional offline
"""

from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback
from sim.opponent import get_opponent


# Default curriculum stages: (timestep_threshold, config)
DEFAULT_STAGES = [
    (0, {
        "opponent_type": "random",
        "failures": {
            "teammate_fail_rate": 0.0,
            "stuck_rate": 0.0,
            "object_stolen_rate": 0.0,
            "teammate_offline_rate": 0.0,
        },
    }),
    (500_000, {
        "opponent_type": "greedy",
        "failures": {
            "teammate_fail_rate": 0.10,
            "stuck_rate": 0.03,
            "object_stolen_rate": 0.05,
            "teammate_offline_rate": 0.0,
        },
    }),
    (1_000_000, {
        "opponent_type": "mixed",
        "failures": {
            "teammate_fail_rate": 0.20,
            "stuck_rate": 0.05,
            "object_stolen_rate": 0.10,
            "teammate_offline_rate": 0.02,
        },
    }),
]


class CurriculumCallback(BaseCallback):
    """SB3 callback that ramps opponent difficulty and failure rates."""

    def __init__(self, stages: list | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.stages = stages or DEFAULT_STAGES
        self._current_stage = -1

    def _on_step(self) -> bool:
        ts = self.num_timesteps

        # Find active stage
        stage_idx = 0
        for i, (threshold, _) in enumerate(self.stages):
            if ts >= threshold:
                stage_idx = i

        if stage_idx == self._current_stage:
            return True

        self._current_stage = stage_idx
        _, config = self.stages[stage_idx]

        if self.verbose:
            print(f"[Curriculum] Stage {stage_idx} at {ts} steps: "
                  f"opponent={config.get('opponent_type', '?')}")

        # Apply to all envs
        for env in self.training_env.envs:
            inner = env.unwrapped
            if hasattr(inner, 'env'):
                inner = inner.env

            if "opponent_type" in config and hasattr(inner, 'opponent_policy'):
                inner.opponent_policy = get_opponent(config["opponent_type"])

            if "failures" in config and hasattr(inner, 'failure_config'):
                for key, val in config["failures"].items():
                    setattr(inner.failure_config, key, val)

        return True
