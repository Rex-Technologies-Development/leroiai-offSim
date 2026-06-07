"""Skill-staged curriculum for VEX Push Back 2-robot team training.

Each stage adds complexity on top of the previous one. The idea is to let the
policy master one skill before having to handle the next.

  Stage 1  (0)        — 2 allies, all blue. No opponents, no failures.
                        Goal: discover collect → score quickly.
  Stage 2  (200K)     — 2 allies, mixed colors. No opponents, no failures.
                        Goal: learn coordinated collect → score.
  Stage 3  (500K)     — 2 allies vs 1 opponent (random). Mixed colors.
                        Goal: learn DESCORE / DEFEND interactions.
  Stage 4  (800K)     — 2 allies vs 2 opponents (mixed). Medium failures.
                        Goal: full 2v2 match conditions.

Apply via CurriculumCallback in train_sim.py. Each config key is read by
SingleAgentWrapper.apply_curriculum_config and pushed into VexAIEnv live.
"""

from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


# (timestep_threshold, config_dict)
DEFAULT_STAGES = [
    # ─── Stage 1: all-blue, solo, no failures — discover collect→score ──
    (0, {
        "all_blue_only":  True,
        "num_allies":     2,
        "num_opponents":  0,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.0,
            "object_stolen_rate":    0.0,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 2: mixed colors, solo, no failures — color-sort + scoring ─
    (200_000, {
        "all_blue_only":  False,
        "num_allies":     2,
        "num_opponents":  0,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.0,
            "object_stolen_rate":    0.0,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 3: 1 opponent (random), light failures — start contested ─
    (500_000, {
        "all_blue_only":  False,
        "num_allies":     2,
        "num_opponents":  1,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.02,
            "object_stolen_rate":    0.03,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 4: 2 opponents (mixed strategy), medium failures ─────────
    (800_000, {
        "all_blue_only":  False,
        "num_allies":     2,
        "num_opponents":  2,
        "opponent_type":  "mixed",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.05,
            "object_stolen_rate":    0.08,
            "teammate_offline_rate": 0.0,
        },
    }),
]


def stages_for_allies(num_allies: int, stages: list | None = None) -> list:
    """Return curriculum stages with num_allies set for the requested team size."""
    base = stages or DEFAULT_STAGES
    out = []
    for threshold, config in base:
        cfg = dict(config)
        cfg["num_allies"] = max(1, min(int(num_allies), 2))
        out.append((threshold, cfg))
    return out


class CurriculumCallback(BaseCallback):
    """SB3 callback that advances curriculum stages over training.

    Each stage update is pushed to every env in the VecEnv via env_method.
    Changes take effect on the NEXT episode reset (mid-episode env state
    is not touched).
    """

    def __init__(self, stages: list | None = None, eval_env=None, verbose: int = 0):
        super().__init__(verbose)
        self.stages = stages or DEFAULT_STAGES
        self.eval_env = eval_env
        self._current_stage = -1

    def _on_step(self) -> bool:
        ts = self.num_timesteps

        # Find the highest-threshold stage we've reached
        stage_idx = 0
        for i, (threshold, _) in enumerate(self.stages):
            if ts >= threshold:
                stage_idx = i

        if stage_idx == self._current_stage:
            return True

        self._current_stage = stage_idx
        _, config = self.stages[stage_idx]

        if self.verbose:
            print(f"[Curriculum] -> Stage {stage_idx + 1} at {ts:,} steps  "
                  f"(allies={config.get('num_allies', 2)}, "
                  f"all_blue={config.get('all_blue_only')}, "
                  f"opps={config.get('num_opponents')}, "
                  f"fails={config.get('failures', {}).get('stuck_rate', 0):.2f})")

        # Push to every worker — DummyVecEnv and SubprocVecEnv both supported.
        self.training_env.env_method("apply_curriculum_config", config)
        if self.eval_env is not None:
            self.eval_env.env_method("apply_curriculum_config", config)
        return True
