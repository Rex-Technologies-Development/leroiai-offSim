"""Skill-staged curriculum for VEX Push Back single-robot training.

Each stage adds complexity on top of the previous one. The idea is to let the
policy master one skill before having to handle the next.

  Stage 1  (0)        — Mixed colors. No opponents, no failures.
                        Goal: learn the collect → score chain AND that red
                        balls should be avoided (via collect_red penalty).
  Stage 2  (300K)     — Mixed colors. No opponents. LIGHT failures.
                        Goal: robustness to stuck/stolen-ball noise.
  Stage 3  (600K)     — Add 1 opponent (random policy). Mixed colors.
                        Goal: learn DESCORE / DEFEND interactions.
  Stage 4  (1M)       — 1 opponent with mixed strategy. Medium failures.
                        Goal: full match conditions.

Opt-in WARMUP stage (not in DEFAULT_STAGES):
  An ALL-BLUE warmup can be prepended for very long training runs where you
  want the policy to discover collect→score before introducing the
  red-discrimination problem. Build a custom stage list and pass it to
  CurriculumCallback if you want it:

      WARMUP_ALL_BLUE = (0, {"all_blue_only": True, "num_opponents": 0,
                             "opponent_type": "random",
                             "failures": {...all zero...}})
      stages = [WARMUP_ALL_BLUE] + DEFAULT_STAGES
      CurriculumCallback(stages=stages)

Apply via CurriculumCallback in train_sim.py. Each config key is read by
SingleAgentWrapper.apply_curriculum_config and pushed into VexAIEnv live.
"""

from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


# (timestep_threshold, config_dict)
DEFAULT_STAGES = [
    # ─── Stage 1: mixed colors, solo, no failures — learn the basics ────
    (0, {
        "all_blue_only":  False,
        "num_opponents":  0,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.0,
            "object_stolen_rate":    0.0,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 2: mixed, solo, light failures — robustness ──────────────
    (300_000, {
        "all_blue_only":  False,
        "num_opponents":  0,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.02,
            "object_stolen_rate":    0.03,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 3: 1 opponent (random), light failures — start contested ─
    (600_000, {
        "all_blue_only":  False,
        "num_opponents":  1,
        "opponent_type":  "random",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.02,
            "object_stolen_rate":    0.03,
            "teammate_offline_rate": 0.0,
        },
    }),
    # ─── Stage 4: 1 opponent (mixed strategy), medium failures ──────────
    (1_000_000, {
        "all_blue_only":  False,
        "num_opponents":  1,
        "opponent_type":  "mixed",
        "failures": {
            "teammate_fail_rate":    0.0,
            "stuck_rate":            0.05,
            "object_stolen_rate":    0.08,
            "teammate_offline_rate": 0.0,
        },
    }),
]


class CurriculumCallback(BaseCallback):
    """SB3 callback that advances curriculum stages over training.

    Each stage update is pushed to every env in the VecEnv via env_method.
    Changes take effect on the NEXT episode reset (mid-episode env state
    is not touched).
    """

    def __init__(self, stages: list | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.stages = stages or DEFAULT_STAGES
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
                  f"(all_blue={config.get('all_blue_only')}, "
                  f"opps={config.get('num_opponents')}, "
                  f"fails={config.get('failures', {}).get('stuck_rate', 0):.2f})")

        # Push to every worker — DummyVecEnv and SubprocVecEnv both supported.
        self.training_env.env_method("apply_curriculum_config", config)
        return True
