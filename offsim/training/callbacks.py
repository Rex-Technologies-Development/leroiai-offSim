"""Training callbacks — TensorBoard extras, render callback, custom logging."""

from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


class ScoreLoggingCallback(BaseCallback):
    """Logs game scores and win rate to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_scores = []
        self._episode_opp_scores = []

    def _on_step(self) -> bool:
        # Check for episode ends
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                env = self.training_env.envs[i].unwrapped
                if hasattr(env, 'env'):
                    env = env.env
                if hasattr(env, 'field'):
                    self._episode_scores.append(env.field.my_score)
                    self._episode_opp_scores.append(env.field.opponent_score)

        # Log every 100 episodes
        if len(self._episode_scores) >= 100:
            import numpy as np
            scores = np.array(self._episode_scores[-100:])
            opp_scores = np.array(self._episode_opp_scores[-100:])
            wins = np.sum(scores > opp_scores)
            avg_margin = np.mean(scores - opp_scores)

            self.logger.record("game/win_rate", wins / 100.0)
            self.logger.record("game/avg_score", np.mean(scores))
            self.logger.record("game/avg_opp_score", np.mean(opp_scores))
            self.logger.record("game/avg_margin", avg_margin)

            self._episode_scores.clear()
            self._episode_opp_scores.clear()

        return True


class RenderCallback(BaseCallback):
    """Renders the env periodically during training."""

    def __init__(self, render_every_episodes: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.render_every = render_every_episodes
        self._episode_count = 0

    def _on_step(self) -> bool:
        for done in self.locals.get("dones", []):
            if done:
                self._episode_count += 1
                if self._episode_count % self.render_every == 0:
                    env = self.training_env.envs[0].unwrapped
                    if hasattr(env, 'env'):
                        env = env.env
                    if hasattr(env, 'render'):
                        env.render()
        return True
