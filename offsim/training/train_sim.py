"""Phase 1: PPO/DQN training in simulation."""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sim.env import SingleAgentWrapper
from sim.failure import FailureConfig
from training.curriculum import CurriculumCallback
from training.callbacks import ScoreLoggingCallback


def make_env(failure_config=None, opponent_type="random", render_mode=None):
    def _init():
        return SingleAgentWrapper(
            render_mode=render_mode,
            failure_config=failure_config or FailureConfig.none(),
            opponent_type=opponent_type,
        )
    return _init


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 4,
    lr: float = 3e-4,
    checkpoint_freq: int = 50_000,
    eval_freq: int = 25_000,
    output_dir: str = "models",
    resume: str | None = None,
):
    env = make_vec_env(make_env(), n_envs=n_envs)

    eval_env = make_vec_env(
        make_env(
            failure_config=FailureConfig(teammate_fail_rate=0.15),
            opponent_type="mixed",
        ),
        n_envs=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="vex_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(output_dir, "best"),
            log_path=os.path.join(output_dir, "logs"),
            eval_freq=eval_freq,
            n_eval_episodes=20,
            deterministic=True,
        ),
        CurriculumCallback(verbose=1),
        ScoreLoggingCallback(),
    ]

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(output_dir, "tb_logs"),
        )

    print(f"Training for {total_timesteps} timesteps with curriculum...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    final_path = os.path.join(output_dir, "final_model")
    model.save(final_path)
    print(f"Saved to {final_path}")

    env.close()
    eval_env.close()
