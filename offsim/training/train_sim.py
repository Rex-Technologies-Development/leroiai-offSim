"""MaskablePPO training integration for OverrideStrategyEnv."""
from __future__ import annotations
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
try:
    from ..sim.env import OverrideStrategyEnv
    from .callbacks import ScoreLoggingCallback
except ImportError:  # direct execution through offsim/main.py
    from sim.env import OverrideStrategyEnv
    from training.callbacks import ScoreLoggingCallback


def make_env(chassis="tank", opponent="mixed", render_mode=None):
    def factory(): return OverrideStrategyEnv(chassis=chassis, opponent=opponent, render_mode=render_mode)
    return factory


def train(total_timesteps=1_000_000, n_envs=4, learning_rate=3e-4, output_dir="models",
          chassis="tank", opponent="mixed", device="auto", resume=None, n_steps=256):
    os.makedirs(output_dir, exist_ok=True)
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    train_env = make_vec_env(make_env(chassis, opponent), n_envs=n_envs, vec_env_cls=vec_cls)
    eval_env = make_vec_env(make_env(chassis, opponent), n_envs=1, vec_env_cls=DummyVecEnv)
    callbacks = [
        CheckpointCallback(save_freq=max(10_000//n_envs, 1), save_path=os.path.join(output_dir, "checkpoints"), name_prefix="override_ppo"),
        MaskableEvalCallback(eval_env, best_model_save_path=os.path.join(output_dir, "best"), log_path=os.path.join(output_dir, "logs"), eval_freq=max(25_000//n_envs, 1), n_eval_episodes=5),
        ScoreLoggingCallback(),
    ]
    if resume:
        model = MaskablePPO.load(resume, env=train_env, device=device)
    else:
        batch_size = min(128, max(8, n_steps*n_envs))
        model = MaskablePPO("MlpPolicy", train_env, learning_rate=learning_rate, n_steps=n_steps,
                            batch_size=batch_size, n_epochs=5, gamma=0.995, gae_lambda=0.95,
                            ent_coef=0.01, verbose=1, tensorboard_log=os.path.join(output_dir, "tb_logs"), device=device)
    interrupted = False
    try:
        model.learn(total_timesteps=int(total_timesteps), callback=callbacks, reset_num_timesteps=resume is None)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        path = os.path.join(output_dir, "interrupted_model" if interrupted else "final_model")
        model.save(path); train_env.close(); eval_env.close(); print(f"saved {path}.zip")
    return path + ".zip"
