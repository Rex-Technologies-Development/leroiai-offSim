"""MaskablePPO training — single blue robot, no opponents, 60-second episodes.

Uses sb3-contrib MaskablePPO so invalid actions (score with no balls,
descore when nothing is scored, etc.) are never sampled by the policy.

Usage:
    python main.py train --timesteps 1000000 --n-envs 4

The robot:
  - Starts with randomised ball layout each episode.
  - Has access to 10 actions (collect, score long/center, descore, eject, regions, idle).
  - Invalid actions are masked per-step via env.action_masks().
  - Receives per-decision reward based on blue-ball scores, avoiding red-ball
    scores (which give the opponent points), and quadrant control.
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
except ImportError as e:
    raise ImportError(
        "sb3-contrib is required. Action masking is central to this project — "
        "falling back to vanilla PPO would silently break the action contract. "
        "Install with: pip install sb3-contrib"
    ) from e
print("[train] Using MaskablePPO (sb3-contrib) — action masks active")

from sim.env import SingleAgentWrapper
from sim.config import Action
from sim.failure import FailureConfig
from training.curriculum import CurriculumCallback


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(render_mode=None, use_timer=True):
    """Return a callable that creates a fresh SingleAgentWrapper episode.

    Starts at curriculum Stage 1 (no opponents, no failures). The
    CurriculumCallback ramps opponents and failure rates as training progresses.

    When render_mode == "human", the renderer is primed inside the worker so
    the pygame window opens on that worker's process (important for
    SubprocVecEnv where each worker is its own process and gets its own window).
    """
    def _init():
        env = SingleAgentWrapper(
            render_mode=render_mode,
            failure_config=FailureConfig.none(),
            opponent_type="random",
            num_allies=1,
            num_opponents=0,
            use_timer=use_timer,
        )
        if render_mode == "human":
            env.render()   # open pygame window in this worker's process
        return env
    return _init


# ---------------------------------------------------------------------------
# Logging / display callbacks
# ---------------------------------------------------------------------------
class EpisodeStatsCallback(BaseCallback):
    """Print per-episode stats and (in render mode) push them to the panel."""

    def __init__(self, log_freq_episodes: int = 50, rendered_env=None, verbose: int = 1):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_blue:    list[float] = []
        self._ep_red:     list[float] = []
        self._log_freq    = log_freq_episodes
        self._rendered_env = rendered_env   # SingleAgentWrapper with render_mode="human"
        # Per-env accumulator: under SubprocVecEnv each worker has its own episode,
        # so a single shared accumulator mixes rewards across envs. Allocated lazily
        # once we see the actual n_envs from the first _on_step call.
        self._cur_ep_rewards: np.ndarray | None = None
        self._n_episodes    = 0
        self._ppo_stats: dict = {}   # filled by _on_rollout_end after each policy update

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones", [])
        n_envs  = len(rewards) if len(rewards) else len(dones)
        if n_envs == 0:
            return True

        # Lazy-init the per-env reward accumulator now that we know n_envs.
        if self._cur_ep_rewards is None or len(self._cur_ep_rewards) != n_envs:
            self._cur_ep_rewards = np.zeros(n_envs, dtype=np.float64)

        # Accumulate per-env rewards independently
        self._cur_ep_rewards += np.asarray(rewards, dtype=np.float64)

        if len(dones) and any(dones):
            # Score lookup: fetch all finished envs' scores in one call
            done_indices = [i for i, d in enumerate(dones) if d]
            try:
                all_scores = self.training_env.env_method(
                    "_get_episode_scores", indices=done_indices
                )
            except Exception:
                all_scores = [(0.0, 0.0)] * len(done_indices)

            for i, scores in zip(done_indices, all_scores):
                self._n_episodes += 1
                self._ep_rewards.append(float(self._cur_ep_rewards[i]))
                b, red = float(scores[0]), float(scores[1])
                self._ep_blue.append(b)
                self._ep_red.append(red)
                self._cur_ep_rewards[i] = 0.0   # reset only the env that finished

            if self.verbose and self._n_episodes % self._log_freq == 0:
                n   = self._log_freq
                r   = np.mean(self._ep_rewards[-n:])
                bl  = np.mean(self._ep_blue[-n:])  if self._ep_blue  else 0
                rd  = np.mean(self._ep_red[-n:])   if self._ep_red   else 0
                print(f"[ep {self._n_episodes:5d}] reward={r:+.2f}  "
                      f"blue={bl:.1f}  red={rd:.1f}  "
                      f"steps={self.num_timesteps:,}")

        self._push_stats_to_all_envs()
        return True

    def _on_rollout_end(self) -> None:
        """Capture PPO training stats from the model logger after each policy update."""
        lv = getattr(self.model.logger, "name_to_value", {})
        self._ppo_stats = {
            "approx_kl":    lv.get("train/approx_kl"),
            "clip_fraction": lv.get("train/clip_fraction"),
            "clip_range":   lv.get("train/clip_range"),
            "entropy_loss": lv.get("train/entropy_loss"),
            "expl_variance": lv.get("train/explained_variance"),
            "learning_rate": lv.get("train/learning_rate"),
            "loss":          lv.get("train/loss"),
            "pg_loss":       lv.get("train/policy_gradient_loss"),
            "value_loss":    lv.get("train/value_loss"),
            "fps":           lv.get("time/fps"),
            "n_updates":     lv.get("train/n_updates"),
        }

    def _push_stats_to_all_envs(self):
        """Push training stats to every env's renderer via env_method.

        Each env gets a panel update tailored to ITS state:
          - Global fields (total_steps, n_episodes, ep_*) are shared
          - last_action is the action this env's policy just took
          - reward_breakdown is this env's last decision's components

        Works under both DummyVecEnv and SubprocVecEnv — each worker's
        SingleAgentWrapper.apply_training_stats merges the dict into its
        own renderer.training_stats (if a renderer exists in that worker).
        """
        from training.reward import REWARD_WEIGHTS
        shared: dict = {
            "total_steps":    self.num_timesteps,
            "n_episodes":     self._n_episodes,
            "ep_reward":      self._ep_rewards[-1] if self._ep_rewards else 0.0,
            "ep_blue":        self._ep_blue[-1]    if self._ep_blue    else 0.0,
            "ep_red":         self._ep_red[-1]     if self._ep_red     else 0.0,
            "reward_weights": REWARD_WEIGHTS,
            "ppo_stats":      self._ppo_stats,
        }

        actions = self.locals.get("actions", None)
        try:
            breakdowns = self.training_env.env_method("_get_reward_breakdown")
        except Exception:
            breakdowns = None

        n_envs = len(breakdowns) if breakdowns is not None else \
                 (len(actions) if actions is not None else 1)

        for i in range(n_envs):
            per_env = dict(shared)
            # Per-env action label (each subproc renders its own robot's choice)
            if actions is not None and i < len(actions):
                try:
                    per_env["last_action"] = Action(int(actions[i])).name[:16]
                except ValueError:
                    per_env["last_action"] = str(actions[i])
            # Per-env reward breakdown (each panel shows its own components)
            if breakdowns is not None and breakdowns[i]:
                per_env["reward_breakdown"] = dict(breakdowns[i])
            try:
                self.training_env.env_method(
                    "apply_training_stats", per_env, indices=[i]
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    lr: float = 3e-4,
    checkpoint_freq: int = 10_000,
    eval_freq: int = 25_000,
    eval_episodes: int = 10,
    output_dir: str = "models",
    resume: str | None = None,
    render: bool = False,
    device: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)

    # Render mode:
    #   n_envs == 1 → in-process DummyVecEnv with one pygame window
    #   n_envs >  1 → SubprocVecEnv, each worker opens its OWN pygame window.
    #                 Stats reach each panel via env_method (see EpisodeStatsCallback).
    rendered_env = None
    if render:
        if n_envs == 1:
            rendered_env = SingleAgentWrapper(
                render_mode="human",
                failure_config=FailureConfig.none(),
                opponent_type="random",
                num_allies=1,
                num_opponents=0,
                use_timer=True,
            )
            rendered_env.render()
            train_env = DummyVecEnv([lambda: rendered_env])
        else:
            train_env = make_vec_env(
                make_env(render_mode="human", use_timer=True),
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,
            )
    else:
        train_env = make_vec_env(
            make_env(use_timer=True),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )

    # Eval env — single process, timer on (always headless)
    eval_env = make_vec_env(
        make_env(use_timer=True),
        n_envs=1,
    )

    callbacks = [
        CurriculumCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(checkpoint_freq // n_envs, 1),
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="vex_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(output_dir, "best"),
            log_path=os.path.join(output_dir, "logs"),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=int(eval_episodes),
            deterministic=True,
            verbose=1,
        ),
        EpisodeStatsCallback(log_freq_episodes=100, rendered_env=rendered_env, verbose=1),
    ]

    # Resolve PyTorch device. SB3 accepts 'auto', 'cpu', 'cuda'.
    # IMPORTANT: if torch was installed as a CPU-only build, SB3 will silently
    # fall back to CPU even if you pass device='cuda'. We detect this and
    # fail loudly when the user explicitly requests CUDA.
    try:
        import torch
        torch_build = getattr(torch, "__version__", "unknown")
        torch_cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_build = "not-installed"
        torch_cuda_ver = None
        cuda_available = False

    if device == "auto":
        actual_device = "cuda" if cuda_available else "cpu"
    elif device == "cuda":
        actual_device = "cuda"
        if not cuda_available:
            raise RuntimeError(
                "--device cuda was requested, but this Python environment does not have CUDA-enabled PyTorch.\n"
                f"Detected torch={torch_build}, torch.version.cuda={torch_cuda_ver}, torch.cuda.is_available()={cuda_available}.\n\n"
                "Fix: install a CUDA-enabled PyTorch build (and a Python version supported by those wheels), or run with --device cpu."
            )
    else:
        actual_device = "cpu"

    # Extra guard: some CUDA wheels don't include kernels for very new GPUs.
    # In that case torch.cuda.is_available() can still be True, but you'll get
    # warnings like "sm_120 is not compatible" and performance/ops may break.
    if actual_device == "cuda":
        try:
            cap_major, cap_minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap_major}{cap_minor}"
            arch_list = []
            try:
                arch_list = list(torch.cuda.get_arch_list())
            except Exception:
                arch_list = []
            if arch_list and arch not in arch_list:
                raise RuntimeError(
                    "CUDA is available, but your GPU compute capability is not included in this PyTorch wheel.\n"
                    f"Detected GPU arch={arch}, torch supports: {', '.join(arch_list)}\n"
                    f"torch={torch_build}, torch.version.cuda={torch_cuda_ver}.\n\n"
                    "Fix: install a newer PyTorch build that supports your GPU (often CUDA 12.6 wheels or nightly builds), "
                    "or run with --device cpu."
                )
        except RuntimeError:
            raise
        except Exception:
            # If we can't query capability/arch list, don't block training.
            pass

    print(f"PyTorch device (requested={device}): {actual_device}  (torch={torch_build}, torch.cuda={torch_cuda_ver})")

    if resume:
        print(f"Resuming from {resume}")
        model = MaskablePPO.load(resume, env=train_env, device=actual_device)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            # Shorter rollouts → faster feedback for a 60s episode game
            n_steps=512,
            batch_size=128,
            n_epochs=8,
            gamma=0.995,        # high gamma: rewards come at end of match
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,      # slightly high entropy → explore speed & routes
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(output_dir, "tb_logs"),
            device=actual_device,
        )

    # Progress bar only if tqdm + rich are installed (SB3 requirement)
    try:
        import tqdm, rich  # noqa: F401
        use_pbar = True
    except ImportError:
        use_pbar = False

    print(f"\nTraining {n_envs} parallel envs × {total_timesteps} timesteps "
          f"(60s/episode, 1 robot, no opponents){' [render]' if render else ''}")
    print(f"Checkpoints every {checkpoint_freq:,} steps → {os.path.join(output_dir, 'checkpoints')}")
    print(f"Ctrl+C at any time saves current model to {os.path.join(output_dir, 'interrupted_model.zip')}\n")

    interrupted = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=use_pbar,
            reset_num_timesteps=resume is None,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[!] Training interrupted by user — saving partial progress...")
    finally:
        save_path = os.path.join(
            output_dir, "interrupted_model" if interrupted else "final_model"
        )
        model.save(save_path)
        print(f"Saved model ({model.num_timesteps:,} steps trained) → {save_path}.zip")
        if interrupted:
            print(f"Resume later with:")
            print(f"  python offsim/main.py train --resume {save_path}.zip "
                  f"--timesteps {total_timesteps}")
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass
