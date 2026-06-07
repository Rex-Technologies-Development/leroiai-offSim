"""MaskablePPO training — 1- or 2-robot blue team, 60-second episodes.

Uses sb3-contrib MaskablePPO so invalid actions (score with no balls,
descore when nothing is scored, etc.) are never sampled by the policy.

Usage:
    python main.py train --num-allies 2 --timesteps 2000000 --n-envs 4
    python main.py train --num-allies 1 --timesteps 1000000 --n-envs 4

2-robot (2v2) training uses TeamAgentWrapper: both allies act each decision,
team reward is the sum of per-robot rewards, obs is concat(robot_0, robot_1).
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

from sim.env import make_training_wrapper
from sim.config import Action
from sim.failure import FailureConfig
from training.curriculum import CurriculumCallback, stages_for_allies
from training.callbacks import ScoreLoggingCallback


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
# Failure-injection presets for fixed (--no-curriculum) runs. The staged
# curriculum sets its own per-stage rates; these mirror those levels so a manual
# run can opt into the same robustness noise without committing to the schedule.
def _failure_config(level: str) -> FailureConfig:
    level = (level or "none").lower()
    if level == "light":    # ~ curriculum Stage 2/3
        return FailureConfig(
            teammate_fail_rate=0.0, action_delay_range=(1.0, 1.0),
            object_stolen_rate=0.03, stuck_rate=0.02, teammate_offline_rate=0.0,
        )
    if level == "medium":   # ~ curriculum Stage 4
        return FailureConfig(
            teammate_fail_rate=0.0, action_delay_range=(1.0, 1.0),
            object_stolen_rate=0.08, stuck_rate=0.05, teammate_offline_rate=0.0,
        )
    return FailureConfig.none()


def make_env(render_mode=None, use_timer=True, num_allies=2, num_opponents=0,
             opponent_type="random", failure_level="none",
             all_blue_only: bool = False,
             log_decisions: bool = False, log_dir: str = "logs"):
    """Return a callable that creates a fresh training-wrapper episode.

    num_allies selects SingleAgentWrapper (1) or TeamAgentWrapper (2).
    num_opponents / opponent_type / failure_level set the FIXED regime for this
    env. Under the staged curriculum these are only the Stage-1 starting point
    and CurriculumCallback overrides them live; with --no-curriculum they are
    the regime for the entire run.

    When render_mode == "human", the renderer is primed inside the worker so
    the pygame window opens on that worker's process (important for
    SubprocVecEnv where each worker is its own process and gets its own window).
    """
    def _init():
        env = make_training_wrapper(
            num_allies=num_allies,
            render_mode=render_mode,
            failure_config=_failure_config(failure_level),
            opponent_type=opponent_type,
            num_opponents=num_opponents,
            use_timer=use_timer,
            log_decisions=log_decisions,
            log_dir=log_dir,
        )
        env.env.all_blue_only = bool(all_blue_only)
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
            done_indices = [i for i, d in enumerate(dones) if d]
            # Read the FINAL scores from info — the env stashes them there on the
            # done step. (env_method("_get_episode_scores") would read 0 because
            # SB3 auto-resets the finished env before this callback runs.)
            infos = self.locals.get("infos", [])
            for i in done_indices:
                info = infos[i] if i < len(infos) else {}
                self._n_episodes += 1
                self._ep_rewards.append(float(self._cur_ep_rewards[i]))
                self._ep_blue.append(float(info.get("episode_score", 0.0)))
                self._ep_red.append(float(info.get("episode_opp_score", 0.0)))
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
                    act = actions[i]
                    if hasattr(act, "__len__") and not isinstance(act, (str, bytes)):
                        labels = []
                        for a in act:
                            try:
                                labels.append(Action(int(a)).name[:12])
                            except (ValueError, TypeError):
                                labels.append(str(a))
                        per_env["last_action"] = "+".join(labels)
                    else:
                        per_env["last_action"] = Action(int(act)).name[:16]
                except (ValueError, TypeError):
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
    use_curriculum: bool = True,
    num_allies: int = 2,
    num_opponents: int = 0,
    opponent_type: str = "random",
    failure_level: str = "none",
    log_decisions: bool = False,
    log_dir: str = "logs",
):
    os.makedirs(output_dir, exist_ok=True)

    # Starting env regime. Under the staged curriculum every env begins at
    # Stage 1 (solo, no failures) and CurriculumCallback ramps it over time.
    # With --no-curriculum the chosen regime is fixed for the whole run, so a
    # solo phase and an opponent phase can be run as separate, compounding
    # invocations (each --resume-ing into the same model).
    if use_curriculum:
        start_stage = stages_for_allies(num_allies)[0][1]
        start_opps = int(start_stage.get("num_opponents", 0))
        start_type = str(start_stage.get("opponent_type", "random"))
        start_fail = "none"
        start_all_blue = bool(start_stage.get("all_blue_only", False))
    else:
        start_opps, start_type, start_fail = num_opponents, opponent_type, failure_level
        start_all_blue = False

    # Render mode:
    #   n_envs == 1 → in-process DummyVecEnv with one pygame window
    #   n_envs >  1 → SubprocVecEnv, each worker opens its OWN pygame window.
    #                 Stats reach each panel via env_method (see EpisodeStatsCallback).
    rendered_env = None
    if render:
        if n_envs == 1:
            rendered_env = make_training_wrapper(
                num_allies=num_allies,
                render_mode="human",
                failure_config=_failure_config(start_fail),
                opponent_type=start_type,
                num_opponents=start_opps,
                use_timer=True,
                log_decisions=log_decisions,
                log_dir=log_dir,
            )
            rendered_env.env.all_blue_only = start_all_blue
            rendered_env.render()
            train_env = DummyVecEnv([lambda: rendered_env])
        else:
            train_env = make_vec_env(
                make_env(render_mode="human", use_timer=True,
                         num_allies=num_allies, num_opponents=start_opps,
                         opponent_type=start_type, failure_level=start_fail,
                         all_blue_only=start_all_blue,
                         log_decisions=log_decisions, log_dir=log_dir),
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,
            )
    else:
        train_env = make_vec_env(
            make_env(use_timer=True, num_allies=num_allies,
                     num_opponents=start_opps, opponent_type=start_type,
                     failure_level=start_fail, all_blue_only=start_all_blue,
                     log_decisions=log_decisions, log_dir=log_dir),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
        )

    # Eval env — single process, timer on (always headless). Matches the run's
    # regime so the "best model" is judged under the same conditions it trains in.
    eval_env = make_vec_env(
        make_env(use_timer=True, num_allies=num_allies,
                 num_opponents=start_opps, opponent_type=start_type,
                 failure_level=start_fail, all_blue_only=start_all_blue),
        n_envs=1,
    )

    callbacks = []
    if use_curriculum:
        callbacks.append(CurriculumCallback(
            stages=stages_for_allies(num_allies), eval_env=eval_env, verbose=1,
        ))
    callbacks += [
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
        # TensorBoard game-score curves (game/avg_score, win_rate, avg_margin) so
        # scoring progress is visible beyond the console readout.
        ScoreLoggingCallback(verbose=1),
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

    team_label = f"{num_allies} ally/allies vs {start_opps} opponent(s)"
    if use_curriculum:
        regime = f"staged curriculum ({team_label}, ramping over time)"
    else:
        regime = f"fixed regime: {team_label}"
        if start_opps > 0:
            regime += f" [{start_type}]"
        regime += f", failures={start_fail}"
    print(f"\nTraining {n_envs} env(s) x {total_timesteps:,} new timesteps -- {regime}"
          f"{' [render]' if render else ''}")
    if render and use_curriculum and start_opps == 0:
        print("[train] Curriculum starts all-blue/no-opponent, adds a random opponent "
              "at ~500K steps, then full 2v2 at ~800K. Use --opponents 2 "
              "for a full 2v2 stress test from the start.")
    print(f"Checkpoints every {checkpoint_freq:,} steps -> {os.path.join(output_dir, 'checkpoints')}")
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
