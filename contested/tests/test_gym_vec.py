"""Gymnasium vector adapter tests (plan Sections 3.12, M2 acceptance)."""
from __future__ import annotations

import numpy as np
import torch

from contested.config import CanonicalConfig
from contested.adapters.gym_vec import CanonicalVectorEnv

F32 = torch.float32


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4,
                horizon_T=2.0, decision_dt=0.5)   # 4 decisions/episode -> fast
    base.update(kw)
    return CanonicalConfig(**base)


def test_reset_and_spaces():
    cfg = _cfg()
    env = CanonicalVectorEnv(cfg, num_envs=5, device="cpu")
    obs, info = env.reset(seed=0)
    spec_keys = set(env.single_observation_space.spaces)
    assert set(obs) == spec_keys
    obs0 = {k: v[0] for k, v in obs.items()}
    assert env.single_observation_space.contains(obs0), "per-env obs not in single_observation_space"
    for k, v in obs.items():
        assert v.shape[0] == 5, f"{k} missing batch dim"


def test_random_policy_rollout_and_autoreset():
    cfg = _cfg()
    env = CanonicalVectorEnv(cfg, num_envs=8, device="cpu")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    boundary_seen = False
    for t in range(cfg.n_decisions + 2):
        actions = rng.integers(0, cfg.action_dim, size=(8, cfg.max_robots))
        obs, reward, terminated, truncated, info = env.step(actions)
        assert reward.shape == (8,) and np.isfinite(reward).all()
        assert not terminated.any(), "fixed-horizon task should truncate, not terminate"
        assert np.isfinite(obs["robot_feat"]).all()
        if truncated.all():
            boundary_seen = True
            assert "telemetry" in info and "final_observation" in info
            assert info["telemetry"]["J_H"].shape == (8,)
    assert boundary_seen, "never reached the episode horizon"


def test_autoreset_advances_seed_between_episodes():
    """Consecutive episodes use different seeds (distinct initial layouts)."""
    cfg = _cfg()
    env = CanonicalVectorEnv(cfg, num_envs=4, device="cpu")
    obs, _ = env.reset(seed=0)
    first_task_pos = env.core.state.task_pos.clone()
    zero = np.zeros((4, cfg.max_robots), dtype=np.int64)
    for _ in range(cfg.n_decisions):
        env.step(zero)
    # after crossing the boundary the batch has auto-reset to a new episode
    assert not torch.allclose(env.core.state.task_pos, first_task_pos), "layout should change across episodes"


def test_runs_on_default_device():
    cfg = _cfg()
    env = CanonicalVectorEnv(cfg, num_envs=4)   # default device (CUDA on the RTX box)
    obs, _ = env.reset(seed=1)
    obs, reward, term, trunc, info = env.step(np.zeros((4, cfg.max_robots), dtype=np.int64))
    assert isinstance(obs["robot_feat"], np.ndarray) and np.isfinite(reward).all()
