"""PettingZoo parallel adapter smoke test (plan Section 3.12)."""
from __future__ import annotations

import numpy as np

from contested.config import CanonicalConfig
from contested.adapters.pettingzoo_par import CanonicalParallelEnv


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4,
                horizon_T=2.0, decision_dt=0.5)
    base.update(kw)
    return CanonicalConfig(**base)


def test_parallel_env_random_rollout():
    cfg = _cfg()
    env = CanonicalParallelEnv(cfg, device="cpu")
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents) == {f"robot_{i}" for i in range(4)}
    a0 = obs["robot_0"]
    assert "robot_feat" in a0 and a0["self_index"].tolist() == [0]
    assert env.observation_space("robot_0").contains(a0)

    rng = np.random.default_rng(0)
    steps = 0
    while env.agents:
        actions = {ag: int(rng.integers(0, cfg.action_dim)) for ag in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1
        assert set(rewards) == set(env.possible_agents) or env.agents == []
        if steps > cfg.n_decisions + 2:
            break
    assert steps == cfg.n_decisions, "should terminate at the horizon"
    assert env.agents == [], "agents cleared once the episode ends"
