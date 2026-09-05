"""OverrideGraphEnv de-risk tests: does TENURE's graph interface represent Override?"""
import numpy as np
import pytest
import torch

from contested.config import CanonicalConfig
from contested.observation import observation_spec
from offsim.sim.graph_env import OverrideGraphEnv, retention_by_site_class


def test_observation_matches_canonical_schema():
    env = OverrideGraphEnv(seed=7)
    obs = env.observation()
    spec = observation_spec(CanonicalConfig(max_robots=8, max_tasks=20, max_adversaries=8))
    assert set(obs) == set(spec)
    for key, shape in spec.items():
        assert tuple(obs[key].shape) == (1, *shape), f"{key} shape mismatch"
    for key in ("robot_feat", "task_feat", "adv_feat", "e_rt", "e_at", "e_tt", "scalar"):
        assert torch.isfinite(obs[key]).all(), f"{key} non-finite"


def test_site_classes_are_the_three_alpha_regimes():
    env = OverrideGraphEnv(seed=1)
    counts = {c: env.site_classes.count(c) for c in set(env.site_classes)}
    assert counts == {"alliance_goal": 4, "neutral_goal": 5, "toggle": 4}   # 13 sites, 3 regimes


def test_action_masking_follows_credit():
    env = OverrideGraphEnv(seed=3)
    T = env.max_tasks
    credit = env._credit_vector()
    mask = env.action_masks()
    for i, c in enumerate(credit):
        assert mask[0, i] == (not c), f"ACQUIRE site {i} should be legal iff uncredited"
        assert mask[0, T + i] == c, f"DEFEND site {i} should be legal iff credited"
    assert mask[:, env.idle_action].all(), "IDLE always legal"


def test_tenure_policy_can_act_on_override():
    """The highest-risk integration item: TENURE's architecture runs on the real game."""
    from tenure.policy import TenurePolicy
    env = OverrideGraphEnv(seed=5)
    policy = TenurePolicy(d_model=32, n_heads=4, n_layers=2, ff=64).eval()
    with torch.no_grad():
        step = policy.act(env.observation())
    assert step["action"].shape == (1, env.max_robots)
    assert step["r_hat"].shape == (1, env.max_tasks)
    assert torch.isfinite(step["value"]).all()


def test_random_policy_steps_the_real_game():
    env = OverrideGraphEnv(seed=9)
    rng = np.random.default_rng(0)
    for _ in range(20):
        if env.field.done:
            break
        m = env.action_masks()
        acts = [int(rng.choice(np.flatnonzero(m[i]))) for i in range(2)]
        obs, reward, done, info = env.step(acts)
        assert np.isfinite(reward)
        assert torch.isfinite(obs["task_feat"]).all()
    assert "blue_score" in info and "red_score" in info


def test_retention_by_site_class_returns_three_regimes():
    from tenure.policy import TenurePolicy
    env = OverrideGraphEnv(seed=2)
    policy = TenurePolicy(d_model=16, n_heads=2, n_layers=1, ff=32).eval()
    out = retention_by_site_class(policy, env, n_decisions=8)
    assert set(out) == {"alliance_goal", "neutral_goal", "toggle"}
    assert all(0.0 <= v <= 1.0 for v in out.values())
