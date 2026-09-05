"""Baseline allocator tests (plan Section 6)."""
from __future__ import annotations

import torch

from contested.actions import action_masks
from contested.config import CanonicalConfig
from contested.core import CanonicalCore
from baselines import REGISTRY, GreedyAllocator, DefensiveHeuristic
from baselines.evaluate import evaluate_policy

F32 = torch.float32


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=8, n_adversaries=3,
                max_robots=6, max_tasks=10, max_adversaries=4, horizon_T=15.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _state(cfg, batch=16, seed=0):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F32)
    core.reset(seed=seed)
    g = torch.Generator().manual_seed(seed)
    core.state.task_c = (torch.rand(core.state.task_c.shape, generator=g) < 0.4) & core.state.task_valid
    return core.state


def test_all_baselines_legal_and_conflict_free():
    cfg = _cfg()
    state = _state(cfg)
    mask = action_masks(state, cfg)
    T = cfg.max_tasks
    for name, cls in REGISTRY.items():
        actions = cls().act(state, cfg)
        assert actions.shape == (16, cfg.max_robots), name
        legal = mask.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        assert legal.all(), f"{name} produced a masked action"
        for b in range(16):
            acquired = [int(a) for a in actions[b] if a < T]
            assert len(acquired) == len(set(acquired)), f"{name} shared an ACQUIRE target"


def test_evaluate_returns_valid_metrics():
    cfg = _cfg()
    res = evaluate_policy(GreedyAllocator().act, cfg, batch_size=32, n_seeds=2, device="cpu")
    assert 0.0 <= res["J_H"] <= 1.0 and 0.0 <= res["retention_rate"] <= 1.0
    assert res["n_reversals"] >= 0 and res["J_H_std"] >= 0


def test_greedy_never_defends_defensive_does():
    """Behavioural distinction: greedy is acquisition-only; defensive uses DEFEND."""
    cfg = _cfg()
    greedy = evaluate_policy(GreedyAllocator().act, cfg, batch_size=48, n_seeds=1, device="cpu")
    defensive = evaluate_policy(DefensiveHeuristic().act, cfg, batch_size=48, n_seeds=1, device="cpu")
    assert greedy["defense_fraction"] < 0.01, "greedy should never defend"
    assert defensive["defense_fraction"] > 0.3, "defensive should defend substantially"
    # defending measurably reduces reversals vs never defending
    assert defensive["n_reversals"] < greedy["n_reversals"], "defense should cut reversals"
