"""Adversary archetype tests (plan Section 3.10)."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore

F64 = torch.float64


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=2, n_tasks=4, n_adversaries=1,
                max_robots=2, max_tasks=4, max_adversaries=1, horizon_T=10.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _core(cfg, batch=1, seed=0):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
    core.reset(seed=seed)
    return core


def test_builder_archetype_acquires_and_holds():
    """PHASE 2: in symmetric mode a builder red targets non-blue tasks and comes to OWN some,
    so there is red-held work for blue to reverse. (No-op archetype without cfg.symmetric.)"""
    cfg = _cfg(symmetric=True, n_robots=2, n_adversaries=2, max_adversaries=2,
               adversary_population=["builder"], horizon_T=30.0, alpha=0.0)
    core = _core(cfg, batch=4, seed=1)
    core.state.adv_archetype[:] = 5                       # force the builder archetype
    idle = torch.full((4, cfg.max_robots), cfg.idle_action, dtype=torch.int64)
    for _ in range(cfg.n_decisions):
        core.step(idle)                                   # blue idles; red builds
    assert core.state.task_c_red.any(), "a builder red should come to own some tasks"
    assert not core.state.task_c.any(), "blue idled, so blue should own nothing (red-built)"


def test_greedy_targets_nearest_complete():
    cfg = _cfg()
    core = _core(cfg)
    s = core.state
    s.adv_archetype[:] = 0
    s.task_pos[0] = torch.tensor([[0.5, 0.5], [3.0, 3.0], [0.6, 0.5], [2.0, 2.0]], dtype=F64)
    s.task_c[0] = torch.tensor([True, True, True, True])
    s.task_valid[0] = torch.tensor([True, True, True, True])
    s.adv_pos[0, 0] = torch.tensor([0.0, 0.0], dtype=F64)      # nearest complete task is index 0
    tgt = core.adversaries.select_targets(s)
    assert tgt[0, 0].item() == 0


def test_value_targets_high_weight_over_near():
    cfg = _cfg()
    core = _core(cfg)
    s = core.state
    s.task_pos[0] = torch.tensor([[0.3, 0.0], [1.0, 0.0], [3.5, 3.5], [3.6, 3.6]], dtype=F64)
    s.task_c[0] = torch.tensor([True, True, False, False])
    s.task_valid[0] = torch.tensor([True, True, True, True])
    s.task_w[0] = torch.tensor([1.0, 5.0, 1.0, 1.0], dtype=F64)  # task 1 is higher value w/(d)
    s.adv_pos[0, 0] = torch.tensor([0.0, 0.0], dtype=F64)

    s.adv_archetype[:] = 0
    assert core.adversaries.select_targets(s)[0, 0].item() == 0   # greedy: nearest
    s.adv_archetype[:] = 1
    assert core.adversaries.select_targets(s)[0, 0].item() == 1   # value: high w/(d)


def test_camper_confined_to_cluster():
    cfg = _cfg()
    core = _core(cfg)
    s = core.state
    s.adv_archetype[:] = 2
    # task 0 is in-cluster but far from adv; task 3 is out-of-cluster but nearer to adv
    s.task_pos[0] = torch.tensor([[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.2, 0.2]], dtype=F64)
    s.task_c[0] = torch.tensor([True, False, False, True])
    s.task_valid[0] = torch.tensor([True, False, False, True])
    s.adv_pos[0, 0] = torch.tensor([0.1, 0.1], dtype=F64)         # closest to task 3
    core.adversaries.camper_center[0, 0] = torch.tensor([3.0, 3.0], dtype=F64)  # cluster around task 0
    tgt = core.adversaries.select_targets(s)
    assert tgt[0, 0].item() == 0, "camper must stay in its cluster, not chase the nearer out-of-cluster task"


def test_no_candidates_holds():
    """With nothing complete, chasers hold; the camper still camps its cluster."""
    cfg = _cfg()
    core = _core(cfg)
    s = core.state
    s.task_c[0] = torch.tensor([False, False, False, False])     # nothing complete to reverse
    for arch in (0, 1, 3):                                        # greedy / value / feinter
        s.adv_archetype[:] = arch
        assert core.adversaries.select_targets(s)[0, 0].item() == -1, f"archetype {arch} should hold"
    # camper is behaviourally distinct: it stays positioned in its cluster
    s.adv_archetype[:] = 2
    core.adversaries.camper_center[0, 0] = s.task_pos[0, 0]
    assert core.adversaries.select_targets(s)[0, 0].item() != -1, "camper should camp an in-cluster task"


def test_feinter_deterministic_across_batch_and_runs():
    """A feinter env behaves identically across batch sizes and repeated runs."""
    cfg = _cfg(alpha=0.0, n_adversaries=1, max_adversaries=1)  # alpha=0: candidate set stays stable

    def rollout(batch, seed, n_dec=6):
        core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
        core.reset(seed=seed)
        core.state.adv_archetype[:] = 3                          # all feinter
        core.state.task_c[:, :3] = True                          # give them something to chase
        idle = torch.full((batch, cfg.max_robots), cfg.idle_action, dtype=torch.int64)
        traj = []
        for _ in range(n_dec):
            core.step(idle)
            traj.append(core.state.adv_target[0, 0].item())
        return traj

    assert rollout(1, 5) == rollout(1, 5), "not reproducible across runs"
    assert rollout(1, 5) == rollout(4, 5), "env 0 differs across batch sizes"
