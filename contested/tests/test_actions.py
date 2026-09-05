"""Masking and greedy-matching tests (plan Sections 3.5, 3.6, 3.13)."""
from __future__ import annotations

import torch

from contested.actions import action_masks, greedy_assign
from contested.config import CanonicalConfig
from contested.core import CanonicalCore

F64 = torch.float64


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4, horizon_T=10.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _state(cfg, batch=16, seed=1):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
    core.reset(seed=seed)
    # make a nontrivial completion vector and an invalid robot to exercise masking
    g = torch.Generator().manual_seed(seed)
    core.state.task_c = (torch.rand(core.state.task_c.shape, generator=g) < 0.4) & core.state.task_valid
    core.state.robot_valid[:, cfg.n_robots - 1] = False
    return core.state


def test_reverse_action_mask_symmetric():
    """Symmetric mode: action layout is 3T+1, ACQUIRE is neutral-only, DEFEND is blue-owned,
    and the new REVERSE block is valid iff the task is red-owned."""
    cfg = _cfg(symmetric=True)
    T = cfg.max_tasks
    assert cfg.action_dim == 3 * T + 1 and cfg.idle_action == 3 * T and cfg.reverse_base == 2 * T
    s = _state(cfg)
    g = torch.Generator().manual_seed(2)
    red = (torch.rand(s.task_c.shape, generator=g) < 0.5) & s.task_c   # split completed -> red-owned
    s.task_c_red = red
    s.task_c = s.task_c & ~red                                         # keep not(task_c & task_c_red)
    m = action_masks(s, cfg)
    rv = s.robot_valid.unsqueeze(-1)
    neutral = s.task_valid & ~s.task_c & ~s.task_c_red
    assert torch.equal(m[:, :, 0:T], neutral.unsqueeze(1) & rv), "ACQUIRE must be neutral-only when symmetric"
    assert torch.equal(m[:, :, T:2 * T], (s.task_valid & s.task_c).unsqueeze(1) & rv), "DEFEND = blue-owned"
    assert torch.equal(m[:, :, 2 * T:3 * T], (s.task_valid & s.task_c_red).unsqueeze(1) & rv), "REVERSE = red-owned"
    assert m[:, :, cfg.idle_action].all(), "IDLE must always be valid"


def test_reverse_action_navigates_and_takes():
    """A robot issued REVERSE drives to the red-owned task and (dominating it) TAKES it: ownership
    transfers to the reverter, so the task becomes BLUE's -- not neutral -- and the taker holds
    what it took. End-to-end mechanic behind the action."""
    cfg = _cfg(symmetric=True, n_robots=1, n_tasks=1, n_adversaries=1,
               max_robots=1, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=20.0)
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.reset(0)
    s = core.state
    W, H = cfg.field_size
    s.task_pos[0, 0] = torch.tensor([W / 2, H / 2], dtype=F64)
    s.robot_pos[0, 0] = torch.tensor([0.0, 0.0], dtype=F64)      # robot starts away from the task
    s.task_c[0, 0] = False
    s.task_c_red[0, 0] = True                                    # red owns the task
    s.adv_pos[0, 0] = torch.tensor([0.0, 0.0], dtype=F64)
    s.adv_target[0, 0] = -1                                      # red stays away -> blue dominates on arrival
    s.robot_action = torch.tensor([[cfg.reverse_base]], dtype=torch.int64)  # REVERSE task 0
    took = False
    for _ in range(400):
        core.tick()
        if bool(s.task_c[0, 0]):                                 # blue now OWNS the task
            took = True
            break
    assert took, "a REVERSE action must drive the robot to the red task and take it"
    assert not bool(s.task_c_red[0, 0]), "red must lose ownership when blue takes the task"


def test_multi_reverse_two_robots_take_defended_task():
    """1v1 on a red robot's own task is a standoff that never flips; a SECOND blue robot
    committing to the same REVERSE target out-numbers red (2v1) and takes it. Validates
    allow_multi_reverse and the rule that a lone reverter cannot beat a sitting defender."""
    def took_within(n_blue: int, ticks: int = 200) -> bool:
        cfg = _cfg(symmetric=True, n_robots=n_blue, n_tasks=1, n_adversaries=1,
                   max_robots=2, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=20.0)
        core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
        core.reset(0)
        s = core.state
        W, H = cfg.field_size
        c = torch.tensor([W / 2, H / 2], dtype=F64)
        s.task_pos[0, 0] = c
        s.task_c[0, 0] = False
        s.task_c_red[0, 0] = True                               # red owns AND defends its task
        s.robot_valid[0] = torch.tensor([r < n_blue for r in range(2)])
        for r in range(n_blue):
            s.robot_pos[0, r] = c.clone()                       # blue robots on the task
        s.adv_pos[0, 0] = c.clone()                             # a red robot sits on it (defends)
        s.adv_target[0, 0] = 0
        s.robot_action = torch.tensor([[cfg.reverse_base] * 2], dtype=torch.int64)[:, :n_blue]
        for _ in range(ticks):
            core.tick()
            s.adv_pos[0, 0] = c.clone()                         # keep the defender pinned on its task
            if bool(s.task_c[0, 0]):
                return True
        return False

    assert not took_within(1), "1v1: a lone reverter cannot take a defended red task"
    assert took_within(2), "2v1: two robots committing to the same reverse target take it"


def test_mask_correctness():
    """Never ACQUIRE a complete task; never DEFEND an incomplete one; IDLE always on."""
    cfg = _cfg()
    s = _state(cfg)
    T = cfg.max_tasks
    m = action_masks(s, cfg)

    acquire_ok = s.task_valid & ~s.task_c            # (B, T)
    defend_ok = s.task_valid & s.task_c
    rv = s.robot_valid.unsqueeze(-1)
    assert torch.equal(m[:, :, 0:T], acquire_ok.unsqueeze(1) & rv)
    assert torch.equal(m[:, :, T:2 * T], defend_ok.unsqueeze(1) & rv)
    assert m[:, :, cfg.idle_action].all(), "IDLE must always be valid"

    # invalid robots: all False except IDLE
    inv = ~s.robot_valid
    inv_rows = m[inv]
    assert not inv_rows[:, :cfg.idle_action].any()
    assert inv_rows[:, cfg.idle_action].all()

    # no complete task is ever ACQUIRE-able, no incomplete task ever DEFEND-able
    for i in range(T):
        if s.task_valid[:, i].any():
            complete = s.task_c[:, i]
            assert not m[complete, :, i].any(), "ACQUIRE offered on a complete task"
            assert not m[~complete & s.task_valid[:, i], :, T + i].any(), "DEFEND offered on an incomplete task"


def test_greedy_conflict_free():
    """No two robots share an ACQUIRE (or DEFEND) target when multi-assign is off."""
    cfg = _cfg(allow_multi_acquire=False, allow_multi_defend=False)
    s = _state(cfg)
    B, R, A, T = s.batch_size, cfg.max_robots, cfg.action_dim, cfg.max_tasks
    mask = action_masks(s, cfg)
    scores = torch.randn(B, R, A, generator=torch.Generator().manual_seed(3), dtype=F64)
    assign = greedy_assign(scores, mask, cfg)

    # every assigned action must have been legal
    legal = mask.gather(2, assign.unsqueeze(-1)).squeeze(-1)
    assert legal.all(), "greedy produced an illegal (masked) assignment"

    for b in range(B):
        acquired = [int(a) for a in assign[b] if a < T]
        defended = [int(a) for a in assign[b] if T <= a < 2 * T]
        assert len(acquired) == len(set(acquired)), "shared ACQUIRE target"
        assert len(defended) == len(set(defended)), "shared DEFEND target"


def test_greedy_allows_shared_when_configured():
    """With multi-defend allowed, two robots may DEFEND the same task."""
    cfg = _cfg(allow_multi_defend=True, n_robots=4, n_tasks=1, n_adversaries=1,
              max_robots=4, max_tasks=2, max_adversaries=2)
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.reset(seed=0)
    core.state.task_c[:, 0] = True                    # task 0 complete -> DEFEND-able
    core.state.task_valid[:, 1] = False               # only one real task
    mask = action_masks(core.state, cfg)
    scores = torch.zeros(1, cfg.max_robots, cfg.action_dim, dtype=F64)
    scores[:, :, cfg.max_tasks + 0] = 10.0            # everyone prefers DEFEND task 0
    assign = greedy_assign(scores, mask, cfg)
    defend0 = (assign[0] == cfg.max_tasks).sum().item()
    assert defend0 >= 2, "multi-defend should let multiple robots share the target"


def test_permutation_equivariance():
    """Permuting robot indices permutes the greedy assignment identically."""
    cfg = _cfg(allow_multi_acquire=False, allow_multi_defend=False)
    s = _state(cfg)
    B, R, A = s.batch_size, cfg.max_robots, cfg.action_dim
    mask = action_masks(s, cfg)
    scores = torch.randn(B, R, A, generator=torch.Generator().manual_seed(9), dtype=F64)

    base = greedy_assign(scores, mask, cfg)
    perm = torch.randperm(R, generator=torch.Generator().manual_seed(11))
    permuted = greedy_assign(scores[:, perm, :], mask[:, perm, :], cfg)
    assert torch.equal(permuted, base[:, perm]), "assignment is not permutation-equivariant in robots"
