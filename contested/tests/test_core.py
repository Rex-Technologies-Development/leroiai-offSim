"""Canonical-core correctness tests (plan Section 3.13).

The reference-match test is the highest-value test in the project: it pins the
batched torch dynamics against the independent NumPy reference over many random
seeds. The rest lock the documented timer/contest/monotonicity invariants.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from contested.config import CanonicalConfig, load_config
from contested.core import CanonicalCore, CanonicalState, sample_initial_arrays
from contested.reference import ReferenceEnv

F64 = torch.float64


# --------------------------------------------------------------------- helpers
def _small_cfg(**overrides) -> CanonicalConfig:
    base = dict(
        n_robots=4, n_tasks=6, n_adversaries=3,
        max_robots=6, max_tasks=8, max_adversaries=4,
        horizon_T=10.0, decision_dt=0.5, dt=0.05,
    )
    base.update(overrides)
    return CanonicalConfig(**base)


def _random_env_dict(cfg: CanonicalConfig, rng: np.random.Generator) -> dict:
    """A fully-randomised padded single-env state, including timers and c, so the
    tick's transition logic is exercised (not just fresh resets)."""
    R, T, K = cfg.max_robots, cfg.max_tasks, cfg.max_adversaries
    W, H = cfg.field_size
    nr, nt, nk = cfg.n_robots, cfg.n_tasks, cfg.n_adversaries
    tau_rev = cfg.tau_rev if math.isfinite(cfg.tau_rev) else cfg.tau_com

    def valid_mask(n, cap):
        m = np.zeros(cap, bool); m[:n] = True; return m

    return dict(
        robot_pos=rng.uniform(0, [W, H], (R, 2)),
        robot_vel=rng.uniform(-cfg.v_max, cfg.v_max, (R, 2)),
        robot_valid=valid_mask(nr, R),
        adv_pos=rng.uniform(0, [W, H], (K, 2)),
        adv_vel=rng.uniform(-cfg.adv_v_max, cfg.adv_v_max, (K, 2)),
        adv_valid=valid_mask(nk, K),
        adv_target=np.where(np.arange(K) < nk, rng.integers(-1, nt, K), -1).astype(np.int64),
        adv_archetype=np.zeros(K, np.int64),
        task_pos=rng.uniform(0, [W, H], (T, 2)),
        task_w=rng.uniform(*cfg.weight_range, T),
        task_c=(rng.random(T) < 0.4) & valid_mask(nt, T),
        task_valid=valid_mask(nt, T),
        task_protected=(rng.random(T) < 0.3) & valid_mask(nt, T),
        sigma=rng.uniform(0, 1.2 * cfg.tau_com, T),
        eta=rng.uniform(0, 1.2 * tau_rev, T),
        t_since_change=rng.uniform(0, cfg.horizon_T, T),
    )


def _state_from_dicts(cfg: CanonicalConfig, envs: list[dict]) -> CanonicalState:
    def stk(key, dtype):
        return torch.tensor(np.stack([e[key] for e in envs]), dtype=dtype)

    B, T = len(envs), cfg.max_tasks
    return CanonicalState(
        robot_pos=stk("robot_pos", F64), robot_vel=stk("robot_vel", F64),
        robot_action=torch.full((B, cfg.max_robots), cfg.idle_action, dtype=torch.int64),
        robot_valid=stk("robot_valid", torch.bool),
        adv_pos=stk("adv_pos", F64), adv_vel=stk("adv_vel", F64),
        adv_target=stk("adv_target", torch.int64), adv_archetype=stk("adv_archetype", torch.int64),
        adv_valid=stk("adv_valid", torch.bool),
        task_pos=stk("task_pos", F64), task_w=stk("task_w", F64),
        task_c=stk("task_c", torch.bool), task_valid=stk("task_valid", torch.bool),
        task_protected=(stk("task_protected", torch.bool) if "task_protected" in envs[0]
                        else torch.zeros((B, T), dtype=torch.bool)),
        sigma=stk("sigma", F64), eta=stk("eta", F64), t_since_change=stk("t_since_change", F64),
        t=torch.zeros(B, dtype=F64), held_integral=torch.zeros(B, dtype=F64),
        first_complete_t=torch.full((B, T), -1.0, dtype=F64),
        task_c_red=(stk("task_c_red", torch.bool) if "task_c_red" in envs[0]
                    else torch.zeros((B, T), dtype=torch.bool)),
        sigma_red=(stk("sigma_red", F64) if "sigma_red" in envs[0]
                   else torch.zeros((B, T), dtype=F64)),
        held_integral_red=torch.zeros(B, dtype=F64),
        neutral_claim=(stk("neutral_claim", torch.int8) if "neutral_claim" in envs[0]
                       else torch.zeros((B, T), dtype=torch.int8)),
        task_height=(stk("task_height", torch.int8) if "task_height" in envs[0]
                     else stk("task_c", torch.bool).to(torch.int8)),
        task_toggle_idx=(stk("task_toggle_idx", torch.int64) if "task_toggle_idx" in envs[0]
                         else torch.arange(T).unsqueeze(0).expand(B, T).contiguous()),
        task_is_toggle=(stk("task_is_toggle", torch.bool) if "task_is_toggle" in envs[0]
                        else torch.zeros((B, T), dtype=torch.bool)),
    )


def _reference_from_dict(cfg: CanonicalConfig, env: dict) -> ReferenceEnv:
    ref = ReferenceEnv(cfg, env)
    ref.sigma = env["sigma"].copy()
    ref.eta = env["eta"].copy()
    ref.t_since_change = env["t_since_change"].copy()
    if "sigma_red" in env:
        ref.sigma_red = env["sigma_red"].copy()
    return ref


def _random_env_dict_sym(cfg: CanonicalConfig, rng: np.random.Generator) -> dict:
    """A randomised SYMMETRIC-mode env: three-valued ownership (neutral/blue/red)
    respecting the invariant not(task_c & task_c_red), plus a red service timer."""
    env = _random_env_dict(cfg, rng)
    T, nt = cfg.max_tasks, cfg.n_tasks
    valid = np.zeros(T, bool); valid[:nt] = True
    owner = rng.integers(0, 3, T)                       # 0 neutral, 1 blue, 2 red
    env["task_c"] = (owner == 1) & valid
    env["task_c_red"] = (owner == 2) & valid
    env["sigma"] = rng.uniform(0, 1.2 * cfg.tau_com, T)
    env["sigma_red"] = rng.uniform(0, 1.2 * cfg.tau_com, T)
    env["neutral_claim"] = rng.integers(0, 3, T).astype(np.int64)   # exercise first-arrival priority
    return env


def _random_env_dict_stack(cfg: CanonicalConfig, rng: np.random.Generator) -> dict:
    """A randomised STACKING env: integer heights consistent with ownership (height>0 iff owned)."""
    env = _random_env_dict_sym(cfg, rng)
    T, nt = cfg.max_tasks, cfg.n_tasks
    valid = np.zeros(T, bool); valid[:nt] = True
    owned = (env["task_c"] | env["task_c_red"]) & valid
    h = np.where(owned, rng.integers(1, cfg.stack_cap + 1, T), 0).astype(np.int8)
    env["task_height"] = h
    env["task_c"] = env["task_c"] & (h > 0)                         # height 0 => neutral
    env["task_c_red"] = env["task_c_red"] & (h > 0)
    return env


# ------------------------------------------------------------------ core tests
def test_batched_matches_reference():
    """500 random seeds: batched core and reference produce identical trajectories."""
    cfg = _small_cfg()
    n_seeds, n_ticks = 500, 20
    master = np.random.default_rng(20260824)
    envs = [_random_env_dict(cfg, master) for _ in range(n_seeds)]

    core = CanonicalCore(cfg, batch_size=n_seeds, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, envs)
    refs = [_reference_from_dict(cfg, e) for e in envs]

    for _ in range(n_ticks):
        acts = master.integers(0, cfg.action_dim, (n_seeds, cfg.max_robots))
        tgts = master.integers(-1, cfg.n_tasks, (n_seeds, cfg.max_adversaries))
        core.state.robot_action = torch.tensor(acts, dtype=torch.int64)
        core.state.adv_target = torch.tensor(tgts, dtype=torch.int64)
        core.tick()
        for b, ref in enumerate(refs):
            ref.robot_action = acts[b].astype(np.int64)
            ref.adv_target = tgts[b].astype(np.int64)
            ref.tick()

    s = core.state
    ref_c = np.stack([r.task_c for r in refs])
    assert np.array_equal(s.task_c.numpy(), ref_c), "completion vectors diverged"
    checks = {
        "robot_pos": np.stack([r.robot_pos for r in refs]),
        "adv_pos": np.stack([r.adv_pos for r in refs]),
        "sigma": np.stack([r.sigma for r in refs]),
        "eta": np.stack([r.eta for r in refs]),
        "t_since_change": np.stack([r.t_since_change for r in refs]),
        "held_integral": np.array([r.held_integral for r in refs]),
        "first_complete_t": np.stack([r.first_complete_t for r in refs]),
    }
    for key, ref_val in checks.items():
        got = getattr(s, key).numpy()
        assert np.allclose(got, ref_val, atol=1e-9, rtol=0), f"{key} diverged (max {np.abs(got-ref_val).max():.2e})"


def test_symmetric_batched_matches_reference():
    """PHASE 2: with symmetric=True, batched core and reference agree over 400 seeds.

    Exercises the full two-team tick — red completions, both reversal directions,
    three-valued ownership — against the independent NumPy mirror. The single-team
    match (above) guards the symmetric=False path; this guards the symmetric one."""
    cfg = _small_cfg(symmetric=True)
    assert cfg.symmetric
    n_seeds, n_ticks = 400, 24
    master = np.random.default_rng(20260827)
    envs = [_random_env_dict_sym(cfg, master) for _ in range(n_seeds)]

    core = CanonicalCore(cfg, batch_size=n_seeds, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, envs)
    refs = [_reference_from_dict(cfg, e) for e in envs]

    for _ in range(n_ticks):
        acts = master.integers(0, cfg.action_dim, (n_seeds, cfg.max_robots))
        tgts = master.integers(-1, cfg.n_tasks, (n_seeds, cfg.max_adversaries))
        core.state.robot_action = torch.tensor(acts, dtype=torch.int64)
        core.state.adv_target = torch.tensor(tgts, dtype=torch.int64)
        core.tick()
        for b, ref in enumerate(refs):
            ref.robot_action = acts[b].astype(np.int64)
            ref.adv_target = tgts[b].astype(np.int64)
            ref.tick()

    s = core.state
    assert np.array_equal(s.task_c.numpy(), np.stack([r.task_c for r in refs])), "blue ownership diverged"
    assert np.array_equal(s.task_c_red.numpy(), np.stack([r.task_c_red for r in refs])), "red ownership diverged"
    assert np.array_equal(s.neutral_claim.numpy(), np.stack([r.neutral_claim for r in refs])), "neutral_claim diverged"
    # invariant: no task is owned by both teams
    assert not bool((s.task_c & s.task_c_red).any()), "ownership invariant violated (both own a task)"
    checks = {
        "sigma": np.stack([r.sigma for r in refs]),
        "sigma_red": np.stack([r.sigma_red for r in refs]),
        "eta": np.stack([r.eta for r in refs]),
        "t_since_change": np.stack([r.t_since_change for r in refs]),
        "held_integral": np.array([r.held_integral for r in refs]),
        "held_integral_red": np.array([r.held_integral_red for r in refs]),
        "first_complete_t": np.stack([r.first_complete_t for r in refs]),
    }
    for key, ref_val in checks.items():
        got = getattr(s, key).numpy()
        assert np.allclose(got, ref_val, atol=1e-9, rtol=0), f"{key} diverged (max {np.abs(got-ref_val).max():.2e})"


def test_stacking_batched_matches_reference():
    """PHASE 3: with stack_cap>1, batched core and reference agree over 400 seeds -- integer stack
    heights, height-dependent dismantle, and height-weighted held value all match the NumPy mirror."""
    cfg = _small_cfg(symmetric=True, stack_cap=5, stack_value_power=2.0)   # convex held value too
    n_seeds, n_ticks = 400, 24
    master = np.random.default_rng(20260902)
    envs = [_random_env_dict_stack(cfg, master) for _ in range(n_seeds)]

    core = CanonicalCore(cfg, batch_size=n_seeds, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, envs)
    refs = [_reference_from_dict(cfg, e) for e in envs]

    for _ in range(n_ticks):
        acts = master.integers(0, cfg.action_dim, (n_seeds, cfg.max_robots))
        tgts = master.integers(-1, cfg.n_tasks, (n_seeds, cfg.max_adversaries))
        core.state.robot_action = torch.tensor(acts, dtype=torch.int64)
        core.state.adv_target = torch.tensor(tgts, dtype=torch.int64)
        core.tick()
        for b, ref in enumerate(refs):
            ref.robot_action = acts[b].astype(np.int64)
            ref.adv_target = tgts[b].astype(np.int64)
            ref.tick()

    s = core.state
    assert np.array_equal(s.task_height.numpy(), np.stack([r.task_height for r in refs])), "stack heights diverged"
    assert np.array_equal(s.task_c.numpy(), np.stack([r.task_c for r in refs])), "blue ownership diverged"
    assert np.array_equal(s.task_c_red.numpy(), np.stack([r.task_c_red for r in refs])), "red ownership diverged"
    assert not bool((s.task_c & s.task_c_red).any()), "ownership invariant violated"
    for key in ("sigma", "sigma_red", "eta", "held_integral", "held_integral_red"):
        got = getattr(s, key).numpy()
        ref_val = (np.array([getattr(r, key) for r in refs]) if got.ndim == 1
                   else np.stack([getattr(r, key) for r in refs]))
        assert np.allclose(got, ref_val, atol=1e-9, rtol=0), f"{key} diverged"


def _stack_env(**over):
    W, H = (3.66, 3.66)
    task = np.array([[W / 2, H / 2]])
    env = dict(robot_pos=task.copy(), robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
               adv_pos=np.array([[0.0, 0.0]]), adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
               adv_target=np.array([-1]), adv_archetype=np.array([0]),
               task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([False]),
               task_c_red=np.array([False]), task_valid=np.array([True]), task_height=np.array([0], np.int8),
               sigma=np.zeros(1), sigma_red=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1))
    env.update(over)
    return env


def test_stacking_build_increments_and_caps():
    """A sole dominator builds up to stack_cap then stops (no notion of 'done' beyond the cap). The
    piece at level L costs tau_com * L to place (reaching higher is slower), so with tau_com=2, dt=.05
    (40 ticks per unit tau_com) level L completes at cumulative ~tau_com*L(L+1)/2 -- ticks 40, 121,
    242 (a 1-tick reset lag each), i.e. heights index 39, 120, 241."""
    cfg = _small_cfg(symmetric=True, stack_cap=3, n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=30.0)
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [_stack_env()])          # neutral, blue robot on the task
    core.state.robot_action = torch.tensor([[0]])               # ACQUIRE task 0 == build a piece
    core.state.adv_target = torch.tensor([[-1]])                # red idle, away -> blue sole dominator
    heights = []
    for _ in range(400):
        core.tick()
        heights.append(int(core.state.task_height[0, 0]))
    assert heights[39] == 1 and heights[120] == 2 and heights[241] == 3, \
        f"build cadence wrong: {heights[39]},{heights[120]},{heights[241]} (want 1,2,3 at tau_com*L)"
    assert heights[119] == 1 and heights[240] == 2, "levels must not complete early (cost grows w/ L)"
    assert max(heights) == 3, "height must cap at stack_cap=3"
    assert bool(core.state.task_c[0, 0]), "a built task is blue-owned"


def test_stacking_dismantle_economies_of_scale():
    """Dismantling removes one piece at a time and the TOP piece of a height-h stack takes tau_rev*h,
    so a 3-stack flattens at cumulative tau_rev*{3,5,6} -- quadratic. Partial reversal leaves
    intermediate heights, and a fully flattened stack becomes neutral (you lose it all at once)."""
    cfg = _small_cfg(symmetric=True, stack_cap=5, n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=40.0)
    tr = cfg.tau_rev                                            # 2.0 at alpha=1
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    # blue owns a 3-stack but sits away; one red robot camps the task and dismantles it.
    core.state = _state_from_dicts(cfg, [_stack_env(
        robot_pos=np.array([[0.0, 0.0]]), task_c=np.array([True]), task_height=np.array([3], np.int8),
        adv_pos=np.array([[3.66 / 2, 3.66 / 2]]))])
    core.state.robot_action = torch.tensor([[cfg.idle_action]])
    core.state.adv_target = torch.tensor([[0]])
    drops, prev = {}, 3
    for k in range(1, 700):
        core.tick()
        core.state.adv_pos[0, 0] = torch.tensor([3.66 / 2, 3.66 / 2], dtype=F64)   # keep red on the task
        h = int(core.state.task_height[0, 0])
        if h != prev:
            drops[prev] = k * cfg.dt
            prev = h
        if h == 0:
            break
    assert abs(drops[3] - 3 * tr) < 0.11, f"3->2 should take tau_rev*3, got {drops.get(3)}"
    assert abs(drops[2] - 5 * tr) < 0.11, f"2->1 cumulative tau_rev*5, got {drops.get(2)}"
    assert abs(drops[1] - 6 * tr) < 0.11, f"1->0 cumulative tau_rev*6, got {drops.get(1)}"
    assert not bool(core.state.task_c[0, 0]), "a fully dismantled stack becomes neutral (lose it all)"


def test_stack_value_power_convex():
    """Held value of a height-h stack is w * h**power: power=1 is linear (h/tick), power=2 is convex
    (h^2/tick) so a tall stack is worth disproportionately more than the same pieces spread flat."""
    for power, per_tick in [(1.0, 3.0), (2.0, 9.0), (0.5, 3.0 ** 0.5)]:
        cfg = _small_cfg(symmetric=True, stack_cap=3, stack_value_power=power, n_robots=1, n_tasks=1,
                         n_adversaries=1, max_robots=1, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=5.0)
        core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
        # blue owns a static height-3 stack; both agents park away so nothing builds/dismantles
        core.state = _state_from_dicts(cfg, [_stack_env(robot_pos=np.array([[0.0, 0.0]]),
                                                        task_c=np.array([True]), task_height=np.array([3], np.int8))])
        core.state.robot_action = torch.tensor([[cfg.idle_action]])
        core.state.adv_target = torch.tensor([[-1]])
        for _ in range(10):
            core.tick()
        assert abs(float(core.state.held_integral[0]) - per_tick * 10 * cfg.dt) < 1e-9, \
            f"power={power}: held/tick should be w*3^power={per_tick:.3f}"
        assert int(core.state.task_height[0, 0]) == 3, "static stack keeps its height"


def _toggle_env(task_c, task_toggle_idx, task_is_toggle, task_c_red=None):
    """A 3-task static env (1 region: task 0 toggle, 1&2 goals), no agents acting, for held-value math."""
    n = len(task_c)
    z2 = np.zeros((1, 2))
    return dict(robot_pos=z2.copy() + 99.0, robot_vel=z2.copy(), robot_valid=np.array([False]),
                adv_pos=z2.copy() + 99.0, adv_vel=z2.copy(), adv_valid=np.array([False]),
                adv_target=np.array([-1]), adv_archetype=np.array([0]),
                task_pos=np.stack([[i * 1.0, 0.0] for i in range(n)]), task_w=np.ones(n),
                task_c=np.array(task_c, bool),
                task_c_red=np.array(task_c_red if task_c_red is not None else [False] * n, bool),
                task_valid=np.ones(n, bool), task_height=np.array(task_c, np.int8),
                sigma=np.zeros(n), sigma_red=np.zeros(n), eta=np.zeros(n), t_since_change=np.zeros(n),
                task_toggle_idx=np.array(task_toggle_idx, np.int64), task_is_toggle=np.array(task_is_toggle, bool))


def test_toggle_multiplier_boosts_owned_cluster():
    """Owning a region's TOGGLE multiplies the held value of that team's GOALS in the region by M; the
    toggle itself is not self-multiplied, and losing the toggle collapses the cluster back to 1x."""
    cfg = _small_cfg(symmetric=True, toggle_regions=1, toggle_multiplier=3.0, n_tasks=3, n_robots=1,
                     n_adversaries=1, max_tasks=3, max_robots=1, max_adversaries=1, horizon_T=5.0)
    tog_idx, is_tog = [0, 0, 0], [True, False, False]

    # (a) blue owns toggle + both goals: value/tick = 1 (toggle) + 3 + 3 (goals x3) = 7
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [_toggle_env([True, True, True], tog_idx, is_tog)])
    for _ in range(10):
        core.tick()
    assert abs(float(core.state.held_integral[0]) - 7.0 * 10 * cfg.dt) < 1e-9, "toggle-owned cluster = 7/tick"

    # (b) blue owns the two goals but NOT the toggle: value/tick = 0 + 1 + 1 = 2 (no multiplier)
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [_toggle_env([False, True, True], tog_idx, is_tog)])
    for _ in range(10):
        core.tick()
    assert abs(float(core.state.held_integral[0]) - 2.0 * 10 * cfg.dt) < 1e-9, "no toggle => goals at 1x"

    # (c) red owns the toggle, blue owns the goals: blue gets 1x, red gets its own multiplier on nothing
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [_toggle_env([False, True, True], tog_idx, is_tog,
                                                     task_c_red=[True, False, False])])
    for _ in range(10):
        core.tick()
    assert abs(float(core.state.held_integral[0]) - 2.0 * 10 * cfg.dt) < 1e-9, "red's toggle doesn't help blue"
    assert abs(float(core.state.held_integral_red[0]) - 1.0 * 10 * cfg.dt) < 1e-9, "red toggle alone = 1/tick"


def _random_env_dict_toggle(cfg, rng, Q):
    """A randomised symmetric env with a Q-region toggle layout (first task of each block is a toggle)."""
    env = _random_env_dict_sym(cfg, rng)
    T, nt = cfg.max_tasks, cfg.n_tasks
    tog_idx = np.arange(T, dtype=np.int64)
    is_tog = np.zeros(T, bool)
    per = max(1, nt // Q)
    for q in range(Q):
        start = q * per
        if start >= nt:
            break
        end = nt if q == Q - 1 else min(nt, (q + 1) * per)
        tog_idx[start:end] = start
        is_tog[start] = True
    env["task_toggle_idx"] = tog_idx
    env["task_is_toggle"] = is_tog
    return env


def test_toggle_batched_matches_reference():
    """PHASE 3 toggles: with toggle_multiplier>1, batched core and the NumPy reference agree over 400
    seeds -- crucially the multiplied held_integral / held_integral_red match piece for piece."""
    cfg = _small_cfg(symmetric=True, toggle_regions=3, toggle_multiplier=3.0)
    n_seeds, n_ticks = 400, 24
    master = np.random.default_rng(20260903)
    envs = [_random_env_dict_toggle(cfg, master, cfg.toggle_regions) for _ in range(n_seeds)]

    core = CanonicalCore(cfg, batch_size=n_seeds, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, envs)
    refs = [_reference_from_dict(cfg, e) for e in envs]

    for _ in range(n_ticks):
        acts = master.integers(0, cfg.action_dim, (n_seeds, cfg.max_robots))
        tgts = master.integers(-1, cfg.n_tasks, (n_seeds, cfg.max_adversaries))
        core.state.robot_action = torch.tensor(acts, dtype=torch.int64)
        core.state.adv_target = torch.tensor(tgts, dtype=torch.int64)
        core.tick()
        for b, ref in enumerate(refs):
            ref.robot_action = acts[b].astype(np.int64)
            ref.adv_target = tgts[b].astype(np.int64)
            ref.tick()

    s = core.state
    assert np.array_equal(s.task_c.numpy(), np.stack([r.task_c for r in refs])), "blue ownership diverged"
    for key in ("held_integral", "held_integral_red"):
        got = getattr(s, key).numpy()
        ref_val = np.array([getattr(r, key) for r in refs])
        assert np.allclose(got, ref_val, atol=1e-9, rtol=0), f"{key} diverged (max {np.abs(got-ref_val).max():.2e})"


def test_symmetric_red_completes_neutral_task():
    """A red adversary camped alone on a neutral task takes ownership at ~tau_com;
    blue's J_H stays zero (red held it, not blue)."""
    cfg = _small_cfg(symmetric=True, n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1, alpha=0.0, horizon_T=20.0)
    W, H = cfg.field_size
    task = np.array([[W / 2, H / 2]])
    env = dict(
        robot_pos=np.array([[0.0, 0.0]]), robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
        adv_pos=task.copy(), adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
        adv_target=np.array([0]), adv_archetype=np.array([0]),
        task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([False]),
        task_c_red=np.array([False]), task_valid=np.array([True]),
        sigma=np.zeros(1), sigma_red=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1),
    )
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [env])
    core.state.robot_action = torch.tensor([[cfg.idle_action]])   # blue idles away from the task
    core.state.adv_target = torch.tensor([[0]])                   # red camps task 0
    red_t = None
    for k in range(1, 201):
        core.tick()
        if bool(core.state.task_c_red[0, 0]):
            red_t = k * cfg.dt
            break
    assert red_t is not None and abs(red_t - cfg.tau_com) < 0.11, f"red should own at ~tau_com, got {red_t}"
    assert not bool(core.state.task_c[0, 0]), "blue must not own a red-completed task"
    assert core.state.held_integral.item() == 0.0, "blue J_H must be zero while red holds the task"
    assert core.state.held_integral_red.item() > 0.0, "red held value must be positive (feeds differential J_H)"


def test_symmetric_blue_takes_red_owned():
    """Blue dominating a red-owned task TAKES it after ~tau_rev: ownership transfers directly to
    the reverter (Override's descore+rescore is one continuous piece of work by one robot), so the
    task becomes BLUE's -- not neutral. This is what lets the taker HOLD what it took (R-take)."""
    cfg = _small_cfg(symmetric=True, n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1, alpha=1.0, horizon_T=20.0)
    assert abs(cfg.tau_rev - cfg.tau_com) < 1e-9   # alpha=1 => tau_rev = tau_com = 2.0
    W, H = cfg.field_size
    task = np.array([[W / 2, H / 2]])
    env = dict(
        robot_pos=task.copy(), robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
        adv_pos=np.array([[0.0, 0.0]]), adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
        adv_target=np.array([-1]), adv_archetype=np.array([0]),
        task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([False]),
        task_c_red=np.array([True]), task_valid=np.array([True]),
        sigma=np.zeros(1), sigma_red=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1),
    )
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [env])
    core.state.robot_action = torch.tensor([[cfg.max_tasks]])   # blue parks on task 0
    core.state.adv_target = torch.tensor([[-1]])                # red leaves -> blue dominates 1v0
    take_t = None
    for k in range(1, 201):
        core.tick()
        if bool(core.state.task_c[0, 0]):                       # blue now OWNS it
            take_t = k * cfg.dt
            break
    expected = cfg.reacquire_cost * cfg.tau_com                 # a take costs rho*tau_com to complete
    assert take_t is not None and abs(take_t - expected) < 0.11, f"blue should TAKE at ~rho*tau_com, got {take_t}"
    assert not bool(core.state.task_c_red[0, 0]), "red must lose ownership when blue takes the task"


def test_symmetric_first_arrival_priority():
    """Consultant round-6 tie-break. An incumbent claimant keeps progressing when a
    matched opponent joins (no freeze); but two robots arriving together with no
    incumbent produce a standoff (no progress) — the residual case telemetry counts."""
    cfg = _small_cfg(symmetric=True, n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1, alpha=0.0, horizon_T=20.0)
    W, H = cfg.field_size
    task = np.array([[W / 2, H / 2]])

    def env_base(blue_on, red_on):
        return dict(
            robot_pos=(task.copy() if blue_on else np.array([[0.0, 0.0]])),
            robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
            adv_pos=(task.copy() if red_on else np.array([[0.0, 0.0]])),
            adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
            adv_target=np.array([-1]), adv_archetype=np.array([0]),
            task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([False]),
            task_c_red=np.array([False]), task_valid=np.array([True]),
            neutral_claim=np.array([0], np.int64),
            sigma=np.zeros(1), sigma_red=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1),
        )

    # (1) blue arrives first (claims the task), red joins matched at tick 10 -> blue STILL
    #     completes at ~tau_com (a pure freeze would stall it forever).
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [env_base(blue_on=True, red_on=False)])
    core.state.robot_action = torch.tensor([[0]])            # blue ACQUIRE task 0
    core.state.adv_target = torch.tensor([[-1]])
    done_t = None
    for k in range(1, 201):
        if k == 10:
            core.state.adv_pos = torch.tensor([[[float(task[0, 0]), float(task[0, 1])]]], dtype=F64)
        core.tick()
        if bool(core.state.task_c[0, 0]):
            done_t = k * cfg.dt
            break
    assert done_t is not None and abs(done_t - cfg.tau_com) < 0.11, \
        f"first-arriver should complete at ~tau_com despite a matched late joiner, got {done_t}"
    assert not bool(core.state.task_c_red[0, 0])

    # (2) both present from t=0 with no incumbent -> standoff -> neither progresses.
    core2 = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core2.state = _state_from_dicts(cfg, [env_base(blue_on=True, red_on=True)])
    core2.state.robot_action = torch.tensor([[0]])
    core2.state.adv_target = torch.tensor([[-1]])
    for _ in range(60):
        core2.tick()
    assert not bool(core2.state.task_c[0, 0]) and not bool(core2.state.task_c_red[0, 0]), \
        "a no-incumbent standoff must not complete for either team"
    assert core2.state.sigma.item() == 0.0 and core2.state.sigma_red.item() == 0.0, \
        "neither timer accumulates in a standoff"


def test_reacquire_cost_delays_recompletion():
    """A previously-completed, now-incomplete task re-completes at ~tau_com*rho (first still tau_com)."""
    def recomplete_time(rho, prior_completed):
        cfg = _small_cfg(reacquire_cost=rho, n_robots=1, n_tasks=1, n_adversaries=1,
                         alpha=0.0, horizon_T=20.0)
        core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
        core.reset(0)
        s = core.state
        s.task_pos[0, 0] = s.robot_pos[0, 0]                 # task under the robot -> continuous service
        s.task_c[0, 0] = False
        s.sigma[0, 0] = 0.0
        s.first_complete_t[0, 0] = 0.0 if prior_completed else -1.0   # rho applies only to a re-completion
        act = torch.full((1, cfg.max_robots), cfg.idle_action, dtype=torch.int64)
        act[0, 0] = 0                                        # robot 0 ACQUIREs task 0
        s.robot_action = act
        for k in range(1, 401):
            core.tick()
            if bool(s.task_c[0, 0]):
                return k * cfg.dt
        return None

    assert abs(recomplete_time(2.0, False) - 2.0) < 0.11    # first completion: tau_com regardless of rho
    assert abs(recomplete_time(1.0, True) - 2.0) < 0.11     # rho=1 re-completion: tau_com
    assert abs(recomplete_time(2.0, True) - 4.0) < 0.11     # rho=2 re-completion: tau_com * rho


def test_determinism_two_runs_and_across_batch():
    """Same seed -> bit-identical rollout across two runs and across batch sizes."""
    cfg = _small_cfg()

    def rollout(batch, seed, n_dec):
        core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
        core.reset(seed=seed)
        idx = torch.arange(cfg.max_robots) % cfg.n_tasks
        acts = idx.unsqueeze(0).expand(batch, -1).contiguous()  # same per env, B-independent
        for _ in range(n_dec):
            core.step(acts)
        return core.state

    a = rollout(4, 123, 8)
    b = rollout(4, 123, 8)
    assert torch.equal(a.task_c, b.task_c) and torch.equal(a.robot_pos, b.robot_pos)
    assert torch.equal(a.held_integral, b.held_integral)

    one = rollout(1, 123, 8)
    four = rollout(4, 123, 8)
    assert torch.equal(one.robot_pos[0], four.robot_pos[0]), "env 0 differs across batch sizes"
    assert torch.equal(one.task_c[0], four.task_c[0])
    assert torch.equal(one.held_integral[0], four.held_integral[0])


def test_alpha_zero_is_monotone():
    """At alpha = 0 (tau_rev = inf), no task ever reverts 1 -> 0."""
    cfg = _small_cfg(alpha=0.0, horizon_T=20.0)
    assert math.isinf(cfg.tau_rev)
    core = CanonicalCore(cfg, batch_size=16, device="cpu", dtype=F64)
    core.reset(seed=7)
    # robots acquire tasks, adversaries attack -> completions happen, reversals must not
    idx = torch.arange(cfg.max_robots) % cfg.n_tasks
    acts = idx.unsqueeze(0).expand(16, -1).contiguous()
    for a in acts:  # ensure every archetype attacks
        pass
    core.state.adv_archetype[:, :cfg.n_adversaries] = 0
    ever_complete = torch.zeros_like(core.state.task_c)
    completions_seen = 0
    for _ in range(cfg.n_decisions):
        core.step(acts)
        c = core.state.task_c
        reverted = ever_complete & ~c & core.state.task_valid
        assert reverted.sum().item() == 0, "a completed task reverted at alpha=0"
        ever_complete = ever_complete | c
        completions_seen = int(ever_complete.sum().item())
    assert completions_seen > 0, "test vacuous: no completions occurred"


def test_protected_tasks_never_revert():
    """Protected tasks hold value permanently even under sustained attack."""
    cfg = _small_cfg(alpha=4.0, protected_fraction=0.5, horizon_T=20.0)
    core = CanonicalCore(cfg, batch_size=16, device="cpu", dtype=F64)
    core.reset(seed=5)
    assert core.state.task_protected.any(), "expected some protected tasks"
    core.state.adv_archetype[:, :cfg.n_adversaries] = 0
    idx = torch.arange(cfg.max_robots) % cfg.n_tasks
    acts = idx.unsqueeze(0).expand(16, -1).contiguous()
    ever_complete = torch.zeros_like(core.state.task_c)
    for _ in range(cfg.n_decisions):
        core.step(acts)
        c = core.state.task_c
        # a protected task, once complete, must never become incomplete again
        reverted_protected = ever_complete & ~c & core.state.task_valid & core.state.task_protected
        assert reverted_protected.sum().item() == 0, "a protected task reverted"
        ever_complete = ever_complete | c


def test_timer_reset_on_leaving_service_region():
    """Leaving the service region resets sigma and eta to zero (Section 3.3 steps 3-4)."""
    cfg = _small_cfg(n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1)
    W, H = cfg.field_size
    task = np.array([[W / 2, H / 2]])

    # sigma: incomplete task, robot parked on it, then teleported away
    env = dict(
        robot_pos=task.copy(), robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
        adv_pos=np.array([[0.0, 0.0]]), adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
        adv_target=np.array([-1]), adv_archetype=np.array([0]),
        task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([False]),
        task_valid=np.array([True]), sigma=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1),
    )
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [env])
    core.state.robot_action = torch.tensor([[0]])  # ACQUIRE task 0 (stay parked)
    core.state.adv_target = torch.tensor([[-1]])
    for _ in range(3):
        core.tick()
    assert core.state.sigma.item() > 0, "sigma should accumulate while parked on an incomplete task"
    core.state.robot_pos = torch.tensor([[[0.0, 0.0]]], dtype=F64)  # leave the region
    core.state.robot_action = torch.tensor([[cfg.idle_action]])     # hold outside
    core.tick()
    assert core.state.sigma.item() == 0.0, "sigma must reset on leaving"

    # eta: complete task, adversary parked on it, then teleported away
    env2 = dict(env)
    env2["task_c"] = np.array([True])
    env2["adv_pos"] = task.copy()
    core2 = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core2.state = _state_from_dicts(cfg, [env2])
    core2.state.robot_pos = torch.tensor([[[0.0, 0.0]]], dtype=F64)  # no defenders
    core2.state.robot_action = torch.tensor([[cfg.idle_action]])
    core2.state.adv_target = torch.tensor([[0]])
    for _ in range(3):
        core2.tick()
    assert core2.state.eta.item() > 0, "eta should accumulate under attack on a complete task"
    core2.state.adv_pos = torch.tensor([[[0.0, 0.0]]], dtype=F64)  # attacker leaves
    core2.state.adv_target = torch.tensor([[-1]])
    core2.tick()
    assert core2.state.eta.item() == 0.0, "eta must reset when the attacker leaves"


@pytest.mark.parametrize("mode,n_team,should_accumulate", [
    ("majority", 0, True), ("majority", 1, False),   # beta=1: 1 adv vs 0/1 team
    ("suppress", 0, True), ("suppress", 1, False),    # a single defender holds indefinitely
    ("none", 0, True), ("none", 1, True),             # defenders have no effect
])
def test_contest_modes(mode, n_team, should_accumulate):
    """Each contest mode produces the documented eta accumulation (Section 3.3 table)."""
    cfg = _small_cfg(contest_mode=mode, beta=1.0,
                     n_robots=1, n_tasks=1, n_adversaries=1,
                     max_robots=1, max_tasks=1, max_adversaries=1)
    W, H = cfg.field_size
    task = np.array([[W / 2, H / 2]])
    env = dict(
        robot_pos=task.copy() if n_team else np.array([[0.0, 0.0]]),
        robot_vel=np.zeros((1, 2)), robot_valid=np.array([True]),
        adv_pos=task.copy(), adv_vel=np.zeros((1, 2)), adv_valid=np.array([True]),
        adv_target=np.array([0]), adv_archetype=np.array([0]),
        task_pos=task.copy(), task_w=np.array([1.0]), task_c=np.array([True]),
        task_valid=np.array([True]), sigma=np.zeros(1), eta=np.zeros(1), t_since_change=np.zeros(1),
    )
    core = CanonicalCore(cfg, batch_size=1, device="cpu", dtype=F64)
    core.state = _state_from_dicts(cfg, [env])
    # keep team robot parked on the task (DEFEND) or idle off-task
    core.state.robot_action = torch.tensor([[cfg.max_tasks]]) if n_team else torch.tensor([[cfg.idle_action]])
    core.state.adv_target = torch.tensor([[0]])
    for _ in range(2):
        core.tick()
    if should_accumulate:
        assert core.state.eta.item() > 0, f"{mode} with n_team={n_team} should accumulate eta"
    else:
        assert core.state.eta.item() == 0.0, f"{mode} with n_team={n_team} should not accumulate eta"
