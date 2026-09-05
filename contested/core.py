"""Batched tensor dynamics for the canonical contested-task environment.

One :class:`CanonicalState` of tensors lives on one device; ``B`` parallel
environments advance together. The physics tick implements plan Section 3.3
*exactly* (order of operations matters because timer semantics depend on it), and
the three ``contest_mode`` variants of step 4.

Everything runs at the padded dimensions (``max_robots``/``max_tasks``/
``max_adversaries``) with validity masks, so a single set of model weights runs
across configurations and padding-invariance is structural, not incidental.

The slow, independent NumPy reference in ``reference.py`` mirrors this file and is
the project's highest-value test (plan Section 3.13).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .config import CanonicalConfig


def default_device() -> torch.device:
    """CUDA when available (the RTX box), else CPU (tests/dev)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CanonicalState:
    """All environment state, padded to ``max_*`` with validity masks.

    ``R = max_robots``, ``T = max_tasks``, ``K = max_adversaries``. Field shapes
    and dtypes follow plan Section 3.2.
    """

    robot_pos: Tensor        # (B, R, 2) f
    robot_vel: Tensor        # (B, R, 2) f
    robot_action: Tensor     # (B, R)    i64  index into action layout (Section 3.5)
    robot_valid: Tensor      # (B, R)    bool

    adv_pos: Tensor          # (B, K, 2) f
    adv_vel: Tensor          # (B, K, 2) f
    adv_target: Tensor       # (B, K)    i64  task slot index or -1
    adv_archetype: Tensor    # (B, K)    i64
    adv_valid: Tensor        # (B, K)    bool

    task_pos: Tensor         # (B, T, 2) f  fixed within an episode
    task_w: Tensor           # (B, T)    f
    task_c: Tensor           # (B, T)    bool  the (blue) completion vector — blue owns the task
    task_valid: Tensor       # (B, T)    bool
    task_protected: Tensor   # (B, T)    bool  protected tasks never revert (tau_rev = inf)

    sigma: Tensor            # (B, T)    f  service timer (blue progress toward owning)
    eta: Tensor              # (B, T)    f  reversal timer (pressure on the current owner)
    t_since_change: Tensor   # (B, T)    f

    t: Tensor                # (B,)      f  elapsed seconds
    held_integral: Tensor    # (B,)      f  running sum_i w_i * int c_i dt   (J_H)
    first_complete_t: Tensor  # (B, T)   f  clock at first 0->1 (by anyone), else -1

    # PHASE 2 (symmetric mode). All-False / all-zero and never read when cfg.symmetric
    # is False, so the single-team dynamics and the reference match are unchanged.
    task_c_red: Tensor       # (B, T)    bool  red owns the task; invariant: not (task_c & task_c_red)
    sigma_red: Tensor        # (B, T)    f     red's service timer (red progress toward owning)
    held_integral_red: Tensor  # (B,)    f     sum_i w_i * int [red owns]_i dt (differential J_H uses it)
    neutral_claim: Tensor    # (B, T)    i8    first-arrival claim on a neutral task: 0 none / 1 blue / 2 red

    # PHASE 3 (stacking). Integer stack height per task, 0..stack_cap. 0 == neutral; >0 == owned
    # (by task_c / task_c_red) with that many pieces. All-<=1 and unused unless cfg.stack_cap > 1.
    task_height: Tensor      # (B, T)    i8

    # PHASE 3 (toggles). ``task_toggle_idx[b,t]`` = index of the TOGGLE governing task t's region;
    # a toggle points at itself. While a team owns that toggle, the team's held value from the
    # region's GOALS is multiplied by cfg.toggle_multiplier. Unused unless toggle_regions > 0.
    task_toggle_idx: Tensor  # (B, T)    i64   (governing toggle index per task; self for toggles)
    task_is_toggle: Tensor   # (B, T)    bool

    @property
    def batch_size(self) -> int:
        return self.robot_pos.shape[0]

    @property
    def device(self) -> torch.device:
        return self.robot_pos.device

    def clone(self) -> "CanonicalState":
        return CanonicalState(**{k: v.clone() for k, v in self.__dict__.items()})

    def to(self, device) -> "CanonicalState":
        return CanonicalState(**{k: v.to(device) for k, v in self.__dict__.items()})


def _integrate_motion(
    pos: Tensor, vel: Tensor, target: Tensor, valid: Tensor,
    v_max: float, a_max: float, dt: float, field_size: Tensor,
) -> tuple[Tensor, Tensor]:
    """First-order accel-limited navigation toward ``target`` (plan Section 3.4).

    Accel-clipped, then velocity-clipped, then walls (Section 3.3 step 1). Invalid
    entities are frozen. Points with a wall clamp lose the offending velocity
    component. No inter-robot collision (deliberately excluded from the formalism).
    """
    eps = 1e-9
    to_target = target - pos                                   # (B, N, 2)
    dist = torch.linalg.vector_norm(to_target, dim=-1)         # (B, N)
    direction = to_target / dist.clamp_min(eps).unsqueeze(-1)  # unit toward target
    # cap desired speed so we never overshoot the target in one dt
    desired_speed = torch.clamp_max(dist / dt, v_max)          # (B, N)
    desired_vel = direction * desired_speed.unsqueeze(-1)
    desired_vel = torch.where((dist > eps).unsqueeze(-1), desired_vel, torch.zeros_like(desired_vel))

    a_needed = (desired_vel - vel) / dt
    a_norm = torch.linalg.vector_norm(a_needed, dim=-1, keepdim=True)
    a = a_needed * torch.clamp_max(a_max / a_norm.clamp_min(eps), 1.0)
    new_vel = vel + a * dt
    v_norm = torch.linalg.vector_norm(new_vel, dim=-1, keepdim=True)
    new_vel = new_vel * torch.clamp_max(v_max / v_norm.clamp_min(eps), 1.0)
    new_pos = pos + new_vel * dt

    # wall clamp to [0, field]; zero the velocity component that hit a wall
    clamped = torch.clamp(torch.minimum(new_pos, field_size), min=torch.zeros_like(field_size))
    hit = clamped != new_pos
    new_vel = torch.where(hit, torch.zeros_like(new_vel), new_vel)
    new_pos = clamped

    keep = valid.unsqueeze(-1)
    new_pos = torch.where(keep, new_pos, pos)
    new_vel = torch.where(keep, new_vel, vel)
    return new_pos, new_vel


def _gather_target(task_pos: Tensor, task_valid: Tensor, idx: Tensor, fallback: Tensor) -> Tensor:
    """Target position for slot ``idx`` (per node); ``fallback`` where idx<0/invalid.

    ``idx`` is (B, N) with -1 meaning "no target". A target pointing at an invalid
    (padded) task falls back to holding position, so padded nodes never leak in.
    """
    T = task_pos.shape[1]
    safe_idx = idx.clamp(min=0, max=T - 1)                       # (B, N)
    gathered = torch.gather(task_pos, 1, safe_idx.unsqueeze(-1).expand(-1, -1, 2))
    valid_idx = torch.gather(task_valid, 1, safe_idx)            # (B, N) bool
    use = (idx >= 0) & valid_idx                                # (B, N)
    return torch.where(use.unsqueeze(-1), gathered, fallback)


class CanonicalCore:
    """Batched dynamics driver. Holds one :class:`CanonicalState`."""

    def __init__(
        self,
        cfg: CanonicalConfig,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.B = int(batch_size)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.field_size = torch.tensor(cfg.field_size, device=self.device, dtype=dtype)
        self.state: CanonicalState  # set by reset() or assigned in tests
        self._rngs: list[np.random.Generator] = []
        self._adv_rngs: list[np.random.Generator] = []
        self.adversaries = None  # AdversaryController, created lazily in reset()
        self.telemetry = None    # EpisodeTelemetry, created in reset(); None => no accumulation

    # ------------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None) -> CanonicalState:
        """Sample a fresh batch. Per-env seeds are independent of ``B`` so a given
        environment's trajectory is bit-identical across batch sizes (Section 3.13).
        """
        base = self.cfg.seed if seed is None else seed
        children = np.random.SeedSequence(base).spawn(self.B)
        self._rngs = [np.random.default_rng(c) for c in children]
        arrays = [sample_initial_arrays(self.cfg, rng) for rng in self._rngs]
        self.state = self._stack_arrays(arrays)
        # separate, batch-size-independent RNG stream for stochastic adversaries
        adv_children = np.random.SeedSequence(entropy=[int(base), 0xADFEED]).spawn(self.B)
        self._adv_rngs = [np.random.default_rng(c) for c in adv_children]
        if self.adversaries is None:
            from .adversaries import AdversaryController  # lazy: avoids import cycle
            self.adversaries = AdversaryController(self.cfg, self.device, self.dtype)
        self.adversaries.reset(self.state, self._adv_rngs)
        from .telemetry import EpisodeTelemetry  # lazy: avoids import cycle
        self.telemetry = EpisodeTelemetry(self.state, self.cfg, self.dtype)
        return self.state

    def _stack_arrays(self, arrays: list[dict]) -> CanonicalState:
        dev, dt = self.device, self.dtype

        def f(key):
            return torch.tensor(np.stack([a[key] for a in arrays]), device=dev, dtype=dt)

        def i(key):
            return torch.tensor(np.stack([a[key] for a in arrays]), device=dev, dtype=torch.int64)

        def b(key):
            return torch.tensor(np.stack([a[key] for a in arrays]), device=dev, dtype=torch.bool)

        B = len(arrays)
        return CanonicalState(
            robot_pos=f("robot_pos"), robot_vel=f("robot_vel"),
            robot_action=torch.full((B, self.cfg.max_robots), self.cfg.idle_action, device=dev, dtype=torch.int64),
            robot_valid=b("robot_valid"),
            adv_pos=f("adv_pos"), adv_vel=f("adv_vel"), adv_target=i("adv_target"),
            adv_archetype=i("adv_archetype"), adv_valid=b("adv_valid"),
            task_pos=f("task_pos"), task_w=f("task_w"), task_c=b("task_c"), task_valid=b("task_valid"),
            task_protected=b("task_protected"),
            sigma=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=dt),
            eta=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=dt),
            t_since_change=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=dt),
            t=torch.zeros(B, device=dev, dtype=dt),
            held_integral=torch.zeros(B, device=dev, dtype=dt),
            first_complete_t=torch.full((B, self.cfg.max_tasks), -1.0, device=dev, dtype=dt),
            task_c_red=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=torch.bool),
            sigma_red=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=dt),
            held_integral_red=torch.zeros(B, device=dev, dtype=dt),
            neutral_claim=torch.zeros((B, self.cfg.max_tasks), device=dev, dtype=torch.int8),
            task_height=b("task_c").to(torch.int8),   # a pre-owned task starts at height 1, else 0
            task_toggle_idx=i("task_toggle_idx"),
            task_is_toggle=b("task_is_toggle"),
        )

    def _held_values(self, s: CanonicalState) -> tuple[Tensor, Tensor]:
        """Per-team instantaneous held value (B,) = sum_i w_i * height_i * mult_i * [team owns i].

        Height (>1 only under stacking) makes each stack piece score, raised to ``stack_value_power``
        so a tall stack can be worth CONVEXLY more than the same pieces spread flat (power 1 = linear).
        TOGGLES: while a team owns a region's toggle, that team's held value from the region's GOALS is
        multiplied by ``toggle_multiplier`` (Override's 3x-a-quadrant). Both reduce to the plain
        weighted objective (mult == 1, height == task_c) when off, so Phase 1/2 are unaffected."""
        cfg = self.cfg
        wv = s.task_w * s.task_valid.to(self.dtype)
        if cfg.stack_cap > 1:
            h = s.task_height.to(self.dtype)
            base = wv * (h.pow(cfg.stack_value_power) if cfg.stack_value_power != 1.0 else h)
        else:
            base = wv
        if cfg.toggle_regions > 0 and cfg.toggle_multiplier > 1.0:
            goal = (~s.task_is_toggle).to(self.dtype)                    # toggles are not self-multiplied
            gov_blue = s.task_c.gather(1, s.task_toggle_idx).to(self.dtype)   # blue owns my region's toggle
            gov_red = s.task_c_red.gather(1, s.task_toggle_idx).to(self.dtype)
            m = cfg.toggle_multiplier - 1.0
            mult_blue = 1.0 + m * goal * gov_blue
            mult_red = 1.0 + m * goal * gov_red
        else:
            mult_blue = mult_red = 1.0
        blue = (base * mult_blue * s.task_c.to(self.dtype)).sum(-1)
        red = (base * mult_red * s.task_c_red.to(self.dtype)).sum(-1)
        return blue, red

    # -------------------------------------------------------------------- tick
    def tick(self) -> None:
        """Advance one ``dt`` using the current ``robot_action`` and ``adv_target``.

        Implements plan Section 3.3 steps 1-7 in order.
        """
        s = self.state
        cfg = self.cfg
        dt = cfg.dt

        # (1) integrate motion for both teams
        robot_target = self._robot_targets()
        adv_target_pos = _gather_target(s.task_pos, s.task_valid, s.adv_target, s.adv_pos)
        s.robot_pos, s.robot_vel = _integrate_motion(
            s.robot_pos, s.robot_vel, robot_target, s.robot_valid,
            cfg.v_max, cfg.a_max, dt, self.field_size)
        s.adv_pos, s.adv_vel = _integrate_motion(
            s.adv_pos, s.adv_vel, adv_target_pos, s.adv_valid,
            cfg.adv_v_max, cfg.adv_a_max, dt, self.field_size)

        # (2) occupancy within the service radius
        n_team = self._occupancy(s.robot_pos, s.robot_valid)   # (B, T) float
        n_adv = self._occupancy(s.adv_pos, s.adv_valid)        # (B, T) float

        if cfg.symmetric and cfg.stack_cap > 1:
            complete, revert, changed, newly, standoff = self._transitions_stacking(s, n_team, n_adv, dt)
        elif cfg.symmetric:
            complete, revert, changed, newly, standoff = self._transitions_symmetric(s, n_team, n_adv, dt)
        else:
            # (3) service timers: accumulate on incomplete tasks with a team present
            service_active = (~s.task_c) & (n_team >= 1) & s.task_valid
            s.sigma = torch.where(service_active, s.sigma + dt, torch.zeros_like(s.sigma))

            # (4) reversal timers: accumulate on complete tasks under contest pressure
            contest = self._contest_condition(n_team, n_adv)       # (B, T) bool
            reversal_active = s.task_c & contest & s.task_valid
            s.eta = torch.where(reversal_active, s.eta + dt, torch.zeros_like(s.eta))

            # (5) transitions (mutually exclusive by construction of steps 3/4)
            # re-acquisition cost rho: a task completed before needs tau_com * rho to complete
            # again (first_complete_t >= 0 marks a prior completion; set later in step 6).
            ever = s.first_complete_t >= 0.0
            thresh = cfg.tau_com * (1.0 + ever.to(s.sigma.dtype) * (cfg.reacquire_cost - 1.0))
            complete = (~s.task_c) & (s.sigma >= thresh) & s.task_valid
            revert = s.task_c & (s.eta >= cfg.tau_rev) & s.task_valid & ~s.task_protected  # protected => never reverts
            s.task_c = (s.task_c & ~revert) | complete
            s.sigma = torch.where(complete, torch.zeros_like(s.sigma), s.sigma)
            s.eta = torch.where(revert, torch.zeros_like(s.eta), s.eta)
            changed = complete | revert
            newly = complete & (s.first_complete_t < 0)
            standoff = None

        # (5b) t_since_change (shared): reset on any ownership change, else advance
        s.t_since_change = torch.where(
            s.task_valid,
            torch.where(changed, torch.zeros_like(s.t_since_change), s.t_since_change + dt),
            s.t_since_change,
        )

        # (6) accumulate held value (post-transition). held_integral is blue-only, kept as
        # telemetry; held_integral_red is red-only. The differential objective is their
        # difference (Override is decided by score DIFFERENCE, and the differential correctly
        # credits a reversal that neutralizes a red task). Single-team: task_c_red is all-False
        # so held_integral_red stays 0 and the objective reduces to blue-only.
        blue_val, red_val = self._held_values(s)
        s.held_integral = s.held_integral + blue_val * dt
        s.held_integral_red = s.held_integral_red + red_val * dt

        # record first completion at the tick-START clock (pre-advance), so that
        # int_{t_first}^T c dt == (T - t_first) exactly when a task is never reverted
        s.first_complete_t = torch.where(newly, s.t.unsqueeze(-1).expand_as(s.first_complete_t), s.first_complete_t)

        # (7) advance clock
        s.t = s.t + dt

        if self.telemetry is not None:
            self.telemetry.on_tick(s, complete, revert, standoff)

    def _robot_targets(self) -> Tensor:
        """Map each robot's action index to a target position (Section 3.5 layout)."""
        s = self.state
        T = self.cfg.max_tasks
        a = s.robot_action
        is_acquire = a < T
        is_defend = (a >= T) & (a < 2 * T)
        is_reverse = ((a >= 2 * T) & (a < 3 * T)) & self.cfg.symmetric   # REVERSE (Phase 2); False single-team
        # ACQUIRE -> a, DEFEND -> a-T, REVERSE -> a-2T, else IDLE (hold, -1)
        task_idx = torch.where(is_acquire, a,
                   torch.where(is_defend, a - T,
                   torch.where(is_reverse, a - 2 * T, torch.full_like(a, -1))))
        return _gather_target(s.task_pos, s.task_valid, task_idx, s.robot_pos)

    def _occupancy(self, pos: Tensor, valid: Tensor) -> Tensor:
        """Count valid agents within ``service_radius`` of each valid task -> (B, T)."""
        s = self.state
        diff = pos.unsqueeze(2) - s.task_pos.unsqueeze(1)      # (B, N, T, 2)
        dist = torch.linalg.vector_norm(diff, dim=-1)          # (B, N, T)
        near = (dist < self.cfg.service_radius) & valid.unsqueeze(-1) & s.task_valid.unsqueeze(1)
        return near.sum(dim=1).to(self.dtype)                  # (B, T)

    def _contest_condition(self, n_team: Tensor, n_adv: Tensor) -> Tensor:
        """Whether ``eta`` accumulates, per ``contest_mode`` (Section 3.3 table).

        This is the single-team rule (red dominating blue); equal to
        ``_dominates(n_adv, n_team)``. Kept as its own method so the single-team
        path is untouched and the reference match cannot regress.
        """
        mode = self.cfg.contest_mode
        if mode == "majority":
            return n_adv > self.cfg.beta * n_team
        if mode == "suppress":
            return (n_adv >= 1) & (n_team == 0)
        if mode == "none":
            return n_adv >= 1
        raise ValueError(f"unknown contest_mode: {mode}")

    def _dominates(self, n_a: Tensor, n_b: Tensor) -> Tensor:
        """Does team A control a task over team B, per ``contest_mode`` (symmetric mode)?

        Mirrors :meth:`_contest_condition` but in both directions, so the same
        contest rule governs which side may service/hold/reverse a task.
        """
        mode = self.cfg.contest_mode
        if mode == "majority":
            return n_a > self.cfg.beta * n_b
        if mode == "suppress":
            return (n_a >= 1) & (n_b == 0)
        if mode == "none":
            return n_a >= 1
        raise ValueError(f"unknown contest_mode: {mode}")

    def _transitions_symmetric(self, s: CanonicalState, n_team: Tensor, n_adv: Tensor, dt: float):
        """Symmetric two-team transitions (Phase 2). Returns
        ``(complete, revert, changed, newly, standoff)`` (blue-centric ``complete``/``revert``
        for telemetry) and mutates ``s`` in place.

        Both teams ACQUIRE neutral tasks, hold their own, and revert the opponent's
        (→ neutral). On a neutral task the sole dominator services it; when both teams are
        matched (neither dominates), **first-arrival priority** lets the incumbent claimant
        keep progressing, so two matched robots never freeze a task forever (model 3A + a
        tie-break — consultant round 6). ``standoff`` flags the residual true deadlock —
        matched presence with no incumbent — which telemetry counts. ``held``/
        ``t_since_change``/``first_complete_t``/clock are applied by the shared tail in
        :meth:`tick`.
        """
        cfg = self.cfg
        NONE, BLUE, RED = 0, 1, 2
        blue_dom = self._dominates(n_team, n_adv)
        red_dom = self._dominates(n_adv, n_team)
        blue_only = blue_dom & ~red_dom
        red_only = red_dom & ~blue_dom
        contested = (n_team >= 1) & (n_adv >= 1) & ~blue_only & ~red_only  # both present, neither dominates
        neutral = (~s.task_c) & (~s.task_c_red) & s.task_valid
        blue_owns = s.task_c & s.task_valid
        red_owns = s.task_c_red & s.task_valid

        # first-arrival claim: the sole dominator (re)claims a neutral task; a contested tie
        # keeps the incumbent claimant (so matched robots don't deadlock); owned tasks carry no
        # claim. The team that loses the claim has its service timer reset below.
        prev = s.neutral_claim
        claim = torch.where(blue_only, torch.full_like(prev, BLUE),
                torch.where(red_only, torch.full_like(prev, RED),
                torch.where(contested, prev, torch.full_like(prev, NONE))))
        claim = torch.where(neutral, claim, torch.full_like(prev, NONE))

        # (3) service on neutral tasks: only the current claimant progresses
        svc_blue = neutral & (claim == BLUE)
        svc_red = neutral & (claim == RED)
        s.sigma = torch.where(svc_blue, s.sigma + dt, torch.zeros_like(s.sigma))
        s.sigma_red = torch.where(svc_red, s.sigma_red + dt, torch.zeros_like(s.sigma_red))
        standoff = neutral & contested & (claim == NONE)   # matched presence, no incumbent

        # (4) reversal pressure on the current owner by the opponent dominating
        under_attack = (blue_owns & red_dom) | (red_owns & blue_dom)
        s.eta = torch.where(under_attack, s.eta + dt, torch.zeros_like(s.eta))

        # (5) transitions. A successful revert TRANSFERS OWNERSHIP to the reverter (Override's
        # descore+rescore is one continuous piece of work by one robot), not to neutral -- no open
        # contest, no deadlock, the taker HOLDS what it took (this gives R-take a real signal).
        # The take COSTS rho*tau_com of sustained domination to complete, so rho is LIVE on the
        # OFFENSIVE channel: at high rho, LOSING a task is expensive (re-taking it costs rho*tau_com)
        # -- the only regime where holding what you take (retention) can pay. Protected never flips;
        # a 1v1 cannot flip (needs the opponent to out-number), two committed robots can.
        ever = s.first_complete_t >= 0.0
        thresh = cfg.tau_com * (1.0 + ever.to(s.sigma.dtype) * (cfg.reacquire_cost - 1.0))
        complete_blue = svc_blue & (s.sigma >= thresh)
        complete_red = svc_red & (s.sigma_red >= thresh)
        take_thresh = cfg.reacquire_cost * cfg.tau_com                         # rho*tau_com to complete a take
        revert_blue = blue_owns & (s.eta >= take_thresh) & ~s.task_protected   # red takes blue's task
        revert_red = red_owns & (s.eta >= take_thresh) & ~s.task_protected     # blue takes red's task
        s.task_c = (blue_owns & ~revert_blue) | complete_blue | revert_red     # keep | acquire | TAKE
        s.task_c_red = (red_owns & ~revert_red) | complete_red | revert_blue   # keep | acquire | TAKE
        s.sigma = torch.where(complete_blue, torch.zeros_like(s.sigma), s.sigma)
        s.sigma_red = torch.where(complete_red, torch.zeros_like(s.sigma_red), s.sigma_red)
        s.eta = torch.where(revert_blue | revert_red, torch.zeros_like(s.eta), s.eta)
        s.neutral_claim = torch.where(complete_blue | complete_red, torch.full_like(prev, NONE), claim)

        changed = complete_blue | complete_red | revert_blue | revert_red
        newly = (complete_blue | complete_red) & (s.first_complete_t < 0)
        return complete_blue, revert_blue, changed, newly, standoff

    def _transitions_stacking(self, s: CanonicalState, n_team: Tensor, n_adv: Tensor, dt: float):
        """PHASE 3 stacking transitions. Placing the piece at level L (height h -> h+1, so L = h+1)
        costs tau_com * L, and dismantling the TOP piece of a height-h stack (level L = h) costs
        tau_rev * L -- reaching higher is slower to add AND to remove. So building a full stack costs
        tau_com * h(h+1)/2 and flattening it costs tau_rev * h(h+1)/2 -- both quadratic, one piece at
        a time (partial reversal). Height 0 == neutral; the held value accumulated by the shared tail
        is height-weighted. A sole dominator BUILDS on its own/claimed task; the opponent DISMANTLES.
        Returns the telemetry-shaped tuple. Mirrored in reference.py."""
        cfg = self.cfg
        NONE, BLUE, RED = 0, 1, 2
        cap = cfg.stack_cap
        z = torch.zeros_like(s.sigma)
        blue_dom = self._dominates(n_team, n_adv)
        red_dom = self._dominates(n_adv, n_team)
        blue_only = blue_dom & ~red_dom
        red_only = red_dom & ~blue_dom
        contested = (n_team >= 1) & (n_adv >= 1) & ~blue_only & ~red_only
        h = s.task_height
        neutral = (h == 0) & s.task_valid
        blue_owns = s.task_c & s.task_valid
        red_owns = s.task_c_red & s.task_valid

        # first-arrival claim on a NEUTRAL task (mirrors _transitions_symmetric)
        prev = s.neutral_claim
        claim = torch.where(blue_only, torch.full_like(prev, BLUE),
                torch.where(red_only, torch.full_like(prev, RED),
                torch.where(contested, prev, torch.full_like(prev, NONE))))
        claim = torch.where(neutral, claim, torch.full_like(prev, NONE))

        # (3) BUILD: the SOLE dominator services a task it can grow (a neutral task it claims, or its
        # own stack below the cap). Placing the piece at level L = h+1 completes after tau_com * L of
        # service (raising the arm higher is slower); height increments, service resets.
        can_h = h < cap
        build_active_blue = (((neutral & (claim == BLUE)) | blue_owns) & blue_only & can_h)
        build_active_red = (((neutral & (claim == RED)) | red_owns) & red_only & can_h)
        build_thresh = cfg.tau_com * (h + 1).to(s.sigma.dtype)   # level L = h+1 costs tau_com * L
        s.sigma = torch.where(build_active_blue, s.sigma + dt, z)
        s.sigma_red = torch.where(build_active_red, s.sigma_red + dt, z)
        build_blue = build_active_blue & (s.sigma >= build_thresh)
        build_red = build_active_red & (s.sigma_red >= build_thresh)

        # (4) DISMANTLE: the opponent dominates an owned stack; its TOP piece comes off after
        # tau_rev * h of sustained pressure. Protected never dismantles.
        under_attack = (blue_owns & red_dom) | (red_owns & blue_dom)
        s.eta = torch.where(under_attack, s.eta + dt, z)
        dismantle_thresh = cfg.tau_rev * h.to(s.eta.dtype).clamp_min(1.0)
        dismantle = under_attack & (s.eta >= dismantle_thresh) & ~s.task_protected

        # (5) apply. build/dismantle are mutually exclusive per task (build needs sole domination of
        # your own/neutral task; dismantle needs the opponent to dominate your task), so delta in {-1,0,1}.
        delta = build_blue.to(h.dtype) + build_red.to(h.dtype) - dismantle.to(h.dtype)
        new_h = (h + delta).clamp(0, cap)
        s.task_height = new_h
        s.task_c = (blue_owns | build_blue) & (new_h > 0)          # dismantled to 0 => neutral (lose it ALL)
        s.task_c_red = (red_owns | build_red) & (new_h > 0)
        s.sigma = torch.where(build_blue, z, s.sigma)
        s.sigma_red = torch.where(build_red, z, s.sigma_red)
        s.eta = torch.where(dismantle, z, s.eta)
        s.neutral_claim = torch.where(build_blue | build_red, torch.full_like(prev, NONE), claim)

        standoff = neutral & contested & (claim == NONE)
        changed = build_blue | build_red | dismantle
        newly = (build_blue | build_red) & (s.first_complete_t < 0)
        return build_blue, (dismantle & blue_owns), changed, newly, standoff

    # -------------------------------------------------------------------- step
    def step(self, robot_actions: Tensor) -> tuple[Tensor, Tensor, dict]:
        """Hold ``robot_actions`` fixed for one decision (``ticks_per_decision``
        ticks), refreshing adversary targets first. Returns ``(reward, done, info)``.

        The graph observation builder (Section 3.7) arrives with Component A/M2;
        for now callers read :pyattr:`state` directly.
        """
        s = self.state
        s.robot_action = robot_actions.to(self.device, torch.int64)
        if self.telemetry is not None:
            self.telemetry.on_decision(s, s.robot_action)
        self._refresh_adversary_targets()

        denom = (s.task_w * s.task_valid.to(self.dtype)).sum(-1).clamp_min(1e-9)
        reward = torch.zeros(self.B, device=self.device, dtype=self.dtype)
        for _ in range(self.cfg.ticks_per_decision):
            self.tick()
            # differential held value (blue - red), height- and toggle-multiplier-weighted; its
            # episode sum is the differential J_H. Reduces to blue-only in the single-team env.
            blue_val, red_val = self._held_values(s)
            held_diff = blue_val - red_val
            reward = reward + held_diff * self.cfg.dt / (self.cfg.horizon_T * denom)
        # half-tick tolerance: robust to float32 accumulation of t over the episode
        done = s.t >= (self.cfg.horizon_T - 0.5 * self.cfg.dt)
        info = {"held_integral": s.held_integral.detach().clone(),
                "held_integral_red": s.held_integral_red.detach().clone()}
        if self.telemetry is not None and bool(done.all()):
            info["telemetry"] = self.telemetry.summary(s)
        return reward, done, info

    def _refresh_adversary_targets(self) -> None:
        """Refresh ``adv_target`` via the full archetype controller (adversaries.py).

        No-op when no controller is attached (e.g. a state assigned directly in a
        unit test), in which case the existing ``adv_target`` is used as given.
        """
        if self.adversaries is None:
            return
        self.state.adv_target = self.adversaries.select_targets(self.state)


# ----------------------------------------------------------------------- layout
def sample_initial_arrays(cfg: CanonicalConfig, rng: np.random.Generator) -> dict:
    """Sample one environment's initial condition as NumPy arrays (padded).

    Shared by the core reset and the reference so both start from identical
    conditions; the *dynamics* are what the two implement independently.
    """
    R, T, K = cfg.max_robots, cfg.max_tasks, cfg.max_adversaries
    W, H = cfg.field_size
    nr, nt, nk = cfg.n_robots, cfg.n_tasks, cfg.n_adversaries

    robot_pos = np.zeros((R, 2), np.float64)
    robot_pos[:nr] = rng.uniform([0, 0], [W, H], size=(nr, 2))
    robot_valid = np.zeros(R, bool); robot_valid[:nr] = True

    adv_pos = np.zeros((K, 2), np.float64)
    adv_pos[:nk] = rng.uniform([0, 0], [W, H], size=(nk, 2))
    adv_valid = np.zeros(K, bool); adv_valid[:nk] = True
    adv_target = np.full(K, -1, np.int64)
    archetypes = _resolve_archetypes(cfg.adversary_population, nk, rng)
    adv_archetype = np.zeros(K, np.int64); adv_archetype[:nk] = archetypes

    task_pos = np.zeros((T, 2), np.float64)
    task_pos[:nt] = _sample_layout(cfg, nt, rng)
    task_valid = np.zeros(T, bool); task_valid[:nt] = True
    task_w = np.zeros(T, np.float64)
    task_w[:nt] = _sample_weights(cfg, nt, rng)

    # mark a fraction of tasks PROTECTED (irreversible) — structural retention variance
    task_protected = np.zeros(T, bool)
    n_prot = int(round(cfg.protected_fraction * nt))
    if n_prot > 0:
        task_protected[rng.choice(nt, size=n_prot, replace=False)] = True

    # PHASE 3 toggles: partition the nt tasks into toggle_regions index-blocks; the FIRST task of each
    # block is its TOGGLE, the rest are GOALS that point at it. Identity default (self) is a no-op.
    task_toggle_idx = np.arange(T, dtype=np.int64)
    task_is_toggle = np.zeros(T, bool)
    Q = cfg.toggle_regions
    if Q > 0 and nt > 0:
        if cfg.dynamic_exposure and nt >= 2 * Q:
            # VARIED cluster sizes (each region >= 2: toggle + >=1 goal) -> a wide range of cluster
            # EXPOSURE, so a policy must price a toggle by its CURRENT load rather than a fixed premium.
            # Drawn from the shared rng, so core and reference partition identically (reference-match kept).
            extra = nt - 2 * Q
            adds = rng.multinomial(extra, np.full(Q, 1.0 / Q)) if extra > 0 else np.zeros(Q, dtype=np.int64)
            start = 0
            for q in range(Q):
                end = nt if q == Q - 1 else start + 2 + int(adds[q])
                task_toggle_idx[start:end] = start
                task_is_toggle[start] = True
                start = end
        else:
            per = max(1, nt // Q)
            for q in range(Q):
                start = q * per
                if start >= nt:
                    break
                end = nt if q == Q - 1 else min(nt, (q + 1) * per)
                task_toggle_idx[start:end] = start
                task_is_toggle[start] = True

    return dict(
        robot_pos=robot_pos, robot_vel=np.zeros((R, 2), np.float64), robot_valid=robot_valid,
        adv_pos=adv_pos, adv_vel=np.zeros((K, 2), np.float64), adv_target=adv_target,
        adv_archetype=adv_archetype, adv_valid=adv_valid,
        task_pos=task_pos, task_w=task_w, task_c=np.zeros(T, bool), task_valid=task_valid,
        task_protected=task_protected, task_toggle_idx=task_toggle_idx, task_is_toggle=task_is_toggle,
    )


_ARCHETYPE_IDS = {
    "greedy_nearest": 0, "value_targeting": 1, "camper": 2, "feinter": 3, "learned_selfplay": 4,
    "builder": 5, "toggle_raider": 6,
}


def _resolve_archetypes(population: tuple[str, ...], nk: int, rng: np.random.Generator) -> np.ndarray:
    ids = np.array([_ARCHETYPE_IDS.get(name, 0) for name in population], np.int64)
    if len(ids) == 0:
        return np.zeros(nk, np.int64)
    return ids[rng.integers(0, len(ids), size=nk)]


def _sample_weights(cfg: CanonicalConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = cfg.weight_range
    if cfg.weight_dist == "uniform":
        return rng.uniform(lo, hi, size=n)
    if cfg.weight_dist == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=0.5, size=n)
        raw = raw / raw.max() if raw.max() > 0 else raw
        return lo + (hi - lo) * np.clip(raw, 0, 1)
    if cfg.weight_dist == "bimodal":
        pick_hi = rng.random(n) < 0.5
        return np.where(pick_hi, hi, lo).astype(np.float64)
    raise ValueError(cfg.weight_dist)


def _sample_layout(cfg: CanonicalConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    W, H = cfg.field_size
    if cfg.layout == "uniform":
        return rng.uniform([0, 0], [W, H], size=(n, 2))
    if cfg.layout == "clustered":
        centers = rng.uniform([0.15 * W, 0.15 * H], [0.85 * W, 0.85 * H], size=(cfg.n_clusters, 2))
        which = rng.integers(0, cfg.n_clusters, size=n)
        spread = 0.06 * min(W, H)
        pts = centers[which] + rng.normal(0, spread, size=(n, 2))
        return np.clip(pts, [0, 0], [W, H])
    if cfg.layout == "polarized":
        side = rng.random(n) < 0.5
        pts = np.empty((n, 2))
        pts[:, 0] = np.where(side, rng.uniform(0, 0.25 * W, n), rng.uniform(0.75 * W, W, n))
        pts[:, 1] = rng.uniform(0, H, n)
        return pts
    raise ValueError(cfg.layout)
