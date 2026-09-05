"""Slow, obvious, single-environment NumPy reference (plan Section 3.13).

Written *independently* of the batched torch core in ``core.py``; its only job is
to be so simple it is obviously correct, so that ``test_batched_matches_reference``
can pin the core's dynamics. Operates on one environment (no batch dimension), at
the padded dimensions with validity masks.
"""
from __future__ import annotations

import numpy as np

from .config import CanonicalConfig


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v * v).sum(axis=-1))


def _navigate(pos, vel, target, valid, v_max, a_max, dt, field):
    """Accel-limited point navigation; walls last; invalid entities frozen."""
    eps = 1e-9
    to_target = target - pos                              # (N, 2)
    dist = _norm(to_target)                               # (N,)
    direction = to_target / np.maximum(dist, eps)[:, None]
    desired_speed = np.minimum(dist / dt, v_max)          # (N,)
    desired_vel = direction * desired_speed[:, None]
    desired_vel[dist <= eps] = 0.0

    a_needed = (desired_vel - vel) / dt
    a_norm = _norm(a_needed)[:, None]
    a = a_needed * np.minimum(a_max / np.maximum(a_norm, eps), 1.0)
    new_vel = vel + a * dt
    v_norm = _norm(new_vel)[:, None]
    new_vel = new_vel * np.minimum(v_max / np.maximum(v_norm, eps), 1.0)
    new_pos = pos + new_vel * dt

    clamped = np.maximum(np.minimum(new_pos, field), 0.0)
    hit = clamped != new_pos
    new_vel = np.where(hit, 0.0, new_vel)
    new_pos = clamped

    keep = valid[:, None]
    new_pos = np.where(keep, new_pos, pos)
    new_vel = np.where(keep, new_vel, vel)
    return new_pos, new_vel


def _target_for(task_pos, task_valid, idx, fallback):
    """Per-node target from slot index (-1 or invalid -> fallback)."""
    T = task_pos.shape[0]
    out = fallback.copy()
    for n in range(idx.shape[0]):
        j = int(idx[n])
        if 0 <= j < T and task_valid[j]:
            out[n] = task_pos[j]
    return out


class ReferenceEnv:
    """One environment's dynamics, NumPy, float64, deliberately literal."""

    def __init__(self, cfg: CanonicalConfig, arrays: dict):
        self.cfg = cfg
        a = {k: np.array(v, np.float64) if v.dtype.kind == "f" else np.array(v) for k, v in arrays.items()}
        self.robot_pos = a["robot_pos"].astype(np.float64)
        self.robot_vel = a["robot_vel"].astype(np.float64)
        self.robot_valid = a["robot_valid"].astype(bool)
        self.adv_pos = a["adv_pos"].astype(np.float64)
        self.adv_vel = a["adv_vel"].astype(np.float64)
        self.adv_valid = a["adv_valid"].astype(bool)
        self.adv_target = a["adv_target"].astype(np.int64)
        self.adv_archetype = a["adv_archetype"].astype(np.int64)
        self.task_pos = a["task_pos"].astype(np.float64)
        self.task_w = a["task_w"].astype(np.float64)
        self.task_c = a["task_c"].astype(bool)
        self.task_valid = a["task_valid"].astype(bool)
        self.task_protected = a.get("task_protected", np.zeros(cfg.max_tasks, bool)).astype(bool)
        # PHASE 2 (symmetric): red ownership + red service timer. All-False/zero and
        # never read when cfg.symmetric is False, so the single-team match is unchanged.
        self.task_c_red = a.get("task_c_red", np.zeros(cfg.max_tasks, bool)).astype(bool)
        self.neutral_claim = a.get("neutral_claim", np.zeros(cfg.max_tasks, np.int64)).astype(np.int64)
        # PHASE 3 (stacking): integer stack height, 0..stack_cap; a pre-owned task starts at height 1.
        self.task_height = (a["task_height"] if "task_height" in a else self.task_c).astype(np.int8)
        # PHASE 3 (toggles): governing-toggle index per task (self by default) + toggle flag.
        self.task_toggle_idx = (a["task_toggle_idx"] if "task_toggle_idx" in a
                                else np.arange(cfg.max_tasks)).astype(np.int64)
        self.task_is_toggle = (a["task_is_toggle"] if "task_is_toggle" in a
                               else np.zeros(cfg.max_tasks, bool)).astype(bool)

        T = cfg.max_tasks
        self.sigma = np.zeros(T)
        self.sigma_red = np.zeros(T)
        self.eta = np.zeros(T)
        self.t_since_change = np.zeros(T)
        self.first_complete_t = np.full(T, -1.0)
        self.t = 0.0
        self.held_integral = 0.0
        self.held_integral_red = 0.0
        self.robot_action = np.full(cfg.max_robots, cfg.idle_action, np.int64)
        self.field = np.array(cfg.field_size, np.float64)

    def _occupancy(self, pos, valid):
        """Number of valid agents within service_radius of each valid task."""
        n_tasks = self.task_pos.shape[0]
        counts = np.zeros(n_tasks)
        for i in range(n_tasks):
            if not self.task_valid[i]:
                continue
            d = _norm(pos - self.task_pos[i][None, :])       # (N,)
            counts[i] = np.sum((d < self.cfg.service_radius) & valid)
        return counts

    def _robot_target(self):
        T = self.cfg.max_tasks
        idx = np.full(self.cfg.max_robots, -1, np.int64)
        for n in range(self.cfg.max_robots):
            act = int(self.robot_action[n])
            if act < T:
                idx[n] = act                                  # ACQUIRE task act
            elif act < 2 * T:
                idx[n] = act - T                              # DEFEND task act-T
            elif self.cfg.symmetric and act < 3 * T:
                idx[n] = act - 2 * T                          # REVERSE task act-2T (Phase 2)
            # else IDLE -> -1 (hold)
        return _target_for(self.task_pos, self.task_valid, idx, self.robot_pos)

    def _contest(self, n_team, n_adv):
        mode = self.cfg.contest_mode
        if mode == "majority":
            return n_adv > self.cfg.beta * n_team
        if mode == "suppress":
            return (n_adv >= 1) & (n_team == 0)
        if mode == "none":
            return n_adv >= 1
        raise ValueError(mode)

    def _dominates(self, n_a, n_b):
        """Team A controls a task over team B, per contest_mode (symmetric mode)."""
        mode = self.cfg.contest_mode
        if mode == "majority":
            return n_a > self.cfg.beta * n_b
        if mode == "suppress":
            return (n_a >= 1) & (n_b == 0)
        if mode == "none":
            return n_a >= 1
        raise ValueError(mode)

    def tick(self):
        cfg = self.cfg
        dt = cfg.dt

        # (1) motion
        rt = self._robot_target()
        at = _target_for(self.task_pos, self.task_valid, self.adv_target, self.adv_pos)
        self.robot_pos, self.robot_vel = _navigate(
            self.robot_pos, self.robot_vel, rt, self.robot_valid, cfg.v_max, cfg.a_max, dt, self.field)
        self.adv_pos, self.adv_vel = _navigate(
            self.adv_pos, self.adv_vel, at, self.adv_valid, cfg.adv_v_max, cfg.adv_a_max, dt, self.field)

        # (2) occupancy
        n_team = self._occupancy(self.robot_pos, self.robot_valid)
        n_adv = self._occupancy(self.adv_pos, self.adv_valid)

        if cfg.symmetric and cfg.stack_cap > 1:
            complete, revert, changed, newly = self._transitions_stacking(n_team, n_adv, dt)
        elif cfg.symmetric:
            complete, revert, changed, newly = self._transitions_symmetric(n_team, n_adv, dt)
        else:
            # (3) service timers
            service = (~self.task_c) & (n_team >= 1) & self.task_valid
            self.sigma = np.where(service, self.sigma + dt, 0.0)

            # (4) reversal timers
            reversal = self.task_c & self._contest(n_team, n_adv) & self.task_valid
            self.eta = np.where(reversal, self.eta + dt, 0.0)

            # (5) transitions
            # re-acquisition cost rho: a previously-completed task needs tau_com * rho again
            ever = self.first_complete_t >= 0.0
            thresh = cfg.tau_com * (1.0 + ever.astype(np.float64) * (cfg.reacquire_cost - 1.0))
            complete = (~self.task_c) & (self.sigma >= thresh) & self.task_valid
            revert = self.task_c & (self.eta >= cfg.tau_rev) & self.task_valid & ~self.task_protected
            self.task_c = (self.task_c & ~revert) | complete
            self.sigma = np.where(complete, 0.0, self.sigma)
            self.eta = np.where(revert, 0.0, self.eta)
            changed = complete | revert
            newly = complete & (self.first_complete_t < 0)

        # (5b) t_since_change (shared)
        self.t_since_change = np.where(
            self.task_valid, np.where(changed, 0.0, self.t_since_change + dt), self.t_since_change)

        # (6) accumulate held value: blue-only (telemetry) + red-only (differential J_H). STACKING:
        # height-weighted; TOGGLES: a region's goals score toggle_multiplier x while that team owns the
        # region's toggle (both reduce to the plain weighted objective when off). Mirrors core._held_values.
        wv = self.task_w * self.task_valid
        if cfg.stack_cap > 1:
            h = self.task_height.astype(np.float64)
            base = wv * (h ** cfg.stack_value_power if cfg.stack_value_power != 1.0 else h)
        else:
            base = wv
        if cfg.toggle_regions > 0 and cfg.toggle_multiplier > 1.0:
            goal = (~self.task_is_toggle).astype(np.float64)
            gov_blue = self.task_c[self.task_toggle_idx].astype(np.float64)
            gov_red = self.task_c_red[self.task_toggle_idx].astype(np.float64)
            m = cfg.toggle_multiplier - 1.0
            mult_blue = 1.0 + m * goal * gov_blue
            mult_red = 1.0 + m * goal * gov_red
        else:
            mult_blue = mult_red = 1.0
        self.held_integral += float(np.sum(base * mult_blue * self.task_c)) * dt
        self.held_integral_red += float(np.sum(base * mult_red * self.task_c_red)) * dt

        # record first completion at the tick-START clock (pre-advance) — matches core
        self.first_complete_t = np.where(newly, self.t, self.first_complete_t)

        # (7) advance clock
        self.t += dt

    def _transitions_symmetric(self, n_team, n_adv, dt):
        """Symmetric two-team transitions with first-arrival priority (Phase 2) — mirrors core."""
        cfg = self.cfg
        NONE, BLUE, RED = 0, 1, 2
        blue_dom = self._dominates(n_team, n_adv)
        red_dom = self._dominates(n_adv, n_team)
        blue_only = blue_dom & ~red_dom
        red_only = red_dom & ~blue_dom
        contested = (n_team >= 1) & (n_adv >= 1) & ~blue_only & ~red_only
        neutral = (~self.task_c) & (~self.task_c_red) & self.task_valid
        blue_owns = self.task_c & self.task_valid
        red_owns = self.task_c_red & self.task_valid

        prev = self.neutral_claim
        claim = np.where(blue_only, BLUE, np.where(red_only, RED, np.where(contested, prev, NONE)))
        claim = np.where(neutral, claim, NONE).astype(np.int64)

        svc_blue = neutral & (claim == BLUE)
        svc_red = neutral & (claim == RED)
        self.sigma = np.where(svc_blue, self.sigma + dt, 0.0)
        self.sigma_red = np.where(svc_red, self.sigma_red + dt, 0.0)

        under_attack = (blue_owns & red_dom) | (red_owns & blue_dom)
        self.eta = np.where(under_attack, self.eta + dt, 0.0)

        ever = self.first_complete_t >= 0.0
        thresh = cfg.tau_com * (1.0 + ever.astype(np.float64) * (cfg.reacquire_cost - 1.0))
        complete_blue = svc_blue & (self.sigma >= thresh)
        complete_red = svc_red & (self.sigma_red >= thresh)
        take_thresh = cfg.reacquire_cost * cfg.tau_com                               # rho*tau_com to take (rho LIVE)
        revert_blue = blue_owns & (self.eta >= take_thresh) & ~self.task_protected   # red takes blue's
        revert_red = red_owns & (self.eta >= take_thresh) & ~self.task_protected     # blue takes red's
        # a successful revert TRANSFERS OWNERSHIP to the reverter (mirrors core): keep | acquire | TAKE
        self.task_c = (blue_owns & ~revert_blue) | complete_blue | revert_red
        self.task_c_red = (red_owns & ~revert_red) | complete_red | revert_blue
        self.sigma = np.where(complete_blue, 0.0, self.sigma)
        self.sigma_red = np.where(complete_red, 0.0, self.sigma_red)
        self.eta = np.where(revert_blue | revert_red, 0.0, self.eta)
        self.neutral_claim = np.where(complete_blue | complete_red, NONE, claim).astype(np.int64)

        changed = complete_blue | complete_red | revert_blue | revert_red
        newly = (complete_blue | complete_red) & (self.first_complete_t < 0)
        return complete_blue, revert_blue, changed, newly

    def _transitions_stacking(self, n_team, n_adv, dt):
        """PHASE 3 stacking transitions (mirrors core._transitions_stacking): place the piece at
        level L = h+1 after tau_com * L, dismantle the top piece of a height-h stack (level L = h)
        after tau_rev * L -- reaching higher is slower to add AND remove; one piece at a time."""
        cfg = self.cfg
        NONE, BLUE, RED = 0, 1, 2
        cap = cfg.stack_cap
        blue_dom = self._dominates(n_team, n_adv)
        red_dom = self._dominates(n_adv, n_team)
        blue_only = blue_dom & ~red_dom
        red_only = red_dom & ~blue_dom
        contested = (n_team >= 1) & (n_adv >= 1) & ~blue_only & ~red_only
        h = self.task_height
        neutral = (h == 0) & self.task_valid
        blue_owns = self.task_c & self.task_valid
        red_owns = self.task_c_red & self.task_valid

        prev = self.neutral_claim
        claim = np.where(blue_only, BLUE, np.where(red_only, RED, np.where(contested, prev, NONE)))
        claim = np.where(neutral, claim, NONE).astype(np.int64)

        can_h = h < cap
        build_active_blue = ((neutral & (claim == BLUE)) | blue_owns) & blue_only & can_h
        build_active_red = ((neutral & (claim == RED)) | red_owns) & red_only & can_h
        build_thresh = cfg.tau_com * (h.astype(np.float64) + 1.0)   # level L = h+1 costs tau_com * L
        self.sigma = np.where(build_active_blue, self.sigma + dt, 0.0)
        self.sigma_red = np.where(build_active_red, self.sigma_red + dt, 0.0)
        build_blue = build_active_blue & (self.sigma >= build_thresh)
        build_red = build_active_red & (self.sigma_red >= build_thresh)

        under_attack = (blue_owns & red_dom) | (red_owns & blue_dom)
        self.eta = np.where(under_attack, self.eta + dt, 0.0)
        dismantle_thresh = cfg.tau_rev * np.maximum(h.astype(np.float64), 1.0)
        dismantle = under_attack & (self.eta >= dismantle_thresh) & ~self.task_protected

        delta = build_blue.astype(np.int8) + build_red.astype(np.int8) - dismantle.astype(np.int8)
        new_h = np.clip(h + delta, 0, cap).astype(np.int8)
        self.task_height = new_h
        self.task_c = (blue_owns | build_blue) & (new_h > 0)
        self.task_c_red = (red_owns | build_red) & (new_h > 0)
        self.sigma = np.where(build_blue, 0.0, self.sigma)
        self.sigma_red = np.where(build_red, 0.0, self.sigma_red)
        self.eta = np.where(dismantle, 0.0, self.eta)
        self.neutral_claim = np.where(build_blue | build_red, NONE, claim).astype(np.int64)

        changed = build_blue | build_red | dismantle
        newly = (build_blue | build_red) & (self.first_complete_t < 0)
        return build_blue, (dismantle & blue_owns), changed, newly
