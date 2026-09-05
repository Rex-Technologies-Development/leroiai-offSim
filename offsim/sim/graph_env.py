"""OverrideGraphEnv — the real Override game behind TENURE's graph interface.

This is the *observation adapter* (plan Section 4.6), pulled forward to de-risk the
project's highest-risk integration question early: **can TENURE's architecture even
represent Override's structure?** It emits the exact canonical Section 3.7 dict
observation and the same flat action layout (Section 3.5), so a canonical-trained
TENURE policy transfers here unchanged — and the retention head is *never told* a
site's class, so predicting retention by class is a genuine geometry read.

Scope, per the consultant's note: the graph observation adapter plus a step that
drives the real ``OverrideField`` — NOT a port of the baselines or of training,
which stay on the fast batched canonical env. Baselines were designed for that
abstraction; adapting them to 13 heterogeneous Override sites is a reviewer trap.

Override contains three alpha regimes *in space* at once (folded into E9/E10):
- **alliance goals** are protected -> tau_rev = inf -> alpha = 0 -> retention ~ 1
- **neutral goals** are removable -> intermediate alpha
- **toggles** flip by the same action both ways -> alpha >= 1 -> low retention
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .config import (
    Alliance, DECISION_INTERVAL, DT, INTERACTION_RANGE, MATCH_DURATION, MAX_FORWARD_SPEED,
)
from .env import ObjectiveController, _body_command
from .field import OverrideField
from .opponent import ScriptedOpponent

_FIELD = 144.0
_V_MAX = float(MAX_FORWARD_SPEED)
_HORIZON = float(MATCH_DURATION)
_SERVICE = float(INTERACTION_RANGE)
_W_MAX = 8.0
_SITE_WEIGHT = {"alliance_goal": 6.0, "neutral_goal": 4.0, "neutral_tall": 8.0, "toggle": 5.0}


class OverrideGraphEnv:
    """Override match exposed as TENURE's graph environment (ally = blue)."""

    def __init__(self, chassis: str = "tank", opponent: str = "mixed", contested: bool = True,
                 max_robots: int = 8, max_tasks: int = 20, max_adversaries: int = 8, seed: int | None = None):
        self.max_robots, self.max_tasks, self.max_adversaries = max_robots, max_tasks, max_adversaries
        self.action_dim = 2 * max_tasks + 1
        self.idle_action = 2 * max_tasks
        self.chassis, self.opponent_style, self.contested = chassis, opponent, contested
        self.ally, self.foe = Alliance.BLUE, Alliance.RED
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        self.field = OverrideField(self.chassis, seed)
        if self.contested:
            self.field.contested_enabled = True
            self.field.contested = {**self.field.contested, "enabled": True}
        self.controller = ObjectiveController(self.field)
        self.opponent = ScriptedOpponent(self.opponent_style)
        self._sites = self._build_sites()                     # list of (kind, obj)
        self.site_classes = [k if k != "neutral_tall" else "neutral_goal" for k, _ in self._sites]
        self.n_sites = len(self._sites)
        self.last_actions = [self.idle_action, self.idle_action]
        self._change_t = [0.0] * self.n_sites
        self._prev_c = self._credit_vector()
        return self.observation()

    def _build_sites(self) -> list:
        sites = []
        for g in self.field.goals:
            kind = "alliance_goal" if g.protected_by is not None else ("neutral_tall" if g.kind == "neutral_tall" else "neutral_goal")
            sites.append((kind, g))
        for t in self.field.toggles:
            sites.append(("toggle", t))
        return sites

    # -------------------------------------------------------- site semantics
    def _credit_vector(self) -> list[bool]:
        """c_i: does the ally currently hold credited value at site i?"""
        out = []
        for kind, obj in self._sites:
            if kind == "toggle":
                out.append(obj.owner is self.ally)
            else:
                halves = obj.placed_pin_halves(self.field.pins)
                pts = halves.count(self.ally.value) + (halves.count("yellow") if self.field.goal_owner(obj) is self.ally else 0)
                out.append(pts > 0)
        return out

    def _timers(self, i: int) -> tuple[float, float]:
        """(sigma, eta) normalized to [0,1]; only toggles have live dwell timers."""
        kind, obj = self._sites[i]
        if kind != "toggle" or not self.contested:
            return 0.0, 0.0
        dwell = max(1e-6, float(self.field.contested["toggle_claim_dwell"]) * float(self.field.contested["alpha_scale"]))
        timer = self.field._toggle_claim_timer.get(obj.toggle_id, 0.0) / dwell
        pending = self.field._toggle_pending.get(obj.toggle_id)
        if obj.owner is self.ally:
            return 0.0, (timer if pending is self.foe else 0.0)          # eta: opponent re-claiming
        return (timer if pending is self.ally else 0.0), 0.0             # sigma: ally claiming

    # ------------------------------------------------------------ observation
    def observation(self) -> dict:
        f = self.field
        R, T, K = self.max_robots, self.max_tasks, self.max_adversaries
        allies = [f.robots[0], f.robots[1]]
        foes = [f.robots[2], f.robots[3]]
        credit = self._credit_vector()

        robot_feat = np.zeros((R, 9), np.float32)
        robot_valid = np.zeros(R, bool)
        for i, r in enumerate(allies):
            tgt = self._ally_target(i, self.last_actions[i])
            a = self.last_actions[i]
            is_acq, is_def = a < T, T <= a < 2 * T
            d = 0.0 if tgt is None else math.hypot(tgt.x - r.x, tgt.y - r.y) / math.hypot(_FIELD, _FIELD)
            robot_feat[i] = [r.x / _FIELD, r.y / _FIELD, *self._world_vel(r), float(is_acq or is_def),
                             float(is_acq), float(is_def), float(not (is_acq or is_def)), d]
            robot_valid[i] = True

        task_feat = np.zeros((T, 7), np.float32)
        task_valid = np.zeros(T, bool)
        site_pos = np.zeros((T, 2), np.float32)
        for i, (kind, obj) in enumerate(self._sites):
            w = _SITE_WEIGHT[kind if kind != "neutral_tall" else "neutral_tall"]
            sigma, eta = self._timers(i)
            tsc = (self.field.elapsed - self._change_t[i]) / _HORIZON
            task_feat[i] = [obj.x / _FIELD, obj.y / _FIELD, w / _W_MAX, float(credit[i]), sigma, eta, tsc]
            task_valid[i] = True
            site_pos[i] = [obj.x, obj.y]

        adv_feat = np.zeros((K, 6), np.float32)
        adv_valid = np.zeros(K, bool)
        adv_pos = np.zeros((K, 2), np.float32)
        for i, r in enumerate(foes):
            vx, vy = self._world_vel(r)
            adv_feat[i] = [r.x / _FIELD, r.y / _FIELD, vx, vy, math.hypot(vx, vy), 1.0]
            adv_valid[i] = True
            adv_pos[i] = [r.x, r.y]

        robot_pos = np.array([[r.x, r.y] for r in allies] + [[0, 0]] * (R - len(allies)), np.float32)
        obs = {
            "robot_feat": robot_feat, "task_feat": task_feat, "adv_feat": adv_feat,
            "e_rt": self._edges_rt(robot_pos, site_pos, robot_valid, task_valid),
            "e_at": self._edges_at(adv_pos, site_pos, adv_valid, task_valid, foes),
            "e_tt": self._edges_tt(site_pos, task_valid),
            "scalar": self._scalar(credit, task_valid),
            "action_mask": self._action_mask(credit, robot_valid, task_valid),
            "robot_valid": robot_valid, "task_valid": task_valid, "adv_valid": adv_valid,
        }
        return {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}   # batch dim of 1

    def _world_vel(self, r) -> tuple[float, float]:
        c, s = math.cos(r.heading), math.sin(r.heading)
        vx = (r.forward_velocity * c - r.lateral_velocity * s) / _V_MAX
        vy = (r.forward_velocity * s + r.lateral_velocity * c) / _V_MAX
        return float(vx), float(vy)

    def _edges_rt(self, rp, tp, rv, tv):
        d = np.linalg.norm(rp[:, None, :] - tp[None, :, :], axis=-1)          # (R, T)
        travel = d / _V_MAX
        hz = np.clip((_HORIZON - (self.field.elapsed + travel)) / _HORIZON, 0, 1)
        e = np.stack([travel / _HORIZON, hz], -1).astype(np.float32)
        return e * (rv[:, None, None] & tv[None, :, None])

    def _edges_at(self, ap, tp, av, tv, foes):
        d = np.linalg.norm(ap[:, None, :] - tp[None, :, :], axis=-1)          # (K, T)
        tau_rev = float(self.field.contested["toggle_claim_dwell"]) if self.contested else 1.5
        threat = np.clip((d / _V_MAX + tau_rev) / _HORIZON, 0, 1)
        vel = np.zeros((ap.shape[0], 2), np.float32)                          # padded to K
        for i, r in enumerate(foes):
            vx, vy = self._world_vel(r)
            vel[i] = (vx * _V_MAX, vy * _V_MAX)
        to_task = tp[None, :, :] - ap[:, None, :]
        closing = ((vel[:, None, :] * to_task).sum(-1) > 0).astype(np.float32)
        e = np.stack([threat, closing], -1).astype(np.float32)
        return e * (av[:, None, None] & tv[None, :, None])

    def _edges_tt(self, tp, tv):
        d = np.linalg.norm(tp[:, None, :] - tp[None, :, :], axis=-1)
        e = np.exp(-d / (3 * _SERVICE))[..., None].astype(np.float32)
        return e * (tv[:, None, None] & tv[None, :, None])

    def _scalar(self, credit, tv):
        frac = sum(c for c, v in zip(credit, tv) if v) / max(1, tv.sum())
        held = self.field.held_value[self.ally] / (_HORIZON * 100.0) if self.contested else 0.0
        return np.array([self.field.elapsed / _HORIZON, frac, min(held, 1.0), 1.0], np.float32)

    def _action_mask(self, credit, rv, tv):
        A, T = self.action_dim, self.max_tasks
        mask = np.zeros((self.max_robots, A), bool)
        acquire = np.array([tv[i] and not credit[i] if i < len(credit) else False for i in range(T)])
        defend = np.array([tv[i] and credit[i] if i < len(credit) else False for i in range(T)])
        mask[:, 0:T] = acquire[None, :] & rv[:, None]
        mask[:, T:2 * T] = defend[None, :] & rv[:, None]
        mask[:, self.idle_action] = True
        return mask

    # ---------------------------------------------------------------- actions
    def _ally_target(self, rid: int, action: int):
        T = self.max_tasks
        robot = self.field.robots[rid]
        if action >= 2 * T:                                   # IDLE
            return None
        site = action if action < T else action - T
        if site >= self.n_sites:
            return None
        kind, obj = self._sites[site]
        if action < T and kind != "toggle":                  # ACQUIRE a goal: fetch first, then deliver
            if robot.held_pin is None and robot.held_cup is None:
                field_obj = (self.field.nearest_object(robot, "pin", self.ally)
                             or self.field.nearest_object(robot, "cup"))
                if field_obj is not None:
                    return field_obj
                inv = self.field.match_loads[self.ally]       # field empty -> go refill at the loader
                if inv["pin"] > 0 or inv["cup"] > 0:
                    return self.field.nearest_loader(robot)
        return obj

    def _ally_command(self, rid: int, action: int):
        tgt = self._ally_target(rid, action)
        if tgt is None:
            return (0.0, 0.0, 0.0)
        return _body_command(self.field.robots[rid], (tgt.x, tgt.y), self.field.static_obstacles())

    def _ally_interact(self, rid: int, action: int) -> None:
        T = self.max_tasks
        robot = self.field.robots[rid]
        if action >= 2 * T:
            return
        site = action if action < T else action - T
        if site >= self.n_sites:
            return
        kind, obj = self._sites[site]
        if kind == "toggle":
            self.field.claim_toggle(robot, obj)               # no-op flip when contested; dwell handles it
        elif action < T:                                      # ACQUIRE goal
            if robot.held_pin is not None or robot.held_cup is not None:
                self.field.place(robot, obj)
            elif not (self.field.collect(robot, "pin", self.ally) or self.field.collect(robot, "cup")):
                self.field.use_loader(robot)                  # field empty -> reload from match loads (in the load zone)
        # DEFEND: hold position only

    def _update_change_times(self) -> None:
        c = self._credit_vector()
        for i, (was, now) in enumerate(zip(self._prev_c, c)):
            if was != now:
                self._change_t[i] = self.field.elapsed
        self._prev_c = c

    def action_masks(self) -> np.ndarray:
        return self._action_mask(self._credit_vector(), np.array([True] * self.max_robots),
                                 np.array([i < self.n_sites for i in range(self.max_tasks)]))

    def step(self, actions):
        actions = [int(a) for a in actions]
        self.last_actions = actions[: self.max_robots]
        reds = {i: self.opponent.action(self, i) for i in (2, 3)}
        ticks = int(round(DECISION_INTERVAL / DT))
        prev = self.field.raw_score(self.ally)
        for _ in range(ticks):
            if self.field.done:
                break
            commands = {0: self._ally_command(0, actions[0]), 1: self._ally_command(1, actions[1])}
            commands.update({i: self.controller.command(i, a) for i, a in reds.items()})
            self.field.physics_tick(commands, DT)
            self._ally_interact(0, actions[0]); self._ally_interact(1, actions[1])
            for i, a in reds.items():
                self.controller.interact(i, a)
        self._update_change_times()
        reward = (self.field.raw_score(self.ally) - prev) / 50.0
        info = {"reversal_events": list(self.field.reversal_events),
                "blue_score": self.field.score(self.ally), "red_score": self.field.score(self.foe)}
        return self.observation(), reward, self.field.done, info


def retention_by_site_class(policy, env: OverrideGraphEnv, n_decisions: int | None = None,
                            deterministic: bool = True) -> dict[str, float]:
    """Mean predicted retention ``R_hat`` per site class over one Override match (E9).

    The head is never told a site's class, so this measures whether it reads
    adversary geometry rather than memorising a layout. A *trained* TENURE policy
    should give ``R_hat`` ~ 1 on alliance goals (protected, alpha=0), intermediate on
    neutral goals, and low on toggle-dependent value (alpha >= 1). An untrained policy
    gives roughly uniform ~0.5 — this returns the tool; the demonstration needs a
    trained checkpoint.
    """
    from collections import defaultdict
    n_decisions = n_decisions or int(round(MATCH_DURATION / DECISION_INTERVAL))
    total: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    obs = env.observation()
    for _ in range(n_decisions):
        if env.field.done:
            break
        with torch.no_grad():
            r_hat = policy(obs)["r_hat"][0].cpu().numpy()
            action = policy.act(obs, deterministic=deterministic)["action"][0].tolist()
        for i, cls in enumerate(env.site_classes):
            total[cls] += float(r_hat[i])
            count[cls] += 1
        obs, _r, _done, _info = env.step(action)
    return {cls: total[cls] / max(1, count[cls]) for cls in sorted(count)}
