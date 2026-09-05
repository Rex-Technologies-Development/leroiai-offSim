"""Gymnasium environments for the Override 2D prototype."""
from __future__ import annotations
import math
from typing import Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import (
    Action, Alliance, ChassisType, DECISION_INTERVAL, DT, FIELD_HEIGHT,
    FIELD_WIDTH, GOAL_CAPACITY, INTERACTION_RANGE, MATCH_DURATION, NUM_ACTIONS,
    OBJECTS, Phase, STATE_DIM, TEAM_STATE_DIM,
)
from .field import OverrideField, Goal
from .opponent import ScriptedOpponent
from .robot import ESCAPE_TICKS, STUCK_MIN_DIST, wrap_angle


def _nearest(items, robot, predicate=lambda _: True):
    choices = [item for item in items if predicate(item)]
    return min(choices, key=lambda item: math.hypot(item.x-robot.x, item.y-robot.y), default=None)


def _avoid_obstacles(robot, tx, ty, ux, uy, obstacles, look=22.0, gain=2.2):
    """Forward look-ahead 'vision' sensor: steer the unit heading (ux, uy) around any
    static obstacle blocking the path to the target. The destination obstacle itself
    is skipped so the robot can still dock at its target Goal/Loader."""
    steer_x = steer_y = 0.0
    for ox, oy, orad in obstacles:
        if math.hypot(ox - tx, oy - ty) < orad + INTERACTION_RANGE:      # this obstacle IS the destination
            continue
        vx, vy = ox - robot.x, oy - robot.y
        dist = math.hypot(vx, vy)
        clearance = robot.radius + orad + 2.0
        if dist < 1e-6 or dist > look + clearance:
            continue
        along = vx * ux + vy * uy                                        # forward distance (inside the vision cone)
        if along <= 0:
            continue                                                     # obstacle is behind us
        perp = vx * (-uy) + vy * ux                                      # signed lateral offset (left positive)
        if abs(perp) >= clearance:
            continue                                                     # not actually in the path
        sgn = -1.0 if perp >= 0 else 1.0                                 # steer toward the clearer side
        strength = (clearance - abs(perp)) / clearance * max(0.0, 1.0 - dist / (look + clearance))
        steer_x += sgn * (-uy) * strength
        steer_y += sgn * (ux) * strength
    ax, ay = ux + gain * steer_x, uy + gain * steer_y
    norm = math.hypot(ax, ay)
    return (ax / norm, ay / norm) if norm > 1e-6 else (ux, uy)


def _escape_heading(robot, ux, uy, obstacles) -> tuple[float, float]:
    """Stuck-breaker heading: the robot is pinned. Return a unit heading along the TANGENT of the
    nearest BLOCKING obstacle, on whichever side better progresses toward the goal (ux, uy), so the
    robot slides free instead of grinding. If nothing is blocking ahead (pinned on a wall), slide
    toward the field centre."""
    best, best_d = None, 1e18
    for ox, oy, orad in obstacles:
        vx, vy = ox - robot.x, oy - robot.y
        d = math.hypot(vx, vy)
        if d < 1e-6 or (vx * ux + vy * uy) <= 0:                         # skip self / obstacles behind us
            continue
        if d <= robot.radius + orad + 4.0 and d < best_d:                # in contact range, nearest
            best_d, best = d, (vx / d, vy / d)
    if best is None:                                                     # not an obstacle -> off a wall, toward centre
        nx, ny = FIELD_WIDTH / 2 - robot.x, FIELD_HEIGHT / 2 - robot.y
        t = (-uy, ux)
        return t if (t[0] * nx + t[1] * ny) >= 0 else (uy, -ux)
    nx, ny = best
    t1, t2 = (-ny, nx), (ny, -nx)                                        # tangents perpendicular to robot->obstacle
    return t1 if (t1[0] * ux + t1[1] * uy) >= (t2[0] * ux + t2[1] * uy) else t2   # goal-ward side


def _body_command(robot, target: tuple[float, float], obstacles=(), escape: bool = True) -> tuple[float, float, float]:
    dx, dy = target[0]-robot.x, target[1]-robot.y
    distance = math.hypot(dx, dy)
    if distance < 0.5:
        robot._escape_ticks = 0                                          # arrived: cancel any escape
        return (0.0, 0.0, 0.0)
    ux, uy = dx/distance, dy/distance
    if obstacles:
        # engage the stuck-breaker only when FAR from target (a non-target obstacle); a robot pressed
        # against its OWN target goal to score is grinding productively and must be left alone. Scoped
        # to the policy-controlled allies (escape=True): it fixes their long persistent grinds; the
        # scripted opponent's short corner bumps are self-resolving and the committed escape only
        # lengthens them, so the red controller keeps the plain reactive avoider (escape=False).
        if escape and distance > STUCK_MIN_DIST and (robot._escape_ticks > 0 or robot.is_grinding()):
            if robot._escape_ticks <= 0:
                robot._escape_ticks = ESCAPE_TICKS                       # commit to a tangential escape
            robot._escape_ticks -= 1
            ux, uy = _escape_heading(robot, ux, uy, obstacles)
        else:                                                            # normal reactive avoidance
            robot._escape_ticks = 0
            ux, uy = _avoid_obstacles(robot, target[0], target[1], ux, uy, obstacles)
        dx, dy = ux*distance, uy*distance
    desired = math.atan2(dy, dx); error = wrap_angle(desired-robot.heading)
    if robot.chassis is ChassisType.MECANUM:
        c, s = math.cos(robot.heading), math.sin(robot.heading)
        forward = (c*dx+s*dy)/max(distance, 1.0)
        lateral = (-s*dx+c*dy)/max(distance, 1.0)
        return (float(np.clip(forward, -1, 1)), float(np.clip(lateral, -1, 1)), float(np.clip(error/1.5, -0.4, 0.4)))
    return (float(max(0.0, math.cos(error))*min(1.0, distance/10.0)), 0.0, float(np.clip(error/0.7, -1, 1)))


def _legal_removal_goal(field: OverrideField, robot, goal: Goal) -> bool:
    if robot.held_pin is not None or goal.protected_by is robot.alliance.opponent:
        return False
    if not goal.stack or goal.stack[-1].kind != "pin":
        return False
    pin = field.pins[goal.stack[-1].object_id]
    return pin.halves == (robot.alliance.value, robot.alliance.value)


def _removable_goal(field: OverrideField, robot, goal: Goal) -> bool:
    """The generalized REMOVE/DESCORE target: remove your own top Pin, OR descore the
    opponent's top Pin (from any Goal that is not their protected Alliance Goal)."""
    return _legal_removal_goal(field, robot, goal) or field.can_descore(robot, goal)


def _robot_observation(field: OverrideField, robot_id: int) -> np.ndarray:
    robot = field.robots[robot_id]
    own_score, opp_score = field.score(robot.alliance), field.score(robot.alliance.opponent)
    data: list[float] = [
        robot.x/FIELD_WIDTH, robot.y/FIELD_HEIGHT, math.sin(robot.heading), math.cos(robot.heading),
        robot.forward_velocity/30.0, robot.lateral_velocity/30.0, robot.yaw_velocity/math.pi,
        float(robot.held_pin is not None), float(robot.held_cup is not None),
        field.time_remaining/MATCH_DURATION, float(field.phase is Phase.OPENING),
        1.0 if robot.alliance is Alliance.BLUE else -1.0,
    ]
    for other in field.robots:
        if other.robot_id == robot_id: continue
        data += [(other.x-robot.x)/FIELD_WIDTH, (other.y-robot.y)/FIELD_HEIGHT,
                 float(other.alliance is robot.alliance), float(other.held_pin is not None or other.held_cup is not None)]
    for goal in field.goals:
        visible = goal.visible_pin_halves(field.pins)
        owner = 1.0 if robot.alliance.value in visible else -1.0 if robot.alliance.opponent.value in visible else 0.0
        data += [(goal.x-robot.x)/FIELD_WIDTH, (goal.y-robot.y)/FIELD_HEIGHT, owner]
    for toggle in field.toggles:
        owner = 1.0 if toggle.owner is robot.alliance else -1.0 if toggle.owner is robot.alliance.opponent else 0.0
        data += [(toggle.x-robot.x)/FIELD_WIDTH, (toggle.y-robot.y)/FIELD_HEIGHT, owner]
    for kind in ("pin", "cup"):
        obj = field.nearest_object(robot, kind)
        if obj is None: data += [0.0, 0.0, 0.0]
        else:
            dx, dy = float(obj.x)-robot.x, float(obj.y)-robot.y
            data += [dx/FIELD_WIDTH, dy/FIELD_HEIGHT, math.hypot(dx, dy)/math.hypot(FIELD_WIDTH, FIELD_HEIGHT)]
    inv = field.match_loads[robot.alliance]
    data += [own_score/300.0, opp_score/300.0,
             inv["pin"]/float(OBJECTS["match_load_pins_per_alliance"]),
             inv["cup"]/float(OBJECTS["match_load_cups_per_alliance"]),
             field.toggle_count(robot.alliance)/4.0, field.toggle_count(robot.alliance.opponent)/4.0]
    if len(data) > STATE_DIM:
        raise RuntimeError(f"observation contract overflow: {len(data)} > {STATE_DIM}")
    data.extend([0.0]*(STATE_DIM-len(data)))
    return np.asarray(data, dtype=np.float32)


class _OverrideGymBase(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, chassis: str = "tank", render_mode: str | None = None, opponent: str = "mixed"):
        self.chassis = ChassisType(chassis)
        self.render_mode = render_mode
        self.opponent = ScriptedOpponent(opponent)
        self.field = OverrideField(self.chassis)
        self._renderer = None
        self._last_score = (0, 0)

    def _reset_field(self, seed: int | None):
        self.field = OverrideField(self.chassis, seed)
        self._last_score = (self.field.score(Alliance.BLUE), self.field.score(Alliance.RED))

    def _obs(self) -> np.ndarray:
        return np.concatenate([_robot_observation(self.field, 0), _robot_observation(self.field, 1)]).astype(np.float32)

    def _info(self) -> dict:
        return {
            "blue_score": self.field.score(Alliance.BLUE), "red_score": self.field.score(Alliance.RED),
            "phase": self.field.phase.value, "opening_bonus": None if self.field.opening_bonus is None else self.field.opening_bonus.value,
            "blue_awp": self.field.awp[Alliance.BLUE], "red_awp": self.field.awp[Alliance.RED],
            "time_remaining": self.field.time_remaining,
        }

    def _reward(self) -> float:
        now = (self.field.score(Alliance.BLUE), self.field.score(Alliance.RED))
        reward = float((now[0]-self._last_score[0])-(now[1]-self._last_score[1]))/5.0
        if self.field.done:
            reward += 10.0 if now[0] > now[1] else -10.0 if now[0] < now[1] else 0.0
        self._last_score = now
        return reward

    def render(self):
        if self.render_mode is None: return None
        if self._renderer is None:
            from .renderer import PygameRenderer
            self._renderer = PygameRenderer(self.render_mode)
        return self._renderer.draw(self.field)

    def close(self):
        if self._renderer is not None: self._renderer.close(); self._renderer = None


class ObjectiveController:
    """Maps one symbolic objective to deterministic navigation and interaction."""
    def __init__(self, field: OverrideField): self.field = field

    def target(self, robot_id: int, action: Action):
        robot = self.field.robots[robot_id]
        if action is Action.COLLECT_PIN:
            return self.field.nearest_object(robot, "pin", robot.alliance)
        if action is Action.COLLECT_CUP:
            return self.field.nearest_object(robot, "cup")
        if action is Action.USE_LOADER:
            return self.field.nearest_loader(robot)
        if action is Action.CLAIM_TOGGLE:
            return _nearest(self.field.toggles, robot, lambda t: t.owner is not robot.alliance)
        if action in (Action.SCORE_NEAREST_GOAL, Action.SCORE_ALLIANCE_GOAL, Action.SCORE_MIDFIELD_GOAL):
            goals = self.field.goals
            if action is Action.SCORE_ALLIANCE_GOAL: goals = [g for g in goals if g.protected_by is robot.alliance]
            if action is Action.SCORE_MIDFIELD_GOAL: goals = [g for g in goals if g.y == 72.0]
            return _nearest(goals, robot, lambda g: len(g.stack) < GOAL_CAPACITY)
        if action is Action.REMOVE_OWN_PIN:
            return _nearest(self.field.goals, robot, lambda g: _removable_goal(self.field, robot, g))
        if action is Action.DEFEND_MIDFIELD:
            return type("Target", (), {"x": 64.0 if robot_id % 2 == 0 else 80.0, "y": 72.0})()
        return None

    def command(self, robot_id: int, action: Action) -> tuple[float, float, float]:
        target = self.target(robot_id, action)
        if target is None:
            return (0.0, 0.0, 0.0)
        # scripted-navigation path (the red opponent): plain reactive avoider, no ally stuck-breaker
        return _body_command(self.field.robots[robot_id], (target.x, target.y),
                             self.field.static_obstacles(), escape=False)

    def interact(self, robot_id: int, action: Action) -> bool:
        robot = self.field.robots[robot_id]; target = self.target(robot_id, action)
        if action is Action.COLLECT_PIN: return self.field.collect(robot, "pin", robot.alliance)
        if action is Action.COLLECT_CUP: return self.field.collect(robot, "cup")
        if action is Action.USE_LOADER: return self.field.use_loader(robot)
        if action is Action.CLAIM_TOGGLE and target is not None: return self.field.claim_toggle(robot, target)
        if action in (Action.SCORE_NEAREST_GOAL, Action.SCORE_ALLIANCE_GOAL, Action.SCORE_MIDFIELD_GOAL) and isinstance(target, Goal):
            return self.field.place(robot, target)
        if action is Action.REMOVE_OWN_PIN and isinstance(target, Goal):
            if self.field.can_descore(robot, target):        # prefer attacking the enemy
                return self.field.descore_pin(robot, target)
            return self.field.remove_own_pin(robot, target)
        return False


class OverrideContinuousEnv(_OverrideGymBase):
    """Low-level two-allied-robot control.

    Per robot controls are ``forward, lateral, yaw, pin, cup, interact`` in [-1, 1].
    Tank ignores lateral. Positive pin/cup collects; interact places/claims/loads.
    """
    def __init__(self, chassis: str = "tank", render_mode: str | None = None, opponent: str = "mixed"):
        super().__init__(chassis, render_mode, opponent)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2, 6), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(TEAM_STATE_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_field(seed)
        return self._obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2, 6): raise ValueError("continuous action must have shape (2, 6)")
        commands = {i: tuple(float(v) for v in action[i, :3]) for i in range(2)}
        controller = ObjectiveController(self.field)
        opponent_actions = {i: self.opponent.action(self, i) for i in (2, 3)}
        commands.update({i: controller.command(i, act) for i, act in opponent_actions.items()})
        self.field.physics_tick(commands, DT)
        for i in range(2):
            robot, controls = self.field.robots[i], action[i]
            if controls[3] > 0.5: self.field.collect(robot, "pin")
            if controls[4] > 0.5: self.field.collect(robot, "cup")
            if controls[5] > 0.5:
                goal = _nearest(self.field.goals, robot)
                toggle = _nearest(self.field.toggles, robot)
                if goal and math.hypot(goal.x-robot.x, goal.y-robot.y) <= 15: self.field.place(robot, goal)
                elif toggle and math.hypot(toggle.x-robot.x, toggle.y-robot.y) <= 10: self.field.claim_toggle(robot, toggle)
                else: self.field.use_loader(robot)
            elif controls[5] < -0.5:
                goal = _nearest(self.field.goals, robot, lambda g: _removable_goal(self.field, robot, g))
                if goal:
                    self.field.descore_pin(robot, goal) if self.field.can_descore(robot, goal) else self.field.remove_own_pin(robot, goal)
        for i, act in opponent_actions.items(): controller.interact(i, act)
        return self._obs(), self._reward(), self.field.done, False, self._info()


class OverrideStrategyEnv(_OverrideGymBase):
    """Centralized 2-allied-team MaskablePPO strategy environment."""
    def __init__(self, chassis: str = "tank", render_mode: str | None = None, opponent: str = "mixed"):
        super().__init__(chassis, render_mode, opponent)
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(TEAM_STATE_DIM,), dtype=np.float32)
        self.last_actions = np.zeros(2, dtype=np.int64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self._reset_field(seed); self.last_actions[:] = 0
        return self._obs(), self._info()

    def action_masks(self) -> np.ndarray:
        masks: list[bool] = []
        for rid in (0, 1):
            robot = self.field.robots[rid]; inv = self.field.match_loads[robot.alliance]
            values = [False]*NUM_ACTIONS
            values[Action.IDLE] = True
            values[Action.COLLECT_PIN] = robot.held_pin is None and self.field.nearest_object(robot, "pin") is not None
            values[Action.COLLECT_CUP] = robot.held_cup is None and self.field.nearest_object(robot, "cup") is not None
            holding = robot.held_pin is not None or robot.held_cup is not None
            values[Action.SCORE_NEAREST_GOAL] = holding and any(len(g.stack) < GOAL_CAPACITY for g in self.field.goals)
            values[Action.SCORE_ALLIANCE_GOAL] = holding and any(g.protected_by is robot.alliance and len(g.stack) < GOAL_CAPACITY for g in self.field.goals)
            values[Action.SCORE_MIDFIELD_GOAL] = holding and any(g.y == 72.0 and len(g.stack) < GOAL_CAPACITY for g in self.field.goals)
            values[Action.USE_LOADER] = (robot.held_pin is None and inv["pin"] > 0) or (robot.held_cup is None and inv["cup"] > 0)
            values[Action.CLAIM_TOGGLE] = any(t.owner is not robot.alliance for t in self.field.toggles)
            values[Action.DEFEND_MIDFIELD] = True
            values[Action.REMOVE_OWN_PIN] = any(_removable_goal(self.field, robot, g) for g in self.field.goals)
            masks.extend(values)
        return np.asarray(masks, dtype=bool)

    def autoplay_actions(self) -> np.ndarray:
        actions = []
        for rid in (0, 1):
            robot = self.field.robots[rid]
            if robot.held_pin is not None or robot.held_cup is not None: action = Action.SCORE_NEAREST_GOAL
            elif any(t.owner is not Alliance.BLUE for t in self.field.toggles) and int(self.field.elapsed//4+rid)%3 == 0: action = Action.CLAIM_TOGGLE
            elif self.field.nearest_object(robot, "pin") is not None: action = Action.COLLECT_PIN
            elif self.field.nearest_object(robot, "cup") is not None: action = Action.COLLECT_CUP
            elif self.field.match_loads[Alliance.BLUE]["pin"] or self.field.match_loads[Alliance.BLUE]["cup"]: action = Action.USE_LOADER
            else: action = Action.DEFEND_MIDFIELD
            actions.append(int(action))
        return np.asarray(actions, dtype=np.int64)

    def step(self, action: Sequence[int]):
        actions = np.asarray(action, dtype=np.int64)
        if actions.shape != (2,): raise ValueError("strategy action must have shape (2,)")
        masks = self.action_masks().reshape(2, NUM_ACTIONS)
        for i in range(2):
            if actions[i] < 0 or actions[i] >= NUM_ACTIONS or not masks[i, actions[i]]: actions[i] = int(Action.IDLE)
        self.last_actions = actions.copy(); controller = ObjectiveController(self.field)
        opponent_actions = {i: self.opponent.action(self, i) for i in (2, 3)}
        ticks = int(round(DECISION_INTERVAL/DT))
        for _ in range(ticks):
            if self.field.done: break
            all_actions = {0: Action(int(actions[0])), 1: Action(int(actions[1])), **opponent_actions}
            commands = {rid: controller.command(rid, act) for rid, act in all_actions.items()}
            self.field.physics_tick(commands, DT)
            for rid, act in all_actions.items(): controller.interact(rid, act)
            if self.render_mode == "human" and self._renderer is not None:
                self.render()
        info = self._info(); info["actions"] = actions.tolist()
        if self.field.done:
            info["episode_score"] = self.field.score(Alliance.BLUE); info["episode_opp_score"] = self.field.score(Alliance.RED)
        return self._obs(), self._reward(), self.field.done, False, info


def make_training_env(**kwargs) -> OverrideStrategyEnv:
    return OverrideStrategyEnv(**kwargs)
