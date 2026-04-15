"""Gymnasium environment for VEX Push Back 2025-2026 dual-agent sim.

Two-level timing:
  DT = 0.05s     — fine-grained sim tick for smooth movement animation
  DECISION_INTERVAL = 3.0s — how often the RL makes a new decision (60 ticks)

Each env.step() = one RL decision cycle:
  1. Receive actions for both allied robots
  2. Simulate 60 ticks of tank-drive movement toward targets
  3. Check collect/score/descore effects each tick
  4. Return obs + rewards after the full interval

For SB3 compatibility, use SingleAgentWrapper (concatenated obs, flat action).
"""

from __future__ import annotations
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.config import (
    Action, NUM_ACTIONS, STATE_DIM,
    FIELD_W, FIELD_H, MATCH_DURATION, DT, TICKS_PER_DECISION, MAX_CARRY,
    MAX_GAME_OBJECTS, OBJ_FEATURES, OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP,
    HEATMAP_W, HEATMAP_H, MAX_SCORE,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    REGION_A_CENTER, REGION_B_CENTER, DEFEND_ZONE_POS,
    LONG_GOAL_POINTS, CENTER_GOAL_POINTS, COLLECT_RANGE,
    LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX,
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W, ROBOT_W,
)
from sim.field import Field
from sim.robot import Robot
from sim.failure import FailureConfig, FailureInjector
from sim.opponent import get_opponent


# ---------------------------------------------------------------------------
# Goal collision resolution
# ---------------------------------------------------------------------------
# Margin = half robot width + 0.25" ghost boundary
_GOAL_MARGIN = ROBOT_W / 2 + 0.25   # 7.75"

# Long goal X extents (precomputed from config)
_RIGHT_GOAL_X_LO = FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH  # inner face
_RIGHT_GOAL_X_HI = FIELD_W - LONG_GOAL_WALL_GAP                     # outer face
_LEFT_GOAL_X_LO  = LONG_GOAL_WALL_GAP                               # outer face
_LEFT_GOAL_X_HI  = LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH             # inner face

# Center goal X-structure
_CX, _CY  = 72.0, 72.0
_ARM_LEN  = CENTER_GOAL_ARM_LEN    # half-length along arm axis
_ARM_HW   = CENTER_GOAL_ARM_W / 2  # half-width across arm

# Center goal approach points — just outside both arm collision zones,
# within SCORE_RANGE (10") of the actual goal centers.
# A robot at these positions can score without entering the X structure.
_CENTER_MID_APPROACH = np.array([72.0, 87.0])   # above CENTER_MID_GOAL (72, 80.94)
_CENTER_LOW_APPROACH = np.array([72.0, 57.0])   # below CENTER_LOW_GOAL (72, 63.06)


def _resolve_goal_collisions(robot) -> None:
    """Push robot out of goal bounding boxes.

    Long goals use AABB. Center X arms are checked simultaneously so that
    the dual-arm overlap case (robot at center intersection) is handled by
    pushing to the nearest cardinal clear position instead of bouncing
    between both arms indefinitely.
    """
    m = _GOAL_MARGIN

    # --- Long goals (axis-aligned rectangles) — re-read position each time ---
    for gx_lo, gx_hi in ((_RIGHT_GOAL_X_LO, _RIGHT_GOAL_X_HI),
                          (_LEFT_GOAL_X_LO,  _LEFT_GOAL_X_HI)):
        rx, ry = robot.x, robot.y
        ex_lo = gx_lo - m;  ex_hi = gx_hi + m
        ey_lo = LONG_GOAL_Y_MIN - m;  ey_hi = LONG_GOAL_Y_MAX + m
        if ex_lo < rx < ex_hi and ey_lo < ry < ey_hi:
            d_lo = rx - ex_lo;  d_hi = ex_hi - rx
            d_yd = ry - ey_lo;  d_yu = ey_hi - ry
            s = min(d_lo, d_hi, d_yd, d_yu)
            if   s == d_lo: robot.x = ex_lo
            elif s == d_hi: robot.x = ex_hi
            elif s == d_yd: robot.y = ey_lo
            else:           robot.y = ey_hi

    # --- Center goal arms — evaluate both before applying any push ---
    # Minimum distance from X centre to be simultaneously clear of BOTH arms:
    # clear_dist = (ARM_HW + margin) / sin(45°) = (ARM_HW + m) * √2
    clear_dist = (_ARM_HW + m) * math.sqrt(2)

    rx, ry = robot.x, robot.y
    arm_data = []   # (angle, along, perp) for each overlapping arm
    for angle in (math.pi / 4, -math.pi / 4):
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = rx - _CX, ry - _CY
        along = dx * ca  + dy * sa
        perp  = dx * -sa + dy * ca
        if abs(along) < _ARM_LEN + m and abs(perp) < _ARM_HW + m:
            arm_data.append((angle, along, perp))

    if not arm_data:
        return

    if len(arm_data) == 2:
        # Robot is inside BOTH arms (centre intersection).
        # Push to the nearest cardinal clear position.
        exits = [
            (abs(ry - (_CY + clear_dist)), _CX,              _CY + clear_dist),
            (abs(ry - (_CY - clear_dist)), _CX,              _CY - clear_dist),
            (abs(rx - (_CX + clear_dist)), _CX + clear_dist, _CY),
            (abs(rx - (_CX - clear_dist)), _CX - clear_dist, _CY),
        ]
        _, nx, ny = min(exits, key=lambda e: e[0])
        robot.x, robot.y = nx, ny
    else:
        # Single arm overlap — push perpendicularly (shortest exit).
        angle, along, perp = arm_data[0]
        ca, sa = math.cos(angle), math.sin(angle)
        exit_perp  = _ARM_HW + m - abs(perp)
        exit_along = _ARM_LEN + m - abs(along)
        if exit_perp <= exit_along:
            sign     = 1 if perp >= 0 else -1
            new_perp = sign * (_ARM_HW + m)
            robot.x  = _CX + along * ca + new_perp * (-sa)
            robot.y  = _CY + along * sa + new_perp * ca
        else:
            sign      = 1 if along >= 0 else -1
            new_along = sign * (_ARM_LEN + m)
            robot.x   = _CX + new_along * ca + perp * (-sa)
            robot.y   = _CY + new_along * sa + perp * ca


# ---------------------------------------------------------------------------
# Action → target mappings
# ---------------------------------------------------------------------------
def _action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    """Convert allied RL action to a moveToPoint target."""
    if action == Action.COLLECT_NEAREST_BALL:
        return field.nearest_on_field_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        return OUR_LONG_GOAL.copy()
    elif action == Action.SCORE_CENTER_GOAL:
        # Navigate to the approach point outside the X structure, within scoring range.
        d_mid = np.linalg.norm(robot.position - _CENTER_MID_APPROACH)
        d_low = np.linalg.norm(robot.position - _CENTER_LOW_APPROACH)
        return _CENTER_MID_APPROACH.copy() if d_mid < d_low else _CENTER_LOW_APPROACH.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OPP_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        d_mid = np.linalg.norm(robot.position - _CENTER_MID_APPROACH)
        d_low = np.linalg.norm(robot.position - _CENTER_LOW_APPROACH)
        return _CENTER_MID_APPROACH.copy() if d_mid < d_low else _CENTER_LOW_APPROACH.copy()
    elif action == Action.DEFEND_ZONE:
        return DEFEND_ZONE_POS.copy()
    elif action == Action.MOVE_TO_REGION_A:
        return REGION_A_CENTER.copy()
    elif action == Action.MOVE_TO_REGION_B:
        return REGION_B_CENTER.copy()
    elif action == Action.IDLE:
        return None
    return None


def _opp_action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    """Convert opponent action to a moveToPoint target (mirror of allied)."""
    if action == Action.COLLECT_NEAREST_BALL:
        return field.nearest_on_field_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        return OPP_LONG_GOAL.copy()
    elif action == Action.SCORE_CENTER_GOAL:
        d_mid = np.linalg.norm(robot.position - _CENTER_MID_APPROACH)
        d_low = np.linalg.norm(robot.position - _CENTER_LOW_APPROACH)
        return _CENTER_MID_APPROACH.copy() if d_mid < d_low else _CENTER_LOW_APPROACH.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OUR_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        d_mid = np.linalg.norm(robot.position - _CENTER_MID_APPROACH)
        d_low = np.linalg.norm(robot.position - _CENTER_LOW_APPROACH)
        return _CENTER_MID_APPROACH.copy() if d_mid < d_low else _CENTER_LOW_APPROACH.copy()
    elif action == Action.DEFEND_ZONE:
        return np.array([24.00, 72.00])   # opp defend zone (left side)
    elif action == Action.MOVE_TO_REGION_A:
        return np.array([36.00, 108.00])
    elif action == Action.MOVE_TO_REGION_B:
        return np.array([108.00, 36.00])
    return None


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class VexAIEnv(gym.Env):
    """Two-agent VEX Push Back field environment.

    Each step() = one RL decision interval (3 seconds / 60 ticks).
    Both allied robots act; opponents are rule-based.
    For SB3, wrap with SingleAgentWrapper.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        opponent_type: str = "mixed",
        failure_config: FailureConfig | None = None,
        num_allies: int = 1,
    ):
        super().__init__()
        self.render_mode    = render_mode
        self.failure_config = failure_config or FailureConfig()
        self.num_allies     = max(1, min(num_allies, 2))  # 1 or 2
        self.num_opponents  = 0                            # no opponents by default

        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])
        self.observation_space = spaces.Dict({
            "robot_0": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            "robot_1": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        })

        self.opponent_policy = get_opponent(opponent_type)
        self.field           = Field()
        self._renderer       = None

        self.score_events          = np.zeros(2)
        self.descore_events        = np.zeros(2)
        self.collected_this_step   = np.zeros(2, dtype=np.int32)
        self.current_actions       = np.array([Action.IDLE, Action.IDLE], dtype=np.int32)
        self.expected_state_delta  = np.zeros(2)
        self.prev_predicted_score  = np.zeros(2)
        self.done = False

        # Decision progress — updated each tick so renderer can show a progress bar
        # 0 = idle/waiting, 1..TICKS_PER_DECISION = executing, -1 = just finished
        self.decision_tick: int = 0
        self.executing: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.field.reset(self.rng)
        self.field.time_remaining = MATCH_DURATION

        self.failure_injector = FailureInjector(self.failure_config, self.rng)
        self.failure_injector.reset()

        self.score_events[:]         = 0
        self.descore_events[:]       = 0
        self.collected_this_step[:]  = 0
        self.current_actions[:]      = Action.IDLE
        self.expected_state_delta[:] = 0
        self.prev_predicted_score[:] = 0
        self.done          = False
        self.decision_tick = 0
        self.executing     = False
        self.opp_targets: list[np.ndarray | None] = [None, None]

        return self._get_obs(), {}

    def setup_reset(self):
        """Partial reset after setup mode: keeps robot/ball positions but clears scores and physics."""
        self.rng = np.random.default_rng(None)
        self.field.time_remaining = MATCH_DURATION
        self.field.my_score       = 0
        self.field.opponent_score = 0

        # Reset robot physics state (NOT position/heading)
        for robot in self.field.allies + self.field.opponents:
            robot.balls_held = 0
            robot.held_object_ids.clear()
            robot.actions_attempted = 0
            robot.actions_succeeded = 0
            robot.target = None
            robot.moving = False

        # Reset ball physics; un-hold any held balls
        for obj in self.field.objects:
            if obj.status == OBJ_HELD:
                obj.status = OBJ_ON_FIELD
            obj.vx = 0.0
            obj.vy = 0.0

        self.failure_injector = FailureInjector(self.failure_config, self.rng)
        self.failure_injector.reset()

        self.score_events[:]         = 0
        self.descore_events[:]       = 0
        self.collected_this_step[:]  = 0
        self.current_actions[:]      = Action.IDLE
        self.expected_state_delta[:] = 0
        self.prev_predicted_score[:] = 0
        self.done          = False
        self.decision_tick = 0
        self.executing     = False
        self.opp_targets   = [None, None]

        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        """One RL decision cycle = TICKS_PER_DECISION sim ticks."""
        a0, a1 = int(actions[0]), int(actions[1])
        self.current_actions = np.array([a0, a1], dtype=np.int32)

        self.score_events[:]        = 0
        self.descore_events[:]      = 0
        self.collected_this_step[:] = 0
        self.decision_tick = 0
        self.executing     = True

        fi = self.failure_injector
        if self.num_allies < 2 or fi.teammate_offline or fi.should_teammate_fail():
            a1 = Action.IDLE

        ally_targets = [
            _action_to_target(Action(a0), self.field, self.field.allies[0]),
            _action_to_target(Action(a1), self.field, self.field.allies[1]),
        ]

        for oi in range(self.num_opponents):
            opp = self.field.opponents[oi]
            opp_state = {
                "position":       opp.position,
                "balls_held":     opp.balls_held,
                "our_score":      self.field.my_score,
                "opp_score":      self.field.opponent_score,
                "us_scored_count": len(self.field.scored_by_us_indices()),
            }
            opp_action = self.opponent_policy(opp_state, self.rng)
            self.opp_targets[oi] = _opp_action_to_target(opp_action, self.field, opp)

        for idx in range(2):
            if fi.should_steal_object():
                self._steal_nearest(idx)

        for tick in range(TICKS_PER_DECISION):
            fi.on_tick(num_robots=2)

            # Save positions before movement for push calculation
            ally_prev = [self.field.allies[i].position.copy()     for i in range(self.num_allies)]
            opp_prev  = [self.field.opponents[i].position.copy()  for i in range(self.num_opponents)]

            for idx in range(self.num_allies):
                if fi.is_stuck(idx):
                    continue
                target = ally_targets[idx]
                if target is not None:
                    self.field.allies[idx].move_toward_point(target)
                _resolve_goal_collisions(self.field.allies[idx])

            for oi in range(self.num_opponents):
                if self.opp_targets[oi] is not None:
                    self.field.opponents[oi].move_toward_point(self.opp_targets[oi])
                _resolve_goal_collisions(self.field.opponents[oi])

            # Apply robot→ball pushes
            for idx in range(self.num_allies):
                self.field.apply_robot_push(self.field.allies[idx], ally_prev[idx])
            for oi in range(self.num_opponents):
                self.field.apply_robot_push(self.field.opponents[oi], opp_prev[oi])

            # Advance ball rolling physics
            self.field.physics_tick(DT)

            self._check_ally_effects(0, Action(a0))
            if self.num_allies >= 2:
                self._check_ally_effects(1, Action(a1))
            self._check_opp_effects()

            self.field.time_remaining -= DT
            self.decision_tick += 1
            if self.field.time_remaining <= 0:
                break

            if self.render_mode == "human" and self._renderer is not None:
                self._renderer.draw(self)

        self.executing = False
        self.done = self.field.time_remaining <= 0

        for idx in range(2):
            self.field.allies[idx].actions_attempted += 1

        for idx in range(2):
            actual = float(self.field.my_score)
            self.expected_state_delta[idx] = actual - self.prev_predicted_score[idx]
            self.prev_predicted_score[idx] = actual

        from training.reward import compute_reward
        r0 = compute_reward(self, robot_id=0)
        r1 = compute_reward(self, robot_id=1)

        return self._get_obs(), (r0, r1), self.done, False, {}

    # ------------------------------------------------------------------
    # Effect checks (called each tick)
    # ------------------------------------------------------------------
    def _check_ally_effects(self, idx: int, action: Action):
        robot = self.field.allies[idx]

        if action == Action.COLLECT_NEAREST_BALL:
            if self.field.try_collect(robot):
                self.collected_this_step[idx] += 1
                robot.actions_succeeded += 1

        elif action == Action.SCORE_LONG_GOAL:
            # Heading constraint: must face up or down (within 30° of ±π/2)
            if abs(math.sin(robot.heading)) > 0.866:
                pts = self.field.try_score(robot, OUR_LONG_GOAL, LONG_GOAL_POINTS)
                if pts > 0:
                    self.score_events[idx] += pts
                    robot.actions_succeeded += 1

        elif action == Action.SCORE_CENTER_GOAL:
            # Heading constraint: must face a diagonal direction (within 30° of 45°/135°/225°/315°)
            if abs(math.sin(2 * robot.heading)) > 0.5:
                pts = self.field.try_score(robot, CENTER_MID_GOAL, CENTER_GOAL_POINTS)
                if pts == 0:
                    pts = self.field.try_score(robot, CENTER_LOW_GOAL, CENTER_GOAL_POINTS)
                if pts > 0:
                    self.score_events[idx] += pts
                    robot.actions_succeeded += 1

        elif action == Action.DESCORE_OPP_LONG:
            pts = self.field.try_descore(robot, OPP_LONG_GOAL, self.rng)
            if pts > 0:
                self.descore_events[idx] += pts
                robot.actions_succeeded += 1

        elif action == Action.DESCORE_CENTER:
            pts = self.field.try_descore(robot, CENTER_MID_GOAL, self.rng)
            if pts == 0:
                pts = self.field.try_descore(robot, CENTER_LOW_GOAL, self.rng)
            if pts > 0:
                self.descore_events[idx] += pts
                robot.actions_succeeded += 1

    def _check_opp_effects(self):
        for oi in range(self.num_opponents):
            opp = self.field.opponents[oi]
            self.field.opp_try_collect(opp, self.rng)
            self.field.opp_try_score(opp, OPP_LONG_GOAL,   LONG_GOAL_POINTS)
            self.field.opp_try_score(opp, CENTER_MID_GOAL, CENTER_GOAL_POINTS)
            self.field.opp_try_score(opp, CENTER_LOW_GOAL, CENTER_GOAL_POINTS)
            self.field.opp_try_descore(opp, OUR_LONG_GOAL,   self.rng)
            self.field.opp_try_descore(opp, CENTER_MID_GOAL, self.rng)

    def _steal_nearest(self, robot_idx: int):
        robot  = self.field.allies[robot_idx]
        target = self.field.nearest_on_field_target(robot.position)
        if target is not None:
            for obj in self.field.objects:
                if obj.status == OBJ_ON_FIELD and np.linalg.norm(obj.position - target) < 1.0:
                    obj.status = OBJ_SCORED_OPP
                    self.field.opponent_score += CENTER_GOAL_POINTS
                    break

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        return {"robot_0": self._build_obs(0), "robot_1": self._build_obs(1)}

    def _build_obs(self, role_id: int) -> np.ndarray:
        robot = self.field.allies[role_id]
        pos   = robot.position

        balls_nearby = self.field.nearby_ball_count(pos)

        # Flatten game objects (up to MAX_GAME_OBJECTS)
        obj_flat = np.zeros(MAX_GAME_OBJECTS * OBJ_FEATURES, dtype=np.float32)
        n_obs    = min(len(self.field.objects), MAX_GAME_OBJECTS)
        for i in range(n_obs):
            obj = self.field.objects[i]
            base = i * OBJ_FEATURES
            obj_flat[base]     = obj.x / FIELD_W
            obj_flat[base + 1] = obj.y / FIELD_H
            obj_flat[base + 2] = float(obj.status) / 4.0   # normalise 0-4
            obj_flat[base + 3] = float(obj.color)           # 0=red, 1=blue

        heatmap = self.field.get_heatmap()

        return np.concatenate([
            np.array([
                float(role_id),
                self.field.time_remaining / MATCH_DURATION,
                float(self.field.my_score)       / MAX_SCORE,
                float(self.field.opponent_score) / MAX_SCORE,
                pos[0] / FIELD_W,
                pos[1] / FIELD_H,
                robot.heading / (2 * np.pi),
                float(robot.balls_held) / MAX_CARRY,
                float(balls_nearby)     / MAX_GAME_OBJECTS,
            ], dtype=np.float32),
            obj_flat,
            heatmap.flatten(),
            np.array([
                self.expected_state_delta[role_id] / 10.0,
                robot.success_ratio(),
            ], dtype=np.float32),
        ])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from sim.renderer import PygameRenderer
                self._renderer = PygameRenderer(self)
            self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ---------------------------------------------------------------------------
# Single-agent wrapper for Stable-Baselines3
# ---------------------------------------------------------------------------
class SingleAgentWrapper(gym.Env):
    """Wraps VexAIEnv for SB3: concatenated obs, single flat discrete action."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.env         = VexAIEnv(render_mode=render_mode, **kwargs)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(STATE_DIM * 2,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS * NUM_ACTIONS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate([obs["robot_0"], obs["robot_1"]]), info

    def step(self, action: int):
        a0  = action // NUM_ACTIONS
        a1  = action % NUM_ACTIONS
        obs, rewards, done, truncated, info = self.env.step(np.array([a0, a1]))
        return np.concatenate([obs["robot_0"], obs["robot_1"]]), rewards[0] + rewards[1], done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
