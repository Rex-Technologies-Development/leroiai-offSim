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
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W, ROBOT_W, TURN_RATE, MAX_SPEED,
    BALL_BLUE, BALL_RED,
)
from sim.field import Field
from sim.robot import Robot
from sim.failure import FailureConfig, FailureInjector
from sim.opponent import get_opponent


# ---------------------------------------------------------------------------
# Goal collision resolution
# ---------------------------------------------------------------------------
# Margin for long goals = half robot width + 0.25" ghost boundary
_GOAL_MARGIN = ROBOT_W / 2 + 0.25   # 7.75"
# Wider margin for center X arms — gives a visually clear gap between robot body and arm body
_CENTER_GOAL_MARGIN = ROBOT_W / 2 + 2.5   # 10.0"

# Long goal X extents (precomputed from config)
_RIGHT_GOAL_X_LO = FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH  # inner face
_RIGHT_GOAL_X_HI = FIELD_W - LONG_GOAL_WALL_GAP                     # outer face
_LEFT_GOAL_X_LO  = LONG_GOAL_WALL_GAP                               # outer face
_LEFT_GOAL_X_HI  = LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH             # inner face

# Center goal X-structure
_CX, _CY  = 72.0, 72.0
_ARM_LEN  = CENTER_GOAL_ARM_LEN    # half-length along arm axis
_ARM_HW   = CENTER_GOAL_ARM_W / 2  # half-width across arm

# ---------------------------------------------------------------------------
# Scoring geometry — robot backs into each goal end so the BACK feeds the goal.
# ---------------------------------------------------------------------------
# Distance from goal opening to robot center when scoring (half robot + clearance)
_SCORE_APPROACH_GAP = ROBOT_W / 2 + 2.0   # ≈ 9.5"

# Long goal scoring positions (top end and bottom end of each vertical goal).
# Robot center sits beyond the open end; back faces into the goal.
_RIGHT_GOAL_CX = (_RIGHT_GOAL_X_LO + _RIGHT_GOAL_X_HI) / 2.0   # ≈ 120.4
_LEFT_GOAL_CX  = (_LEFT_GOAL_X_LO  + _LEFT_GOAL_X_HI)  / 2.0   # ≈ 23.6

_RIGHT_GOAL_TOP_POS = np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MAX + _SCORE_APPROACH_GAP])
_RIGHT_GOAL_BOT_POS = np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MIN - _SCORE_APPROACH_GAP])
_LEFT_GOAL_TOP_POS  = np.array([_LEFT_GOAL_CX,  LONG_GOAL_Y_MAX + _SCORE_APPROACH_GAP])
_LEFT_GOAL_BOT_POS  = np.array([_LEFT_GOAL_CX,  LONG_GOAL_Y_MIN - _SCORE_APPROACH_GAP])

# Required headings (sim radians, 0=E, CCW). Robot faces AWAY from goal so back enters.
# At TOP end: robot above goal → faces NORTH (+y) → sim heading +π/2.
# At BOTTOM end: robot below goal → faces SOUTH (-y) → sim heading -π/2.
_HDG_NORTH = math.pi / 2.0
_HDG_SOUTH = -math.pi / 2.0

# Center X-structure scoring — 4 arm tips, robot backs into each tip along the arm axis.
# Arm half-length _ARM_LEN ≈ 23.18, tips at distance _ARM_LEN from (72, 72) at ±45°.
_X_TIP_DIST = _ARM_LEN + _SCORE_APPROACH_GAP
_X_DIAG = _X_TIP_DIST / math.sqrt(2.0)   # decompose along axes

# Mid goal (upper X half) tips: NE and NW
_MID_NE_POS = np.array([_CX + _X_DIAG, _CY + _X_DIAG])
_MID_NW_POS = np.array([_CX - _X_DIAG, _CY + _X_DIAG])
# Low goal (lower X half) tips: SE and SW
_LOW_SE_POS = np.array([_CX + _X_DIAG, _CY - _X_DIAG])
_LOW_SW_POS = np.array([_CX - _X_DIAG, _CY - _X_DIAG])

# Required headings for each X tip (robot's back points toward the center).
_HDG_NE =  math.pi / 4.0     # face NE (back faces SW into NE arm)
_HDG_NW =  3.0 * math.pi / 4.0
_HDG_SE = -math.pi / 4.0
_HDG_SW = -3.0 * math.pi / 4.0

_SCORE_HDG_TOL      = math.pi / 6.0   # 30° tolerance
_SCORE_ARRIVAL_DIST = 6.0
_SCORE_INTERVAL     = 1.5


def _nearest_long_goal_target(robot):
    """Pick nearest long-goal scoring end for this robot.

    Returns (approach_pos, goal_end_pos, required_heading, goal_name).
    """
    options = [
        (_RIGHT_GOAL_TOP_POS, np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MAX]), _HDG_NORTH, "our_long"),
        (_RIGHT_GOAL_BOT_POS, np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MIN]), _HDG_SOUTH, "our_long"),
        (_LEFT_GOAL_TOP_POS,  np.array([_LEFT_GOAL_CX,  LONG_GOAL_Y_MAX]), _HDG_NORTH, "opp_long"),
        (_LEFT_GOAL_BOT_POS,  np.array([_LEFT_GOAL_CX,  LONG_GOAL_Y_MIN]), _HDG_SOUTH, "opp_long"),
    ]
    return min(options, key=lambda o: float(np.linalg.norm(robot.position - o[0])))


_ARM_TIP_X = _ARM_LEN / math.sqrt(2.0)
_TIP_NE = np.array([_CX + _ARM_TIP_X, _CY + _ARM_TIP_X])
_TIP_NW = np.array([_CX - _ARM_TIP_X, _CY + _ARM_TIP_X])
_TIP_SE = np.array([_CX + _ARM_TIP_X, _CY - _ARM_TIP_X])
_TIP_SW = np.array([_CX - _ARM_TIP_X, _CY - _ARM_TIP_X])


def _nearest_center_tip(robot, lower: bool):
    """Pick nearest center goal tip.

    MID goal = NE–SW bar (+45° diagonal): NE tip and SW tip.
    LOW goal = NW–SE bar (-45° diagonal): NW tip and SE tip.
    lower=True → prefer LOW goal; lower=False → prefer MID goal.
    Returns (approach_pos, tip_pos, required_heading, goal_name).
    """
    if lower:   # LOW goal = NW-SE bar
        options = [
            (_MID_NW_POS, _TIP_NW, _HDG_NW, "center_low"),
            (_LOW_SE_POS, _TIP_SE, _HDG_SE, "center_low"),
        ]
    else:       # MID goal = NE-SW bar
        options = [
            (_MID_NE_POS, _TIP_NE, _HDG_NE, "center_mid"),
            (_LOW_SW_POS, _TIP_SW, _HDG_SW, "center_mid"),
        ]
    return min(options, key=lambda o: float(np.linalg.norm(robot.position - o[0])))


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Navigation waypoints (path avoidance around center X)
# ---------------------------------------------------------------------------
# These approach-corridor points sit in the clear field quadrants, safely
# outside all X-arm and long-goal collision zones.  Verified:
#   (95,  40): below X arms (arm tips at ~y=55.6),  right side
#   (95, 104): above X arms (arm tips at ~y=88.4),  right side
#   (49,  40): below X arms, left side
#   (49, 104): above X arms, left side
# Generic center corridors kept for COLLECT-route detours.
_NAV_ABOVE_X     = np.array([ 72.0, 103.0])
_NAV_BELOW_X     = np.array([ 72.0,  41.0])
_NAV_RIGHT_LOW   = np.array([ 95.0,  40.0])  # right-goal approach, below X
_NAV_RIGHT_HIGH  = np.array([ 95.0, 104.0])  # right-goal approach, above X
_NAV_LEFT_LOW    = np.array([ 49.0,  40.0])  # left-goal approach,  below X
_NAV_LEFT_HIGH   = np.array([ 49.0, 104.0])  # left-goal approach,  above X


def _build_nav_waypoints(start: np.ndarray, final: np.ndarray) -> list[np.ndarray]:
    """Build waypoint list from `start` to `final` avoiding all goal obstacles.

    Strategy:
      1. If LOS to final is clear, go direct.
      2. Otherwise pick an approach corridor in the appropriate field quadrant.
      3. If corridor→final is still blocked, add the central go-around point.

    Corridor points (90% guaranteed clear of X arms AND long-goal bodies for a
    15"-wide robot):
      • (95, 40)  / (95, 104)  — right side, below / above X
      • (49, 40)  / (49, 104)  — left side,  below / above X
    """
    from sim.route_planner import _los_blocked, _NAV_MARGIN

    # Direct path?
    if not _los_blocked(start[0], start[1], final[0], final[1], margin=_NAV_MARGIN):
        return [final]

    use_low = (start[1] + final[1]) / 2.0 <= 72.0
    if final[0] > 72.0:
        corridor = _NAV_RIGHT_LOW.copy() if use_low else _NAV_RIGHT_HIGH.copy()
    else:
        corridor = _NAV_LEFT_LOW.copy()  if use_low else _NAV_LEFT_HIGH.copy()

    # If approaching a long-goal end, swap corridor side based on final.y
    if abs(final[0] - _RIGHT_GOAL_CX) < 6.0 or abs(final[0] - _LEFT_GOAL_CX) < 6.0:
        # We want the corridor on the same side as the final's Y
        use_low = final[1] < 72.0
        if final[0] > 72.0:
            corridor = _NAV_RIGHT_LOW.copy() if use_low else _NAV_RIGHT_HIGH.copy()
        else:
            corridor = _NAV_LEFT_LOW.copy()  if use_low else _NAV_LEFT_HIGH.copy()

    # Verify corridor→final is clear; if not, add center go-around
    if _los_blocked(corridor[0], corridor[1], final[0], final[1], margin=_NAV_MARGIN):
        center = _NAV_BELOW_X.copy() if start[1] <= 72.0 else _NAV_ABOVE_X.copy()
        return [center, corridor, final]
    # Verify start→corridor is clear too
    if _los_blocked(start[0], start[1], corridor[0], corridor[1], margin=_NAV_MARGIN):
        center = _NAV_BELOW_X.copy() if start[1] <= 72.0 else _NAV_ABOVE_X.copy()
        return [center, corridor, final]
    return [corridor, final]


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

    # --- Center goal arms — use wider margin for visible clearance ---
    mc = _CENTER_GOAL_MARGIN
    # Minimum distance from X centre to be simultaneously clear of BOTH arms:
    # clear_dist = (ARM_HW + mc) / sin(45°) = (ARM_HW + mc) * √2
    clear_dist = (_ARM_HW + mc) * math.sqrt(2)

    rx, ry = robot.x, robot.y
    arm_data = []   # (angle, along, perp) for each overlapping arm
    for angle in (math.pi / 4, -math.pi / 4):
        ca, sa = math.cos(angle), math.sin(angle)
        dx, dy = rx - _CX, ry - _CY
        along = dx * ca  + dy * sa
        perp  = dx * -sa + dy * ca
        if abs(along) < _ARM_LEN + mc and abs(perp) < _ARM_HW + mc:
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
        exit_perp  = _ARM_HW + mc - abs(perp)
        exit_along = _ARM_LEN + mc - abs(along)
        if exit_perp <= exit_along:
            sign     = 1 if perp >= 0 else -1
            new_perp = sign * (_ARM_HW + mc)
            robot.x  = _CX + along * ca + new_perp * (-sa)
            robot.y  = _CY + along * sa + new_perp * ca
        else:
            sign      = 1 if along >= 0 else -1
            new_along = sign * (_ARM_LEN + mc)
            robot.x   = _CX + new_along * ca + perp * (-sa)
            robot.y   = _CY + new_along * sa + perp * ca


# ---------------------------------------------------------------------------
# Action → target mappings
# ---------------------------------------------------------------------------
def _action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    """Convert allied RL action to a moveToPoint target."""
    if action == Action.COLLECT_NEAREST_BALL:
        # Priority 1: route planner with full vision cone + LOS + strategic scoring
        from sim.route_planner import compute_collection_route
        route = compute_collection_route(
            robot.position, field,
            already_held=robot.balls_held,
            max_volley=1,
            robot=robot,
        )
        if route:
            return route[0][1].copy()   # centroid of best cluster
        # Priority 2: nearest ball with LOS check (ball not behind a goal)
        return field.nearest_navigable_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        approach, _, _, _ = _nearest_long_goal_target(robot)
        return approach.copy()
    elif action == Action.SCORE_CENTER_GOAL:
        # MID = NE+SW tips; LOW = NW+SE tips — pick globally nearest
        d_mid = min(np.linalg.norm(robot.position - _MID_NE_POS),
                    np.linalg.norm(robot.position - _LOW_SW_POS))
        d_low = min(np.linalg.norm(robot.position - _MID_NW_POS),
                    np.linalg.norm(robot.position - _LOW_SE_POS))
        lower = d_low < d_mid
        approach, _, _, _ = _nearest_center_tip(robot, lower=lower)
        return approach.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OPP_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        approach, _, _, _ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
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
        approach, _, _ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OUR_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        approach, _, _ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
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
        self.decision_tick: int = 0
        self.executing: bool = False

        # Scoring animations — list of dicts the renderer consumes and removes.
        # Each: {x0,y0, x1,y1, color, elapsed, duration}
        self.score_animations: list[dict] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.field.reset(self.rng)
        self.field.time_remaining = MATCH_DURATION   # display only — not decremented

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
            robot.target      = None
            robot.moving      = False
            robot.score_timer = 0.0

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
        self.score_animations.clear()

        return self._get_obs(), {}

    def manual_tick(self, renderer) -> None:
        """One physics tick driven by WASD input from the renderer.

        Called from the demo loop instead of step() when renderer.wasd_mode is True.
        Handles movement, goal collision, intake, and scoring for the selected robot.
        Does NOT render — the caller's env.render() already drew this frame.
        """
        import math

        # Tuned for ~60fps feel: 65% speed, 30% turn rate so driving is smooth
        _WASD_SPEED = MAX_SPEED * 0.65
        _WASD_TURN  = TURN_RATE * 0.30

        ridx  = renderer.wasd_robot_idx
        robot = self.field.allies[ridx]
        prev_pos = robot.position.copy()

        # --- Movement ---
        fwd  = renderer.wasd_fwd    # -1 / 0 / +1
        turn = renderer.wasd_turn   # -1=A / 0 / +1=D

        if turn != 0:
            robot.heading = _wrap_angle(robot.heading + turn * _WASD_TURN * DT)

        if fwd != 0:
            fwd_vec = np.array([math.cos(robot.heading), math.sin(robot.heading)])
            robot.x = round(robot.x + fwd_vec[0] * fwd * _WASD_SPEED * DT, 2)
            robot.y = round(robot.y + fwd_vec[1] * fwd * _WASD_SPEED * DT, 2)
            robot.moving = True
        else:
            robot.speed  = 0.0
            robot.moving = False

        robot._clamp_to_field()
        _resolve_goal_collisions(robot)

        # --- Intake (collect balls when near) ---
        robot.intake_active = renderer.wasd_intake_on and robot.balls_held < MAX_CARRY
        if robot.intake_active:
            self.field.try_collect(robot)

        # --- Scoring (hold F) — eject balls if robot leaves position while F held ---
        if renderer.wasd_score_on and robot.balls_held > 0:
            robot.intake_active = True   # show intake animation while scoring
            timer_before = robot.score_timer
            balls_before = robot.balls_held
            self._do_manual_scoring(robot)
            # If timer reset to 0 without a ball leaving, the robot moved away
            left_position = (timer_before > 0.0 and robot.score_timer == 0.0
                             and robot.balls_held == balls_before)
            if left_position:
                self._eject_held_balls(robot)
        elif robot.score_timer > 0.0:
            # F was released — timer stays, balls stay, just stop the clock
            robot.score_timer = 0.0

        # --- Ball push & rolling physics ---
        self.field.apply_robot_push(robot, prev_pos)
        self.field.physics_tick(DT)

    def _eject_held_balls(self, robot) -> None:
        """Scatter all balls the robot is holding out from its back face.

        Called when the robot moves away from a goal while F is still held.
        Each ball gets an independent random scatter so they must be
        re-collected individually.
        """
        import math
        bx = robot.x - math.cos(robot.heading) * (ROBOT_W / 2)
        by = robot.y - math.sin(robot.heading) * (ROBOT_W / 2)

        rng = getattr(self, "rng", None)
        for obj_idx in robot.held_object_ids:
            obj = self.field.objects[obj_idx]
            obj.status = OBJ_ON_FIELD
            scatter = 8.0
            ox = bx + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            oy = by + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            obj.x = round(float(np.clip(ox, ROBOT_W / 2 + 1, FIELD_W - ROBOT_W / 2 - 1)), 2)
            obj.y = round(float(np.clip(oy, ROBOT_W / 2 + 1, FIELD_H - ROBOT_W / 2 - 1)), 2)
            # Velocity: primarily outward from the back face, with random spread
            speed = 12.0 + (float(rng.uniform(0, 8)) if rng else 0.0)
            obj.vx = float(-math.cos(robot.heading) * speed)  # -heading = back direction
            obj.vy = float(-math.sin(robot.heading) * speed)
            if rng:
                obj.vx += float(rng.uniform(-4, 4))
                obj.vy += float(rng.uniform(-4, 4))

        robot.held_object_ids.clear()
        robot.balls_held = 0
        robot.score_timer = 0.0

    def _do_manual_scoring(self, robot) -> None:
        """Back-face proximity scoring for WASD mode.

        Checks if the robot's BACK FACE CENTER is near any goal scoring surface
        (long goal field-side face, long goal ends, or center X arm tips) and the
        robot heading points roughly away from that surface (back INTO goal).

        Balls score in FIFO order (first collected → first ejected). No specific
        approach waypoint required — just get the back face close and face away.
        """
        import math

        _BACK_DIST = 12.0            # back-face-center → goal-surface threshold (in)
        _HDG_TOL   = math.pi / 4.0  # ±45° heading tolerance (generous for manual)

        # Robot back-face centre
        bx = robot.x - math.cos(robot.heading) * (ROBOT_W / 2)
        by = robot.y - math.sin(robot.heading) * (ROBOT_W / 2)
        back = np.array([bx, by])

        # Collect all goal surfaces the back face is near: (req_heading, anim_target, gname, pts)
        candidates: list[tuple] = []
        in_y = LONG_GOAL_Y_MIN - _BACK_DIST < by < LONG_GOAL_Y_MAX + _BACK_DIST

        # ── Right long goal ──
        # Field side: back faces east (heading ≈ π)
        if abs(bx - _RIGHT_GOAL_X_LO) < _BACK_DIST and in_y:
            gy = float(np.clip(by, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX))
            candidates.append((math.pi,
                                np.array([_RIGHT_GOAL_CX, gy]),
                                "our_long", LONG_GOAL_POINTS))
        # Top opening: back faces south (heading ≈ +π/2)
        if abs(by - LONG_GOAL_Y_MAX) < _BACK_DIST and abs(bx - _RIGHT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_NORTH,
                                np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MAX]),
                                "our_long", LONG_GOAL_POINTS))
        # Bottom opening: back faces north (heading ≈ −π/2)
        if abs(by - LONG_GOAL_Y_MIN) < _BACK_DIST and abs(bx - _RIGHT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_SOUTH,
                                np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MIN]),
                                "our_long", LONG_GOAL_POINTS))

        # ── Left long goal ──
        # Field side: back faces west (heading ≈ 0)
        if abs(bx - _LEFT_GOAL_X_HI) < _BACK_DIST and in_y:
            gy = float(np.clip(by, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX))
            candidates.append((0.0,
                                np.array([_LEFT_GOAL_CX, gy]),
                                "opp_long", LONG_GOAL_POINTS))
        # Top opening
        if abs(by - LONG_GOAL_Y_MAX) < _BACK_DIST and abs(bx - _LEFT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_NORTH,
                                np.array([_LEFT_GOAL_CX, LONG_GOAL_Y_MAX]),
                                "opp_long", LONG_GOAL_POINTS))
        # Bottom opening
        if abs(by - LONG_GOAL_Y_MIN) < _BACK_DIST and abs(bx - _LEFT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_SOUTH,
                                np.array([_LEFT_GOAL_CX, LONG_GOAL_Y_MIN]),
                                "opp_long", LONG_GOAL_POINTS))

        # ── Center X arm tips — MID=NE-SW bar, LOW=NW-SE bar ──
        for tip, req_hdg, gname in [
            (_TIP_NE, _HDG_NE, "center_mid"),
            (_TIP_SW, _HDG_SW, "center_mid"),
            (_TIP_NW, _HDG_NW, "center_low"),
            (_TIP_SE, _HDG_SE, "center_low"),
        ]:
            if float(np.linalg.norm(back - tip)) < _BACK_DIST:
                candidates.append((req_hdg, tip.copy(), gname, CENTER_GOAL_POINTS))

        if not candidates:
            robot.score_timer = 0.0
            return

        # Best candidate = smallest heading error
        req_hdg, anim_target, gname, points = min(
            candidates,
            key=lambda c: abs(_wrap_angle(c[0] - robot.heading)),
        )

        if abs(_wrap_angle(req_hdg - robot.heading)) > _HDG_TOL:
            robot.score_timer = 0.0
            return

        robot.score_timer += DT
        interval = _SCORE_INTERVAL / max(robot.balls_held, 1)
        if robot.score_timer < interval:
            return

        robot.score_timer = 0.0
        if robot.balls_held <= 0 or not robot.held_object_ids:
            return

        # heading < 0 → enters from the 'start' end (S / SW / SE) → prepend
        prepend    = req_hdg < 0.0

        # Score oldest held ball (FIFO: first collected = first ejected)
        idx        = robot.held_object_ids.pop(0)
        ball_color = self.field.objects[idx].color
        robot.balls_held -= 1
        self.field.objects[idx].status         = OBJ_SCORED_US
        self.field.objects[idx].scored_in_goal = gname
        ejected = self.field.goal_state.score_ball(gname, idx, ball_color, prepend=prepend)
        # Ball color determines which alliance earns the points
        if ball_color == BALL_BLUE:
            self.field.my_score += points
        else:
            self.field.opponent_score += points

        if ejected is not None:
            self._handle_overflow(ejected, gname, prepend, points)

        # Ball flies from robot back face to goal entry position
        self.score_animations.append({
            'x0': bx, 'y0': by,
            'x1': float(anim_target[0]), 'y1': float(anim_target[1]),
            'color':    ball_color,
            'start_ms': None,
            'duration': 0.4,
        })

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

        # ── Waypoint queues: each robot gets an ordered list of nav points. ──
        # SCORE_LONG_GOAL may insert a detour waypoint to avoid the center X.
        # live_actions mirrors the current per-robot action and updates on replanning.
        live_actions = [a0, a1]

        def _build_wq(idx: int) -> list:
            """Build the waypoint queue for robot idx using its current live action."""
            robot  = self.field.allies[idx]
            act    = Action(live_actions[idx])
            final  = _action_to_target(act, self.field, robot)
            if final is None:
                return []
            if act in (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_GOAL,
                        Action.COLLECT_NEAREST_BALL):
                return _build_nav_waypoints(robot.position, final)
            return [final]

        ally_wq: list[list] = [_build_wq(0), _build_wq(1)]

        def _current_target(idx: int):
            return ally_wq[idx][0] if ally_wq[idx] else None

        # ── Dynamic step length ──────────────────────────────────────────
        # Timer disabled: simply run until robots stop moving or a tick cap is hit.
        MAX_STEP_TICKS = TICKS_PER_DECISION * 8

        for tick in range(MAX_STEP_TICKS):
            if tick >= TICKS_PER_DECISION:
                still_moving = any(
                    self.field.allies[i].moving and bool(ally_wq[i])
                    for i in range(self.num_allies)
                )
                if not still_moving:
                    break

            fi.on_tick(num_robots=2)

            # ── Mid-step adaptive replanning ──────────────────────────────
            for idx in range(self.num_allies):
                robot = self.field.allies[idx]
                act   = Action(live_actions[idx])
                new_act = None
                # Full robot → always score, regardless of current action
                if robot.balls_held >= MAX_CARRY and act != Action.SCORE_LONG_GOAL:
                    new_act = Action.SCORE_LONG_GOAL
                # Finished scoring → go collect
                elif act == Action.SCORE_LONG_GOAL and robot.balls_held == 0:
                    new_act = Action.COLLECT_NEAREST_BALL
                if new_act is not None:
                    live_actions[idx]          = int(new_act)
                    self.current_actions[idx]  = live_actions[idx]
                    ally_wq[idx]               = _build_wq(idx)

            # Save positions before movement for push calculation
            ally_prev = [self.field.allies[i].position.copy()     for i in range(self.num_allies)]
            opp_prev  = [self.field.opponents[i].position.copy()  for i in range(self.num_opponents)]

            for idx in range(self.num_allies):
                if fi.is_stuck(idx):
                    continue
                target = _current_target(idx)
                if target is not None:
                    arrived = self.field.allies[idx].move_toward_point(target)
                    if arrived and len(ally_wq[idx]) > 1:
                        ally_wq[idx].pop(0)      # advance to next waypoint
                    elif arrived:
                        ally_wq[idx].clear()     # reached final destination
                _resolve_goal_collisions(self.field.allies[idx])

            for oi in range(self.num_opponents):
                if self.opp_targets[oi] is not None:
                    self.field.opponents[oi].move_toward_point(self.opp_targets[oi])
                _resolve_goal_collisions(self.field.opponents[oi])

            # Update intake state: spinning only while actively collecting and not full
            for idx in range(self.num_allies):
                robot = self.field.allies[idx]
                robot.intake_active = (
                    Action(live_actions[idx]) == Action.COLLECT_NEAREST_BALL
                    and robot.balls_held < MAX_CARRY
                )

            # Check collect/score before push so balls in intake range don't get kicked away
            self._check_ally_effects(0, Action(live_actions[0]))
            if self.num_allies >= 2:
                self._check_ally_effects(1, Action(live_actions[1]))
            self._check_opp_effects()

            # Apply robot→ball pushes (already-held balls are skipped)
            for idx in range(self.num_allies):
                self.field.apply_robot_push(self.field.allies[idx], ally_prev[idx])
            for oi in range(self.num_opponents):
                self.field.apply_robot_push(self.field.opponents[oi], opp_prev[oi])

            # Advance ball rolling physics
            self.field.physics_tick(DT)

            # Timer disabled — clock does not decrement, episode does not end.
            self.decision_tick += 1

            if self.render_mode == "human" and self._renderer is not None:
                self._renderer.draw(self)

        self.executing = False
        self.done = False   # timer disabled

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
            if robot.balls_held <= 0:
                robot.score_timer = 0.0
                return
            score_pos, goal_pos, required_heading, gname = _nearest_long_goal_target(robot)
            self._do_back_in_scoring(idx, robot, score_pos, goal_pos, gname,
                                     required_heading, LONG_GOAL_POINTS)

        elif action == Action.SCORE_CENTER_GOAL:
            if robot.balls_held <= 0:
                robot.score_timer = 0.0
                return
            d_mid = min(np.linalg.norm(robot.position - _MID_NE_POS),
                        np.linalg.norm(robot.position - _LOW_SW_POS))
            d_low = min(np.linalg.norm(robot.position - _MID_NW_POS),
                        np.linalg.norm(robot.position - _LOW_SE_POS))
            lower = d_low < d_mid
            score_pos, goal_pos, required_heading, gname = _nearest_center_tip(robot, lower)
            self._do_back_in_scoring(idx, robot, score_pos, goal_pos, gname,
                                     required_heading, CENTER_GOAL_POINTS)

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

    def _do_back_in_scoring(self, idx: int, robot, score_pos: np.ndarray,
                              goal_pos: np.ndarray, gname: str,
                              required_heading: float, points: int) -> None:
        """Common back-in scoring routine.

        Robot arrives at score_pos, turns to required_heading (back faces goal),
        then scores one ball at a time on a timer.
        """
        dist = float(np.linalg.norm(robot.position - score_pos))
        if dist >= _SCORE_ARRIVAL_DIST:
            robot.score_timer = 0.0
            return

        angle_err  = _wrap_angle(required_heading - robot.heading)
        turn_delta = float(np.clip(angle_err * 4.0, -TURN_RATE, TURN_RATE)) * DT
        robot.heading = _wrap_angle(robot.heading + turn_delta)

        if abs(_wrap_angle(required_heading - robot.heading)) >= _SCORE_HDG_TOL:
            robot.score_timer = 0.0
            return

        robot.score_timer += DT
        interval = _SCORE_INTERVAL / max(robot.balls_held, 1)
        if robot.score_timer < interval:
            return
        robot.score_timer = 0.0

        # heading < 0 → scores from the "start" end (S / SW / SE) → prepend
        prepend    = required_heading < 0.0
        ball_color = (self.field.objects[robot.held_object_ids[0]].color
                      if robot.held_object_ids else None)
        pts, ejected = self.field.try_score_one(robot, goal_pos, points,
                                                gname=gname, prepend=prepend)
        if pts > 0:
            self.score_events[idx] += pts
            robot.actions_succeeded += 1
            if ejected is not None:
                self._handle_overflow(ejected, gname, prepend, points)
            if ball_color is not None:
                self.score_animations.append({
                    'x0': robot.x, 'y0': robot.y,
                    'x1': float(goal_pos[0]), 'y1': float(goal_pos[1]),
                    'color': ball_color,
                    'start_ms': None,
                    'duration': 0.5,
                })

    def _handle_overflow(self, ejected: tuple, gname: str,
                          prepend: bool, points: int) -> None:
        """Put an overflow ball back on the field at the goal exit end.

        prepend=True means the ball entered from the 'start' end, so it exits
        from the 'end' end, and vice versa.
        """
        import math
        ej_idx, ej_color = ejected
        obj = self.field.objects[ej_idx]
        obj.status         = OBJ_ON_FIELD
        obj.scored_in_goal = None
        # Subtract from whichever team earned this ball's points
        if ej_color == BALL_BLUE:
            self.field.my_score -= points
        else:
            self.field.opponent_score -= points

        # Place ball just outside the exit end of each goal
        gap = ROBOT_W + 4.0
        if gname == "our_long":
            if prepend:   # entered S → exits N
                obj.x, obj.y = _RIGHT_GOAL_CX, LONG_GOAL_Y_MAX + gap
                obj.vy = 12.0
            else:         # entered N → exits S
                obj.x, obj.y = _RIGHT_GOAL_CX, LONG_GOAL_Y_MIN - gap
                obj.vy = -12.0
            obj.vx = 0.0
        elif gname == "opp_long":
            if prepend:
                obj.x, obj.y = _LEFT_GOAL_CX, LONG_GOAL_Y_MAX + gap
                obj.vy = 12.0
            else:
                obj.x, obj.y = _LEFT_GOAL_CX, LONG_GOAL_Y_MIN - gap
                obj.vy = -12.0
            obj.vx = 0.0
        elif gname == "center_mid":
            exit_tip = _TIP_NE if prepend else _TIP_SW
            d = gap / math.sqrt(2.0)
            sign = 1.0 if prepend else -1.0
            obj.x, obj.y = float(exit_tip[0]) + sign * d, float(exit_tip[1]) + sign * d
            obj.vx =  sign * 8.0;  obj.vy =  sign * 8.0
        elif gname == "center_low":
            exit_tip = _TIP_NW if prepend else _TIP_SE
            d = gap / math.sqrt(2.0)
            if prepend:   # exits NW
                obj.x, obj.y = float(exit_tip[0]) - d, float(exit_tip[1]) + d
                obj.vx = -8.0;  obj.vy =  8.0
            else:         # exits SE
                obj.x, obj.y = float(exit_tip[0]) + d, float(exit_tip[1]) - d
                obj.vx =  8.0;  obj.vy = -8.0
        else:
            obj.vx = obj.vy = 0.0

        # Scatter each ball with a random offset + lateral velocity so
        # consecutive overflow balls land at different spots and must be
        # collected individually.
        rng = getattr(self, "rng", None)
        if rng is not None:
            scatter = 10.0  # max scatter radius in inches
            obj.x += float(rng.uniform(-scatter, scatter))
            obj.y += float(rng.uniform(-scatter, scatter))
            obj.vx += float(rng.uniform(-8.0, 8.0))
            obj.vy += float(rng.uniform(-8.0, 8.0))

        obj.x = round(float(np.clip(obj.x, ROBOT_W / 2 + 1, FIELD_W - ROBOT_W / 2 - 1)), 2)
        obj.y = round(float(np.clip(obj.y, ROBOT_W / 2 + 1, FIELD_H - ROBOT_W / 2 - 1)), 2)

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
