"""Field state model — objects, goals, zones, robot positions.

VEX Push Back 2025-2026.
Central state container. Owns all robots and game objects.
The env (env.py) orchestrates stepping; the field tracks state.
"""

from __future__ import annotations
import numpy as np

import math

from sim.config import (
    FIELD_W, FIELD_H, MAX_GAME_OBJECTS, INITIAL_OBJECTS,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP, OBJ_REMOVED,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    LONG_GOAL_POINTS, CENTER_GOAL_POINTS,
    COLLECT_RANGE, SCORE_RANGE, MAX_CARRY, ROBOT_W, Action,
    BALL_RED, BALL_BLUE,
    MATCHLOAD_TUBES, MATCHLOAD_TUBE_RADIUS,
    LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX, LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH,
    LONG_GOAL_CAPACITY, CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W,
    HALF_BOTTOM_Y_MAX, HALF_TOP_Y_MIN,
    HALF_SHARED_STRIP_Y_LO, HALF_SHARED_STRIP_Y_HI,
    ALLY_START_R0, ALLY_START_R1, OPP_START_R0, OPP_START_R1,
    ALLY_START_HEADING_R0, ALLY_START_HEADING_R1,
    OPP_START_HEADING_R0, OPP_START_HEADING_R1,
    HALF_LEFT_X_MAX, HALF_RIGHT_X_MIN,
    HALF_SHARED_STRIP_X_LO, HALF_SHARED_STRIP_X_HI,
    DESCORE_P0, DESCORE_P1, DESCORE_P2, DESCORE_P3,
    SLAM_SPEED_DIVISOR, SLAM_MIN_SPEED,
)
from sim.game_object import BALL_RADIUS, GameObject
from sim.action_helpers import is_descore_action, is_ram_action

# Staging / approach margin beyond long-goal ends (matches env _STAGE_* offsets).
_LONG_GOAL_NAV_MARGIN = 25.0

# Long-goal center X (matches env scoring geometry)
_RIGHT_GOAL_CX = (FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH + FIELD_W - LONG_GOAL_WALL_GAP) / 2.0
_LEFT_GOAL_CX  = (LONG_GOAL_WALL_GAP + LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH) / 2.0
_CENTER_X, _CENTER_Y = 72.0, 72.0
_ARM_TIP = CENTER_GOAL_ARM_LEN / math.sqrt(2.0)
_APPROACH_GAP = min(ROBOT_W / 2.0 + 4.5, SCORE_RANGE - 1.0)
_CENTER_TIP_DIAG = (CENTER_GOAL_ARM_LEN + _APPROACH_GAP) / math.sqrt(2.0)
CENTER_MID_ACCESS_Y_MAX = _CENTER_Y + _CENTER_TIP_DIAG + ROBOT_W / 2.0 + 1.0
CENTER_LOW_ACCESS_Y_MIN = _CENTER_Y - _CENTER_TIP_DIAG - ROBOT_W / 2.0 - 1.0
_TIP_NE = (_CENTER_X + _ARM_TIP, _CENTER_Y + _ARM_TIP)
_TIP_SW = (_CENTER_X - _ARM_TIP, _CENTER_Y - _ARM_TIP)
_TIP_NW = (_CENTER_X - _ARM_TIP, _CENTER_Y + _ARM_TIP)
_TIP_SE = (_CENTER_X + _ARM_TIP, _CENTER_Y - _ARM_TIP)
_SPILL_GAP = ROBOT_W + 4.0


def exit_end_opposite_entry(entry_end: int) -> int:
    """Return the goal end balls spill from when entering at entry_end."""
    return DESCORE_P3 if entry_end == DESCORE_P0 else DESCORE_P0


def exit_end_for_scoring_prepend(prepend: bool) -> int:
    """Return spill end when a ball enters with prepend=True (start) or False (end)."""
    return DESCORE_P3 if prepend else DESCORE_P0


def spill_ball_at_goal_exit(obj: GameObject, gname: str, exit_end: int,
                            rng: np.random.Generator) -> None:
    """Place a ball just outside the goal exit with spill velocity."""
    scatter = 8.0
    if gname == "our_long":
        cx = _RIGHT_GOAL_CX
        if exit_end == DESCORE_P3:
            obj.x, obj.y = cx, LONG_GOAL_Y_MAX + _SPILL_GAP
            obj.vx, obj.vy = 0.0, 12.0
        else:
            obj.x, obj.y = cx, LONG_GOAL_Y_MIN - _SPILL_GAP
            obj.vx, obj.vy = 0.0, -12.0
    elif gname == "opp_long":
        cx = _LEFT_GOAL_CX
        if exit_end == DESCORE_P3:
            obj.x, obj.y = cx, LONG_GOAL_Y_MAX + _SPILL_GAP
            obj.vx, obj.vy = 0.0, 12.0
        else:
            obj.x, obj.y = cx, LONG_GOAL_Y_MIN - _SPILL_GAP
            obj.vx, obj.vy = 0.0, -12.0
    elif gname == "center_mid":
        if exit_end == DESCORE_P3:
            tip, vx, vy = _TIP_NE, 8.0, 8.0
        else:
            tip, vx, vy = _TIP_SW, -8.0, -8.0
        d = _SPILL_GAP / math.sqrt(2.0)
        sign = 1.0 if exit_end == DESCORE_P3 else -1.0
        obj.x = float(tip[0]) + sign * d
        obj.y = float(tip[1]) + sign * d
        obj.vx, obj.vy = sign * vx, sign * vy
    elif gname == "center_low":
        d = _SPILL_GAP / math.sqrt(2.0)
        if exit_end == DESCORE_P3:
            obj.x = float(_TIP_NW[0]) - d
            obj.y = float(_TIP_NW[1]) + d
            obj.vx, obj.vy = -8.0, 8.0
        else:
            obj.x = float(_TIP_SE[0]) + d
            obj.y = float(_TIP_SE[1]) - d
            obj.vx, obj.vy = 8.0, -8.0
    else:
        obj.vx = obj.vy = 0.0

    obj.x += float(rng.uniform(-scatter, scatter))
    obj.y += float(rng.uniform(-scatter, scatter))
    obj.vx += float(rng.uniform(-6.0, 6.0))
    obj.vy += float(rng.uniform(-6.0, 6.0))
    half = ROBOT_W / 2.0 + 1.0
    obj.x = round(float(np.clip(obj.x, half, FIELD_W - half)), 2)
    obj.y = round(float(np.clip(obj.y, half, FIELD_H - half)), 2)


def infer_scoring_prepend(gname: str, goal_pos: np.ndarray) -> bool:
    """Infer whether scoring enters from the 'start' end (prepend) at goal_pos."""
    gy = float(goal_pos[1])
    gx = float(goal_pos[0])
    if "long" in gname:
        return gy < (_CENTER_Y)
    if gname == "center_mid":
        return gy < _CENTER_Y or (abs(gx - _CENTER_X) < 1.0 and gy <= _CENTER_Y)
    if gname == "center_low":
        return gy > _CENTER_Y
    return False

# ---------------------------------------------------------------------------
# Ball-position randomisation
# ---------------------------------------------------------------------------
# Per-episode jitter so each match is a slightly different scenario — keeps
# the policy from overfitting to the 44 fixed INITIAL_OBJECTS coordinates.
_JITTER_RADIUS = 20.0  # inches — max offset from anchor position
_JITTER_ATTEMPTS = 12  # if jittered pos lands on a goal, try again N times
_BALL_WALL_MARGIN = 4.0

# Pre-computed goal bounding boxes for collision testing during jitter.
# Long goals = two vertical bars near the side walls.
_R_GOAL_X_LO = FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH
_R_GOAL_X_HI = FIELD_W - LONG_GOAL_WALL_GAP
_L_GOAL_X_LO = LONG_GOAL_WALL_GAP
_L_GOAL_X_HI = LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH
_GOAL_PAD    = BALL_RADIUS + 2.0


def _ball_pos_clear(x: float, y: float) -> bool:
    """True if (x, y) is safely on the field — not inside a goal body."""
    # Walls
    if x < _BALL_WALL_MARGIN or x > FIELD_W - _BALL_WALL_MARGIN:
        return False
    if y < _BALL_WALL_MARGIN or y > FIELD_H - _BALL_WALL_MARGIN:
        return False
    # Long goal bodies
    if (_R_GOAL_X_LO - _GOAL_PAD <= x <= _R_GOAL_X_HI + _GOAL_PAD
            and LONG_GOAL_Y_MIN - _GOAL_PAD <= y <= LONG_GOAL_Y_MAX + _GOAL_PAD):
        return False
    if (_L_GOAL_X_LO - _GOAL_PAD <= x <= _L_GOAL_X_HI + _GOAL_PAD
            and LONG_GOAL_Y_MIN - _GOAL_PAD <= y <= LONG_GOAL_Y_MAX + _GOAL_PAD):
        return False
    # Center X arms — rotated rectangle test, expanded by ball radius
    dx, dy = x - 72.0, y - 72.0
    half_w = CENTER_GOAL_ARM_W / 2 + _GOAL_PAD
    for angle in (math.pi / 4.0, -math.pi / 4.0):
        ca, sa = math.cos(angle), math.sin(angle)
        along = dx * ca + dy * sa
        perp  = dx * (-sa) + dy * ca
        if abs(along) <= CENTER_GOAL_ARM_LEN + _GOAL_PAD and abs(perp) <= half_w:
            return False
    return True


def _jitter_position(anchor_x: float, anchor_y: float,
                     rng: np.random.Generator) -> tuple[float, float]:
    """Return a randomly jittered position near (anchor_x, anchor_y).

    Falls back to the anchor if every jitter attempt lands on a goal body.
    """
    for _ in range(_JITTER_ATTEMPTS):
        dx = float(rng.uniform(-_JITTER_RADIUS, _JITTER_RADIUS))
        dy = float(rng.uniform(-_JITTER_RADIUS, _JITTER_RADIUS))
        x = anchor_x + dx
        y = anchor_y + dy
        if _ball_pos_clear(x, y):
            return round(x, 2), round(y, 2)
    return anchor_x, anchor_y

# How close a ball must be to a tube to be considered "at the tube"
_TUBE_SNAP_DIST = MATCHLOAD_TUBE_RADIUS * 3.0
# Heading tolerance for facing a matchload tube (radians)
_TUBE_FACE_TOL  = math.radians(30.0)

# Robot push / intake constants
_PUSH_RADIUS  = ROBOT_W / 2 + BALL_RADIUS + 1.0   # 11.0" — radial trigger for side/back push
_PUSH_SCALE   = 3.5                                 # gentle deflect (side/back only)
_MAX_BALL_SPD = 60.0                                # in/s cap

# Front-face intake zone.
# A ball is collectable when its centre is within the 15"-wide stripe directly
# in front of the leading face, extending _INTAKE_DEPTH inches beyond it.
#   along  ∈ [ROBOT_W/2 − BALL_RADIUS,  ROBOT_W/2 + _INTAKE_DEPTH + BALL_RADIUS]
#   |perp| ≤  ROBOT_W/2 + BALL_RADIUS
_INTAKE_DEPTH = 4.0   # inches beyond front face that the intake can reach


def _in_intake_zone(robot, ball_pos: np.ndarray) -> bool:
    """True if ball_pos falls inside the robot's front-face intake rectangle."""
    dx, dy   = ball_pos[0] - robot.x, ball_pos[1] - robot.y
    fwd_x    = math.cos(robot.heading)
    fwd_y    = math.sin(robot.heading)
    rgt_x    =  math.sin(robot.heading)
    rgt_y    = -math.cos(robot.heading)
    along    = dx * fwd_x + dy * fwd_y
    perp     = dx * rgt_x + dy * rgt_y
    half     = ROBOT_W / 2
    return (half - BALL_RADIUS <= along <= half + _INTAKE_DEPTH + BALL_RADIUS
            and abs(perp) <= half + BALL_RADIUS)

def _goal_name(goal_pos: np.ndarray) -> str:
    """Map goal position to a short string key for per-goal counts."""
    if np.allclose(goal_pos, OUR_LONG_GOAL):   return "our_long"
    if np.allclose(goal_pos, OPP_LONG_GOAL):   return "opp_long"
    if np.allclose(goal_pos, CENTER_MID_GOAL):  return "center_mid"
    if np.allclose(goal_pos, CENTER_LOW_GOAL):  return "center_low"
    return "other"
from sim.robot import Robot
from sim.heatmap import compute_heatmap


def field_half(robot: Robot, is_opponent: bool) -> str:
    """Return alliance half: allies bottom, opponents top (VAIRC midfield at y=72)."""
    return "top" if is_opponent else "bottom"


def field_side(robot: Robot) -> str:
    """Return 'left' or 'right' lane for this robot (role 0 = left, role 1 = right)."""
    return "left" if robot.role_id == 0 else "right"


def cross_alliance_pair(ally_idx: int, opp_idx: int) -> bool:
    """True when ally and opponent share a field side (R0↔R0 left, R1↔R1 right)."""
    return ally_idx == opp_idx


def y_bounds_for_robot(robot: Robot, is_opponent: bool,
                       action: Action | None = None) -> tuple[float, float]:
    """Allowed y-range for this robot (inclusive), with center-goal extensions."""
    half_w = ROBOT_W / 2.0 + 1.0
    if field_half(robot, is_opponent) == "bottom":
        y_min = half_w
        y_max = HALF_BOTTOM_Y_MAX
        if action is not None:
            if action in (Action.SCORE_MID_LEFT, Action.SCORE_MID_RIGHT):
                y_max = max(y_max, CENTER_MID_ACCESS_Y_MAX)
            elif action in (Action.SCORE_LOW_LEFT, Action.SCORE_LOW_RIGHT):
                y_max = max(y_max, HALF_SHARED_STRIP_Y_HI)
            elif is_descore_action(action) or is_ram_action(action):
                y_max = max(y_max, LONG_GOAL_Y_MAX + _LONG_GOAL_NAV_MARGIN)
    else:
        y_min = HALF_TOP_Y_MIN
        y_max = FIELD_H - half_w
        if action is not None:
            if action in (Action.SCORE_LOW_LEFT, Action.SCORE_LOW_RIGHT):
                y_min = min(y_min, CENTER_LOW_ACCESS_Y_MIN)
            elif action in (Action.SCORE_MID_LEFT, Action.SCORE_MID_RIGHT):
                y_min = min(y_min, HALF_SHARED_STRIP_Y_LO)
            elif is_descore_action(action) or is_ram_action(action):
                y_min = min(y_min, LONG_GOAL_Y_MIN - _LONG_GOAL_NAV_MARGIN)
    return y_min, y_max


def x_bounds_for_robot(robot: Robot,
                       action: Action | None = None) -> tuple[float, float]:
    """Allowed x-range for this robot's field side (inclusive)."""
    half_w = ROBOT_W / 2.0 + 1.0
    center_intent = action in (
        Action.SCORE_MID_LEFT, Action.SCORE_MID_RIGHT,
        Action.SCORE_LOW_LEFT, Action.SCORE_LOW_RIGHT,
    ) if action is not None else False
    if field_side(robot) == "left":
        x_min = half_w
        x_max = HALF_SHARED_STRIP_X_HI if center_intent else HALF_LEFT_X_MAX
    else:
        x_min = HALF_SHARED_STRIP_X_LO if center_intent else HALF_RIGHT_X_MIN
        x_max = FIELD_W - half_w
    return x_min, x_max


def clip_xy_to_zone(x: float, y: float,
                    x_min: float, x_max: float,
                    y_min: float, y_max: float) -> tuple[float, float]:
    """Clip position to allowed x/y bounds."""
    return (
        round(float(np.clip(x, x_min, x_max)), 2),
        round(float(np.clip(y, y_min, y_max)), 2),
    )


def clip_xy_to_half(x: float, y: float, y_min: float, y_max: float) -> tuple[float, float]:
    """Clip position to field x-bounds and robot half y-bounds."""
    half_w = ROBOT_W / 2.0 + 1.0
    return clip_xy_to_zone(x, y, half_w, FIELD_W - half_w, y_min, y_max)


def clamp_robot_to_half(robot: Robot, is_opponent: bool,
                        action: Action | None = None,
                        enforce_x_lane: bool = False) -> None:
    """Clamp robot center to alliance y-bounds and optional left/right x-lane."""
    y_min, y_max = y_bounds_for_robot(robot, is_opponent, action)
    if enforce_x_lane:
        x_min, x_max = x_bounds_for_robot(robot, action)
    else:
        half_w = ROBOT_W / 2.0 + 1.0
        x_min, x_max = half_w, FIELD_W - half_w
    robot.x, robot.y = clip_xy_to_zone(robot.x, robot.y, x_min, x_max, y_min, y_max)


def resolve_robot_pair_collision(robot_a: Robot, robot_b: Robot) -> bool:
    """Push apart two overlapping robots (center distance < ROBOT_W). Returns True if moved."""
    dx = float(robot_b.x - robot_a.x)
    dy = float(robot_b.y - robot_a.y)
    dist = math.hypot(dx, dy)
    min_dist = float(ROBOT_W)
    if dist >= min_dist:
        return False
    if dist < 1e-6:
        dx, dy, dist = 1.0, 0.0, 1.0
    overlap = min_dist - dist
    nx, ny = dx / dist, dy / dist
    push = overlap / 2.0
    robot_a.x = round(robot_a.x - nx * push, 2)
    robot_a.y = round(robot_a.y - ny * push, 2)
    robot_b.x = round(robot_b.x + nx * push, 2)
    robot_b.y = round(robot_b.y + ny * push, 2)
    return True


def _robots_overlap(robots: list[Robot]) -> bool:
    min_dist = float(ROBOT_W)
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            a, b = robots[i], robots[j]
            if math.hypot(b.x - a.x, b.y - a.y) < min_dist - 1e-6:
                return True
    return False


def resolve_all_robot_collisions(allies: list[Robot],
                                 opponents: list[Robot],
                                 max_passes: int = 16) -> None:
    """Push apart every overlapping robot pair (allies, opponents, and cross-alliance)."""
    robots = list(allies) + list(opponents)
    if len(robots) < 2:
        return
    for _ in range(max_passes):
        if not _robots_overlap(robots):
            break
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                resolve_robot_pair_collision(robots[i], robots[j])


class GoalState:
    """Tracks the ordered ball contents of each goal for VEX AI control scoring.

    Each goal holds an ordered list of (ball_idx, color) from first-scored
    (south/left outer) to last-scored (north/right outer).

    Control bonus: a quadrant is controlled by the color that matches the
    outermost position of BOTH the adjacent long goal AND the adjacent center goal.
    If only 1 ball in a goal, it counts as both outer positions.
    """

    def __init__(self):
        # ordered (ball_idx, color) — index 0 = south/left outer
        self.our_long:   list[tuple[int, int]] = []   # right long goal (x≈120")
        self.opp_long:   list[tuple[int, int]] = []   # left long goal  (x≈23.5")
        self.center_mid: list[tuple[int, int]] = []   # upper center goal
        self.center_low: list[tuple[int, int]] = []   # lower center goal

    def _list(self, gname: str) -> list:
        return getattr(self, gname)

    def reset(self):
        self.our_long.clear()
        self.opp_long.clear()
        self.center_mid.clear()
        self.center_low.clear()

    def score_ball(self, gname: str, ball_idx: int, color: int,
                   prepend: bool = False) -> tuple | None:
        """Score a ball into the goal, respecting capacity limits.

        prepend=False (default): ball enters from the 'end' (N for long, NE for MID, NW for LOW).
        prepend=True: ball enters from the 'start' (S for long, SW for MID, SE for LOW).

        Capacities: long goals = 12 balls, center goals = 7 balls.
        If full, the ball at the OPPOSITE end is ejected (rolls out).
        Returns (ejected_ball_idx, ejected_color) on overflow, else None.
        """
        _LONG_CAP   = LONG_GOAL_CAPACITY
        _CENTER_CAP = 7
        lst = self._list(gname)
        cap = _LONG_CAP if "long" in gname else _CENTER_CAP

        ejected = None
        if len(lst) >= cap:
            # Eject from the end opposite the entry
            ejected = lst.pop(-1) if prepend else lst.pop(0)

        if prepend:
            lst.insert(0, (ball_idx, color))
        else:
            lst.append((ball_idx, color))

        return ejected

    def remove_ball(self, gname: str, ball_idx: int):
        """Remove a specific ball from this goal by ball_idx."""
        lst = self._list(gname)
        for i, (idx, _) in enumerate(lst):
            if idx == ball_idx:
                lst.pop(i)
                return

    @staticmethod
    def partition_boundaries(n_balls: int) -> list[int]:
        """Return [P0, P1, P2, P3] list-index boundaries for n scored balls."""
        if n_balls <= 0:
            return [0, 0, 0, 0]
        b0 = (n_balls + 2) // 3
        b1 = b0 + (n_balls - b0 + 1) // 2
        return [0, b0, b1, n_balls]

    @staticmethod
    def segment_bounds(n_balls: int) -> list[tuple[int, int]]:
        """Three equal-ish index segments for n balls in a goal."""
        if n_balls <= 0:
            return [(0, -1), (0, -1), (0, -1)]
        b = GoalState.partition_boundaries(n_balls)
        return [(b[0], b[1] - 1), (b[1], b[2] - 1), (b[2], b[3] - 1)]

    @staticmethod
    def slide_index_range(n_balls: int, start_pt: int, end_pt: int) -> tuple[int, int] | None:
        """Inclusive index span of balls between partition points start_pt → end_pt."""
        if n_balls <= 0 or start_pt == end_pt:
            return None
        b = GoalState.partition_boundaries(n_balls)
        if end_pt < start_pt:
            lo, hi = b[end_pt], b[start_pt] - 1
        else:
            lo, hi = b[start_pt], b[end_pt] - 1
        if lo > hi:
            return None
        return lo, hi

    @staticmethod
    def is_slide_allowed(gname: str, start_pt: int, end_pt: int) -> bool:
        """Validate slide partition pairs per goal type."""
        if gname in ("our_long", "opp_long"):
            return (start_pt, end_pt) in ((DESCORE_P1, DESCORE_P0), (DESCORE_P2, DESCORE_P0))
        if gname == "center_mid":
            return start_pt in (DESCORE_P1, DESCORE_P2) and end_pt in (DESCORE_P0, DESCORE_P3)
        return False

    def outer_colors(self, gname: str) -> tuple[int | None, int | None]:
        """Return (south/left outer color, north/right outer color).

        If 1 ball, same color for both. If empty, (None, None).
        """
        lst = self._list(gname)
        if not lst:
            return None, None
        if len(lst) == 1:
            return lst[0][1], lst[0][1]
        return lst[0][1], lst[-1][1]

    def compute_quadrant_control(self) -> dict[str, int | None]:
        """Return the controlling color for each quadrant, or None.

        A quadrant is controlled when the outermost ball at the adjacent long-goal
        end AND the adjacent center-goal arm tip are the SAME color.

        Goal list layout (lst[0] = first-entry end, lst[-1] = last-entry end):
          our_long:    lst[0] = South end,  lst[-1] = North end
          opp_long:    lst[0] = South end,  lst[-1] = North end
          center_mid:  lst[0] = SW  end,    lst[-1] = NE  end   (NE-SW bar)
          center_low:  lst[0] = SE  end,    lst[-1] = NW  end   (NW-SE bar)

        Quadrant pairings:
          bottom-right (BR): our_long[South]  +  center_low[SE]
          top-right    (TR): our_long[North]  +  center_mid[NE]
          bottom-left  (BL): opp_long[South]  +  center_mid[SW]
          top-left     (TL): opp_long[North]  +  center_low[NW]
        """
        our_s, our_n  = self.outer_colors("our_long")   # S=lst[0], N=lst[-1]
        opp_s, opp_n  = self.outer_colors("opp_long")
        mid_sw, mid_ne = self.outer_colors("center_mid") # SW=lst[0], NE=lst[-1]
        low_se, low_nw = self.outer_colors("center_low") # SE=lst[0], NW=lst[-1]

        def _ctrl(a, b):
            return a if (a is not None and a == b) else None

        return {
            "bottom_right": _ctrl(our_s,  low_se),
            "top_right":    _ctrl(our_n,  mid_ne),
            "bottom_left":  _ctrl(opp_s,  mid_sw),
            "top_left":     _ctrl(opp_n,  low_nw),
        }


class Field:
    """Manages all field state: 4 robots + 44 game objects + scores."""

    def __init__(self):
        # Allied robots — blue team, bottom park zone (VAIRC Section 7)
        self.allies: list[Robot] = [
            Robot(x=float(ALLY_START_R0[0]), y=float(ALLY_START_R0[1]),
                  heading=ALLY_START_HEADING_R0, role_id=0),
            Robot(x=float(ALLY_START_R1[0]), y=float(ALLY_START_R1[1]),
                  heading=ALLY_START_HEADING_R1, role_id=1),
        ]
        # Opponent robots — red team, top half (VAIRC Section 7)
        self.opponents: list[Robot] = [
            Robot(x=float(OPP_START_R0[0]), y=float(OPP_START_R0[1]),
                  heading=OPP_START_HEADING_R0, role_id=0),
            Robot(x=float(OPP_START_R1[0]), y=float(OPP_START_R1[1]),
                  heading=OPP_START_HEADING_R1, role_id=1),
        ]
        # 44 colored balls
        self.objects: list[GameObject] = []
        for i in range(len(INITIAL_OBJECTS)):
            row = INITIAL_OBJECTS[i]
            self.objects.append(GameObject(i, row[0], row[1], int(row[2])))

        self.my_score: int = 0
        self.opponent_score: int = 0
        self.time_remaining: float = 0.0
        self.goal_state = GoalState()

    def reset(self, rng: np.random.Generator):
        """Reset field to starting positions (clears any editor changes)."""
        self.allies[0].reset(float(ALLY_START_R0[0]), float(ALLY_START_R0[1]),
                               heading=ALLY_START_HEADING_R0)
        self.allies[1].reset(float(ALLY_START_R1[0]), float(ALLY_START_R1[1]),
                               heading=ALLY_START_HEADING_R1)
        self.opponents[0].reset(float(OPP_START_R0[0]), float(OPP_START_R0[1]),
                                heading=OPP_START_HEADING_R0)
        self.opponents[1].reset(float(OPP_START_R1[0]), float(OPP_START_R1[1]),
                                heading=OPP_START_HEADING_R1)

        # Rebuild from INITIAL_OBJECTS with randomized position+color assignment.
        # Each ball is jittered ±_JITTER_RADIUS from its anchor — keeps the
        # general spread but gives every episode a slightly different layout.
        all_pos    = [(float(row[0]), float(row[1])) for row in INITIAL_OBJECTS]
        all_colors = [int(row[2]) for row in INITIAL_OBJECTS]
        pos_perm   = rng.permutation(len(all_pos))
        col_perm   = rng.permutation(len(all_colors))
        self.objects = []
        for i, (pi, ci) in enumerate(zip(pos_perm, col_perm)):
            anchor_x, anchor_y = all_pos[pi]
            x, y  = _jitter_position(anchor_x, anchor_y, rng)
            color = all_colors[ci]
            self.objects.append(GameObject(i, x, y, color))

        self.my_score = 0
        self.opponent_score = 0
        self.goal_state.reset()

    # ------------------------------------------------------------------
    # State editor helpers
    # ------------------------------------------------------------------
    def add_ball(self, x: float, y: float, color: int):
        """Add a ball via the state editor."""
        obj_id = len(self.objects)
        self.objects.append(GameObject(obj_id, x, y, color))

    def remove_ball(self, idx: int):
        """Mark a ball as removed via the state editor."""
        if 0 <= idx < len(self.objects):
            obj = self.objects[idx]
            if obj.status in (OBJ_SCORED_US, OBJ_SCORED_OPP) and obj.scored_in_goal:
                self.goal_state.remove_ball(obj.scored_in_goal, idx)
            obj.status = OBJ_REMOVED

    def change_ball_color(self, idx: int):
        """Toggle a ball's color between red and blue."""
        if 0 <= idx < len(self.objects):
            obj = self.objects[idx]
            obj.color = BALL_BLUE if obj.color == BALL_RED else BALL_RED

    def clear_all_balls(self):
        """Remove all game objects from the field (for setup mode)."""
        self.objects.clear()
        self.goal_state.reset()

    def set_robot_start(self, idx: int, x: float, y: float, heading: float | None = None):
        """Reposition (and optionally re-orient) an allied robot (for setup mode)."""
        if 0 <= idx < len(self.allies):
            half = ROBOT_W / 2
            self.allies[idx].x = round(float(np.clip(x, half, FIELD_W - half)), 2)
            self.allies[idx].y = round(float(np.clip(y, half, FIELD_H - half)), 2)
            if heading is not None:
                self.allies[idx].heading = float(heading)

    def set_robot_heading(self, idx: int, heading: float):
        """Set heading of an allied robot without moving it."""
        if 0 <= idx < len(self.allies):
            self.allies[idx].heading = float(heading)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_obj_positions(self) -> np.ndarray:
        return np.array([o.position for o in self.objects], dtype=np.float64)

    def get_obj_statuses(self) -> np.ndarray:
        return np.array([o.status for o in self.objects], dtype=np.int32)

    def get_heatmap(self) -> np.ndarray:
        return compute_heatmap(self.get_obj_positions(), self.get_obj_statuses())

    def on_field_indices(self) -> np.ndarray:
        return np.array([i for i, o in enumerate(self.objects) if o.status == OBJ_ON_FIELD])

    def scored_by_us_indices(self) -> np.ndarray:
        return np.array([i for i, o in enumerate(self.objects) if o.status == OBJ_SCORED_US])

    def scored_by_opp_indices(self) -> np.ndarray:
        return np.array([i for i, o in enumerate(self.objects) if o.status == OBJ_SCORED_OPP])

    def nearby_ball_count(self, pos: np.ndarray, radius: float = 36.00) -> int:
        """Count on-field balls within radius inches of pos."""
        count = 0
        for obj in self.objects:
            if obj.status == OBJ_ON_FIELD:
                if np.linalg.norm(obj.position - pos) < radius:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Allied action effects
    # ------------------------------------------------------------------
    def _ball_at_tube(self, obj) -> np.ndarray | None:
        """Return tube position if this ball is sitting at a matchload tube, else None."""
        for tube in MATCHLOAD_TUBES:
            if np.linalg.norm(obj.position - tube) < _TUBE_SNAP_DIST:
                return tube
        return None

    def _facing_toward(self, robot: Robot, target_pos: np.ndarray) -> bool:
        """True if robot heading is within _TUBE_FACE_TOL of the direction to target."""
        vec = target_pos - robot.position
        if np.linalg.norm(vec) < 0.1:
            return True
        angle_to = math.atan2(vec[1], vec[0])
        diff = abs(((robot.heading - angle_to + math.pi) % (2 * math.pi)) - math.pi)
        return diff <= _TUBE_FACE_TOL

    def try_collect(self, robot: Robot) -> int | bool:
        """Pick up a blue ball or automatically reject a wrong-color ball.

        The real robot color-sorts at the intake. In sim, a red ball that enters
        the blue robot's intake is dropped just behind the robot with a small
        spill velocity instead of being held or requiring a separate action.

        Only the front face of the robot (full width, _INTAKE_DEPTH reach) can
        collect. Tube balls still require the robot to face the tube within 30°.
        """
        if robot.balls_held >= robot.capacity:
            return False

        best_idx, best_dist = -1, float("inf")
        for i, obj in enumerate(self.objects):
            if obj.status == OBJ_ON_FIELD and _in_intake_zone(robot, obj.position):
                d = np.linalg.norm(obj.position - robot.position)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0:
            obj = self.objects[best_idx]
            tube = self._ball_at_tube(obj)
            if tube is not None and not self._facing_toward(robot, tube):
                return False   # must face the tube head-on

            if obj.color != BALL_BLUE:
                # Automatic color-sort reject: drop the ball out the back face.
                back_x = -math.cos(robot.heading)
                back_y = -math.sin(robot.heading)
                side_x = math.sin(robot.heading)
                side_y = -math.cos(robot.heading)
                offset = ROBOT_W / 2.0 + BALL_RADIUS + 1.5
                obj.x = round(float(robot.x + back_x * offset + side_x * 2.0), 2)
                obj.y = round(float(robot.y + back_y * offset + side_y * 2.0), 2)
                half = ROBOT_W / 2.0 + 1.0
                obj.x = round(float(np.clip(obj.x, half, FIELD_W - half)), 2)
                obj.y = round(float(np.clip(obj.y, half, FIELD_H - half)), 2)
                obj.vx = float(back_x * 18.0 + side_x * 4.0)
                obj.vy = float(back_y * 18.0 + side_y * 4.0)
                obj.status = OBJ_ON_FIELD
                return -1

            obj.status = OBJ_HELD
            robot.balls_held += 1
            robot.held_object_ids.append(best_idx)
            return True
        return False

    def try_score_one(self, robot: Robot, goal_pos: np.ndarray, points: int,
                       gname: str | None = None,
                       prepend: bool = False) -> tuple[int, tuple | None]:
        """Score exactly one held ball at goal.

        Returns (points_scored, ejected_ball) where ejected_ball is
        (ball_idx, color) if the goal overflowed, else None.
        goal_pos is the scoring reference point.
        gname overrides goal-name lookup when goal_pos isn't the centre.
        prepend=True: ball enters from the 'start' end (S / SW / SE).
        """
        if robot.balls_held <= 0:
            return 0, None
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0, None

        idx = robot.held_object_ids.pop(0)
        robot.balls_held -= 1
        self.objects[idx].status = OBJ_SCORED_US
        if gname is None:
            gname = _goal_name(goal_pos)
        self.objects[idx].scored_in_goal = gname
        ball_color = self.objects[idx].color
        ejected = self.goal_state.score_ball(gname, idx, ball_color, prepend=prepend)
        # Ball color determines which alliance earns the points
        if ball_color == BALL_BLUE:
            self.my_score += points
        else:
            self.opponent_score += points
        return points, ejected

    def _goal_points(self, gname: str) -> int:
        return LONG_GOAL_POINTS if "long" in gname else CENTER_GOAL_POINTS

    def _eject_ball_from_goal(self, gname: str, ball_idx: int, exit_end: int,
                              rng: np.random.Generator) -> None:
        obj = self.objects[ball_idx]
        obj.status = OBJ_ON_FIELD
        obj.scored_in_goal = ""
        self.goal_state.remove_ball(gname, ball_idx)
        spill_ball_at_goal_exit(obj, gname, exit_end, rng)

    def spill_overflow_ball(self, gname: str, ball_idx: int, prepend: bool,
                            rng: np.random.Generator) -> None:
        """Spill an overflow ball out the end opposite the scoring entry."""
        obj = self.objects[ball_idx]
        obj.status = OBJ_ON_FIELD
        obj.scored_in_goal = ""
        spill_ball_at_goal_exit(obj, gname, exit_end_for_scoring_prepend(prepend), rng)

    @staticmethod
    def _descore_target_color(remove_ally_scored: bool,
                              *, blue_alliance: bool = True) -> int:
        """Ball color removable for this descore intent (matches belief/planning)."""
        alliance_color = BALL_BLUE if blue_alliance else BALL_RED
        opponent_color = BALL_RED if blue_alliance else BALL_BLUE
        return alliance_color if remove_ally_scored else opponent_color

    def _ball_removable_for_descore(self, ball_idx: int, remove_ally_scored: bool,
                                    *, blue_alliance: bool = True) -> bool:
        """True when this scored ball matches the descore action's color filter."""
        obj = self.objects[ball_idx]
        if obj.status not in (OBJ_SCORED_US, OBJ_SCORED_OPP):
            return False
        return obj.color == self._descore_target_color(
            remove_ally_scored, blue_alliance=blue_alliance,
        )

    def _removable_in_goal(self, gname: str, remove_ally_scored: bool,
                           *, blue_alliance: bool = True) -> list[int]:
        """Ball indices in goal list that the descoring alliance may remove."""
        lst = self.goal_state._list(gname)
        out: list[int] = []
        for ball_idx, _ in lst:
            if (self.objects[ball_idx].scored_in_goal == gname
                    and self._ball_removable_for_descore(
                        ball_idx, remove_ally_scored, blue_alliance=blue_alliance,
                    )):
                out.append(ball_idx)
        return out

    def _remove_balls_from_goal(self, gname: str, ball_indices: list[int],
                                exit_end: int, rng: np.random.Generator) -> int:
        """Eject balls and subtract score from whichever alliance earned the points."""
        if not ball_indices:
            return 0
        pts_each = self._goal_points(gname)
        total = 0
        for ball_idx in sorted(ball_indices, reverse=True):
            color = self.objects[ball_idx].color
            self._eject_ball_from_goal(gname, ball_idx, exit_end, rng)
            if color == BALL_BLUE:
                self.my_score = max(0, self.my_score - pts_each)
            else:
                self.opponent_score = max(0, self.opponent_score - pts_each)
            total += pts_each
        return total

    def try_descore_slide(self, robot: Robot, gname: str,
                          start_pt: int, end_pt: int,
                          approach_pos: np.ndarray,
                          rng: np.random.Generator,
                          remove_ally_scored: bool = False,
                          *, blue_alliance: bool = True) -> int:
        """Slide descore: eject removable balls in the partition span start_pt → end_pt."""
        drop_ref = OPP_LONG_GOAL if gname == "opp_long" else OUR_LONG_GOAL
        dist = min(
            float(np.linalg.norm(robot.position - approach_pos)),
            float(np.linalg.norm(robot.position - drop_ref)),
        )
        if dist >= SCORE_RANGE:
            return 0
        if not GoalState.is_slide_allowed(gname, start_pt, end_pt):
            return 0
        lst = self.goal_state._list(gname)
        span = GoalState.slide_index_range(len(lst), start_pt, end_pt)
        if span is None:
            return 0
        lo, hi = span
        to_remove: list[int] = []
        for i, (ball_idx, _) in enumerate(lst):
            if (lo <= i <= hi
                    and self._ball_removable_for_descore(
                        ball_idx, remove_ally_scored, blue_alliance=blue_alliance,
                    )):
                to_remove.append(ball_idx)
        return self._remove_balls_from_goal(gname, to_remove, end_pt, rng)

    def try_descore_slam(self, robot: Robot, gname: str, entry_end: int,
                         impact_speed: float, approach_pos: np.ndarray,
                         rng: np.random.Generator,
                         remove_ally_scored: bool = False,
                         *, blue_alliance: bool = True) -> int:
        """Slam descore: eject floor(speed/K) balls from the end opposite entry_end."""
        drop_ref = OPP_LONG_GOAL if gname == "opp_long" else OUR_LONG_GOAL
        dist = min(
            float(np.linalg.norm(robot.position - approach_pos)),
            float(np.linalg.norm(robot.position - drop_ref)),
        )
        if dist >= SCORE_RANGE:
            return 0
        if impact_speed < SLAM_MIN_SPEED:
            return 0
        if entry_end not in (DESCORE_P0, DESCORE_P3):
            return 0
        n_eject = int(math.floor(impact_speed / SLAM_SPEED_DIVISOR))
        if n_eject <= 0:
            return 0
        lst = self.goal_state._list(gname)
        if not lst:
            return 0
        exit_end = exit_end_opposite_entry(entry_end)
        indices = range(len(lst) - 1, -1, -1) if exit_end == DESCORE_P3 else range(len(lst))
        to_remove: list[int] = []
        for i in indices:
            ball_idx, _ = lst[i]
            if self._ball_removable_for_descore(
                ball_idx, remove_ally_scored, blue_alliance=blue_alliance,
            ):
                to_remove.append(ball_idx)
                if len(to_remove) >= n_eject:
                    break
        return self._remove_balls_from_goal(gname, to_remove, exit_end, rng)

    def nearest_on_field_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return position of nearest on-field ball, or None (no obstacle check)."""
        on_field = self.on_field_indices()
        if len(on_field) == 0:
            return None
        positions = self.get_obj_positions()[on_field]
        dists = np.linalg.norm(positions - pos, axis=1)
        return positions[np.argmin(dists)].copy()

    def nearest_navigable_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return nearest BLUE on-field ball whose direct path is NOT blocked by a goal.

        Only considers BALL_BLUE (allied robot only collects its own color).
        Uses the route planner's LOS+margin check to skip balls behind/inside goal
        structures. Returns None if all blue balls are blocked — the caller's
        exploration fallback handles that case.
        """
        from sim.route_planner import _los_blocked, _NAV_MARGIN, _near_any_goal
        from sim.config import BALL_BLUE
        rx, ry = float(pos[0]), float(pos[1])
        best_pos: np.ndarray | None = None
        best_dist = float("inf")
        for obj in self.objects:
            if obj.status != OBJ_ON_FIELD or obj.color != BALL_BLUE:
                continue
            if _near_any_goal(obj.x, obj.y):
                continue
            if _los_blocked(rx, ry, obj.x, obj.y, margin=_NAV_MARGIN):
                continue
            d = float(np.linalg.norm(obj.position - pos))
            if d < best_dist:
                best_dist = d
                best_pos = obj.position.copy()
        return best_pos

    # ------------------------------------------------------------------
    # Opponent action effects
    # ------------------------------------------------------------------
    def opp_try_collect(self, robot: Robot, rng: np.random.Generator) -> bool:
        if robot.balls_held >= robot.capacity:
            return False
        best_idx, best_dist = -1, float("inf")
        for i, obj in enumerate(self.objects):
            if obj.status == OBJ_ON_FIELD and _in_intake_zone(robot, obj.position):
                d = np.linalg.norm(obj.position - robot.position)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx >= 0:
            self.objects[best_idx].status = OBJ_HELD
            robot.balls_held += 1
            robot.held_object_ids.append(best_idx)
            return True
        return False

    def opp_try_score(self, robot: Robot, goal_pos: np.ndarray, points: int,
                       gname: str | None = None, rng: np.random.Generator | None = None,
                       prepend: bool | None = None) -> int:
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0
        if gname is None:
            gname = _goal_name(goal_pos)
        if prepend is None:
            prepend = infer_scoring_prepend(gname, goal_pos)
        n = robot.balls_held
        scored = 0
        for idx in list(robot.held_object_ids):
            self.objects[idx].status = OBJ_SCORED_OPP
            self.objects[idx].scored_in_goal = gname
            ejected = self.goal_state.score_ball(
                gname, idx, self.objects[idx].color, prepend=prepend,
            )
            scored += points
            if ejected is not None and rng is not None:
                ej_idx, ej_color = ejected
                self.spill_overflow_ball(gname, ej_idx, prepend, rng)
                if ej_color == BALL_BLUE:
                    self.my_score = max(0, self.my_score - points)
                else:
                    self.opponent_score = max(0, self.opponent_score - points)
        self.opponent_score += scored
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return scored

    def opp_try_descore_slide(self, robot: Robot, gname: str,
                              start_pt: int, end_pt: int,
                              approach_pos: np.ndarray,
                              rng: np.random.Generator,
                              remove_ally_scored: bool) -> int:
        return self.try_descore_slide(
            robot, gname, start_pt, end_pt, approach_pos, rng,
            remove_ally_scored=remove_ally_scored, blue_alliance=False,
        )

    def opp_try_descore_slam(self, robot: Robot, gname: str, entry_end: int,
                             impact_speed: float, approach_pos: np.ndarray,
                             rng: np.random.Generator,
                             remove_ally_scored: bool) -> int:
        return self.try_descore_slam(
            robot, gname, entry_end, impact_speed, approach_pos, rng,
            remove_ally_scored=remove_ally_scored, blue_alliance=False,
        )

    # ------------------------------------------------------------------
    # Ball rolling physics
    # ------------------------------------------------------------------
    def physics_tick(self, dt: float):
        """Advance ball velocities one sim tick (friction + wall bounce)."""
        for obj in self.objects:
            if obj.status == OBJ_ON_FIELD:
                obj.apply_physics(dt, FIELD_W, FIELD_H)

    def apply_robot_push(self, robot: Robot, prev_pos: np.ndarray):
        """Push nearby on-field balls based on how far the robot moved this tick.

        Intake zone behaviour:
          - intake_active=True  → skip (try_collect handles collection)
          - intake_active=False → treat front face as a soft wall; push ball
            outward (forward) with a gentler impulse so it bounces slowly away.
        """
        delta = robot.position - prev_pos
        robot_speed = np.linalg.norm(delta)

        # Forward unit vector for stopped-intake bounce
        fwd = np.array([math.cos(robot.heading), math.sin(robot.heading)])

        for obj in self.objects:
            if obj.status != OBJ_ON_FIELD:
                continue
            diff = obj.position - robot.position
            dist = np.linalg.norm(diff)
            if dist < 0.5 or dist >= _PUSH_RADIUS:
                continue

            if _in_intake_zone(robot, obj.position):
                if robot.intake_active:
                    # Spinning intake — let try_collect handle this ball
                    continue
                else:
                    # Stopped intake / wall mode: push ball forward out of intake
                    # Impulse is position-based (overlap spring) + any robot motion
                    overlap  = max(0.0, _PUSH_RADIUS - dist)
                    impulse  = (robot_speed * _PUSH_SCALE + overlap * 1.5) * 0.4
                    if impulse > 0.01:
                        obj.vx += fwd[0] * impulse
                        obj.vy += fwd[1] * impulse
                        spd = math.sqrt(obj.vx ** 2 + obj.vy ** 2)
                        cap = _MAX_BALL_SPD * 0.35   # slow bounce cap
                        if spd > cap:
                            obj.vx *= cap / spd
                            obj.vy *= cap / spd
                    continue

            # Normal body-push: away from robot centre
            push_dir = diff / dist
            overlap  = _PUSH_RADIUS - dist
            impulse  = robot_speed * _PUSH_SCALE * (1.0 + overlap / _PUSH_RADIUS)
            obj.vx  += push_dir[0] * impulse
            obj.vy  += push_dir[1] * impulse

            spd = np.sqrt(obj.vx ** 2 + obj.vy ** 2)
            if spd > _MAX_BALL_SPD:
                scale  = _MAX_BALL_SPD / spd
                obj.vx *= scale
                obj.vy *= scale

