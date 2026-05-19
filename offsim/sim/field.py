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
    COLLECT_RANGE, SCORE_RANGE, MAX_CARRY, ROBOT_W,
    BALL_RED, BALL_BLUE,
    MATCHLOAD_TUBES, MATCHLOAD_TUBE_RADIUS,
    LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX, LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH,
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W,
)
from sim.game_object import BALL_RADIUS

# ---------------------------------------------------------------------------
# Ball-position randomisation
# ---------------------------------------------------------------------------
# Per-episode jitter so each match is a slightly different scenario — keeps
# the policy from overfitting to the 44 fixed INITIAL_OBJECTS coordinates.
_JITTER_RADIUS = 12.0  # inches — max offset from anchor position
_JITTER_ATTEMPTS = 6   # if jittered pos lands on a goal, try again N times
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
from sim.game_object import GameObject
from sim.heatmap import compute_heatmap


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

        Capacities: long goals = 14 balls, center goals = 7 balls.
        If full, the ball at the OPPOSITE end is ejected (rolls out).
        Returns (ejected_ball_idx, ejected_color) on overflow, else None.
        """
        _LONG_CAP   = 14
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
        # Allied robots — blue team, start right side
        self.allies: list[Robot] = [
            Robot(x=108.00, y=18.00,  heading=np.pi, role_id=0),
            Robot(x=108.00, y=126.00, heading=np.pi, role_id=1),
        ]
        # Opponent robots — red team, start left side
        self.opponents: list[Robot] = [
            Robot(x=24.00, y=126.00, heading=0.0, role_id=0),
            Robot(x=24.00, y=18.00,  heading=0.0, role_id=1),
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
        self.allies[0].reset(108.00,  18.00, heading=np.pi)
        self.allies[1].reset(108.00, 126.00, heading=np.pi)
        self.opponents[0].reset(24.00, 126.00, heading=0.0)
        self.opponents[1].reset(24.00,  18.00, heading=0.0)

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

    def try_collect(self, robot: Robot) -> bool:
        """Pick up the nearest ball (any color) inside the front-face intake zone.

        Intake is colorblind — the *policy* learns to avoid red balls through
        the collect_red / holding_wrong_color penalties, and the route planner
        only targets blue balls so the robot rarely drives toward reds in the
        first place. If a red ball ends up in the intake (e.g. opp pushed it
        in front of us), EJECT_WRONG_COLOR clears it.

        Only the front face of the robot (full width, _INTAKE_DEPTH reach) can
        collect. Tube balls still require the robot to face the tube within 30°.
        """
        if robot.balls_held >= MAX_CARRY:
            return False

        best_idx, best_dist = -1, float("inf")
        for i, obj in enumerate(self.objects):
            if obj.status == OBJ_ON_FIELD and _in_intake_zone(robot, obj.position):
                d = np.linalg.norm(obj.position - robot.position)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0:
            tube = self._ball_at_tube(self.objects[best_idx])
            if tube is not None and not self._facing_toward(robot, tube):
                return False   # must face the tube head-on
            self.objects[best_idx].status = OBJ_HELD
            robot.balls_held += 1
            robot.held_object_ids.append(best_idx)
            return True
        return False

    def try_score(self, robot: Robot, goal_pos: np.ndarray, points: int) -> int:
        """Score held balls at goal. Ball color determines which team scores.

        Blue balls → my_score, Red balls → opponent_score.
        Returns total points added across both teams.
        """
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        gname = _goal_name(goal_pos)
        total = 0
        for idx in robot.held_object_ids:
            self.objects[idx].status       = OBJ_SCORED_US
            self.objects[idx].scored_in_goal = gname
            color = self.objects[idx].color
            self.goal_state.score_ball(gname, idx, color)
            if color == BALL_BLUE:
                self.my_score += points
            else:
                self.opponent_score += points
            total += points
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return total

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

    def try_descore(self, robot: Robot, goal_pos: np.ndarray, rng: np.random.Generator) -> int:
        """Remove an opponent-scored ball from the SPECIFIC targeted goal.

        Returns points removed (0 if out of range or no opp ball in that goal).
        Only removes balls whose .scored_in_goal matches the goal_pos —
        the previous version removed any opp-scored ball anywhere on the field,
        which made the cause-effect chain meaningless for the policy.
        """
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        target_gname = _goal_name(goal_pos)
        # Only consider opp-scored balls actually sitting in the targeted goal
        candidates = [
            i for i, obj in enumerate(self.objects)
            if obj.status == OBJ_SCORED_OPP and obj.scored_in_goal == target_gname
        ]
        if not candidates:
            return 0

        idx = candidates[0]
        self.objects[idx].status = OBJ_ON_FIELD
        self.objects[idx].scored_in_goal = ""
        self.goal_state.remove_ball(target_gname, idx)
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos

        # Point value matches the goal type — long goals are worth different
        # points than center goals.
        if target_gname in ("our_long", "opp_long"):
            pts = LONG_GOAL_POINTS
        else:
            pts = CENTER_GOAL_POINTS
        self.opponent_score = max(0, self.opponent_score - pts)
        return pts

    def nearest_on_field_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return position of nearest on-field ball, or None (no obstacle check)."""
        on_field = self.on_field_indices()
        if len(on_field) == 0:
            return None
        positions = self.get_obj_positions()[on_field]
        dists = np.linalg.norm(positions - pos, axis=1)
        return positions[np.argmin(dists)].copy()

    def nearest_navigable_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return nearest on-field ball whose direct path is NOT blocked by a goal.

        Uses the route planner's LOS check to skip balls behind goal structures.
        Falls back to nearest_on_field_target if every ball is behind a goal (rare).
        """
        from sim.route_planner import _los_blocked, _NAV_MARGIN
        rx, ry = float(pos[0]), float(pos[1])
        best_pos: np.ndarray | None = None
        best_dist = float("inf")
        for obj in self.objects:
            if obj.status != OBJ_ON_FIELD:
                continue
            if _los_blocked(rx, ry, obj.x, obj.y, margin=_NAV_MARGIN):
                continue
            d = float(np.linalg.norm(obj.position - pos))
            if d < best_dist:
                best_dist = d
                best_pos = obj.position.copy()
        return best_pos if best_pos is not None else self.nearest_on_field_target(pos)

    # ------------------------------------------------------------------
    # Opponent action effects
    # ------------------------------------------------------------------
    def opp_try_collect(self, robot: Robot, rng: np.random.Generator) -> bool:
        if robot.balls_held >= MAX_CARRY:
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

    def opp_try_score(self, robot: Robot, goal_pos: np.ndarray, points: int) -> int:
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0
        n = robot.balls_held
        scored = points * n
        self.opponent_score += scored
        gname = _goal_name(goal_pos)
        for idx in robot.held_object_ids:
            self.objects[idx].status = OBJ_SCORED_OPP
            self.objects[idx].scored_in_goal = gname
            self.goal_state.score_ball(gname, idx, self.objects[idx].color)
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return scored

    def opp_try_descore(self, robot: Robot, goal_pos: np.ndarray, rng: np.random.Generator) -> int:
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0
        us_scored = self.scored_by_us_indices()
        if len(us_scored) == 0:
            return 0
        idx = us_scored[0]
        gname_remove = self.objects[idx].scored_in_goal
        self.objects[idx].status = OBJ_ON_FIELD
        self.objects[idx].scored_in_goal = ""
        self.goal_state.remove_ball(gname_remove, idx)
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos
        self.my_score = max(0, self.my_score - CENTER_GOAL_POINTS)
        return CENTER_GOAL_POINTS

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
