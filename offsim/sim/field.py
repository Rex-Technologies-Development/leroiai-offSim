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
)

# How close a ball must be to a tube to be considered "at the tube"
_TUBE_SNAP_DIST = MATCHLOAD_TUBE_RADIUS * 3.0
# Heading tolerance for facing a matchload tube (radians)
_TUBE_FACE_TOL  = math.radians(30.0)
from sim.game_object import BALL_RADIUS

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

    def score_ball(self, gname: str, ball_idx: int, color: int):
        """Append a ball to this goal (becomes the new north/right outer)."""
        self._list(gname).append((ball_idx, color))

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

        Quadrant layout (y=0 at bottom of field):
          bottom-left:  opp_long south-outer  & center_low left-outer
          bottom-right: our_long south-outer  & center_low right-outer
          top-left:     opp_long north-outer  & center_mid left-outer
          top-right:    our_long north-outer  & center_mid right-outer
        """
        opp_s, opp_n = self.outer_colors("opp_long")
        our_s, our_n = self.outer_colors("our_long")
        low_l, low_r = self.outer_colors("center_low")
        mid_l, mid_r = self.outer_colors("center_mid")

        return {
            "bottom_left":  opp_s if (opp_s is not None and opp_s == low_l) else None,
            "bottom_right": our_s if (our_s is not None and our_s == low_r) else None,
            "top_left":     opp_n if (opp_n is not None and opp_n == mid_l) else None,
            "top_right":    our_n if (our_n is not None and our_n == mid_r) else None,
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

        # Rebuild from INITIAL_OBJECTS (removes any editor additions)
        self.objects = []
        for i in range(len(INITIAL_OBJECTS)):
            row = INITIAL_OBJECTS[i]
            self.objects.append(GameObject(i, row[0], row[1], int(row[2])))

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
        """Pick up the nearest ball inside the front-face intake zone.

        Only the front face of the robot (full width, _INTAKE_DEPTH reach) can
        collect. The other three sides only push. Tube balls still require the
        robot to be facing the tube within 30°.
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
        """Score held balls at goal. Returns points scored."""
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        n = robot.balls_held
        scored = points * n
        self.my_score += scored
        gname = _goal_name(goal_pos)
        for idx in robot.held_object_ids:
            self.objects[idx].status = OBJ_SCORED_US
            self.objects[idx].scored_in_goal = gname
            self.goal_state.score_ball(gname, idx, self.objects[idx].color)
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return scored

    def try_score_one(self, robot: Robot, goal_pos: np.ndarray, points: int) -> int:
        """Score exactly one held ball at goal. Returns points scored (0 if none)."""
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        idx = robot.held_object_ids.pop(0)
        robot.balls_held -= 1
        self.objects[idx].status = OBJ_SCORED_US
        gname = _goal_name(goal_pos)
        self.objects[idx].scored_in_goal = gname
        self.goal_state.score_ball(gname, idx, self.objects[idx].color)
        self.my_score += points
        return points

    def try_descore(self, robot: Robot, goal_pos: np.ndarray, rng: np.random.Generator) -> int:
        """Remove an opponent-scored ball from goal. Returns points removed."""
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        opp_scored = self.scored_by_opp_indices()
        if len(opp_scored) == 0:
            return 0

        idx = opp_scored[0]
        gname_remove = self.objects[idx].scored_in_goal
        self.objects[idx].status = OBJ_ON_FIELD
        self.objects[idx].scored_in_goal = ""
        self.goal_state.remove_ball(gname_remove, idx)
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos
        self.opponent_score = max(0, self.opponent_score - CENTER_GOAL_POINTS)
        return CENTER_GOAL_POINTS

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
