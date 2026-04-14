"""Field state model — objects, goals, zones, robot positions.

VEX Push Back 2025-2026.
Central state container. Owns all robots and game objects.
The env (env.py) orchestrates stepping; the field tracks state.
"""

from __future__ import annotations
import numpy as np

from sim.config import (
    FIELD_W, FIELD_H, MAX_GAME_OBJECTS, INITIAL_OBJECTS,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP, OBJ_REMOVED,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    LONG_GOAL_POINTS, CENTER_GOAL_POINTS,
    COLLECT_RANGE, SCORE_RANGE, MAX_CARRY, ROBOT_W,
    BALL_RED, BALL_BLUE,
)
from sim.game_object import BALL_RADIUS

# Robot push constants
_PUSH_RADIUS  = ROBOT_W / 2 + BALL_RADIUS + 1.0  # inches — overlap triggers push
_PUSH_SCALE   = 22.0   # in/s impulse per in/s of robot speed
_MAX_BALL_SPD = 120.0  # in/s cap so balls don't fly off the field

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


class Field:
    """Manages all field state: 4 robots + 44 game objects + scores."""

    def __init__(self):
        # Allied robots — blue team, start right side
        self.allies: list[Robot] = [
            Robot(x=120.00, y=18.00,  heading=np.pi, role_id=0),
            Robot(x=120.00, y=126.00, heading=np.pi, role_id=1),
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

    def reset(self, rng: np.random.Generator):
        """Reset field to starting positions (clears any editor changes)."""
        self.allies[0].reset(120.00,  18.00, heading=np.pi)
        self.allies[1].reset(120.00, 126.00, heading=np.pi)
        self.opponents[0].reset(24.00, 126.00, heading=0.0)
        self.opponents[1].reset(24.00,  18.00, heading=0.0)

        # Rebuild from INITIAL_OBJECTS (removes any editor additions)
        self.objects = []
        for i in range(len(INITIAL_OBJECTS)):
            row = INITIAL_OBJECTS[i]
            self.objects.append(GameObject(i, row[0], row[1], int(row[2])))

        self.my_score = 0
        self.opponent_score = 0

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
            self.objects[idx].status = OBJ_REMOVED

    def change_ball_color(self, idx: int):
        """Toggle a ball's color between red and blue."""
        if 0 <= idx < len(self.objects):
            obj = self.objects[idx]
            obj.color = BALL_BLUE if obj.color == BALL_RED else BALL_RED

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
    def try_collect(self, robot: Robot) -> bool:
        """Pick up nearest on-field ball. Returns True if collected."""
        if robot.balls_held >= MAX_CARRY:
            return False

        best_idx, best_dist = -1, float("inf")
        for i, obj in enumerate(self.objects):
            if obj.status == OBJ_ON_FIELD:
                d = np.linalg.norm(obj.position - robot.position)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0 and best_dist < COLLECT_RANGE:
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
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return scored

    def try_descore(self, robot: Robot, goal_pos: np.ndarray, rng: np.random.Generator) -> int:
        """Remove an opponent-scored ball from goal. Returns points removed."""
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        opp_scored = self.scored_by_opp_indices()
        if len(opp_scored) == 0:
            return 0

        idx = opp_scored[0]
        self.objects[idx].status = OBJ_ON_FIELD
        self.objects[idx].scored_in_goal = ""
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos
        self.opponent_score = max(0, self.opponent_score - CENTER_GOAL_POINTS)
        return CENTER_GOAL_POINTS

    def nearest_on_field_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return position of nearest on-field ball, or None."""
        on_field = self.on_field_indices()
        if len(on_field) == 0:
            return None
        positions = self.get_obj_positions()[on_field]
        dists = np.linalg.norm(positions - pos, axis=1)
        return positions[np.argmin(dists)].copy()

    # ------------------------------------------------------------------
    # Opponent action effects
    # ------------------------------------------------------------------
    def opp_try_collect(self, robot: Robot, rng: np.random.Generator) -> bool:
        if robot.balls_held >= MAX_CARRY:
            return False
        best_idx, best_dist = -1, float("inf")
        for i, obj in enumerate(self.objects):
            if obj.status == OBJ_ON_FIELD:
                d = np.linalg.norm(obj.position - robot.position)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        if best_idx >= 0 and best_dist < COLLECT_RANGE:
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
        self.objects[idx].status = OBJ_ON_FIELD
        self.objects[idx].scored_in_goal = ""
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
        """Push nearby on-field balls based on how far the robot moved this tick."""
        delta = robot.position - prev_pos
        robot_speed = np.linalg.norm(delta)
        if robot_speed < 0.01:
            return

        for obj in self.objects:
            if obj.status != OBJ_ON_FIELD:
                continue
            diff = obj.position - robot.position
            dist = np.linalg.norm(diff)
            if dist < 0.5 or dist >= _PUSH_RADIUS:
                continue

            # Push direction: away from robot centre
            push_dir = diff / dist
            # Scale by overlap and robot speed
            overlap  = _PUSH_RADIUS - dist
            impulse  = robot_speed * _PUSH_SCALE * (1.0 + overlap / _PUSH_RADIUS)
            obj.vx  += push_dir[0] * impulse
            obj.vy  += push_dir[1] * impulse

            # Cap so balls don't teleport
            spd = np.sqrt(obj.vx ** 2 + obj.vy ** 2)
            if spd > _MAX_BALL_SPD:
                scale  = _MAX_BALL_SPD / spd
                obj.vx *= scale
                obj.vy *= scale
