"""Field state model — objects, goals, zones, robot positions.

Central state container for the entire VEX AI field. Owns all
robots and game objects. The env (env.py) orchestrates stepping;
the field tracks state.
"""

from __future__ import annotations
import numpy as np

from sim.config import (
    FIELD_W, FIELD_H, MAX_GAME_OBJECTS, INITIAL_OBJECT_POSITIONS,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP,
    OUR_LONG_GOAL, OUR_MID_GOAL, OPP_LONG_GOAL, OPP_MID_GOAL,
    LONG_GOAL_POINTS, MID_GOAL_POINTS,
    COLLECT_RANGE, SCORE_RANGE, MAX_CARRY,
)
from sim.robot import Robot
from sim.game_object import GameObject
from sim.heatmap import compute_heatmap


class Field:
    """Manages all field state: 4 robots + 24 game objects + scores."""

    def __init__(self):
        # Allied robots
        self.allies: list[Robot] = [
            Robot(x=12.00, y=12.00, heading=0.0, role_id=0),
            Robot(x=12.00, y=132.00, heading=0.0, role_id=1),
        ]
        # Opponent robots
        self.opponents: list[Robot] = [
            Robot(x=132.00, y=132.00, heading=np.pi, role_id=0),
            Robot(x=132.00, y=12.00, heading=np.pi, role_id=1),
        ]
        # Game objects
        self.objects: list[GameObject] = []
        for i in range(MAX_GAME_OBJECTS):
            pos = INITIAL_OBJECT_POSITIONS[i]
            self.objects.append(GameObject(i, pos[0], pos[1]))

        self.my_score: int = 0
        self.opponent_score: int = 0
        self.time_remaining: float = 0.0

    def reset(self, rng: np.random.Generator):
        """Reset field to starting positions."""
        self.allies[0].reset(12.00, 12.00, heading=0.0)
        self.allies[1].reset(12.00, 132.00, heading=0.0)
        self.opponents[0].reset(132.00, 132.00, heading=np.pi)
        self.opponents[1].reset(132.00, 12.00, heading=np.pi)

        for i, obj in enumerate(self.objects):
            pos = INITIAL_OBJECT_POSITIONS[i]
            obj.reset(pos[0], pos[1])

        self.my_score = 0
        self.opponent_score = 0

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
        """Count on-field objects within radius inches of pos."""
        count = 0
        for obj in self.objects:
            if obj.status == OBJ_ON_FIELD:
                if np.linalg.norm(obj.position - pos) < radius:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Action effects
    # ------------------------------------------------------------------
    def try_collect(self, robot: Robot) -> bool:
        """Try to pick up the nearest on-field object. Returns True if collected."""
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
        """Try to score held objects at goal. Returns points scored."""
        if robot.balls_held <= 0:
            return 0
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        n = robot.balls_held
        scored = points * n
        self.my_score += scored
        for idx in robot.held_object_ids:
            self.objects[idx].status = OBJ_SCORED_US
        robot.held_object_ids.clear()
        robot.balls_held = 0
        return scored

    def try_descore(self, robot: Robot, goal_pos: np.ndarray, rng: np.random.Generator) -> int:
        """Try to remove an opponent-scored object from goal. Returns points removed."""
        if np.linalg.norm(robot.position - goal_pos) >= SCORE_RANGE:
            return 0

        opp_scored = self.scored_by_opp_indices()
        if len(opp_scored) == 0:
            return 0

        idx = opp_scored[0]
        self.objects[idx].status = OBJ_ON_FIELD
        # Drop near goal
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos
        pts = MID_GOAL_POINTS
        self.opponent_score = max(0, self.opponent_score - pts)
        return pts

    def nearest_on_field_target(self, pos: np.ndarray) -> np.ndarray | None:
        """Return position of nearest on-field object, or None."""
        on_field = self.on_field_indices()
        if len(on_field) == 0:
            return None
        positions = self.get_obj_positions()[on_field]
        dists = np.linalg.norm(positions - pos, axis=1)
        return positions[np.argmin(dists)].copy()

    # ------------------------------------------------------------------
    # Opponent actions
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
        for idx in robot.held_object_ids:
            self.objects[idx].status = OBJ_SCORED_OPP
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
        drop_pos = goal_pos + rng.uniform(-8.0, 8.0, size=2)
        drop_pos = np.clip(drop_pos, [0.0, 0.0], [FIELD_W, FIELD_H])
        self.objects[idx].position = drop_pos
        pts = MID_GOAL_POINTS
        self.my_score = max(0, self.my_score - pts)
        return pts
