"""Ball collection route planner for VEX Push Back 2025-2026.

Scores on-field balls by strategic value and returns an ordered
waypoint list for the robot to follow to collect efficiently.

Priority rules (highest to lowest):
  1. Skip balls in corners (near 2+ walls) or adjacent to goal structures
  2. Prefer our-color (BALL_BLUE) balls in clusters
  3. Wall balls (against exactly 1 wall) at reduced priority
  4. Single balls only if isolated or ≤1 opponent ball nearby
  5. Cap route at max_volley (target volley size)

Route ordering: greedy score/distance — highest (score / dist+20) wins.
"""

from __future__ import annotations
import math
import numpy as np

from sim.config import (
    FIELD_W, FIELD_H, BALL_BLUE, BALL_RED, OBJ_ON_FIELD, MAX_CARRY,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX,
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W,
    MATCHLOAD_TUBES, MATCHLOAD_TUBE_RADIUS,
)

# ---- Proximity thresholds --------------------------------------------------
_WALL_MARGIN  = 24.0    # 1 tile from any wall → "near a wall"
_GOAL_MARGIN  = 16.0    # inches from a goal center → "near goal"
_CLUSTER_DIST = 36.0    # inches — balls within this are "in the same cluster"

# Pre-computed goal extents (long goal bodies)
_R_X_LO = FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH   # inner face right goal
_R_X_HI = FIELD_W - LONG_GOAL_WALL_GAP                      # outer face right goal
_L_X_LO = LONG_GOAL_WALL_GAP                                 # outer face left goal
_L_X_HI = LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH               # inner face left goal
_G_Y_LO = LONG_GOAL_Y_MIN
_G_Y_HI = LONG_GOAL_Y_MAX

_GOAL_CENTERS = [OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL]

# Center X geometry for proximity check
_CX, _CY = 72.0, 72.0
_ARM_LEN  = CENTER_GOAL_ARM_LEN
_ARM_HW   = CENTER_GOAL_ARM_W / 2


# ---- Helpers ---------------------------------------------------------------

def _near_wall_count(x: float, y: float) -> int:
    """Return number of field walls within _WALL_MARGIN of this position."""
    n = 0
    if x < _WALL_MARGIN:              n += 1
    if x > FIELD_W - _WALL_MARGIN:    n += 1
    if y < _WALL_MARGIN:              n += 1
    if y > FIELD_H - _WALL_MARGIN:    n += 1
    return n


def _near_any_goal(x: float, y: float) -> bool:
    """True if position is within _GOAL_MARGIN of any goal or inside a goal body."""
    pos = np.array([x, y])

    for g in _GOAL_CENTERS:
        if np.linalg.norm(pos - g) < _GOAL_MARGIN:
            return True

    # Long goal body bounding boxes (with small margin)
    pad = 4.0
    if (_R_X_LO - pad <= x <= _R_X_HI + pad and _G_Y_LO - pad <= y <= _G_Y_HI + pad):
        return True
    if (_L_X_LO - pad <= x <= _L_X_HI + pad and _G_Y_LO - pad <= y <= _G_Y_HI + pad):
        return True

    # Center X arms (rotated AABBs)
    dx, dy = x - _CX, y - _CY
    for angle in (math.pi / 4, -math.pi / 4):
        ca, sa = math.cos(angle), math.sin(angle)
        along = dx * ca + dy * sa
        perp  = dx * (-sa) + dy * ca
        if abs(along) < _ARM_LEN + pad and abs(perp) < _ARM_HW + pad:
            return True

    return False


def _cluster_counts(obj, on_field_objs):
    """Return (same_color_nearby, opp_color_nearby) within _CLUSTER_DIST."""
    pos = np.array([obj.x, obj.y])
    same = opp = 0
    for other in on_field_objs:
        if other is obj:
            continue
        d = np.linalg.norm(np.array([other.x, other.y]) - pos)
        if d < _CLUSTER_DIST:
            if other.color == obj.color:
                same += 1
            else:
                opp += 1
    return same, opp


def _score_ball(obj, on_field_objs: list) -> float:
    """Return strategic desirability score for collecting this ball.

    Returns -1.0 if the ball should be skipped entirely.
    Higher score = higher priority.
    """
    x, y = obj.x, obj.y
    nw = _near_wall_count(x, y)

    # Rule 1: skip corners (near 2+ walls)
    if nw >= 2:
        return -1.0

    # Rule 1: skip balls inside / adjacent to goal structures
    if _near_any_goal(x, y):
        return -1.0

    same, opp = _cluster_counts(obj, on_field_objs)

    # Rule 2: our-color (blue) clusters
    if obj.color == BALL_BLUE:
        if same >= 3:
            base = 12.0 + same     # big cluster — top priority
        elif same >= 1:
            base = 8.0 + same      # small cluster
        else:
            # Rule 4: single ball
            if opp == 0:
                base = 5.0         # isolated
            elif opp == 1:
                base = 3.5         # one opponent ball nearby
            else:
                base = 0.5         # surrounded — low value
    else:
        # Red ball scoring
        if same >= 2:
            base = 4.0
        elif opp <= 1:
            base = 2.0
        else:
            base = 0.5

    # Rule 3: reduce priority for balls against one wall
    if nw == 1:
        base *= 0.6

    return base


# ---- Public API ------------------------------------------------------------

def compute_collection_route(
    robot_pos: np.ndarray,
    field,
    already_held: int = 0,
    max_volley: int = 5,
) -> list[tuple[int, float]]:
    """Return an ordered list of (ball_index, score) representing the best
    collection route starting from robot_pos.

    Picks up to min(MAX_CARRY - already_held, max_volley - already_held) balls.
    Uses greedy selection weighted by score / (distance + 20).
    """
    slots_left = min(MAX_CARRY - already_held, max_volley - already_held)
    if slots_left <= 0:
        return []

    on_field = [o for o in field.objects if o.status == OBJ_ON_FIELD]
    if not on_field:
        return []

    # Score all eligible balls
    def _make_candidates(pool):
        result = []
        for i, obj in enumerate(field.objects):
            if obj.status != OBJ_ON_FIELD:
                continue
            s = _score_ball(obj, pool)
            if s > 0:
                result.append((i, obj, s))
        return result

    available = _make_candidates(on_field)
    if not available:
        return []

    route: list[tuple[int, float]] = []
    current_pos = robot_pos.copy()

    for _ in range(slots_left):
        if not available:
            break

        best_j   = -1
        best_val = -1.0
        for j, (idx, obj, s) in enumerate(available):
            dist = float(np.linalg.norm(np.array([obj.x, obj.y]) - current_pos))
            val  = s / (dist + 20.0)
            if val > best_val:
                best_val = val
                best_j   = j

        if best_j < 0:
            break

        idx, obj, s = available.pop(best_j)
        route.append((idx, s))
        current_pos = np.array([obj.x, obj.y])

        # Re-score remaining with updated cluster context
        remaining_pool = [entry[1] for entry in available]
        new_available  = []
        for entry_idx, entry_obj, _ in available:
            new_s = _score_ball(entry_obj, remaining_pool)
            if new_s > 0:
                new_available.append((entry_idx, entry_obj, new_s))
        available = new_available

    return route
