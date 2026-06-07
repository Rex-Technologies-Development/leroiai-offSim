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
    VISION_HALF_ANGLE, VISION_RANGE,
    ROBOT_W,
    PARK_ZONE_X_MIN, PARK_ZONE_X_MAX, PARK_ZONE_BOTTOM, PARK_ZONE_TOP,
)

# Margin used to expand obstacles so the path planner routes the robot's
# CENTRE along a path that keeps the whole body clear. Uses the half-diagonal
# (ROBOT_W * √2 / 2) instead of the half-width because when the robot rotates,
# its corners extend further out than its sides — half-width was clipping
# goal bodies on diagonal headings.
_NAV_MARGIN: float = ROBOT_W / 2.0 + 1.0   # ≈ 8.5" (half-width + buffer)

# ---- Goal obstacle definitions for line-of-sight occlusion ----------------
# Long goal AABBs [x_lo, x_hi, y_lo, y_hi]
_LOS_AABB_OBSTACLES = [
    (LONG_GOAL_WALL_GAP, LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH,
     LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX),
    (FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH, FIELD_W - LONG_GOAL_WALL_GAP,
     LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX),
]
# Center X arms: (cx, cy, angle, half_len, half_w)
_LOS_ARM_OBSTACLES = [
    (72.0, 72.0, math.pi / 4,  CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W / 2),
    (72.0, 72.0, -math.pi / 4, CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W / 2),
]


def _seg_hits_aabb(p1x: float, p1y: float, p2x: float, p2y: float,
                   ax0: float, ax1: float, ay0: float, ay1: float) -> bool:
    """Liang-Barsky: does segment p1→p2 intersect AABB [ax0,ax1]×[ay0,ay1]?"""
    dx, dy = p2x - p1x, p2y - p1y
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p1x - ax0), (dx, ax1 - p1x),
                  (-dy, p1y - ay0), (dy, ay1 - p1y)):
        if abs(p) < 1e-9:
            if q < 0:
                return False
        elif p < 0:
            r = q / p
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            r = q / p
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return t0 <= t1


def _seg_hits_arm(p1x: float, p1y: float, p2x: float, p2y: float,
                  cx: float, cy: float, angle: float,
                  half_len: float, half_w: float) -> bool:
    """Does segment p1→p2 intersect a rotated-rectangle obstacle?"""
    ca, sa = math.cos(angle), math.sin(angle)
    def to_local(x: float, y: float):
        dx, dy = x - cx, y - cy
        return dx * ca + dy * sa, dx * (-sa) + dy * ca
    lp1 = to_local(p1x, p1y)
    lp2 = to_local(p2x, p2y)
    return _seg_hits_aabb(lp1[0], lp1[1], lp2[0], lp2[1],
                           -half_len, half_len, -half_w, half_w)


def _los_blocked(rx: float, ry: float, bx: float, by: float,
                 margin: float = 0.0) -> bool:
    """True if the path from (rx,ry) to (bx,by) is blocked by any goal obstacle.

    margin: expand each obstacle outward by this many inches before checking.
    Pass margin=_NAV_MARGIN (7.5") for navigation paths so the full robot body
    is accounted for, not just its centre point.
    """
    for ax0, ax1, ay0, ay1 in _LOS_AABB_OBSTACLES:
        if _seg_hits_aabb(rx, ry, bx, by,
                          ax0 - margin, ax1 + margin,
                          ay0 - margin, ay1 + margin):
            return True
    for cx, cy, angle, hl, hw in _LOS_ARM_OBSTACLES:
        if _seg_hits_arm(rx, ry, bx, by, cx, cy, angle,
                         hl + margin, hw + margin):
            return True
    return False


# ---- Parking zones (soft-avoid) -------------------------------------------
# The top/bottom-centre park platforms are NOT hard obstacles — robots park on
# them at endgame — but during play we PREFER not to drive through them. The nav
# planner treats them as soft obstacles (a routing penalty) and the collection
# scorer deprioritises balls inside them. Flip _PARK_AVOID off to ignore parks.
_PARK_AVOID  = True
_PARK_MARGIN = ROBOT_W / 2.0   # 7.5" — keep the robot BODY out of the platform
_PARK_AABBS  = [
    (PARK_ZONE_X_MIN, PARK_ZONE_X_MAX, 0.0,           PARK_ZONE_BOTTOM),  # bottom
    (PARK_ZONE_X_MIN, PARK_ZONE_X_MAX, PARK_ZONE_TOP, FIELD_H),           # top
]


def point_in_park(x: float, y: float, margin: float = _PARK_MARGIN) -> bool:
    """True if (x, y) is inside (or within `margin` of) a park platform."""
    for ax0, ax1, ay0, ay1 in _PARK_AABBS:
        if (ax0 - margin <= x <= ax1 + margin and
                ay0 - margin <= y <= ay1 + margin):
            return True
    return False


def seg_crosses_park(x0: float, y0: float, x1: float, y1: float,
                     margin: float = _PARK_MARGIN) -> bool:
    """True if the segment (x0,y0)->(x1,y1) passes through a park platform."""
    for ax0, ax1, ay0, ay1 in _PARK_AABBS:
        if _seg_hits_aabb(x0, y0, x1, y1,
                          ax0 - margin, ax1 + margin,
                          ay0 - margin, ay1 + margin):
            return True
    return False


# ---- Proximity thresholds --------------------------------------------------
_WALL_MARGIN         = 24.0   # 1 tile from any wall → "near a wall"
_GOAL_MARGIN         = 16.0   # inches from a goal center → "near goal"
_CLUSTER_DIST        = 36.0   # scoring cluster distance (strategic grouping)
_CLUSTER_PICKUP_DIST = 15.0   # balls this close → single drive-to waypoint
_MIN_ROUTE_VALUE     = 0.06   # score/dist ratio below this → stop adding waypoints

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

    # Long goal body bounding boxes — pad matches env's _GOAL_MARGIN (8.5") so
    # that balls inside the physical exclusion zone are never added as targets.
    # A robot approaching such a ball would be pushed back by _resolve_goal_collisions
    # every tick and oscillate against the goal face.
    pad = 8.5
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


def _in_vision(robot, ball_x: float, ball_y: float) -> bool:
    """True if the ball is within the robot's vision cone AND not occluded
    by any goal obstacle."""
    dx = ball_x - robot.x
    dy = ball_y - robot.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > VISION_RANGE:
        return False
    angle_to = math.atan2(dy, dx)
    diff = abs(((robot.heading - angle_to + math.pi) % (2 * math.pi)) - math.pi)
    if diff > VISION_HALF_ANGLE:
        return False
    # Goal structures block line of sight
    if _los_blocked(robot.x, robot.y, ball_x, ball_y):
        return False
    return True


def _los_blocked_by_arms(rx: float, ry: float, bx: float, by: float,
                         margin: float = 0.0) -> bool:
    """True if the segment is blocked by a center-X arm only (ignores long goals)."""
    for cx, cy, angle, hl, hw in _LOS_ARM_OBSTACLES:
        if _seg_hits_arm(rx, ry, bx, by, cx, cy, angle, hl + margin, hw + margin):
            return True
    return False


def goal_in_fov(robot, gx: float, gy: float, blocked_by_arms: bool = True) -> bool:
    """True if the goal point (gx, gy) is within the robot's camera FOV.

    Unlike _in_vision() (used for balls), this does NOT treat long-goal bodies as
    occluders — the robot is looking AT a goal, so a goal must not self-occlude.
    The center X still blocks long-goal sightlines across the field, so callers
    pass blocked_by_arms=True for long goals (and False for the center goals,
    which ARE the X structure). Used to update the robot's perceived goal state.
    """
    dx = gx - robot.x
    dy = gy - robot.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > VISION_RANGE:
        return False
    angle_to = math.atan2(dy, dx)
    diff = abs(((robot.heading - angle_to + math.pi) % (2 * math.pi)) - math.pi)
    if diff > VISION_HALF_ANGLE:
        return False
    if blocked_by_arms and _los_blocked_by_arms(robot.x, robot.y, gx, gy):
        return False
    return True


def _score_ball(obj, on_field_objs: list) -> float:
    """Return strategic desirability score for collecting this ball.

    Returns -1.0 if the ball should be skipped entirely.
    Higher score = higher priority.

    Blue-alliance allies never route to red balls — the intake refuses
    them anyway, so targeting them just wastes a decision.
    """
    # Rule 0: reject wrong-color balls — allies (blue) only collect blue
    if obj.color != BALL_BLUE:
        return -1.0

    x, y = obj.x, obj.y
    nw = _near_wall_count(x, y)

    # Rule 1: skip corners (near 2+ walls)
    if nw >= 2:
        return -1.0

    # Rule 1: skip balls inside / adjacent to goal structures
    if _near_any_goal(x, y):
        return -1.0

    same, opp = _cluster_counts(obj, on_field_objs)

    # Rule 2: blue clusters
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

    # Rule 3: reduce priority for balls against one wall
    if nw == 1:
        base *= 0.6

    # Soft park-zone avoidance: heavily deprioritise balls sitting on a park
    # platform — collecting one means driving in, which we prefer not to do, so
    # the robot picks them up only when nothing better is left.
    if _PARK_AVOID and point_in_park(x, y, margin=0.0):
        base *= 0.1

    return base


# ---- Cluster builder -------------------------------------------------------

def _build_pickup_clusters(
    candidates: list,   # [(field_idx, obj, score), ...]
) -> list[tuple[list[int], np.ndarray, float]]:
    """Group nearby candidates into single-waypoint clusters.

    Two balls belong to the same cluster when any pair within the group is
    within _CLUSTER_PICKUP_DIST of each other (single-linkage).

    Returns list of (field_indices, centroid_pos, combined_score).
    Single-ball groups are returned as clusters of size 1.
    """
    unvisited = list(candidates)   # copy
    clusters: list[tuple[list[int], np.ndarray, float]] = []

    while unvisited:
        # Seed a new cluster with the first unvisited ball
        cluster_entries = [unvisited.pop(0)]
        changed = True
        while changed:
            changed = False
            still_outside = []
            for candidate in unvisited:
                _, c_obj, _ = candidate
                # Check against every member already in this cluster
                merged = False
                for _, m_obj, _ in cluster_entries:
                    if math.hypot(c_obj.x - m_obj.x, c_obj.y - m_obj.y) <= _CLUSTER_PICKUP_DIST:
                        cluster_entries.append(candidate)
                        merged = True
                        changed = True
                        break
                if not merged:
                    still_outside.append(candidate)
            unvisited = still_outside

        idxs  = [e[0] for e in cluster_entries]
        cx    = sum(e[1].x for e in cluster_entries) / len(cluster_entries)
        cy    = sum(e[1].y for e in cluster_entries) / len(cluster_entries)
        total = sum(e[2] for e in cluster_entries)
        clusters.append((idxs, np.array([cx, cy]), total))

    return clusters


# ---- Public API ------------------------------------------------------------

def compute_collection_route(
    robot_pos: np.ndarray,
    field,
    already_held: int = 0,
    max_volley: int = 5,
    robot=None,
    y_min: float | None = None,
    y_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[tuple[list[int], np.ndarray, float]]:
    """Return an ordered list of (ball_indices, waypoint_pos, score).

    Nearby balls are merged into cluster waypoints so the robot drives to a
    single centroid to collect all of them, rather than threading each center.

    Only considers balls visible within the robot's vision cone (if robot is
    provided). Returns at most max_volley waypoints.
    """
    slots_left = min(MAX_CARRY - already_held, max_volley - already_held)
    if slots_left <= 0:
        return []

    on_field = [o for o in field.objects if o.status == OBJ_ON_FIELD]
    if not on_field:
        return []

    # Score all eligible, navigable balls.
    # LOS check is always applied (even without a robot heading) so the planner
    # never routes toward balls that are physically behind goal structures.
    def _make_candidates(pool):
        result = []
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        for i, obj in enumerate(field.objects):
            if obj.status != OBJ_ON_FIELD:
                continue
            if y_min is not None and y_max is not None:
                if obj.y < y_min or obj.y > y_max:
                    continue
            if x_min is not None and x_max is not None:
                if obj.x < x_min or obj.x > x_max:
                    continue
            # Skip balls whose direct path is blocked when accounting for robot width
            if _los_blocked(rx, ry, obj.x, obj.y, margin=_NAV_MARGIN):
                continue
            # Vision cone + range check only when a robot object is provided
            if robot is not None and not _in_vision(robot, obj.x, obj.y):
                continue
            s = _score_ball(obj, pool)
            if s > 0:
                result.append((i, obj, s))
        return result

    candidates = _make_candidates(on_field)
    if not candidates:
        return []

    # Group into pickup clusters (nearby balls → single waypoint)
    clusters = _build_pickup_clusters(candidates)

    route: list[tuple[list[int], np.ndarray, float]] = []
    used_indices: set[int] = set()
    current_pos = robot_pos.copy()
    waypoints_added = 0

    for _ in range(len(clusters)):
        if waypoints_added >= max_volley:
            break

        # Filter clusters that still have unused balls
        available = [
            (idxs, pos, sc)
            for idxs, pos, sc in clusters
            if any(i not in used_indices for i in idxs)
        ]
        if not available:
            break

        best_j   = -1
        best_val = -1.0
        for j, (idxs, pos, sc) in enumerate(available):
            dist = float(np.linalg.norm(pos - current_pos))
            val  = sc / (dist + 20.0)
            if val > best_val:
                best_val = val
                best_j   = j

        if best_j < 0 or best_val < _MIN_ROUTE_VALUE:
            break

        idxs, pos, sc = available[best_j]
        fresh = [i for i in idxs if i not in used_indices]
        route.append((fresh, pos, sc))
        used_indices.update(fresh)
        current_pos = pos
        waypoints_added += 1

    return route
