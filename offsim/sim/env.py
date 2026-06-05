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
    N_NEAREST_BLUE, N_NEAREST_RED,
    FIELD_W, FIELD_H, MATCH_DURATION, DT, TICKS_PER_DECISION, MAX_CARRY,
    MAX_GAME_OBJECTS, OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP,
    HEATMAP_W, HEATMAP_H, INCLUDE_HEATMAP, REPLAN_LOCK_TICKS, SCORING_DWELL_TICKS,
    SCORE_COMMIT_DECISIONS, SCORE_COMMIT_UNTIL_EMPTY, SCORE_COMMIT_MAX_DECISIONS, MAX_SCORE,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    DEFEND_ZONE_POS,
    LONG_GOAL_POINTS, CENTER_GOAL_POINTS, COLLECT_RANGE,
    SCORE_RANGE,
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
# Margin for long goals — keeps robot body ~4" clear of the goal face
_GOAL_MARGIN = ROBOT_W / 2 + 4.0   # 11.5"
# Wider margin for center X arms — keeps robot body ~5" clear of each arm face
_CENTER_GOAL_MARGIN = ROBOT_W / 2 + 5.0   # 12.5"

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
# Distance from goal opening to robot center when scoring.
# NOTE: This MUST remain < SCORE_RANGE, or scoring will never fire.
# We keep a small safety margin so collision resolution jitter doesn't
# kick the robot out of scoring range right at the threshold.
_APPROACH_SAFETY = 0.75
_SCORE_APPROACH_GAP = min(ROBOT_W / 2 + 4.0, SCORE_RANGE - _APPROACH_SAFETY)   # long goals
_CENTER_APPROACH_GAP = min(ROBOT_W / 2 + 4.5, SCORE_RANGE - _APPROACH_SAFETY)  # center X arms

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
_X_TIP_DIST = _ARM_LEN + _CENTER_APPROACH_GAP
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

# Action-change lock-in: when the policy picks a different action than last
# decision, the robot freezes for this many sim ticks before executing the new
# one. Forces deliberate decisions and prevents thrashing between actions.
# 15 ticks × DT(0.05s) = 0.75s of in-sim pause.
_ACTION_CHANGE_PAUSE_TICKS = 15


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

# Staging waypoints — one per goal end, placed 12" beyond the approach position
# so the robot arrives already aligned on the correct axis before the last leg.
# Layout:  field → corridor → staging → approach_pos → (score)
_STAGE_R_TOP = np.array([_RIGHT_GOAL_CX,  LONG_GOAL_Y_MAX + 22.0])  # above right goal
_STAGE_R_BOT = np.array([_RIGHT_GOAL_CX,  LONG_GOAL_Y_MIN - 22.0])  # below right goal
_STAGE_L_TOP = np.array([_LEFT_GOAL_CX,   LONG_GOAL_Y_MAX + 22.0])  # above left goal
_STAGE_L_BOT = np.array([_LEFT_GOAL_CX,   LONG_GOAL_Y_MIN - 22.0])  # below left goal


def _build_nav_waypoints(start: np.ndarray, final: np.ndarray) -> list[np.ndarray]:
    """Build waypoint list from `start` to `final` avoiding all goal obstacles.

    Strategy:
      1. Direct LOS clear → single-leg path.
      2. Final is a long-goal end approach → use a fixed two-step staging path
         (corridor → staging → final) that guarantees correct heading alignment.
      3. Otherwise: corridor detour, with a centre go-around if still blocked.
    """
    from sim.route_planner import _los_blocked, _NAV_MARGIN

    # If we're already at (or extremely close to) the final point, don't run the
    # LOS/detour planner. Some edge cases around zero-length segments can cause
    # the planner to incorrectly return a detour, sending the robot away from a
    # scoring pose and preventing the score timer from ever firing.
    if float(np.linalg.norm(final - start)) < 0.75:
        return [final]

    # ── Center-goal scoring approach: force 2 intermediate waypoints ───────
    # The generic LOS planner treats the center X arms as hard obstacles (with
    # a large navigation margin). For scoring, we *must* be able to approach a
    # tip even though the segment naturally intersects that geometry.
    # We therefore route via two safe corridor points, then allow a final leg
    # into the score pose.
    center_score_targets = (_MID_NE_POS, _MID_NW_POS, _LOW_SE_POS, _LOW_SW_POS)
    if min(float(np.linalg.norm(final - t)) for t in center_score_targets) < 2.0:
        pre = _NAV_ABOVE_X if float(final[1]) >= _CY else _NAV_BELOW_X
        if float(final[0]) >= _CX:
            corridor = _NAV_RIGHT_HIGH if float(final[1]) >= _CY else _NAV_RIGHT_LOW
        else:
            corridor = _NAV_LEFT_HIGH if float(final[1]) >= _CY else _NAV_LEFT_LOW

        legs: list[np.ndarray] = []
        for p in (pre, corridor, final):
            if not legs or float(np.linalg.norm(p - legs[-1])) > 0.75:
                # Skip if we're already basically at this waypoint
                if float(np.linalg.norm(p - start)) > 0.75:
                    legs.append(p.copy())
        if not legs:
            return [final]
        # Always end exactly at final
        if float(np.linalg.norm(legs[-1] - final)) > 1e-6:
            legs.append(final)
        return legs

    # ── Long-goal scoring approach: force 2 intermediate waypoints ──────────
    # Layout: start → (optional center) → corridor → stage → final
    is_right_long = abs(float(final[0] - _RIGHT_GOAL_CX)) < 6.0 and (
        float(final[1]) > LONG_GOAL_Y_MAX or float(final[1]) < LONG_GOAL_Y_MIN
    )
    is_left_long = abs(float(final[0] - _LEFT_GOAL_CX)) < 6.0 and (
        float(final[1]) > LONG_GOAL_Y_MAX or float(final[1]) < LONG_GOAL_Y_MIN
    )
    if is_right_long or is_left_long:
        is_top = float(final[1]) > LONG_GOAL_Y_MAX
        corridor = (_NAV_RIGHT_HIGH if is_top else _NAV_RIGHT_LOW).copy() \
            if is_right_long else \
            (_NAV_LEFT_HIGH if is_top else _NAV_LEFT_LOW).copy()
        # If the direct path to this corridor is blocked by the center X,
        # route via the OTHER corridor on the same side (high↔low) to go
        # around the X rather than trying to cut through its expanded margin.
        go_around = (_NAV_RIGHT_LOW if is_top else _NAV_RIGHT_HIGH).copy() \
            if is_right_long else \
            (_NAV_LEFT_LOW if is_top else _NAV_LEFT_HIGH).copy()
        stage = np.array([float(corridor[0]), float(final[1])], dtype=np.float64)

        # If the robot has already passed the corridor toward the goal (its x is
        # between the corridor and the goal), skip the corridor (and go-around)
        # entirely so the robot doesn't get sent backward on subsequent steps.
        start_x = float(start[0])
        corr_x  = float(corridor[0])
        past_corridor = (
            (is_right_long and start_x >= corr_x) or
            (is_left_long  and start_x <= corr_x)
        )
        if past_corridor:
            compact: list[np.ndarray] = []
            for p in [final]:
                if not compact or float(np.linalg.norm(p - compact[-1])) > 0.75:
                    compact.append(p.copy())
            return compact

        legs: list[np.ndarray] = []
        if _los_blocked(start[0], start[1], corridor[0], corridor[1], margin=_NAV_MARGIN):
            legs.append(go_around)
        legs.append(corridor)
        legs.append(stage)
        legs.append(final)
        # Compact duplicates / near-zero legs
        compact: list[np.ndarray] = []
        for p in legs:
            if not compact or float(np.linalg.norm(p - compact[-1])) > 0.75:
                compact.append(p)
        return compact

    # Direct path (most common for collect actions)
    if not _los_blocked(start[0], start[1], final[0], final[1], margin=_NAV_MARGIN):
        return [final]

    # NOTE: Long-goal scoring is handled by the block above. For non-scoring
    # targets we keep using the generic LOS/detour planner below.

    # ── Generic corridor detour ─────────────────────────────────────────────
    # Search all safe corridor points for the shortest clear 1-stop or 2-stop
    # path. The old single-heuristic approach could leave one leg still blocked,
    # causing the robot to be collision-resolved sideways and appear to orbit.
    _pool = [
        _NAV_RIGHT_LOW, _NAV_RIGHT_HIGH,
        _NAV_LEFT_LOW,  _NAV_LEFT_HIGH,
        _NAV_BELOW_X,   _NAV_ABOVE_X,
    ]
    sx, sy, fx, fy = float(start[0]), float(start[1]), float(final[0]), float(final[1])

    best_dist = 1e9
    best_path: list = []

    # 1-stop search
    for c in _pool:
        cx, cy = float(c[0]), float(c[1])
        if (not _los_blocked(sx, sy, cx, cy, margin=_NAV_MARGIN) and
                not _los_blocked(cx, cy, fx, fy, margin=_NAV_MARGIN)):
            d = math.hypot(cx - sx, cy - sy) + math.hypot(fx - cx, fy - cy)
            if d < best_dist:
                best_dist = d
                best_path = [c.copy(), final.copy()]

    if best_path:
        return best_path

    # 2-stop search
    for c1 in _pool:
        c1x, c1y = float(c1[0]), float(c1[1])
        if _los_blocked(sx, sy, c1x, c1y, margin=_NAV_MARGIN):
            continue
        for c2 in _pool:
            if c2 is c1:
                continue
            c2x, c2y = float(c2[0]), float(c2[1])
            if (not _los_blocked(c1x, c1y, c2x, c2y, margin=_NAV_MARGIN) and
                    not _los_blocked(c2x, c2y, fx, fy, margin=_NAV_MARGIN)):
                d = (math.hypot(c1x - sx, c1y - sy) +
                     math.hypot(c2x - c1x, c2y - c1y) +
                     math.hypot(fx - c2x, fy - c2y))
                if d < best_dist:
                    best_dist = d
                    best_path = [c1.copy(), c2.copy(), final.copy()]

    return best_path if best_path else [final]


def _resolve_goal_collisions(robot, skip_center: bool = False,
                              skip_long: bool = False) -> bool:
    """Push robot out of goal bounding boxes.

    Long goals use AABB. Center X arms are checked simultaneously so that
    the dual-arm overlap case (robot at center intersection) is handled by
    pushing to the nearest cardinal clear position instead of bouncing
    between both arms indefinitely.

    skip_center: suppress center X arm push (used while scoring at a center tip).
    skip_long: suppress long-goal push (used while the robot is at a long-goal
               scoring position so the goal margin can't eject it mid-shot).

    Returns True if the robot was actually pushed (i.e. was inside a goal margin).
    """
    pre_x, pre_y = robot.x, robot.y
    m = _GOAL_MARGIN

    # --- Long goals (axis-aligned rectangles) — re-read position each time ---
    # skip_long=True when the robot is at a long-goal scoring position so the
    # collision margin does not eject it mid-shot.
    if not skip_long:
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
    # When skip_center is set, the robot is mid-score at a center tip and we
    # want to let it nestle into the approach without being shoved away.
    if skip_center:
        return (robot.x, robot.y) != (pre_x, pre_y)
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
        return (robot.x, robot.y) != (pre_x, pre_y)

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

    return (robot.x, robot.y) != (pre_x, pre_y)


# ---------------------------------------------------------------------------
# Observation helpers — relative nearest-object features
# ---------------------------------------------------------------------------
# Field diagonal — used to normalise distance to [0, 1]
_FIELD_DIAG = math.sqrt(FIELD_W * FIELD_W + FIELD_H * FIELD_H)


def _nearest_balls_of_color(robot, objects, color: int, n: int) -> list:
    """Return n on-field balls of `color` sorted by distance. Pads with None."""
    cands = [
        (float(np.linalg.norm(obj.position - robot.position)), obj)
        for obj in objects
        if obj.status == OBJ_ON_FIELD and obj.color == color
    ]
    cands.sort(key=lambda c: c[0])
    out = [obj for _, obj in cands[:n]]
    while len(out) < n:
        out.append(None)
    return out


def _relative_obj_features(obj, robot) -> list:
    """Return [dx, dy, dist, bearing_sin, bearing_cos] relative to robot.

    Bearing is the angle from robot's heading to the object — sin/cos form so
    the policy sees orientation continuously.  None objects pad with
    distance=1.0 (sentinel for "no such ball").
    """
    if obj is None:
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    dx = float(obj.x - robot.x)
    dy = float(obj.y - robot.y)
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        bsin, bcos = 0.0, 1.0
    else:
        rel_angle = math.atan2(dy, dx) - robot.heading
        bsin = math.sin(rel_angle)
        bcos = math.cos(rel_angle)
    return [dx / FIELD_W, dy / FIELD_H, min(dist / _FIELD_DIAG, 1.0), bsin, bcos]


def _relative_goal_features(goal_pos: np.ndarray, robot) -> list:
    """Return [dx, dy] relative position of a fixed goal."""
    return [
        float(goal_pos[0] - robot.x) / FIELD_W,
        float(goal_pos[1] - robot.y) / FIELD_H,
    ]


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
        nav = field.nearest_navigable_target(robot.position)
        if nav is not None:
            return nav

        # Priority 3 — no accessible target in the vision cone.  Look beyond
        # the cone: head toward the centroid of all remaining blue balls on
        # the field (even ones behind obstacles — the detour planner will
        # route around them). This is the "we don't see anything good
        # nearby, go to where the blue balls actually are" fallback.
        blue_positions = [
            obj.position for obj in field.objects
            if obj.status == OBJ_ON_FIELD and obj.color == BALL_BLUE
        ]
        if blue_positions:
            centroid = np.mean(blue_positions, axis=0)
            # Don't return a target right next to us — if centroid is too close,
            # we're already there; push it 24" past the robot in centroid's direction.
            vec = centroid - robot.position
            dist = float(np.linalg.norm(vec))
            if dist < 24.0 and dist > 1e-3:
                centroid = robot.position + (vec / dist) * 24.0
            return np.array([
                float(np.clip(centroid[0], ROBOT_W, FIELD_W - ROBOT_W)),
                float(np.clip(centroid[1], ROBOT_W, FIELD_H - ROBOT_W)),
            ])
        # No blue balls left on field at all — drift toward field center
        # (where the goal X is) so the policy gets a fresh observation.
        return np.array([FIELD_W * 0.5, FIELD_H * 0.5])
    elif action == Action.SCORE_LONG_GOAL:
        approach, _, _, _ = _nearest_long_goal_target(robot)
        return approach.copy()
    elif action == Action.SCORE_CENTER_MID:
        approach, _, _, _ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
    elif action == Action.SCORE_CENTER_LOW:
        approach, _, _, _ = _nearest_center_tip(robot, lower=True)
        return approach.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OPP_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        approach, _, _, _ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
    elif action == Action.DEFEND_ZONE:
        return DEFEND_ZONE_POS.copy()
    elif action == Action.EJECT_WRONG_COLOR:
        return None   # robot stays in place; eject happens in _check_ally_effects
    elif action == Action.IDLE:
        return None
    return None


def _opp_action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    """Convert opponent action to a moveToPoint target (mirror of allied)."""
    if action == Action.COLLECT_NEAREST_BALL:
        return field.nearest_on_field_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        return OPP_LONG_GOAL.copy()
    elif action == Action.SCORE_CENTER_MID:
        approach, *_ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
    elif action == Action.SCORE_CENTER_LOW:
        approach, *_ = _nearest_center_tip(robot, lower=True)
        return approach.copy()
    elif action == Action.DESCORE_OPP_LONG:
        return OUR_LONG_GOAL.copy()
    elif action == Action.DESCORE_CENTER:
        approach, *_ = _nearest_center_tip(robot, lower=False)
        return approach.copy()
    elif action == Action.DEFEND_ZONE:
        return np.array([24.00, 72.00])   # opp defend zone (left side)
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
        num_opponents: int = 0,
        use_timer: bool = False,
        alliance: str = "blue",
    ):
        super().__init__()
        self.render_mode    = render_mode
        self.failure_config = failure_config or FailureConfig()
        self.num_allies     = max(1, min(num_allies, 2))      # 1 or 2
        self.num_opponents  = max(0, min(num_opponents, 2))   # 0, 1, or 2
        self.use_timer      = use_timer                       # if True, timer decrements and episode ends
        # Alliance color: 1.0 = blue (we score blue balls), 0.0 = red.
        # Currently always blue — kwarg reserved for future symmetric training.
        self.alliance_color = 1.0 if alliance == "blue" else 0.0

        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])
        self.observation_space = spaces.Dict({
            "robot_0": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            "robot_1": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        })

        self.opponent_policy = get_opponent(opponent_type)
        self.field           = Field()
        self._renderer       = None
        # Action-change lock-in counter and last-action snapshot — see
        # _ACTION_CHANGE_PAUSE_TICKS at module top.
        self.action_pause_ticks    = np.zeros(2, dtype=np.int32)
        self.prev_action_for_pause = np.array([Action.IDLE, Action.IDLE], dtype=np.int32)
        # Curriculum stage-1 mode: convert all red balls to blue at reset so the
        # policy can master collect/score without the red-ball discrimination
        # problem getting in the way. Toggled live by CurriculumCallback.
        self.all_blue_only   = False

        self.score_events          = np.zeros(2)
        self.descore_events        = np.zeros(2)
        self.eject_events          = np.zeros(2, dtype=np.int32)
        self.collected_this_step   = np.zeros(2, dtype=np.int32)
        self.red_collected_this_step = np.zeros(2, dtype=np.int32)
        self.current_actions       = np.array([Action.IDLE, Action.IDLE], dtype=np.int32)

        # Scoring target cache (per robot): locks the chosen goal END while a
        # SCORE_* action is active, so the robot doesn't thrash between ends as
        # its position changes (e.g., long-goal top vs bottom).
        # Each entry: (action_int, score_pos, goal_pos, required_heading, gname)
        self._score_target_cache: list[tuple[int, np.ndarray, np.ndarray, float, str] | None] = [None, None]
        # Extra route hints used to build deterministic approach waypoints while scoring.
        # For long-goal scoring: {"route_low": bool} selected when the score target is first chosen.
        self._score_route_hint: list[dict[str, object] | None] = [None, None]

        # Scoring commitment: when the agent chooses a SCORE_* action, we hold it
        # for a couple of decisions so it can align + complete scoring instead
        # of thrashing to a different goal next decision.
        self._score_commit_left = np.zeros(2, dtype=np.int32)
        self._score_committed_action = np.array([Action.IDLE, Action.IDLE], dtype=np.int32)
        self.expected_state_delta  = np.zeros(2)
        self.prev_predicted_score  = np.zeros(2)
        # Per-decision shaping signals (read by compute_reward)
        self.goal_bump_ticks       = np.zeros(2, dtype=np.int32)
        self.low_progress          = np.zeros(2, dtype=bool)
        self.balls_held_at_step_start = np.zeros(2, dtype=np.int32)
        self.center_score_events   = np.zeros(2, dtype=np.int32)   # pts scored in center this step
        # Sum of |heading delta| across all ticks in the step (rad).
        # Compared to "normal" travel turn to detect erratic wiggling.
        self.heading_turn_amount   = np.zeros(2, dtype=np.float32)
        # Previous step's controlled-quadrant count (for ctrl_gain bonus)
        self.prev_ctrl_us          = 0
        self.prev_ctrl_opp         = 0
        # Last step's gain (computed inside step, read by reward)
        self.ctrl_gain_us          = 0
        self.ctrl_gain_opp         = 0
        # Most recent reward breakdown per robot (component → value) for renderer.
        self.last_reward_breakdown: list[dict[str, float]] = [{}, {}]
        # Score snapshots for color-aware reward delta computation
        self.prev_my_score:  float = 0.0
        self.prev_opp_score: float = 0.0
        # Phase-objective milestones (one-time bonuses per episode)
        # Key = margin threshold, value = already-awarded flag.
        self.score_milestones_hit: dict[int, bool] = {10: False, 20: False}
        self.done = False

        # Decision progress — updated each tick so renderer can show a progress bar
        self.decision_tick: int = 0
        self.executing: bool = False

        # Scoring animations — list of dicts the renderer consumes and removes.
        # Each: {x0,y0, x1,y1, color, elapsed, duration}
        self.score_animations: list[dict] = []

        # Per-decision one-shot effect guards (some actions should not spam-fire
        # every tick once stationary).
        self._eject_fired_this_decision = np.zeros(2, dtype=bool)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.field.reset(self.rng)
        # Curriculum stage 1: paint every ball blue so the policy can practice
        # collect→score chains without needing to discriminate by color yet.
        if self.all_blue_only:
            for obj in self.field.objects:
                obj.color = BALL_BLUE
        self.field.time_remaining = MATCH_DURATION   # decremented if use_timer=True

        self.failure_injector = FailureInjector(self.failure_config, self.rng)
        self.failure_injector.reset()

        self.score_events[:]           = 0
        self.descore_events[:]         = 0
        self.eject_events[:]           = 0
        self.collected_this_step[:]    = 0
        self.red_collected_this_step[:] = 0
        self.current_actions[:]      = Action.IDLE
        self._score_commit_left[:]   = 0
        self._score_committed_action[:] = Action.IDLE
        self._score_target_cache = [None, None]
        self._score_route_hint   = [None, None]
        self.expected_state_delta[:] = 0
        self.prev_predicted_score[:] = 0
        self.goal_bump_ticks[:]      = 0
        self.low_progress[:]         = False
        self.balls_held_at_step_start[:] = 0
        self.center_score_events[:]  = 0
        self.heading_turn_amount[:]  = 0.0
        self.prev_ctrl_us            = 0
        self.prev_ctrl_opp           = 0
        self.ctrl_gain_us            = 0
        self.ctrl_gain_opp           = 0
        self.prev_my_score           = 0.0
        self.prev_opp_score          = 0.0
        self.score_milestones_hit    = {10: False, 20: False}
        self.action_pause_ticks[:]   = 0
        self.prev_action_for_pause[:] = Action.IDLE
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

        self.score_events[:]            = 0
        self.descore_events[:]          = 0
        self.eject_events[:]            = 0
        self.collected_this_step[:]     = 0
        self.red_collected_this_step[:] = 0
        self.current_actions[:]         = Action.IDLE
        self._score_target_cache = [None, None]
        self._score_route_hint   = [None, None]
        self.expected_state_delta[:]    = 0
        self.prev_predicted_score[:]    = 0
        self.goal_bump_ticks[:]         = 0
        self.low_progress[:]            = False
        self.balls_held_at_step_start[:] = 0
        self.center_score_events[:]     = 0
        self.heading_turn_amount[:]     = 0.0
        self.prev_ctrl_us               = 0
        self.prev_ctrl_opp           = 0
        self.ctrl_gain_us            = 0
        self.ctrl_gain_opp           = 0
        self.score_milestones_hit    = {10: False, 20: False}
        self.action_pause_ticks[:]   = 0
        self.prev_action_for_pause[:] = Action.IDLE
        self.done          = False
        self.decision_tick = 0
        self.executing     = False
        self.opp_targets   = [None, None]
        self.score_animations.clear()
        self._eject_fired_this_decision[:] = False

        return self._get_obs(), {}

    def manual_tick(self, renderer) -> None:
        """One physics tick driven by WASD input from the renderer.

        Called from the demo loop instead of step() when renderer.wasd_mode is True.
        Handles movement, goal collision, intake, and scoring for the selected robot.
        Does NOT render — the caller's env.render() already drew this frame.

        If renderer.queued_manual_action is set (from clicking an action button in the
        panel), that RL action runs for TICKS_PER_DECISION ticks using the normal
        nav/effect pipeline, then control returns to WASD.
        """
        import math

        # Tuned for ~60fps feel: 65% speed, 30% turn rate so driving is smooth
        _WASD_SPEED = MAX_SPEED * 0.65
        _WASD_TURN  = TURN_RATE * 0.30

        ridx  = renderer.wasd_robot_idx
        robot = self.field.allies[ridx]

        # ── Pick up a newly queued action button press ───────────────────
        queued = getattr(renderer, "queued_manual_action", None)
        if queued is not None:
            renderer.queued_manual_action = None
            act    = Action(queued)
            target = _action_to_target(act, self.field, robot)
            self._manual_rl_action    = int(act)
            self._manual_rl_ticks     = 0
            self._manual_rl_wq        = (
                _build_nav_waypoints(robot.position, target)
                if target is not None else []
            )
            robot.score_timer = 0.0   # reset score timer for fresh approach
            self._eject_fired_this_decision[ridx] = False

        # ── Run active manual RL action (overrides WASD movement) ────────
        if getattr(self, "_manual_rl_action", None) is not None:
            act  = Action(self._manual_rl_action)
            wq   = self._manual_rl_wq
            prev_pos = robot.position.copy()

            # Movement: follow waypoint queue
            if wq:
                arrived = robot.move_toward_point(wq[0])
                if arrived and len(wq) > 1:
                    wq.pop(0)
                elif arrived:
                    wq.clear()

            _resolve_goal_collisions(robot)

            # Effects (collect, score, eject, etc.) — same logic as RL step
            self.decision_tick = self._manual_rl_ticks
            self._check_ally_effects(ridx, act)

            self.field.apply_robot_push(robot, prev_pos)
            self.field.physics_tick(DT)

            self._manual_rl_ticks += 1
            ticks_done = self._manual_rl_ticks >= TICKS_PER_DECISION
            nav_done   = not wq
            if ticks_done and nav_done:
                self._manual_rl_action = None
                self.decision_tick     = 0

            # Expose ticks-left for the panel status line
            self._manual_rl_ticks_left = max(0, TICKS_PER_DECISION - self._manual_rl_ticks)
            return

        # ── Normal WASD control ───────────────────────────────────────────
        self._manual_rl_action     = None
        self._manual_rl_ticks_left = 0
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

        # --- Scoring (hold F) ---
        # Timer always runs while F held. On each fire: score into goal if
        # positioned correctly, otherwise eject one ball from the back (free-shoot).
        if renderer.wasd_score_on and robot.balls_held > 0:
            robot.intake_active = True
            self._do_manual_scoring(robot)
        elif robot.score_timer > 0.0:
            # F released — freeze timer, keep balls in robot
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

    def _eject_wrong_color_balls(self, robot) -> int:
        """Eject all wrong-color (red for blue alliance) balls out the robot's back face.

        Returns the count of balls ejected.  Used by EJECT_WRONG_COLOR action and
        by WASD free-shoot when the robot is not near a goal.
        """
        to_eject = [oid for oid in list(robot.held_object_ids)
                    if self.field.objects[oid].color != BALL_BLUE]
        if not to_eject:
            return 0

        bx = robot.x - math.cos(robot.heading) * (ROBOT_W / 2)
        by = robot.y - math.sin(robot.heading) * (ROBOT_W / 2)
        rng = getattr(self, "rng", None)

        for obj_idx in to_eject:
            robot.held_object_ids.remove(obj_idx)
            robot.balls_held -= 1
            obj = self.field.objects[obj_idx]
            obj.status = OBJ_ON_FIELD
            scatter = 6.0
            ox = bx + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            oy = by + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            obj.x = round(float(np.clip(ox, ROBOT_W / 2 + 1, FIELD_W - ROBOT_W / 2 - 1)), 2)
            obj.y = round(float(np.clip(oy, ROBOT_W / 2 + 1, FIELD_H - ROBOT_W / 2 - 1)), 2)
            speed = 14.0 + (float(rng.uniform(0, 6)) if rng else 0.0)
            obj.vx = float(-math.cos(robot.heading) * speed)
            obj.vy = float(-math.sin(robot.heading) * speed)
            if rng:
                obj.vx += float(rng.uniform(-3, 3))
                obj.vy += float(rng.uniform(-3, 3))

        robot.score_timer = 0.0
        return len(to_eject)

    def _do_manual_scoring(self, robot) -> None:
        """Scoring when F is held — timer runs regardless of position.

        On each timer fire:
        - If the robot's back face is near a valid goal surface with correct
          heading → score one ball into that goal.
        - Otherwise → eject one ball out the back (free-shoot onto the field).

        This means holding F anywhere produces a steady stream of balls exiting
        the robot, going into a goal when positioned correctly, or onto the floor
        when not.  Same FIFO order either way.
        """
        import math

        _BACK_DIST = 12.0
        _HDG_TOL   = math.pi / 4.0

        # Robot back-face centre
        bx = robot.x - math.cos(robot.heading) * (ROBOT_W / 2)
        by = robot.y - math.sin(robot.heading) * (ROBOT_W / 2)
        back = np.array([bx, by])

        # ── Detect nearby goal surfaces ──
        candidates: list[tuple] = []
        in_y = LONG_GOAL_Y_MIN - _BACK_DIST < by < LONG_GOAL_Y_MAX + _BACK_DIST

        # Right long goal — field side
        if abs(bx - _RIGHT_GOAL_X_LO) < _BACK_DIST and in_y:
            gy = float(np.clip(by, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX))
            candidates.append((math.pi, np.array([_RIGHT_GOAL_CX, gy]),
                                "our_long", LONG_GOAL_POINTS))
        if abs(by - LONG_GOAL_Y_MAX) < _BACK_DIST and abs(bx - _RIGHT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_NORTH, np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MAX]),
                                "our_long", LONG_GOAL_POINTS))
        if abs(by - LONG_GOAL_Y_MIN) < _BACK_DIST and abs(bx - _RIGHT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_SOUTH, np.array([_RIGHT_GOAL_CX, LONG_GOAL_Y_MIN]),
                                "our_long", LONG_GOAL_POINTS))

        # Left long goal — field side
        if abs(bx - _LEFT_GOAL_X_HI) < _BACK_DIST and in_y:
            gy = float(np.clip(by, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX))
            candidates.append((0.0, np.array([_LEFT_GOAL_CX, gy]),
                                "opp_long", LONG_GOAL_POINTS))
        if abs(by - LONG_GOAL_Y_MAX) < _BACK_DIST and abs(bx - _LEFT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_NORTH, np.array([_LEFT_GOAL_CX, LONG_GOAL_Y_MAX]),
                                "opp_long", LONG_GOAL_POINTS))
        if abs(by - LONG_GOAL_Y_MIN) < _BACK_DIST and abs(bx - _LEFT_GOAL_CX) < _BACK_DIST:
            candidates.append((_HDG_SOUTH, np.array([_LEFT_GOAL_CX, LONG_GOAL_Y_MIN]),
                                "opp_long", LONG_GOAL_POINTS))

        # Center X arm tips
        for tip, req_hdg, gname in [
            (_TIP_NE, _HDG_NE, "center_mid"),
            (_TIP_SW, _HDG_SW, "center_mid"),
            (_TIP_NW, _HDG_NW, "center_low"),
            (_TIP_SE, _HDG_SE, "center_low"),
        ]:
            if float(np.linalg.norm(back - tip)) < _BACK_DIST:
                candidates.append((req_hdg, tip.copy(), gname, CENTER_GOAL_POINTS))

        # ── Pick best goal candidate (if any is within heading tolerance) ──
        best_goal = None
        if candidates:
            req_hdg, anim_target, gname, points = min(
                candidates, key=lambda c: abs(_wrap_angle(c[0] - robot.heading))
            )
            if abs(_wrap_angle(req_hdg - robot.heading)) <= _HDG_TOL:
                best_goal = (req_hdg, anim_target, gname, points)

        # ── Timer always runs while F is held ──
        robot.score_timer += DT
        interval = _SCORE_INTERVAL / max(robot.balls_held, 1)
        if robot.score_timer < interval:
            return

        robot.score_timer = 0.0
        if robot.balls_held <= 0 or not robot.held_object_ids:
            return

        # When free-shooting (no goal), prefer ejecting wrong-color (red) balls first.
        if best_goal is None:
            wrong_i = next(
                (i for i, oid in enumerate(robot.held_object_ids)
                 if self.field.objects[oid].color != BALL_BLUE), None
            )
            pop_i = wrong_i if wrong_i is not None else 0
        else:
            pop_i = 0
        idx        = robot.held_object_ids.pop(pop_i)
        ball_color = self.field.objects[idx].color
        robot.balls_held -= 1

        if best_goal is not None:
            # ── Score into goal ──
            req_hdg, anim_target, gname, points = best_goal
            prepend = req_hdg < 0.0
            self.field.objects[idx].status         = OBJ_SCORED_US
            self.field.objects[idx].scored_in_goal = gname
            ejected = self.field.goal_state.score_ball(gname, idx, ball_color, prepend=prepend)
            if ball_color == BALL_BLUE:
                self.field.my_score += points
            else:
                self.field.opponent_score += points
            if ejected is not None:
                self._handle_overflow(ejected, gname, prepend, points)
            self.score_animations.append({
                'x0': bx, 'y0': by,
                'x1': float(anim_target[0]), 'y1': float(anim_target[1]),
                'color': ball_color, 'start_ms': None, 'duration': 0.4,
            })
        else:
            # ── Free-shoot: eject ONE ball from back face onto field ──
            rng = getattr(self, "rng", None)
            scatter = 4.0
            ox = bx + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            oy = by + (float(rng.uniform(-scatter, scatter)) if rng else 0.0)
            obj = self.field.objects[idx]
            obj.status = OBJ_ON_FIELD
            obj.x = round(float(np.clip(ox, ROBOT_W/2+1, FIELD_W-ROBOT_W/2-1)), 2)
            obj.y = round(float(np.clip(oy, ROBOT_W/2+1, FIELD_H-ROBOT_W/2-1)), 2)
            speed = 15.0 + (float(rng.uniform(0, 8)) if rng else 0.0)
            obj.vx = float(-math.cos(robot.heading) * speed)
            obj.vy = float(-math.sin(robot.heading) * speed)
            if rng:
                obj.vx += float(rng.uniform(-4, 4))
                obj.vy += float(rng.uniform(-4, 4))

    def action_masks(self, robot_id: int = 0) -> np.ndarray:
        """Boolean mask of shape (NUM_ACTIONS,) — True = action valid this decision.

        Rules:
          DESCORE_OPP_LONG  only when opp's long goal has balls
          DESCORE_CENTER    only when center goals contain opp-colored balls
          EJECT_WRONG_COLOR only when robot holds wrong-color balls
          SCORE_*           always available — auto-chains to COLLECT first if empty
          All others always available (COLLECT, DEFEND, IDLE).
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        if robot_id >= self.num_allies:
            return mask
        robot = self.field.allies[robot_id]
        gs    = self.field.goal_state

        # Descore opp long: only when their goal has balls to remove
        if len(gs.opp_long) == 0:
            mask[int(Action.DESCORE_OPP_LONG)] = False

        # Descore center: only when center goals hold opponent-colored balls
        opp_in_center = (
            any(color != BALL_BLUE for _, color in gs.center_mid) or
            any(color != BALL_BLUE for _, color in gs.center_low)
        )
        if not opp_in_center:
            mask[int(Action.DESCORE_CENTER)] = False

        # Eject wrong color: only when holding wrong-color balls
        wrong_held = any(
            self.field.objects[oid].color != BALL_BLUE
            for oid in robot.held_object_ids
        )
        if not wrong_held:
            mask[int(Action.EJECT_WRONG_COLOR)] = False

        return mask

    def step(self, actions: np.ndarray):
        """One RL decision cycle = TICKS_PER_DECISION sim ticks."""
        raw_a0, raw_a1 = int(actions[0]), int(actions[1])
        a0, a1 = raw_a0, raw_a1

        SCORE_INTENTS = (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW)

        # Scoring commitment across DECISIONS (env.step calls).
        # Goal: once the agent selects SCORE_*, don't allow thrashing to other
        # objectives while the robot is still aligning/finishing scoring.
        for idx, chosen in enumerate((a0, a1)):
            if idx >= self.num_allies:
                continue
            if self._score_commit_left[idx] > 0:
                committed = int(self._score_committed_action[idx])
                if Action(committed) in SCORE_INTENTS and self.field.allies[idx].balls_held > 0:
                    if idx == 0:
                        a0 = committed
                    else:
                        a1 = committed
                    self._score_commit_left[idx] -= 1
                else:
                    self._score_commit_left[idx] = 0
                    self._score_committed_action[idx] = int(Action.IDLE)

        # Safety remap: if the chosen action is currently masked, fall back to
        # COLLECT_NEAREST_BALL. MaskablePPO won't ever send a masked action, but
        # this protects eval runs with old models and manual action-button clicks.
        if not self.action_masks(robot_id=0)[a0]:
            a0 = int(Action.COLLECT_NEAREST_BALL)
        if not self.action_masks(robot_id=1)[a1]:
            a1 = int(Action.COLLECT_NEAREST_BALL)

        self.current_actions = np.array([a0, a1], dtype=np.int32)

        # Prime/clear scoring target caches for this decision.
        for ridx in range(self.num_allies):
            act_i = Action(self.current_actions[ridx])
            robot = self.field.allies[ridx]
            if act_i in SCORE_INTENTS and robot.balls_held > 0:
                self._get_scoring_target(ridx, act_i)
            else:
                self._score_target_cache[ridx] = None

        # (Re)start scoring commitment when a score action is chosen.
        # Two modes:
        #  - until_empty: keep scoring intent until balls are emptied, with a timeout
        #  - fixed decisions: keep for SCORE_COMMIT_DECISIONS decisions
        if SCORE_COMMIT_UNTIL_EMPTY:
            raw_commit_a0, raw_commit_a1 = raw_a0, raw_a1
            if not self.action_masks(robot_id=0)[raw_commit_a0]:
                raw_commit_a0 = int(Action.COLLECT_NEAREST_BALL)
            if not self.action_masks(robot_id=1)[raw_commit_a1]:
                raw_commit_a1 = int(Action.COLLECT_NEAREST_BALL)

            for idx, act in enumerate((raw_commit_a0, raw_commit_a1)):
                if idx >= self.num_allies:
                    continue
                if self._score_commit_left[idx] > 0:
                    continue
                try:
                    if Action(act) in SCORE_INTENTS and self.field.allies[idx].balls_held > 0:
                        self._score_committed_action[idx] = int(act)
                        self._score_commit_left[idx] = SCORE_COMMIT_MAX_DECISIONS
                except ValueError:
                    pass

        elif SCORE_COMMIT_DECISIONS > 1:
            raw_commit_a0, raw_commit_a1 = raw_a0, raw_a1
            if not self.action_masks(robot_id=0)[raw_commit_a0]:
                raw_commit_a0 = int(Action.COLLECT_NEAREST_BALL)
            if not self.action_masks(robot_id=1)[raw_commit_a1]:
                raw_commit_a1 = int(Action.COLLECT_NEAREST_BALL)

            for idx, act in enumerate((raw_commit_a0, raw_commit_a1)):
                if idx >= self.num_allies:
                    continue
                if self._score_commit_left[idx] > 0:
                    continue
                try:
                    if Action(act) in SCORE_INTENTS and self.field.allies[idx].balls_held > 0:
                        self._score_committed_action[idx] = int(act)
                        # We already execute this decision now; hold for N-1 future decisions.
                        self._score_commit_left[idx] = SCORE_COMMIT_DECISIONS - 1
                except ValueError:
                    pass

        # Action-change lock-in: if a robot's action differs from last decision,
        # freeze it for _ACTION_CHANGE_PAUSE_TICKS ticks before executing the
        # new action. Forces the policy's "intent" to settle before motion.
        for idx, new_act in enumerate((a0, a1)):
            if new_act != int(self.prev_action_for_pause[idx]):
                self.action_pause_ticks[idx] = _ACTION_CHANGE_PAUSE_TICKS

        self.score_events[:]            = 0
        self.descore_events[:]          = 0
        self.eject_events[:]            = 0
        self.collected_this_step[:]     = 0
        self.red_collected_this_step[:] = 0
        self.goal_bump_ticks[:]         = 0
        self.low_progress[:]        = False
        self.center_score_events[:] = 0
        self.heading_turn_amount[:] = 0.0
        # Per-tick prev heading snapshot (for accumulating |heading delta|)
        prev_tick_headings = [self.field.allies[i].heading for i in range(2)]
        for idx in range(2):
            self.balls_held_at_step_start[idx] = int(self.field.allies[idx].balls_held)
        # Snapshot positions for end-of-decision low-progress detection
        step_start_pos = [self.field.allies[i].position.copy() for i in range(2)]
        self.decision_tick = 0
        self.executing     = True
        self._eject_fired_this_decision[:] = False

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
            """Build the waypoint queue for robot idx using its current live action.

            All movement actions get the detour planner so the robot routes
            around long-goal bodies and the center X structure.
            """
            robot  = self.field.allies[idx]
            act    = Action(live_actions[idx])
            if act in SCORE_INTENTS and robot.balls_held > 0:
                return self._build_scoring_waypoints(idx, act)
            else:
                final  = _action_to_target(act, self.field, robot)
            if final is None:
                return []
            return _build_nav_waypoints(robot.position, final)

        ally_wq: list[list] = [_build_wq(0), _build_wq(1)]

        def _current_target(idx: int):
            return ally_wq[idx][0] if ally_wq[idx] else None

        # ── Dynamic step length ──────────────────────────────────────────
        # Step runs for at least TICKS_PER_DECISION (3s) and extends until either
        # robots stop moving OR MAX_STEP_TICKS is hit. The hard cap prevents the
        # agent from getting locked into one decision for 20+ seconds when its
        # chosen action drives into an obstacle — instead it gets to re-decide
        # roughly every 5 seconds.
        # Hard cap on per-decision simulation time. Keep it high enough to
        # honor post-arrival scoring dwell even if that value is increased.
        MAX_STEP_TICKS = max(100, TICKS_PER_DECISION + SCORING_DWELL_TICKS)

        # How long since the mid-step replanner last changed the executing action.
        # This is independent of the RL decision interval; it only affects the
        # adaptive "auto-chain" logic inside the decision.
        live_action_age = [REPLAN_LOCK_TICKS, REPLAN_LOCK_TICKS]

        # Post-arrival scoring dwell: once in scoring pose (arrived + aligned),
        # keep the decision alive for a minimum duration so scoring timers can fire.
        scoring_dwell_left = [0, 0]

        for tick in range(MAX_STEP_TICKS):
            if tick >= TICKS_PER_DECISION:
                def _busy(i: int) -> bool:
                    """Return True if robot i should keep this decision alive.

                    Some actions (notably scoring/turning-in-place and one-shot eject)
                    have important effects even when the robot is not moving.
                    """
                    robot = self.field.allies[i]
                    act_i = Action(live_actions[i])
                    if scoring_dwell_left[i] > 0:
                        return True
                    if act_i in SCORE_INTENTS and robot.balls_held > 0:
                        score_pos, *_ = self._get_scoring_target(i, act_i)
                        dist = float(np.linalg.norm(robot.position - score_pos))
                        # Keep sim running while we are close enough to be aligning/scoring.
                        return dist < (_SCORE_ARRIVAL_DIST * 2.0)

                    if act_i == Action.EJECT_WRONG_COLOR and not self._eject_fired_this_decision[i]:
                        wrong_held = any(
                            self.field.objects[oid].color != BALL_BLUE
                            for oid in robot.held_object_ids
                        )
                        return wrong_held

                    return False

                still_active = any(
                    (self.field.allies[i].moving and bool(ally_wq[i])) or _busy(i)
                    for i in range(self.num_allies)
                )
                if not still_active:
                    break

            fi.on_tick(num_robots=2)

            # ── Mid-step adaptive replanning ──────────────────────────────
            # `current_actions` keeps the agent's original choice so the
            # wasted-action penalty still fires. `live_actions` is what's
            # actually executed this tick.
            n_opp_scored = len(self.field.scored_by_opp_indices()) if self.num_opponents > 0 else 0
            our_long_count = len(self.field.goal_state.our_long)
            defend_worth_it = (self.num_opponents > 0) and (our_long_count > 0)
            SCORE_INTENTS = (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_MID,
                             Action.SCORE_CENTER_LOW)
            for idx in range(self.num_allies):
                robot    = self.field.allies[idx]
                act      = Action(live_actions[idx])
                original = Action(self.current_actions[idx])
                new_act  = None
                force_replan = False
                # Chain-back: SCORE intent paused for COLLECT, balls now held → resume SCORE
                if (act == Action.COLLECT_NEAREST_BALL
                        and original in SCORE_INTENTS
                        and robot.balls_held > 0):
                    new_act = original
                # SCORE_* with no balls → auto-collect first
                elif act in SCORE_INTENTS and robot.balls_held == 0:
                    new_act = Action.COLLECT_NEAREST_BALL
                    force_replan = True
                # Full robot → score, unless already scoring or ejecting wrong-color
                elif robot.balls_held >= MAX_CARRY and act not in SCORE_INTENTS + (Action.EJECT_WRONG_COLOR,):
                    # Respect RL's SCORE intent if it had one, else default to long goal
                    new_act = original if original in SCORE_INTENTS else Action.SCORE_LONG_GOAL
                # DESCORE_* with nothing to descore → go collect instead of
                # driving into the goal body
                elif act in (Action.DESCORE_OPP_LONG, Action.DESCORE_CENTER) \
                        and n_opp_scored == 0:
                    new_act = Action.COLLECT_NEAREST_BALL
                    force_replan = True
                # DEFEND_ZONE only makes sense when (a) an opponent exists AND
                # (b) we have something scored worth defending. Otherwise route
                # to COLLECT so we don't waste a decision parked uselessly.
                elif act == Action.DEFEND_ZONE and not defend_worth_it:
                    new_act = Action.COLLECT_NEAREST_BALL
                if new_act is not None:
                    # Replan lock: don't flip executing intent too rapidly.
                    # Still allow forced replans when the current intent is invalid.
                    if force_replan or live_action_age[idx] >= REPLAN_LOCK_TICKS:
                        live_actions[idx] = int(new_act)
                        ally_wq[idx]      = _build_wq(idx)
                        live_action_age[idx] = 0

            # Save positions before movement for push calculation
            ally_prev = [self.field.allies[i].position.copy()     for i in range(self.num_allies)]
            opp_prev  = [self.field.opponents[i].position.copy()  for i in range(self.num_opponents)]

            for idx in range(self.num_allies):
                if fi.is_stuck(idx):
                    continue
                # Action-change lock-in — freeze movement until pause expires
                if self.action_pause_ticks[idx] > 0:
                    self.field.allies[idx].moving = False
                    continue
                robot = self.field.allies[idx]

                # If we're close enough to a scoring pose, stop driving and let
                # the scoring routine rotate-in-place + score. Without this,
                # move_toward_point() keeps steering toward score_pos while the
                # scoring code simultaneously steers toward required_heading,
                # producing an orbiting/circling failure mode.
                act_i = Action(live_actions[idx])
                if act_i in (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW) and robot.balls_held > 0:
                    score_pos, goal_pos, *_ = self._get_scoring_target(idx, act_i)
                    dist_to_score = float(np.linalg.norm(robot.position - score_pos))
                    dist_to_goal  = float(np.linalg.norm(robot.position - goal_pos))

                    # Blocking scoring behavior:
                    # - Keep following the fixed scoring waypoint sequence until we're
                    #   truly within scoring range of the goal reference point.
                    # - Only then stop driving and let _do_back_in_scoring() rotate + score.
                    if dist_to_score < _SCORE_ARRIVAL_DIST and dist_to_goal < (SCORE_RANGE - 0.25):
                        ally_wq[idx].clear()
                        robot.moving = False
                        robot.speed = 0.0
                        robot.target = None
                        target = None
                    else:
                        target = _current_target(idx)
                else:
                    target = _current_target(idx)
                if target is not None:
                    arrived = robot.move_toward_point(target)
                    if arrived and len(ally_wq[idx]) > 1:
                        ally_wq[idx].pop(0)      # advance to next waypoint
                    elif arrived:
                        ally_wq[idx].clear()     # reached final destination
                # Suppress center-arm push when scoring/descoring at a center tip.
                # Suppress long-goal push when the robot has reached the long-goal
                # scoring position so the collision margin can't kick it out mid-shot.
                skip_center = Action(live_actions[idx]) in (
                    Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW, Action.DESCORE_CENTER
                )
                _act_i_now = Action(live_actions[idx])
                skip_long = False
                if _act_i_now == Action.SCORE_LONG_GOAL and robot.balls_held > 0:
                    _cache_i = self._score_target_cache[idx]
                    if _cache_i is not None:
                        _sp = _cache_i[1]
                        skip_long = float(np.linalg.norm(robot.position - _sp)) < (_SCORE_ARRIVAL_DIST * 2.0)
                if _resolve_goal_collisions(self.field.allies[idx],
                                            skip_center=skip_center,
                                            skip_long=skip_long):
                    self.goal_bump_ticks[idx] += 1
                # Accumulate per-tick |heading delta| for jitter penalty
                cur_h = self.field.allies[idx].heading
                self.heading_turn_amount[idx] += abs(
                    _wrap_angle(cur_h - prev_tick_headings[idx])
                )
                prev_tick_headings[idx] = cur_h

                # Start (or refresh) scoring dwell once the scoring pose is achieved.
                # This is intentionally stricter than the generic "near" check: we
                # only dwell after arrival AND heading alignment are both satisfied.
                # Start (or refresh) scoring dwell once the scoring pose is achieved.
                # Note: `robot` is the per-idx robot defined above.
                if act_i in (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW) and robot.balls_held > 0:
                    score_pos, _, required_heading, _ = self._get_scoring_target(idx, act_i)

                    dist = float(np.linalg.norm(robot.position - score_pos))
                    aligned = abs(_wrap_angle(required_heading - robot.heading)) < _SCORE_HDG_TOL
                    if dist < _SCORE_ARRIVAL_DIST and aligned:
                        scoring_dwell_left[idx] = max(scoring_dwell_left[idx], SCORING_DWELL_TICKS)

            # Age the replanning lock timers (cap to avoid unbounded growth)
            for idx in range(self.num_allies):
                live_action_age[idx] = min(live_action_age[idx] + 1, REPLAN_LOCK_TICKS * 10 + 1)
                if scoring_dwell_left[idx] > 0:
                    scoring_dwell_left[idx] -= 1

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
            # Suppress effects during the action-change lock-in (matches the
            # frozen movement above — robot is "thinking", not acting).
            if self.action_pause_ticks[0] == 0:
                self._check_ally_effects(0, Action(live_actions[0]))
            if self.num_allies >= 2 and self.action_pause_ticks[1] == 0:
                self._check_ally_effects(1, Action(live_actions[1]))
            self._check_opp_effects()

            # Tick down the pause counters at the END of the tick so a robot
            # that started this tick paused stays paused for the full tick.
            for idx in range(self.num_allies):
                if self.action_pause_ticks[idx] > 0:
                    self.action_pause_ticks[idx] -= 1

            # Apply robot→ball pushes (already-held balls are skipped)
            for idx in range(self.num_allies):
                self.field.apply_robot_push(self.field.allies[idx], ally_prev[idx])
            for oi in range(self.num_opponents):
                self.field.apply_robot_push(self.field.opponents[oi], opp_prev[oi])

            # Advance ball rolling physics
            self.field.physics_tick(DT)

            # Tick timer (only active when use_timer=True)
            if self.use_timer:
                self.field.time_remaining -= DT
            self.decision_tick += 1

            if self.render_mode == "human" and self._renderer is not None:
                self._renderer.draw(self)

        self.executing = False
        self.done = self.use_timer and self.field.time_remaining <= 0

        # Snapshot this decision's actions for next-step pause comparison
        self.prev_action_for_pause[:] = self.current_actions

        # Low-progress detection: non-IDLE action, robot still trying to move at
        # end of decision, but covered < 6" total. Exclude failure-injected stucks.
        for idx in range(self.num_allies):
            travelled = float(np.linalg.norm(
                self.field.allies[idx].position - step_start_pos[idx]
            ))
            self.low_progress[idx] = (
                self.current_actions[idx] != Action.IDLE
                and self.field.allies[idx].moving
                and travelled < 6.0
                and not fi.is_stuck(idx)
            )

        for idx in range(2):
            self.field.allies[idx].actions_attempted += 1

        for idx in range(2):
            actual = float(self.field.my_score)
            self.expected_state_delta[idx] = actual - self.prev_predicted_score[idx]
            self.prev_predicted_score[idx] = actual

        # Compute change in controlled-quadrant counts since last step
        ctrl_now = self.field.goal_state.compute_quadrant_control()
        ctrl_us_now  = sum(1 for c in ctrl_now.values() if c == BALL_BLUE)
        ctrl_opp_now = sum(1 for c in ctrl_now.values() if c == BALL_RED)
        self.ctrl_gain_us  = max(0, ctrl_us_now  - self.prev_ctrl_us)
        self.ctrl_gain_opp = max(0, ctrl_opp_now - self.prev_ctrl_opp)
        self.prev_ctrl_us  = ctrl_us_now
        self.prev_ctrl_opp = ctrl_opp_now

        from training.reward import compute_reward
        r0 = compute_reward(self, robot_id=0)
        r1 = compute_reward(self, robot_id=1)

        # Snapshot scores AFTER reward computation so next step can compute delta
        self.prev_my_score  = float(self.field.my_score)
        self.prev_opp_score = float(self.field.opponent_score)

        return self._get_obs(), (r0, r1), self.done, False, {}

    # ------------------------------------------------------------------
    # Effect checks (called each tick)
    # ------------------------------------------------------------------
    def _get_scoring_target(
        self,
        idx: int,
        action: Action,
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Return (score_pos, goal_pos, required_heading, gname) for a SCORE_* action.

        The target is cached per robot while the scoring action is active so the
        robot doesn't thrash between goal ends across decisions (e.g. long-goal
        top vs bottom) as its position changes.
        """
        if idx >= self.num_allies:
            raise IndexError(f"robot idx {idx} out of range")

        robot = self.field.allies[idx]
        if robot.balls_held <= 0:
            self._score_target_cache[idx] = None

        cached = self._score_target_cache[idx]
        if cached is not None and cached[0] == int(action):
            _, score_pos, goal_pos, required_heading, gname = cached
            return score_pos, goal_pos, float(required_heading), gname

        if action == Action.SCORE_LONG_GOAL:
            score_pos, goal_pos, required_heading, gname = _nearest_long_goal_target(robot)
            # Stable detour choice: route via the corridor that matches the
            # chosen long-goal END. Top end → high corridor, bottom end → low.
            self._score_route_hint[idx] = {"route_low": float(score_pos[1]) < _CY}
        elif action == Action.SCORE_CENTER_MID:
            score_pos, goal_pos, required_heading, gname = _nearest_center_tip(robot, lower=False)
            self._score_route_hint[idx] = None
        elif action == Action.SCORE_CENTER_LOW:
            score_pos, goal_pos, required_heading, gname = _nearest_center_tip(robot, lower=True)
            self._score_route_hint[idx] = None
        else:
            raise ValueError(f"Unsupported scoring action: {action}")

        self._score_target_cache[idx] = (
            int(action),
            score_pos.copy(),
            goal_pos.copy(),
            float(required_heading),
            str(gname),
        )
        return score_pos, goal_pos, float(required_heading), str(gname)

    def _build_scoring_waypoints(self, idx: int, action: Action) -> list[np.ndarray]:
        """Deterministic scoring approach waypoints.

        We avoid dynamic LOS-based replanning for scoring because it can flip
        between corridor variants as the robot moves, creating a stable orbit
        and preventing scoring.

        Returns an ordered waypoint list ending at score_pos.
        """
        robot = self.field.allies[idx]
        score_pos, _, _, _ = self._get_scoring_target(idx, action)

        def _compact(pts: list[np.ndarray]) -> list[np.ndarray]:
            out: list[np.ndarray] = []
            prev = robot.position
            for p in pts:
                if float(np.linalg.norm(p - prev)) <= 0.75:
                    continue
                if out and float(np.linalg.norm(p - out[-1])) <= 0.75:
                    continue
                out.append(p.copy())
            return out

        if action == Action.SCORE_LONG_GOAL:
            hint = self._score_route_hint[idx] or {"route_low": float(robot.y) <= 72.0}
            route_low = bool(hint.get("route_low", True))

            is_right = float(score_pos[0]) > _CX
            corridor = (_NAV_RIGHT_LOW if route_low else _NAV_RIGHT_HIGH).copy() \
                if is_right else \
                (_NAV_LEFT_LOW if route_low else _NAV_LEFT_HIGH).copy()
            stage = np.array([float(corridor[0]), float(score_pos[1])], dtype=np.float64)

            # If the robot has already passed the corridor on its way to the goal
            # (x-position is between the corridor and the goal), skip the corridor
            # waypoint so the robot doesn't get sent backward on subsequent decisions.
            rx = float(robot.x)
            corr_x = float(corridor[0])
            already_past = (is_right and rx >= corr_x) or (not is_right and rx <= corr_x)
            if already_past:
                return _compact([score_pos])
            return _compact([corridor, stage, score_pos])

        if action in (Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW):
            pre = _NAV_ABOVE_X if float(score_pos[1]) >= _CY else _NAV_BELOW_X
            if float(score_pos[0]) >= _CX:
                corridor = _NAV_RIGHT_HIGH if float(score_pos[1]) >= _CY else _NAV_RIGHT_LOW
            else:
                corridor = _NAV_LEFT_HIGH if float(score_pos[1]) >= _CY else _NAV_LEFT_LOW
            return _compact([pre.copy(), corridor.copy(), score_pos])

        # Fallback (shouldn't happen)
        return [score_pos]

    def _check_ally_effects(self, idx: int, action: Action):
        robot = self.field.allies[idx]

        if action == Action.COLLECT_NEAREST_BALL:
            if self.field.try_collect(robot):
                self.collected_this_step[idx] += 1
                robot.actions_succeeded += 1
                # Track wrong-color (red) collections for penalty
                if self.field.objects[robot.held_object_ids[-1]].color != BALL_BLUE:
                    self.red_collected_this_step[idx] += 1

        elif action == Action.SCORE_LONG_GOAL:
            if robot.balls_held <= 0:
                robot.score_timer = 0.0
                return
            score_pos, goal_pos, required_heading, gname = self._get_scoring_target(idx, action)
            self._do_back_in_scoring(idx, robot, score_pos, goal_pos, gname,
                                     required_heading, LONG_GOAL_POINTS)

        elif action == Action.SCORE_CENTER_MID:
            if robot.balls_held <= 0:
                robot.score_timer = 0.0
                return
            score_pos, goal_pos, required_heading, gname = self._get_scoring_target(idx, action)
            self._do_back_in_scoring(idx, robot, score_pos, goal_pos, gname,
                                     required_heading, CENTER_GOAL_POINTS)

        elif action == Action.SCORE_CENTER_LOW:
            if robot.balls_held <= 0:
                robot.score_timer = 0.0
                return
            score_pos, goal_pos, required_heading, gname = self._get_scoring_target(idx, action)
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

        elif action == Action.EJECT_WRONG_COLOR:
            # One-shot per decision. Important: the action-change pause can
            # suppress effects on tick 0, so gate by a flag rather than
            # decision_tick==0.
            if not self._eject_fired_this_decision[idx]:
                self._eject_fired_this_decision[idx] = True
                ejected = self._eject_wrong_color_balls(robot)
                if ejected > 0:
                    self.eject_events[idx] += ejected
                    robot.actions_succeeded += 1

    def _do_back_in_scoring(self, idx: int, robot, score_pos: np.ndarray,
                              goal_pos: np.ndarray, gname: str,
                              required_heading: float, points: int) -> None:
        """Common back-in scoring routine.

        Robot arrives at score_pos, turns to required_heading (back faces goal),
        then scores one ball at a time on a timer.
        """
        dist_to_score = float(np.linalg.norm(robot.position - score_pos))
        if dist_to_score >= _SCORE_ARRIVAL_DIST:
            robot.score_timer = 0.0
            return

        # Even if we're near score_pos, scoring can only occur when we're within
        # SCORE_RANGE of the goal reference point used by Field.try_score_one().
        if float(np.linalg.norm(robot.position - goal_pos)) >= SCORE_RANGE:
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
            if gname in ("center_mid", "center_low"):
                self.center_score_events[idx] += pts
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

        # Relative features — N nearest blue and red balls, sorted by distance.
        # Replaces the old fixed-index flat-object array, which was not
        # permutation-invariant (index 0 was arbitrary, not strategically meaningful).
        nearest_blue = _nearest_balls_of_color(robot, self.field.objects, BALL_BLUE, N_NEAREST_BLUE)
        nearest_red  = _nearest_balls_of_color(robot, self.field.objects, BALL_RED,  N_NEAREST_RED)
        blue_flat = np.array(
            [v for ball in nearest_blue for v in _relative_obj_features(ball, robot)],
            dtype=np.float32,
        )
        red_flat = np.array(
            [v for ball in nearest_red for v in _relative_obj_features(ball, robot)],
            dtype=np.float32,
        )

        # Goal relative positions — 4 fixed goals, just (dx, dy) each.
        goal_rel = np.array([
            *_relative_goal_features(OUR_LONG_GOAL,    robot),
            *_relative_goal_features(OPP_LONG_GOAL,    robot),
            *_relative_goal_features(CENTER_MID_GOAL,  robot),
            *_relative_goal_features(CENTER_LOW_GOAL,  robot),
        ], dtype=np.float32)

        heatmap = None
        if INCLUDE_HEATMAP:
            heatmap = self.field.get_heatmap()

        # Current quadrant control snapshot (normalised — 4 quadrants total).
        ctrl_now = self.field.goal_state.compute_quadrant_control()
        ctrl_us  = sum(1 for c in ctrl_now.values() if c == BALL_BLUE) / 4.0
        ctrl_opp = sum(1 for c in ctrl_now.values() if c == BALL_RED)  / 4.0

        wrong_color_held = sum(
            1 for oid in robot.held_object_ids
            if self.field.objects[oid].color != BALL_BLUE
        )

        parts = [
            np.array([
                float(role_id),
                self.alliance_color,                  # 1.0 = blue, 0.0 = red
                self.field.time_remaining / MATCH_DURATION,
                float(self.field.my_score)       / MAX_SCORE,
                float(self.field.opponent_score) / MAX_SCORE,
                pos[0] / FIELD_W,
                pos[1] / FIELD_H,
                # Heading as (sin, cos) avoids the ±π discontinuity.
                math.sin(robot.heading),
                math.cos(robot.heading),
                float(robot.balls_held) / MAX_CARRY,
                float(balls_nearby)     / MAX_GAME_OBJECTS,
                ctrl_us,
                ctrl_opp,
            ], dtype=np.float32),
            blue_flat,         # 8 × 5 = 40
            red_flat,          # 4 × 5 = 20
            goal_rel,          # 4 × 2 = 8
        ]

        if INCLUDE_HEATMAP and heatmap is not None:
            parts.append(heatmap.flatten().astype(np.float32, copy=False))

        # Goal-state features — give the policy direct visibility into how full
        # each goal is and how many opponent-colored balls are in each.
        # Without these, the policy can only infer goal fill indirectly from
        # the score value, which is too late for proactive descoring decisions.
        _LONG_CAP   = 14.0
        _CTR_CAP    = 14.0  # center_mid (7) + center_low (7)
        gs = self.field.goal_state
        opp_long_fill      = len(gs.opp_long) / _LONG_CAP
        opp_balls_opp_long = sum(1 for _, c in gs.opp_long if c == BALL_RED) / _LONG_CAP
        our_long_fill      = len(gs.our_long) / _LONG_CAP
        our_balls_our_long = sum(1 for _, c in gs.our_long if c == BALL_BLUE) / _LONG_CAP
        center_fill        = (len(gs.center_mid) + len(gs.center_low)) / _CTR_CAP
        opp_balls_center   = sum(
            1 for _, c in gs.center_mid + gs.center_low if c == BALL_RED
        ) / _CTR_CAP
        parts.append(np.array([
            opp_long_fill,       # how full is the opponent's long goal?
            opp_balls_opp_long,  # how many of those are the opponent's color? (threat)
            our_long_fill,       # how full is our long goal?
            our_balls_our_long,  # how many of ours are in our goal? (progress)
            center_fill,         # combined center goal fill
            opp_balls_center,    # opp-colored balls in center goals (threat)
        ], dtype=np.float32))

        parts.append(
            np.array([
                self.expected_state_delta[role_id] / 10.0,
                robot.success_ratio(),
                float(wrong_color_held) / MAX_CARRY,  # how many red balls held
            ], dtype=np.float32),
        )

        obs = np.concatenate(parts)
        # Guard against silent obs-shape drift — catches missing fields immediately.
        assert obs.shape == (STATE_DIM,), (
            f"obs shape {obs.shape}, expected ({STATE_DIM},)"
        )
        return obs

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

    def __init__(self, render_mode: str | None = None,
                 num_allies: int = 1, num_opponents: int = 0,
                 use_timer: bool = False, **kwargs):
        super().__init__()
        self.env         = VexAIEnv(render_mode=render_mode,
                                    num_allies=num_allies,
                                    num_opponents=num_opponents,
                                    use_timer=use_timer,
                                    **kwargs)
        self.render_mode = render_mode

        # Single-robot training: obs = one robot state, actions = NUM_ACTIONS
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["robot_0"], info

    def step(self, action: int):
        # Single robot: robot_0 gets the chosen action, robot_1 is idle
        obs, rewards, done, truncated, info = self.env.step(np.array([action, Action.IDLE]))
        return obs["robot_0"], rewards[0], done, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the action validity mask for robot 0. Required by MaskablePPO."""
        return self.env.action_masks(robot_id=0)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def _get_episode_scores(self) -> tuple[float, float]:
        """Return (my_score, opp_score) from the inner field. Called via env_method."""
        return (float(self.env.field.my_score),
                float(self.env.field.opponent_score))

    def _get_reward_breakdown(self) -> dict:
        """Return robot 0's last reward breakdown. Called via env_method."""
        bd = getattr(self.env, "last_reward_breakdown", None)
        if bd and bd[0]:
            return dict(bd[0])
        return {}

    def apply_training_stats(self, stats: dict) -> None:
        """Merge stats into this env's renderer.training_stats (if rendering).

        Used by EpisodeStatsCallback via VecEnv.env_method so the TRAINING
        panel updates in every worker — including SubprocVecEnv workers where
        the main process can't touch their pygame state directly.
        """
        renderer = getattr(self.env, "_renderer", None)
        if renderer is None:
            return
        renderer.training_stats.update(stats)

    def apply_curriculum_config(self, config: dict) -> None:
        """Apply a curriculum stage config to the underlying VexAIEnv.

        Called by CurriculumCallback via VecEnv.env_method so it works under
        both DummyVecEnv and SubprocVecEnv. Mutates opponent_policy,
        num_opponents, and/or failure_config fields in place.
        """
        from sim.opponent import get_opponent

        if "opponent_type" in config:
            self.env.opponent_policy = get_opponent(config["opponent_type"])
        if "num_opponents" in config:
            self.env.num_opponents = max(0, min(int(config["num_opponents"]), 2))
        if "all_blue_only" in config:
            self.env.all_blue_only = bool(config["all_blue_only"])
        if "failures" in config:
            for key, val in config["failures"].items():
                setattr(self.env.failure_config, key, val)
