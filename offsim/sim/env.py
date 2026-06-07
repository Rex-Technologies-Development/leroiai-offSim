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
    SCORE_RANGE, VISION_RANGE, VISION_HALF_ANGLE,
    LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH, LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX,
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W, ROBOT_W, TURN_RATE, MAX_SPEED,
    BALL_BLUE, BALL_RED,
)
from sim.field import Field, GoalState
from sim.robot import Robot
from sim.failure import FailureConfig, FailureInjector
from sim.opponent import get_opponent, DEFAULT_OPPONENT_PROFILE


# ---------------------------------------------------------------------------
# Goal collision resolution
# ---------------------------------------------------------------------------
# Margin for long goals — keeps robot body ~1" clear of the goal face
_GOAL_MARGIN = ROBOT_W / 2 + 1.0   # 8.5"
# Wider margin for center X arms — keeps robot body ~2" clear of each arm face
_CENTER_GOAL_MARGIN = ROBOT_W / 2 + 2.0   # 9.5"

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
# Final-approach stop tolerance for the scoring leg. The default move tolerance
# (ARRIVAL_DIST, 2") is wider than the standoff's margin inside SCORE_RANGE, so a
# robot stopping on the default tolerance lands short of scoring range and can't
# fire _do_back_in_scoring(). Nose in to within that margin instead. Derived from
# geometry (floored) so it tracks SCORE_RANGE / robot size changes.
_SCORE_FINAL_ARRIVAL = max(0.3, (SCORE_RANGE - _SCORE_APPROACH_GAP) - 0.25)

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


# Opponent deposit references: the SAME scoring approach poses the opponent
# navigates to (long-goal ends / center tips), each with its goal name + points.
# The opponent's deposit range is checked against these — matching where it
# actually drives — instead of the goal CENTERS (which sit inside the goal body
# and are far from the ends, so the opponent could camp a goal and never score).
_OPP_SCORE_REFS = [
    (_RIGHT_GOAL_TOP_POS, "our_long",   LONG_GOAL_POINTS),
    (_RIGHT_GOAL_BOT_POS, "our_long",   LONG_GOAL_POINTS),
    (_LEFT_GOAL_TOP_POS,  "opp_long",   LONG_GOAL_POINTS),
    (_LEFT_GOAL_BOT_POS,  "opp_long",   LONG_GOAL_POINTS),
    (_MID_NE_POS,         "center_mid", CENTER_GOAL_POINTS),
    (_LOW_SW_POS,         "center_mid", CENTER_GOAL_POINTS),
    (_MID_NW_POS,         "center_low", CENTER_GOAL_POINTS),
    (_LOW_SE_POS,         "center_low", CENTER_GOAL_POINTS),
]


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
# Side-lane bridges: the LOW/HIGH corridors above sit inside the center-X arm
# margin for a straight vertical run, so bottom↔top traversal needs a midpoint in
# the clear lane between each long goal's nav margin and the X-arm margin. Verified
# clear end-to-end (y=40↔104): x≈35 on the left (goal margin ends ~34.1, arm margin
# starts ~44) and x≈109 on the right (arm margin ends ~100, goal margin starts
# ~109.9). Without these the planner can't route a robot from one half to the other.
_NAV_LEFT_LANE   = np.array([ 35.0,  72.0])  # left  side-lane (goal↔X gap)
_NAV_RIGHT_LANE  = np.array([109.0,  72.0])  # right side-lane (X↔goal gap)

# ── Creeping exploration ────────────────────────────────────────────────────
# When a robot keeps choosing COLLECT but finds no reachable ball, it has cleaned
# out its area. Robot.explore_barren counts those consecutive "blind" decisions;
# the scan frontier (explore_y) then creeps from the bottom band toward the top so
# the robot expands into unexplored territory (notably the top half) instead of
# oscillating in the picked-clean bottom. Resets the moment it collects / sees a
# reachable ball (see the barren update at the end of step()).
_EXPLORE_Y_LO        = 36.0   # bottom scan band y
_EXPLORE_Y_HI        = 108.0  # top scan band y
_EXPLORE_BARREN_FULL = 6      # barren decisions for the frontier to reach the top
_EXPLORE_CREEP_AT    = 2      # barren decisions before creep overrides a known quadrant
# Reachable explore anchors spanning the field bottom→top (used by the creep to
# pick the point nearest the rising frontier). Mid-field side lanes are included
# so the robot has a routable stepping stone between halves.
_EXPLORE_POOL = [
    np.array([95.0,  36.0]), np.array([49.0,  36.0]),   # SE, SW   (bottom)
    _NAV_LEFT_LANE.copy(),   _NAV_RIGHT_LANE.copy(),     # side lanes (mid)
    np.array([72.0, 103.0]),                             # above center X
    np.array([95.0, 108.0]), np.array([49.0, 108.0]),   # NE, NW   (top)
]

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

    Park platforms are soft-avoided: a direct/detour leg that cuts through a park
    zone is penalised so a park-free route wins when one exists, but the robot
    still falls back to crossing (or to a target inside a park) rather than
    stranding itself.
    """
    from sim.route_planner import (_los_blocked, _NAV_MARGIN,
                                   seg_crosses_park, point_in_park, _PARK_AVOID)

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
        # Staging point: directly below (bottom end) or above (top end) the goal
        # at the goal's centre x. The final leg is therefore a straight N/S drive
        # so the robot's heading aligns with the goal opening before arrival.
        # Path layout: start → corridor → stage → final (approach).
        if is_right_long:
            stage = (_STAGE_R_TOP if is_top else _STAGE_R_BOT).copy()
        else:
            stage = (_STAGE_L_TOP if is_top else _STAGE_L_BOT).copy()

        # If the robot has already passed the corridor toward the goal (its x is
        # between the corridor and the goal), skip the corridor (and go-around)
        # but still route via the stage unless the robot is already in the goal's
        # x-column (where only a short vertical drive remains).
        start_x = float(start[0])
        corr_x  = float(corridor[0])
        past_corridor = (
            (is_right_long and start_x >= corr_x) or
            (is_left_long  and start_x <= corr_x)
        )
        if past_corridor:
            goal_cx_approx = float(stage[0])
            in_goal_column = abs(float(start[0]) - goal_cx_approx) < 3.0
            pts = [final] if in_goal_column else [stage, final]
            compact: list[np.ndarray] = []
            for p in pts:
                if not compact:
                    if float(np.linalg.norm(p - start)) > 0.75:
                        compact.append(p.copy())
                elif float(np.linalg.norm(p - compact[-1])) > 0.75:
                    compact.append(p.copy())
            return compact if compact else [final.copy()]

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

    # Soft park avoidance applies unless the destination itself is in a park zone
    # (then the robot must enter, so don't fight it).
    avoid_park = _PARK_AVOID and not point_in_park(float(final[0]), float(final[1]))
    direct_clear = not _los_blocked(start[0], start[1], final[0], final[1], margin=_NAV_MARGIN)

    # Direct path (most common for collect actions) — take it when clear, and
    # (unless heading into a park) when it doesn't cut through a park zone.
    if direct_clear and not (avoid_park and
                             seg_crosses_park(float(start[0]), float(start[1]),
                                              float(final[0]), float(final[1]))):
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
        _NAV_LEFT_LANE, _NAV_RIGHT_LANE,   # side-lane bridges for bottom↔top
    ]
    sx, sy, fx, fy = float(start[0]), float(start[1]), float(final[0]), float(final[1])

    # Park crossing adds a large but FINITE cost so a park-free route always wins
    # when one exists, yet a crossing route is still chosen over stranding.
    _PARK_PENALTY = 1000.0
    def _park_pen(x0: float, y0: float, x1: float, y1: float) -> float:
        return _PARK_PENALTY if (avoid_park and seg_crosses_park(x0, y0, x1, y1)) else 0.0

    best_cost = 1e18
    best_path: list = []

    # 1-stop search
    for c in _pool:
        cx, cy = float(c[0]), float(c[1])
        if (not _los_blocked(sx, sy, cx, cy, margin=_NAV_MARGIN) and
                not _los_blocked(cx, cy, fx, fy, margin=_NAV_MARGIN)):
            cost = (math.hypot(cx - sx, cy - sy) + math.hypot(fx - cx, fy - cy)
                    + _park_pen(sx, sy, cx, cy) + _park_pen(cx, cy, fx, fy))
            if cost < best_cost:
                best_cost = cost
                best_path = [c.copy(), final.copy()]

    # 2-stop search — run even when a 1-stop exists so a park-free 2-stop can
    # beat a park-crossing 1-stop (distances otherwise keep the 1-stop ahead).
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
                cost = (math.hypot(c1x - sx, c1y - sy) +
                        math.hypot(c2x - c1x, c2y - c1y) +
                        math.hypot(fx - c2x, fy - c2y) +
                        _park_pen(sx, sy, c1x, c1y) +
                        _park_pen(c1x, c1y, c2x, c2y) +
                        _park_pen(c2x, c2y, fx, fy))
                if cost < best_cost:
                    best_cost = cost
                    best_path = [c1.copy(), c2.copy(), final.copy()]

    # Prefer a park-free corridor route (cost under the penalty floor).
    if best_path and best_cost < _PARK_PENALTY:
        return best_path

    # No park-free corridor route exists. If the direct path is obstacle-clear,
    # take it (shortest park crossing) instead of a longer crossing detour.
    if direct_clear:
        return [final]

    # Goal-blocked AND every corridor route crosses a park → best available
    # route (may cross a park), or [] if nothing routes at all. Returning [] makes
    # the caller keep the robot put rather than drive through a goal face.
    return best_path


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

# Opponent jam threshold: if an opponent moves less than this (inches) over a full
# decision AND scores nothing, it's pinned against a goal body/wall and gets peeled
# toward field center next decision (see the opponent loop in step()).
_OPP_JAM_MIN_DIST = 3.0


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

        # Priority 3 — no accessible target in vision cone or via LOS.
        # Explore: go to the scan position in the field quadrant that has the
        # most remaining blue balls AND that the robot can actually route to.
        #
        # The expanded center X arm (_NAV_MARGIN ≈ 11.6") can block paths to
        # some quadrant scan positions depending on the robot's location (e.g.
        # the NE arm tip with margin reaches x≈100", blocking paths from the
        # right side to the upper-left/right scan positions). We verify each
        # candidate has a non-empty nav path before committing.
        _SCAN_POS = {
            "SE": np.array([ 95.0,  36.0]),
            "NE": np.array([ 95.0, 108.0]),
            "NW": np.array([ 49.0, 108.0]),
            "SW": np.array([ 49.0,  36.0]),
        }
        on_blue = [obj for obj in field.objects
                   if obj.status == OBJ_ON_FIELD and obj.color == BALL_BLUE]
        if not on_blue:
            # No blue balls at all — stay near center so policy sees goals.
            return np.array([FIELD_W * 0.5, FIELD_H * 0.5])

        # Priority 2.5 — rotate-to-look before committing to a long scan drive.
        # For each of 8 evenly-spaced candidate headings, count how many on-field
        # blue balls would fall in the vision cone from the robot's current position
        # if it were facing that direction (uses bare goal LOS, no nav margin).
        # The look-point is 30" in the winning direction, but ONLY returned if it
        # is actually navigable — if the nav planner returns [] (e.g. the robot is
        # pressed against a goal wall and every northward look-point is inside the
        # expanded obstacle zone), we fall through to P3 which routes to a verified
        # scan position rather than leaving the wq empty and freezing the robot.
        from sim.route_planner import _los_blocked as _cam_los, _near_any_goal as _rtl_near
        _best_look_cnt   = 0
        _best_look_lx: float | None = None
        _best_look_ly: float | None = None
        for _k in range(8):
            _theta = _k * (math.pi / 4)
            _cnt = 0
            for _obj in on_blue:
                _dx, _dy = _obj.x - robot.x, _obj.y - robot.y
                _dist = math.sqrt(_dx * _dx + _dy * _dy)
                if _dist < 3.0 or _dist > VISION_RANGE:
                    continue
                if _rtl_near(_obj.x, _obj.y):
                    continue
                _ang_to = math.atan2(_dy, _dx)
                _diff   = abs(((_theta - _ang_to + math.pi) % (2 * math.pi)) - math.pi)
                if _diff > VISION_HALF_ANGLE:
                    continue
                if _cam_los(robot.x, robot.y, _obj.x, _obj.y):
                    continue
                _cnt += 1
            if _cnt > _best_look_cnt:
                _lx = float(np.clip(robot.x + 30.0 * math.cos(_theta),
                                    7.5, FIELD_W - 7.5))
                _ly = float(np.clip(robot.y + 30.0 * math.sin(_theta),
                                    7.5, FIELD_H - 7.5))
                if _build_nav_waypoints(robot.position, np.array([_lx, _ly])):
                    _best_look_cnt = _cnt
                    _best_look_lx  = _lx
                    _best_look_ly  = _ly
        if _best_look_lx is not None:
            return np.array([_best_look_lx, _best_look_ly])

        counts = {q: 0 for q in _SCAN_POS}
        for obj in on_blue:
            q = ("N" if obj.y >= FIELD_H * 0.5 else "S") + \
                ("E" if obj.x >= FIELD_W * 0.5 else "W")
            counts[q] += 1

        robot_q = ("N" if robot.y >= FIELD_H * 0.5 else "S") + \
                  ("E" if robot.x >= FIELD_W * 0.5 else "W")
        robot_near_scan = (
            float(np.linalg.norm(robot.position - _SCAN_POS[robot_q])) < 30.0
        )

        # Richest reachable quadrant that has KNOWN balls; skip the one the robot
        # is already scanning, and any whose nav path would be empty (unreachable).
        known_target: np.ndarray | None = None
        for q in sorted(counts, key=counts.__getitem__, reverse=True):
            if counts[q] == 0:
                break                       # no more known balls anywhere
            if q == robot_q and robot_near_scan:
                continue
            cand = _SCAN_POS[q].copy()
            if _build_nav_waypoints(robot.position, cand):
                known_target = cand
                break

        # Creeping exploration: once the robot has cycled several decisions without
        # finding a reachable ball (explore_barren), or when no known quadrant is
        # routable, push a frontier northward so it expands into new territory (the
        # top half) instead of oscillating in the cleaned-out bottom. The frontier
        # y creeps from the bottom band to the top as barren grows; we head for the
        # reachable explore anchor nearest that frontier, preferring the opposite
        # side and skipping anywhere we're already sitting.
        barren = int(getattr(robot, "explore_barren", 0))
        if known_target is None or barren >= _EXPLORE_CREEP_AT:
            # Triangle-wave frontier: sweep the band UP to the top then back DOWN,
            # repeating, so a persistently barren robot covers the whole field
            # instead of camping at the top once it maxes out. A monotonic creep
            # could strand it in a top corner pocket, never re-acquiring balls that
            # are only visible/navigable from mid- or low-field.
            period   = 2 * _EXPLORE_BARREN_FULL
            phase    = barren % period
            frontier = (phase if phase <= _EXPLORE_BARREN_FULL
                        else period - phase) / _EXPLORE_BARREN_FULL
            explore_y = _EXPLORE_Y_LO + frontier * (_EXPLORE_Y_HI - _EXPLORE_Y_LO)
            robot_on_left = robot.x < FIELD_W * 0.5
            best_pt: np.ndarray | None = None
            best_key = float("inf")
            for p in _EXPLORE_POOL:
                if float(np.linalg.norm(robot.position - p)) < 16.0:
                    continue                # already basically here — keep moving
                if not _build_nav_waypoints(robot.position, p):
                    continue                # not routable from here
                same_side = (float(p[0]) < FIELD_W * 0.5) == robot_on_left
                key = abs(float(p[1]) - explore_y) + (6.0 if same_side else 0.0)
                if key < best_key:
                    best_key = key
                    best_pt  = p
            if best_pt is not None:
                return best_pt.copy()

        if known_target is not None:
            return known_target.copy()

        # No quadrant scan position is routable — fall back to the nearest
        # corridor point that IS reachable and meaningfully far from the robot.
        # Corridor points are the pre-verified safe waypoints used by the path
        # planner itself, so at least one will always be reachable.
        _CORRIDOR_FALLBACKS = [
            _NAV_RIGHT_LOW, _NAV_RIGHT_HIGH,
            _NAV_LEFT_LOW,  _NAV_LEFT_HIGH,
            _NAV_BELOW_X,   _NAV_ABOVE_X,
            _NAV_LEFT_LANE, _NAV_RIGHT_LANE,
        ]
        best_corr: np.ndarray | None = None
        best_dist = float("inf")
        for c in _CORRIDOR_FALLBACKS:
            d = float(np.linalg.norm(c - robot.position))
            if d < 12.0:          # already here — skip
                continue
            if not _build_nav_waypoints(robot.position, c):
                continue          # not routable
            if d < best_dist:
                best_dist = d
                best_corr = c
        if best_corr is not None:
            return best_corr.copy()

        # Absolute last resort: idle (caller treats None as no waypoints).
        return None
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
        # Reachable approach pose for the nearest long-goal end (either wall — the
        # opponent can score in both). The raw goal reference sits against the wall
        # and pins the robot; the approach point is navigable.
        approach, *_ = _nearest_long_goal_target(robot)
        return approach.copy()
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
        log_decisions: bool = False,
        log_dir: str = "logs",
        opponent_profile=None,
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

        # Decision logger — disabled during training, enable in demo/eval via
        # env.enable_logging() or log_decisions=True constructor arg.
        self._decision_logger = None
        if log_decisions:
            from sim.decision_logger import DecisionLogger
            self._decision_logger = DecisionLogger(log_dir)

        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])
        self.observation_space = spaces.Dict({
            "robot_0": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            "robot_1": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        })

        self.opponent_policy = get_opponent(opponent_type)
        self.field           = Field()
        # Opponent physical profile (speed / scoring rate / capacity). Stamped
        # onto the opponent robots so they can differ from our robot. Swap the
        # profile later to train against different opponent types.
        self.opponent_profile = opponent_profile or DEFAULT_OPPONENT_PROFILE
        self._apply_opponent_profile()
        self._renderer       = None

        # ── Perceived goal state (camera-based belief) ──────────────────────
        # field.goal_state is GROUND TRUTH. goal_belief is what the TEAM believes
        # is scored in each goal, updated only from the robots' camera FOV (and
        # their own scoring/descoring). It can go STALE: if an opponent descores a
        # goal no ally is looking at, the belief keeps the last-seen contents until
        # a robot sees that goal again. The policy/observation and decision
        # heuristics act on this belief — the robot never assumes a scored ball
        # stays forever; what it last saw in its FOV is its game-state knowledge.
        # Toggle off to fall back to omniscient ground-truth (e.g. for an
        # easier full-observability training curriculum).
        self.use_goal_belief = True
        self.goal_belief     = GoalState()
        self._goal_in_view   = {"our_long": False, "opp_long": False,
                                "center_mid": False, "center_low": False}
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
        # Goals start empty — that's known, so the belief starts accurate (empty).
        self.goal_belief.reset()
        for _g in self._goal_in_view:
            self._goal_in_view[_g] = False
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
        self.opp_prev_pos:  list[np.ndarray | None] = [None, None]
        self.opp_prev_oscore: list[float] = [0.0, 0.0]

        if self._decision_logger is not None:
            self._decision_logger.new_episode()

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
        self.opp_prev_pos   = [None, None]
        self.opp_prev_oscore = [0.0, 0.0]
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

    def enable_logging(self, log_dir: str = "logs") -> None:
        """Activate the decision logger (call after reset, before stepping)."""
        from sim.decision_logger import DecisionLogger
        if self._decision_logger is not None:
            self._decision_logger.close()
        self._decision_logger = DecisionLogger(log_dir)

    def action_masks(self, robot_id: int = 0) -> np.ndarray:
        """Boolean mask of shape (NUM_ACTIONS,) — True = action valid this decision.

        Rules:
          DESCORE_OPP_LONG  only when opp's long goal has opp-colored balls
          DESCORE_CENTER    only when center goals contain opp-colored balls
          EJECT_WRONG_COLOR only when robot holds wrong-color balls
          SCORE_*           always available — auto-chains to COLLECT first if empty
          All others always available (COLLECT, DEFEND, IDLE).
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        if robot_id >= self.num_allies:
            return mask
        robot = self.field.allies[robot_id]
        # Mask descore on the PERCEIVED goal state — the robot can only choose to
        # descore a goal it BELIEVES holds opponent balls (what it has seen), not
        # via omniscient truth. If its belief is stale it may drive over and find
        # nothing (and the mid-step replanner re-routes); if it hasn't seen an
        # opponent score yet, descore stays unavailable until it looks.
        gs = self.goal_belief if self.use_goal_belief else self.field.goal_state

        # Descore opp long: only when their goal holds OPPONENT-colored balls.
        # (try_descore only removes opp-scored balls, so descoring a goal full of
        # our own balls is a no-op that just parks the robot at the wall — and
        # with no opponents there's never anything to descore. Mirror the center
        # rule, which already checks color.)
        opp_in_long = any(color != BALL_BLUE for _, color in gs.opp_long)
        if not opp_in_long:
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
        if self._decision_logger is not None:
            self._decision_logger.mark_step_start()

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
        # Track action source for the logger BEFORE masking overrides
        _src = ["policy", "policy"]
        for _i, (_r, _a) in enumerate(((raw_a0, a0), (raw_a1, a1))):
            if _r != _a:
                _src[_i] = "committed"
        if not self.action_masks(robot_id=0)[a0]:
            a0 = int(Action.COLLECT_NEAREST_BALL)
            _src[0] = "mask_fallback"
        if not self.action_masks(robot_id=1)[a1]:
            a1 = int(Action.COLLECT_NEAREST_BALL)
            _src[1] = "mask_fallback"

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

        # Obstacle-aware nav queue per opponent (mirrors ally_wq). Without this the
        # opponent drives in a straight line and jams against the center X / long-goal
        # bodies, appearing to "stop" until it happens to pick an unobstructed target.
        opp_wq: list[list] = [[] for _ in range(2)]
        for oi in range(self.num_opponents):
            opp = self.field.opponents[oi]
            # Jam detection (across decisions): if the opponent barely moved AND
            # didn't score last decision, it's pinned against a goal body / wall
            # (its raw goal-reference target is an unreachable pose). Peel it toward
            # field center to break the deadlock; normal targeting resumes next
            # decision. A productive goal-camp (motionless but scoring) is NOT
            # treated as a jam.
            jammed = False
            if self.opp_prev_pos[oi] is not None:
                moved  = float(np.linalg.norm(opp.position - self.opp_prev_pos[oi]))
                scored = self.field.opponent_score > self.opp_prev_oscore[oi]
                jammed = moved < _OPP_JAM_MIN_DIST and not scored
            self.opp_prev_pos[oi]    = opp.position.copy()
            self.opp_prev_oscore[oi] = float(self.field.opponent_score)

            opp_state = {
                "position":       opp.position,
                "balls_held":     opp.balls_held,
                "our_score":      self.field.my_score,
                "opp_score":      self.field.opponent_score,
                "us_scored_count": len(self.field.scored_by_us_indices()),
            }
            opp_action = self.opponent_policy(opp_state, self.rng)
            tgt = _opp_action_to_target(opp_action, self.field, opp)
            if tgt is None:
                # Never let the opponent freeze (IDLE/EJECT, or COLLECT with no
                # reachable ball all yield no target). Fall back to the nearest
                # ball, or failing that the nearest goal, so it always keeps
                # moving instead of stopping in place.
                tgt = self.field.nearest_on_field_target(opp.position)
                if tgt is None:
                    tgt = min(
                        (OPP_LONG_GOAL, OUR_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL),
                        key=lambda g: float(np.linalg.norm(opp.position - g)),
                    ).copy()
            if jammed:
                tgt = np.array([FIELD_W * 0.5, FIELD_H * 0.5])
            self.opp_targets[oi] = tgt
            opp_wq[oi] = _build_nav_waypoints(opp.position, tgt)

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
            elif act in SCORE_INTENTS and robot.balls_held == 0:
                # SCORE chosen but nothing to score — navigate to collect instead
                # so the robot finds balls rather than running empty to a goal.
                final = _action_to_target(Action.COLLECT_NEAREST_BALL, self.field, robot)
            else:
                final = _action_to_target(act, self.field, robot)
            if final is None:
                return []
            wq = _build_nav_waypoints(robot.position, final)
            # If the target is geometrically unreachable (arm / goal body blocks
            # every corridor), fall back to COLLECT so the robot explores instead
            # of freezing. Applies to DESCORE and DEFEND — not COLLECT itself
            # (would loop) or IDLE/EJECT (those have no nav target anyway).
            if not wq and act not in (Action.COLLECT_NEAREST_BALL,
                                      Action.IDLE, Action.EJECT_WRONG_COLOR):
                live_actions[idx] = int(Action.COLLECT_NEAREST_BALL)
                fb = _action_to_target(Action.COLLECT_NEAREST_BALL, self.field, robot)
                if fb is not None:
                    wq = _build_nav_waypoints(robot.position, fb)
            return wq

        ally_wq: list[list] = [_build_wq(0), _build_wq(1)]
        # Capture first waypoint as the decision-level "target" for logging
        _log_targets = [ally_wq[i][0] if ally_wq[i] else None for i in range(2)]

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

        # Set when a scoring approach is abandoned mid-decision (timeout, see the
        # stall handler). Suppresses the chain-back replanner so the robot doesn't
        # immediately re-commit to the same unreachable score for the rest of the
        # decision after we've handed it off to COLLECT.
        abandon_score = [False, False]

        # Stall detection: if a robot hasn't moved STALL_MIN_DIST inches in
        # STALL_CHECK_INTERVAL ticks, that's one "strike". After STALL_MAX_STRIKES
        # consecutive strikes the waypoint queue is cleared so the robot gives up
        # on the current target and idles until the next decision.
        # At 60 Hz: 30 ticks = 1.5 s, 2 strikes = 3 s total stall.
        _STALL_CHECK_INTERVAL = 30    # ticks between progress samples
        _STALL_MIN_DIST       = 4.0   # inches required per interval (not stalled)
        _STALL_MAX_STRIKES    = 2     # consecutive missed intervals before clearing
        stall_pos_snap   = [r.position.copy() for r in self.field.allies]
        stall_tick_count = [0, 0]
        stall_strikes    = [0, 0]

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
            # Descore/defend decisions act on the PERCEIVED goal state (what the
            # team has seen), not omniscient truth — the robot can't react to an
            # opponent score it hasn't observed yet.
            _gs_dec = self.goal_belief if self.use_goal_belief else self.field.goal_state
            n_opp_scored = (
                sum(1 for _, c in (_gs_dec.our_long + _gs_dec.opp_long
                                   + _gs_dec.center_mid + _gs_dec.center_low)
                    if c == BALL_RED)
                if self.num_opponents > 0 else 0
            )
            our_long_count = len(_gs_dec.our_long)
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
                # (skipped once a score approach has been abandoned this decision).
                if (act == Action.COLLECT_NEAREST_BALL
                        and original in SCORE_INTENTS
                        and robot.balls_held > 0
                        and not abandon_score[idx]):
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
                        # New nav target — reset stall tracking so the robot gets
                        # a fresh 3-second window to reach the new destination.
                        stall_strikes[idx]    = 0
                        stall_tick_count[idx] = 0
                        stall_pos_snap[idx]   = self.field.allies[idx].position.copy()

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
                reverse_drive = False
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
                        # Back-in approach: on the final leg to the scoring pose
                        # (last waypoint, on the goal axis) drive REAR-first so the
                        # robot's back leads into the goal mouth and it arrives
                        # already aligned to score — no ~180° spin at the opening,
                        # which is the failure mode where it freezes "facing the
                        # intermediate point" instead of feeding the goal.
                        if (target is not None and len(ally_wq[idx]) <= 1
                                and float(np.linalg.norm(
                                    np.asarray(target, dtype=np.float64) - score_pos)) < 1.5):
                            reverse_drive = True
                else:
                    target = _current_target(idx)
                if target is not None:
                    if reverse_drive:
                        # Back in and nose fully onto the score pose (tight
                        # tolerance) so we land within scoring range.
                        arrived = robot.move_toward_point(
                            target, reverse=True, arrival_dist=_SCORE_FINAL_ARRIVAL)
                    else:
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

            # ── Stall detection ───────────────────────────────────────────────
            # Skip counting during intentional pauses (action lock-in, scoring
            # dwell). Accumulated movement is sampled every _STALL_CHECK_INTERVAL
            # ticks; if the robot covered < _STALL_MIN_DIST twice in a row its
            # waypoint queue is cleared so it stops fighting an unreachable target.
            for idx in range(self.num_allies):
                # Intentional pauses: reset snapshot so pause ticks aren't counted.
                if self.action_pause_ticks[idx] > 0 or scoring_dwell_left[idx] > 0:
                    stall_pos_snap[idx]   = self.field.allies[idx].position.copy()
                    stall_tick_count[idx] = 0
                    continue
                # No waypoints → nothing to stall on.
                if not ally_wq[idx]:
                    stall_strikes[idx]    = 0
                    stall_tick_count[idx] = 0
                    stall_pos_snap[idx]   = self.field.allies[idx].position.copy()
                    continue
                stall_tick_count[idx] += 1
                if stall_tick_count[idx] >= _STALL_CHECK_INTERVAL:
                    moved = float(np.linalg.norm(
                        self.field.allies[idx].position - stall_pos_snap[idx]
                    ))
                    stall_pos_snap[idx]   = self.field.allies[idx].position.copy()
                    stall_tick_count[idx] = 0
                    if moved < _STALL_MIN_DIST:
                        stall_strikes[idx] += 1
                        if stall_strikes[idx] >= _STALL_MAX_STRIKES:
                            ally_wq[idx].clear()
                            stall_strikes[idx] = 0
                            # Timeout: no progress for _STALL_MAX_STRIKES intervals.
                            # If we were scoring but never reached scoring range and
                            # still have room to carry, abandon the approach and go
                            # collect instead of idling out the rest of the decision.
                            # Drop the cross-decision commit so it isn't re-imposed.
                            act_to = Action(live_actions[idx])
                            if act_to in SCORE_INTENTS:
                                robot_to = self.field.allies[idx]
                                _, goal_pos_to, *_ = self._get_scoring_target(idx, act_to)
                                out_of_range = float(np.linalg.norm(
                                    robot_to.position - goal_pos_to)) >= SCORE_RANGE
                                if out_of_range and robot_to.balls_held < MAX_CARRY:
                                    self._score_commit_left[idx] = 0
                                    self._score_committed_action[idx] = int(Action.IDLE)
                                    abandon_score[idx] = True
                                    live_actions[idx]  = int(Action.COLLECT_NEAREST_BALL)
                                    ally_wq[idx]       = _build_wq(idx)
                                    live_action_age[idx] = 0
                                    stall_pos_snap[idx] = self.field.allies[idx].position.copy()
                    else:
                        stall_strikes[idx] = 0

            for oi in range(self.num_opponents):
                opp_r = self.field.opponents[oi]
                if opp_wq[oi]:
                    # Follow the obstacle-aware route, advancing waypoints on arrival.
                    arrived = opp_r.move_toward_point(opp_wq[oi][0])
                    if arrived and len(opp_wq[oi]) > 1:
                        opp_wq[oi].pop(0)
                    elif arrived:
                        opp_wq[oi].clear()
                elif self.opp_targets[oi] is not None:
                    # No route found (target unreachable) — straight-line fallback.
                    opp_r.move_toward_point(self.opp_targets[oi])
                _resolve_goal_collisions(opp_r)

            # Refresh perceived goal state from camera FOV (after movement, so
            # headings are current). Scanning/driving past a goal updates what the
            # team knows is scored in it; goals out of view keep their last belief.
            self._update_goal_belief()

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

        # Creeping-exploration barren counter. Grows each decision the robot is
        # trying to collect but has no reachable ball (vision cone via P1, or
        # navigable LOS via P2) — i.e. it's cycling without finding anything new.
        # Resets the instant it collects or a reachable target reappears. Read by
        # _action_to_target() to push the scan frontier north (explore the top).
        from sim.route_planner import compute_collection_route
        for idx in range(self.num_allies):
            robot = self.field.allies[idx]
            if self.collected_this_step[idx] > 0:
                robot.explore_barren = 0
                continue
            collecting = (
                Action(live_actions[idx]) == Action.COLLECT_NEAREST_BALL
                or Action(self.current_actions[idx]) == Action.COLLECT_NEAREST_BALL
            )
            if not collecting:
                continue
            route = compute_collection_route(
                robot.position, self.field,
                already_held=robot.balls_held, max_volley=1, robot=robot,
            )
            has_target = bool(route) or (
                self.field.nearest_navigable_target(robot.position) is not None
            )
            if has_target:
                robot.explore_barren = 0
            else:
                # Keep counting well past _EXPLORE_BARREN_FULL so the triangle-wave
                # frontier in _action_to_target() keeps cycling (sweeps up↔down)
                # rather than saturating at the top. Capped only to stay tidy.
                robot.explore_barren = min(robot.explore_barren + 1, 600)

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

        # Decision logging (no-op when logger is None)
        if self._decision_logger is not None:
            for _li in range(self.num_allies):
                _travelled = float(np.linalg.norm(
                    self.field.allies[_li].position - step_start_pos[_li]
                ))
                self._decision_logger.log(
                    env=self,
                    robot_id=_li,
                    policy_action=raw_a0 if _li == 0 else raw_a1,
                    executed_action=int(self.current_actions[_li]),
                    action_source=_src[_li],
                    target=_log_targets[_li] if _li < len(_log_targets) else None,
                    reward=r0 if _li == 0 else r1,
                    travelled_in=_travelled,
                )

        # On episode end, stash the FINAL scores in info. SB3's VecEnv auto-resets
        # a done env before the training callbacks run, so reading env.field after
        # the step yields the freshly-reset 0 — callbacks must read these instead.
        info: dict = {}
        if self.done:
            info["episode_score"]     = float(self.field.my_score)
            info["episode_opp_score"] = float(self.field.opponent_score)
        return self._get_obs(), (r0, r1), self.done, False, info

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

    # ------------------------------------------------------------------
    # Perceived goal state (camera FOV belief)
    # ------------------------------------------------------------------
    # Sample points per goal that the camera must see to "read" its contents.
    # Long goals: bar centre + both ends. Center goals: their reference points.
    _GOAL_VIEW_POINTS = {
        "our_long":   [(_RIGHT_GOAL_CX, _CY),
                       (_RIGHT_GOAL_CX, LONG_GOAL_Y_MIN),
                       (_RIGHT_GOAL_CX, LONG_GOAL_Y_MAX)],
        "opp_long":   [(_LEFT_GOAL_CX, _CY),
                       (_LEFT_GOAL_CX, LONG_GOAL_Y_MIN),
                       (_LEFT_GOAL_CX, LONG_GOAL_Y_MAX)],
        "center_mid": [(float(CENTER_MID_GOAL[0]), float(CENTER_MID_GOAL[1]))],
        "center_low": [(float(CENTER_LOW_GOAL[0]), float(CENTER_LOW_GOAL[1]))],
    }

    def _observe_goal(self, gname: str) -> None:
        """Copy one goal's ground-truth contents into the belief (robot read it)."""
        self.goal_belief._list(gname)[:] = list(getattr(self.field.goal_state, gname))

    def _update_goal_belief(self) -> None:
        """Refresh the perceived goal state from the robots' camera FOV.

        For each goal, if ANY ally currently sees it (cone + range, with the
        center X occluding long-goal sightlines), copy ground truth into the
        belief. Goals nobody is looking at keep their last-seen contents and may
        therefore be stale (e.g. after an unseen opponent descore).
        """
        from sim.route_planner import goal_in_fov
        for gname, pts in self._GOAL_VIEW_POINTS.items():
            is_long = gname.endswith("long")
            seen = False
            for r in self.field.allies:
                for gx, gy in pts:
                    if goal_in_fov(r, gx, gy, blocked_by_arms=is_long):
                        seen = True
                        break
                if seen:
                    break
            self._goal_in_view[gname] = seen
            if seen:
                self._observe_goal(gname)

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
                self._observe_goal("opp_long")   # we know what we just removed

        elif action == Action.DESCORE_CENTER:
            pts = self.field.try_descore(robot, CENTER_MID_GOAL, self.rng)
            if pts == 0:
                pts = self.field.try_descore(robot, CENTER_LOW_GOAL, self.rng)
            if pts > 0:
                self.descore_events[idx] += pts
                robot.actions_succeeded += 1
                self._observe_goal("center_mid")
                self._observe_goal("center_low")

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
            # The robot knows what it just deposited even though the goal is now
            # behind it (it backed in) — update the belief from its own scoring.
            self._observe_goal(gname)
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

    def _apply_opponent_profile(self) -> None:
        """Stamp the opponent physical profile onto the opponent robots.

        Capabilities persist across episodes (Robot.reset() leaves them alone),
        so this only needs to run once at construction. Swap self.opponent_profile
        and call again to retune (e.g. for a different opponent type later).
        """
        p = self.opponent_profile
        for opp in self.field.opponents:
            opp.speed_scale    = p.speed_scale
            opp.capacity       = p.capacity
            opp.score_interval = p.score_interval

    def _check_opp_effects(self):
        for oi in range(self.num_opponents):
            opp = self.field.opponents[oi]
            self.field.opp_try_collect(opp, self.rng)

            # Scoring: the opponent must DWELL in range of a goal for its
            # score_interval (profile) before depositing its load — a bit slower
            # than our timed back-in, instead of the old instant dump.
            in_range = False
            if opp.balls_held > 0:
                # Opponent can score in EVERY goal (either color scores in any
                # goal). Deposit range is checked against the goal ENDS/TIPS it
                # actually backs into (the same poses the ally scores at), not the
                # goal centers — so it can no longer camp a goal end out of range
                # and never score.
                for ref, gname, pts in _OPP_SCORE_REFS:
                    if float(np.linalg.norm(opp.position - ref)) < SCORE_RANGE:
                        in_range = True
                        opp.score_timer += DT
                        if opp.score_timer >= opp.score_interval:
                            opp.score_timer = 0.0
                            self.field.opp_try_score(opp, ref, pts, gname=gname)
                        break
            if not in_range:
                opp.score_timer = 0.0   # not lined up on a goal — reset the dwell

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
        # From the PERCEIVED goal state — the policy only knows control it has seen.
        _gs_obs  = self.goal_belief if self.use_goal_belief else self.field.goal_state
        ctrl_now = _gs_obs.compute_quadrant_control()
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
        # Sourced from the PERCEIVED goal state (camera belief) so the policy acts
        # on what it has actually seen, not omniscient truth.
        _LONG_CAP   = 14.0
        _CTR_CAP    = 14.0  # center_mid (7) + center_low (7)
        gs = self.goal_belief if self.use_goal_belief else self.field.goal_state
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
