"""Reward function for VEX Push Back 2025-2026 single-robot RL.

All weights live in REWARD_WEIGHTS so the panel can display them alongside
the per-step contribution values. Per-step reward = weight × what_happened.

Ball-color scoring model:
  Blue balls  → our points  (reward)
  Red balls   → opp points  (penalty — don't feed them)
  Control zones add 10 pts each → incremental reward for holding zones.

Shaping penalties:
  - Goal-collision penalty: per-tick when the robot is pushed out of a goal.
  - Low-progress penalty:   non-IDLE action that moved <6" in a decision.
  - Wasted-action penalty:  SCORE/DESCORE/DEFEND with no effect possible.
  - Holding urgency:        per-ball-held per decision (score what you have).
  - Heading jitter:         penalty for total per-tick heading change above
                            ~one full travel turn (catches circling/wiggling).

Bonuses with time/state context:
  - Center bonus:           extra reward when balls land in mid/low goals.
  - Control gain:           one-time bonus when we *take* a new quadrant.
  - Late-game ctrl boost:   ctrl_us reward grows as time runs out.
  - Smooth win margin:      sigmoid bonus so a 20-pt win is much more than a
                            5-pt win (linear made margin under-rewarded).

Each call to compute_reward stores a {component: value} breakdown on
env.last_reward_breakdown[robot_id] so the renderer panel can show *why*
the last decision earned what it did.
"""

import math
from sim.config import Action, BALL_BLUE, BALL_RED, CONTROL_BONUS_PTS, MATCH_DURATION, OBJ_ON_FIELD


# ---------------------------------------------------------------------------
# Reward weights — single source of truth (panel reads from here too)
# ---------------------------------------------------------------------------
REWARD_WEIGHTS: dict[str, float] = {
    # ── Scoring (the actual game goal — generous per-action shaping) ─────
    "score_blue":           +2.0,    # +6.0 per blue ball scored (3 pts)
    "score_red":            -3.0,    # -9.0 per red ball scored — opp's points hurt
    "center_bonus":         +0.5,    # +1.5 extra per ball scored in mid/low goal
    # ── Field activity ──────────────────────────────────────────────────
    "collect":              +0.10,   # per blue ball collected (try_collect skips red)
    "collect_red":          -0.50,   # safety: penalty if a red ball is held anyway
    # ── Quadrant control — VEX endgame is decided here ─────────────────
    "ctrl_us":              +0.20,   # +2.0 per held quadrant per decision
    "ctrl_opp":             -0.15,   # opp holding a quadrant hurts
    "ctrl_gain":            +1.0,    # one-time bonus when we TAKE a quadrant
    "ctrl_loss":            -1.0,    # one-time penalty when we lose one
    "all_quadrants":        +0.5,    # per-decision bonus when we hold all 4
    # ── Wrong-color ball management (rare with intake filter) ───────────
    "eject_wrong_color":    +0.30,   # per red ball ejected
    "holding_wrong_color":  -0.10,   # per red ball held each decision
    # ── Per-decision shaping penalties (kept light to not dominate) ─────
    "idle":                 -0.03,
    "idle_while_losing":    -0.20,
    "idle_with_balls":      -0.15,
    "goal_bump":            -0.005,  # per tick scraping a goal body (cap 60)
    "low_progress":         -0.20,
    "wasted_action":        -0.10,
    "holding_urgency":      -0.02,   # per ball held (any color)
    "heading_jitter":       -0.02,   # per radian of turn beyond normal travel
    # ── Opponent goal threat — continuous pressure to descore proactively ──
    "opp_goal_threat":      -0.20,   # per decision: -(opp_balls_opp_long + opp_balls_center)
    # ── Phase objectives — "win early and decisively" ───────────────────
    "margin_milestone_10":  +5.0,    # one-time at margin ≥ 10 with > 50% time left
    "margin_milestone_20":  +10.0,   # one-time at margin ≥ 20 with > 25% time left
    # ── Episode end — DOMINANT signal so the outcome matters ────────────
    "episode_end_win":      +20.0,
    # Penalize ties so the agent learns it must *score* (and not settle).
    "episode_end_tie":      -5.0,
    "episode_end_loss":     -15.0,
    # Extra penalty when we ended with *zero* scoring points.
    "episode_end_zero":     -10.0,
    "margin_curve":         +15.0,   # sigmoid amplitude on top of win/loss base
}

_GOAL_BUMP_CAP_TICKS  = 60     # bump penalty caps at this many ticks per decision
_NORMAL_TRAVEL_TURN   = math.pi   # robots typically turn up to ~π rad per decision
_LATE_GAME_MAX_BOOST  = 0.5    # late-game ctrl_us multiplier doubles (1.0 → 2.0×)
_MARGIN_CURVE_CENTER  = 12.0   # win margin where sigmoid hits 50%
_MARGIN_CURVE_SLOPE   = 3.0   # higher = steeper. 3.0 spreads 5→20pt leads smoothly


def _control_pts(env) -> tuple[int, int]:
    """(our_ctrl_pts, opp_ctrl_pts) — points (10 per quadrant)."""
    gs = getattr(env.field, "goal_state", None)
    if gs is None:
        return 0, 0
    ctrl = gs.compute_quadrant_control()
    ours = sum(1 for c in ctrl.values() if c == BALL_BLUE) * CONTROL_BONUS_PTS
    opps = sum(1 for c in ctrl.values() if c == BALL_RED)  * CONTROL_BONUS_PTS
    return ours, opps


def _time_factor(env) -> float:
    """0.0 at match start, 1.0 at match end. Used to ramp late-game urgency."""
    remaining = float(getattr(env.field, "time_remaining", MATCH_DURATION))
    return max(0.0, min(1.0, 1.0 - remaining / MATCH_DURATION))


def _sigmoid(x: float) -> float:
    """Stable sigmoid."""
    if x >= 0:
        z = math.exp(-x);  return 1.0 / (1.0 + z)
    z = math.exp(x);       return z / (1.0 + z)


def _reward_components(env, robot_id: int) -> dict[str, float]:
    """Return per-component reward breakdown for robot_id (sums to total)."""
    w = REWARD_WEIGHTS
    comps: dict[str, float] = {}

    # Score deltas this decision (computed from stored prev values)
    blue_delta = float(env.field.my_score)       - float(env.prev_my_score)
    red_delta  = float(env.field.opponent_score) - float(env.prev_opp_score)
    comps["score_blue"] =  blue_delta * w["score_blue"]
    comps["score_red"]  = -red_delta  * abs(w["score_red"])

    # Center-goal bonus
    center_pts = float(getattr(env, "center_score_events", [0, 0])[robot_id])
    comps["center_bonus"] = center_pts * w["center_bonus"]

    # Collection signal (blue only — try_collect filters red at the action level)
    red_collected  = int(getattr(env, "red_collected_this_step", [0, 0])[robot_id])
    blue_collected = max(0, int(env.collected_this_step[robot_id]) - red_collected)
    comps["collect"]     = float(blue_collected) * w["collect"]
    comps["collect_red"] = float(red_collected)  * w["collect_red"]

    # Quadrant control snapshot — late-game multiplier scales up ctrl_us.
    # Early match: 1.0× weight. Match end: (1 + _LATE_GAME_MAX_BOOST)× weight.
    our_ctrl, opp_ctrl = _control_pts(env)
    late_mult = 1.0 + _time_factor(env) * _LATE_GAME_MAX_BOOST
    comps["ctrl_us"]  =  our_ctrl * w["ctrl_us"]  * late_mult
    comps["ctrl_opp"] = -opp_ctrl * abs(w["ctrl_opp"])

    # Control gain/loss — one-time bonus when we take or lose a quadrant
    ctrl_gain_us  = int(getattr(env, "ctrl_gain_us", 0))
    ctrl_gain_opp = int(getattr(env, "ctrl_gain_opp", 0))
    comps["ctrl_gain"] = ctrl_gain_us  * w["ctrl_gain"]
    comps["ctrl_loss"] = ctrl_gain_opp * w["ctrl_loss"]

    # All-quadrants bonus — strong incentive to dominate control
    our_quadrants = our_ctrl // CONTROL_BONUS_PTS    # 0..4
    comps["all_quadrants"] = w["all_quadrants"] if our_quadrants == 4 else 0.0

    # Opponent goal threat — continuous penalty while the opponent has their colored
    # balls sitting in goals. Encourages proactive descoring, not just reactive.
    gs = env.field.goal_state
    opp_in_long   = sum(1 for _, c in gs.opp_long if c == BALL_RED)
    opp_in_center = sum(1 for _, c in gs.center_mid + gs.center_low if c == BALL_RED)
    threat = opp_in_long / 14.0 + opp_in_center / 14.0   # 0..2
    comps["opp_goal_threat"] = threat * w["opp_goal_threat"]

    # Context-aware idle penalty
    action = env.current_actions[robot_id]
    if action == Action.IDLE:
        comps["idle"] = w["idle"]
        # Extra penalty when idling while losing (including control-zone score)
        effective_my  = float(env.field.my_score)       + our_ctrl
        effective_opp = float(env.field.opponent_score) + opp_ctrl
        comps["idle_while_losing"] = (
            w["idle_while_losing"] if effective_my < effective_opp else 0.0
        )
        # Extra penalty when idling while collectible balls are still on the field
        balls_on_field = sum(1 for obj in env.field.objects if obj.status == OBJ_ON_FIELD)
        comps["idle_with_balls"] = (
            w["idle_with_balls"] if balls_on_field > 0 else 0.0
        )
    else:
        comps["idle"]              = 0.0
        comps["idle_while_losing"] = 0.0
        comps["idle_with_balls"]   = 0.0

    # Goal-collision penalty (capped)
    bump_ticks = int(getattr(env, "goal_bump_ticks", [0, 0])[robot_id])
    comps["goal_bump"] = w["goal_bump"] * min(bump_ticks, _GOAL_BUMP_CAP_TICKS)

    # Low-progress penalty
    comps["low_progress"] = (
        w["low_progress"]
        if bool(getattr(env, "low_progress", [False, False])[robot_id])
        else 0.0
    )

    # Wasted-action penalty
    score_actions   = (Action.SCORE_LONG_GOAL, Action.SCORE_CENTER_MID, Action.SCORE_CENTER_LOW)
    descore_actions = (Action.DESCORE_OPP_LONG, Action.DESCORE_CENTER)
    held_at_start = int(getattr(env, "balls_held_at_step_start", [0, 0])[robot_id])
    scored   = float(env.score_events[robot_id])
    descored = float(env.descore_events[robot_id])
    wasted = 0.0
    # SCORE penalty only fires if no collect happened either (chain-back didn't get a ball)
    if action in score_actions and held_at_start == 0 and scored == 0:
        collected = int(getattr(env, "collected_this_step", [0, 0])[robot_id])
        if collected == 0:
            wasted += w["wasted_action"]
    if action in descore_actions and descored == 0:
        wasted += w["wasted_action"]
    if action == Action.DEFEND_ZONE:
        num_opp = int(getattr(env, "num_opponents", 0))
        our_long_balls = len(getattr(env.field.goal_state, "our_long", []))
        if num_opp == 0 or our_long_balls == 0:
            wasted += w["wasted_action"]
    if action == Action.EJECT_WRONG_COLOR:
        ejected = float(getattr(env, "eject_events", [0, 0])[robot_id])
        if ejected == 0:
            wasted += w["wasted_action"]
    comps["wasted_action"] = wasted

    # Holding urgency
    robot_r = env.field.allies[robot_id]
    balls_now = int(robot_r.balls_held)
    comps["holding_urgency"] = balls_now * w["holding_urgency"]

    # Extra penalty per wrong-color (red) ball held — encourages ejecting ASAP
    wrong_color_held = sum(
        1 for oid in robot_r.held_object_ids
        if env.field.objects[oid].color != BALL_BLUE
    )
    comps["holding_wrong_color"] = wrong_color_held * w["holding_wrong_color"]

    # Reward for successfully ejecting wrong-color balls this decision
    comps["eject_wrong_color"] = (
        float(getattr(env, "eject_events", [0, 0])[robot_id]) * w["eject_wrong_color"]
    )

    # Heading jitter — penalty for total |heading change| beyond ~one travel turn
    turn = float(getattr(env, "heading_turn_amount", [0.0, 0.0])[robot_id])
    extra_turn = max(0.0, turn - _NORMAL_TRAVEL_TURN)
    comps["heading_jitter"] = extra_turn * w["heading_jitter"]

    # Phase-objective margin milestones — one-time bonuses for early dominance.
    # Encourages the policy to score fast & hard rather than coasting to a thin win.
    margin    = float(env.field.my_score) - float(env.field.opponent_score)
    t_left    = float(getattr(env.field, "time_remaining", MATCH_DURATION))
    t_frac    = max(0.0, t_left / MATCH_DURATION)   # 1.0 at start, 0.0 at end
    milestones = getattr(env, "score_milestones_hit", None)
    milestone_bonus = 0.0
    if milestones is not None and robot_id == 0:   # award once, attributed to robot 0
        if not milestones.get(10, False) and margin >= 10 and t_frac > 0.50:
            milestone_bonus += w["margin_milestone_10"]
            milestones[10] = True
        if not milestones.get(20, False) and margin >= 20 and t_frac > 0.25:
            milestone_bonus += w["margin_milestone_20"]
            milestones[20] = True
    comps["margin_milestone"] = milestone_bonus

    # End-of-episode outcome — smooth sigmoid margin curve
    if env.done:
        blue_total = env.field.my_score       + our_ctrl
        red_total  = env.field.opponent_score + opp_ctrl
        diff = blue_total - red_total
        if diff > 0:
            margin_bonus = w["margin_curve"] * _sigmoid(
                (diff - _MARGIN_CURVE_CENTER) / _MARGIN_CURVE_SLOPE
            )
            comps["episode_end"] = w["episode_end_win"] + margin_bonus
        elif diff == 0:
            comps["episode_end"] = w["episode_end_tie"]
        else:
            margin_pen = w["margin_curve"] * _sigmoid(
                (abs(diff) - _MARGIN_CURVE_CENTER) / _MARGIN_CURVE_SLOPE
            )
            comps["episode_end"] = w["episode_end_loss"] - margin_pen

        # If we never scored any points, add an extra penalty.
        comps["episode_end_zero"] = w["episode_end_zero"] if env.field.my_score <= 0 else 0.0
    else:
        comps["episode_end"] = 0.0
        comps["episode_end_zero"] = 0.0

    return comps


def compute_reward(env, robot_id: int) -> float:
    """Per-decision reward for robot_id (sum of all components).

    Also stores the breakdown dict on env.last_reward_breakdown[robot_id] so
    the renderer panel can show which terms contributed.
    """
    comps = _reward_components(env, robot_id)
    env.last_reward_breakdown[robot_id] = comps
    return float(sum(comps.values()))
