"""Action-space classifiers and routing metadata for the 23-action enum."""

from __future__ import annotations

from sim.config import Action, DESCORE_MODE_SLIDE, DESCORE_MODE_SLAM

SCORE_ACTIONS: frozenset[Action] = frozenset({
    Action.SCORE_LEFT_LONG_GOAL_ALLIANCE,
    Action.SCORE_LEFT_LONG_GOAL_OPPONENT,
    Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE,
    Action.SCORE_RIGHT_LONG_GOAL_OPPONENT,
    Action.SCORE_MID_LEFT,
    Action.SCORE_MID_RIGHT,
    Action.SCORE_LOW_LEFT,
    Action.SCORE_LOW_RIGHT,
})

DESCORE_ACTIONS: frozenset[Action] = frozenset({
    Action.DESCORE_LEFT_LONG_GOAL_ALLIANCE_FIELD,
    Action.DESCORE_LEFT_LONG_GOAL_ALLIANCE_WALL,
    Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD,
    Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL,
    Action.DESCORE_RIGHT_LONG_GOAL_ALLIANCE_FIELD,
    Action.DESCORE_RIGHT_LONG_GOAL_ALLIANCE_WALL,
    Action.DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIELD,
    Action.DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL,
})

RAM_ACTIONS: frozenset[Action] = frozenset({
    Action.RAM_RIGHT_LONG_GOAL_OPPONENT,
    Action.RAM_RIGHT_LONG_GOAL_ALLIANCE,
    Action.RAM_LEFT_LONG_GOAL_OPPONENT,
    Action.RAM_LEFT_LONG_GOAL_ALLIANCE,
})

INTAKE_OFF_ACTIONS: frozenset[Action] = frozenset({
    Action.STOP,
    Action.MOVE_CHASSIS_ONLY,
    *RAM_ACTIONS,
})


def _as_action(action: Action | int) -> Action:
    return action if isinstance(action, Action) else Action(int(action))


def is_stop(action: Action | int) -> bool:
    return _as_action(action) == Action.STOP


def is_collect(action: Action | int) -> bool:
    return _as_action(action) == Action.COLLECT_BLOCKS


def is_chassis_only(action: Action | int) -> bool:
    return _as_action(action) == Action.MOVE_CHASSIS_ONLY


def is_score_action(action: Action | int) -> bool:
    return _as_action(action) in SCORE_ACTIONS


def is_descore_action(action: Action | int) -> bool:
    return _as_action(action) in DESCORE_ACTIONS


def is_ram_action(action: Action | int) -> bool:
    return _as_action(action) in RAM_ACTIONS


def wants_intake(action: Action | int) -> bool:
    """True when the robot should run its intake roller."""
    return _as_action(action) == Action.COLLECT_BLOCKS


def long_gname_for(action: Action | int) -> str:
    """Return 'opp_long' (left wall) or 'our_long' (right wall)."""
    name = _as_action(action).name
    if "LEFT_LONG" in name:
        return "opp_long"
    if "RIGHT_LONG" in name:
        return "our_long"
    raise ValueError(f"Action {name} has no long-goal side")


def descore_remove_ally_scored(action: Action | int) -> bool:
    return "ALLIANCE" in _as_action(action).name


def ram_remove_ally_scored(action: Action | int) -> bool:
    return "ALLIANCE" in _as_action(action).name


def descore_mode_for_action(action: Action | int) -> str:
    name = _as_action(action).name
    if name.endswith("_FIELD"):
        return DESCORE_MODE_SLIDE
    if name.endswith("_WALL"):
        return DESCORE_MODE_SLAM
    raise ValueError(f"Action {name} is not a descore action")


def score_intent_alliance(action: Action | int) -> bool | None:
    """True=alliance-colored balls, False=opponent-colored, None=center goals."""
    act = _as_action(action)
    if not is_score_action(act):
        return None
    name = act.name
    if "ALLIANCE" in name:
        return True
    if "OPPONENT" in name:
        return False
    return True  # center goals: alliance scoring intent
