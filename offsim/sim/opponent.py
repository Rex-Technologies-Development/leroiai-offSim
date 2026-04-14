"""Rule-based opponent policies: random, greedy, defensive, mixed.

Opponent difficulty ramps during curriculum training to prevent
overfitting to one strategy.
"""

from __future__ import annotations
import numpy as np
from sim.config import (
    Action, MAX_CARRY,
    OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL, OUR_LONG_GOAL,
)


class RandomOpponent:
    """Picks a random valid action each decision."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        return Action(rng.integers(0, len(Action)))


class GreedyOpponent:
    """Always collects nearest ball, then scores at closest goal."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state["balls_held"] >= MAX_CARRY:
            pos   = state["position"]
            d_long = np.linalg.norm(pos - OPP_LONG_GOAL)
            d_mid  = np.linalg.norm(pos - CENTER_MID_GOAL)
            d_low  = np.linalg.norm(pos - CENTER_LOW_GOAL)
            best   = min(d_long, d_mid, d_low)
            if best == d_long:
                return Action.SCORE_LONG_GOAL
            else:
                return Action.SCORE_CENTER_GOAL
        return Action.COLLECT_NEAREST_BALL


class DefensiveOpponent:
    """Prioritises descoring our goals and blocking."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state.get("us_scored_count", 0) > 0:
            pos    = state["position"]
            d_long = np.linalg.norm(pos - OUR_LONG_GOAL)
            d_ctr  = np.linalg.norm(pos - CENTER_MID_GOAL)
            return Action.DESCORE_OPP_LONG if d_long < d_ctr else Action.DESCORE_CENTER

        if rng.random() < 0.5:
            return Action.DEFEND_ZONE
        return Action.COLLECT_NEAREST_BALL


class MixedOpponent:
    """Switches greedy/defensive based on score differential."""

    def __init__(self):
        self.greedy    = GreedyOpponent()
        self.defensive = DefensiveOpponent()

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        diff = state.get("opp_score", 0) - state.get("our_score", 0)
        if diff >= 5:
            return self.defensive(state, rng)
        elif diff <= -5:
            return self.greedy(state, rng)
        else:
            return self.greedy(state, rng) if rng.random() < 0.6 else self.defensive(state, rng)


OPPONENT_CLASSES = {
    "random":    RandomOpponent,
    "greedy":    GreedyOpponent,
    "defensive": DefensiveOpponent,
    "mixed":     MixedOpponent,
}


def get_opponent(name: str):
    return OPPONENT_CLASSES[name]()
