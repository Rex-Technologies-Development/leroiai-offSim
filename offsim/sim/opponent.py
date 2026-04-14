"""Rule-based opponent policies: random, greedy, defensive, mixed.

Opponent difficulty is randomized per episode during training to prevent
overfitting to one strategy.
"""

from __future__ import annotations
import numpy as np
from sim.config import (
    Action, MAX_CARRY,
    OPP_LONG_GOAL, OPP_MID_GOAL, OUR_LONG_GOAL, OUR_MID_GOAL,
)


class RandomOpponent:
    """Picks a random valid action each decision."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        return Action(rng.integers(0, len(Action)))


class GreedyOpponent:
    """Always collects nearest object, then scores at closest goal."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state["balls_held"] >= MAX_CARRY:
            pos = state["position"]
            d_long = np.linalg.norm(pos - OPP_LONG_GOAL)
            d_mid = np.linalg.norm(pos - OPP_MID_GOAL)
            return Action.SCORE_LONG_GOAL if d_long < d_mid else Action.SCORE_MID_GOAL
        return Action.COLLECT_NEAREST_BALL


class DefensiveOpponent:
    """Prioritises descoring our goals and blocking."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state.get("us_scored_count", 0) > 0:
            pos = state["position"]
            d_long = np.linalg.norm(pos - OUR_LONG_GOAL)
            d_mid = np.linalg.norm(pos - OUR_MID_GOAL)
            return Action.DESCORE_OPPONENT_LONG if d_long < d_mid else Action.DESCORE_OPPONENT_MID

        if rng.random() < 0.5:
            return Action.DEFEND_ZONE
        return Action.COLLECT_NEAREST_BALL


class MixedOpponent:
    """Switches greedy/defensive based on score differential."""

    def __init__(self):
        self.greedy = GreedyOpponent()
        self.defensive = DefensiveOpponent()

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        diff = state.get("opp_score", 0) - state.get("our_score", 0)
        if diff >= 5:
            return self.defensive(state, rng)
        elif diff <= -5:
            return self.greedy(state, rng)
        else:
            if rng.random() < 0.6:
                return self.greedy(state, rng)
            return self.defensive(state, rng)


OPPONENT_CLASSES = {
    "random": RandomOpponent,
    "greedy": GreedyOpponent,
    "defensive": DefensiveOpponent,
    "mixed": MixedOpponent,
}

def get_opponent(name: str):
    return OPPONENT_CLASSES[name]()
