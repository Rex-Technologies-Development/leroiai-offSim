"""Rule-based opponent policies: random, greedy, defensive, mixed."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sim.config import (
    Action, MAX_CARRY,
    OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL, OUR_LONG_GOAL,
)


@dataclass(frozen=True)
class OpponentProfile:
    name:           str   = "standard"
    speed_scale:    float = 0.75
    score_interval: float = 2.25
    capacity:       int   = MAX_CARRY


DEFAULT_OPPONENT_PROFILE = OpponentProfile()

OPPONENT_PROFILES = {
    "standard": DEFAULT_OPPONENT_PROFILE,
}


def get_opponent_profile(name: str = "standard") -> OpponentProfile:
    return OPPONENT_PROFILES.get(name, DEFAULT_OPPONENT_PROFILE)


class RandomOpponent:
    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        return Action(rng.integers(0, len(Action)))


class GreedyOpponent:
    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state["balls_held"] >= MAX_CARRY:
            pos = state["position"]
            d_right = np.linalg.norm(pos - OUR_LONG_GOAL)
            d_left = np.linalg.norm(pos - OPP_LONG_GOAL)
            d_mid_l = np.linalg.norm(pos - CENTER_MID_GOAL)
            d_low_l = np.linalg.norm(pos - CENTER_LOW_GOAL)
            best = min(d_right, d_left, d_mid_l, d_low_l)
            if best == d_right:
                return Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE
            if best == d_left:
                return Action.SCORE_LEFT_LONG_GOAL_ALLIANCE
            if best == d_mid_l:
                return Action.SCORE_MID_LEFT
            return Action.SCORE_LOW_LEFT
        return Action.COLLECT_BLOCKS


class DefensiveOpponent:
    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state.get("us_scored_count", 0) >= 2:
            pos = state["position"]
            d_right = np.linalg.norm(pos - OUR_LONG_GOAL)
            d_left = np.linalg.norm(pos - OPP_LONG_GOAL)
            if d_right < d_left:
                return Action.DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIELD
            return Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD

        if rng.random() < 0.5:
            return Action.RAM_RIGHT_LONG_GOAL_OPPONENT
        return Action.COLLECT_BLOCKS


class MixedOpponent:
    def __init__(self):
        self.greedy    = GreedyOpponent()
        self.defensive = DefensiveOpponent()

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        diff = state.get("opp_score", 0) - state.get("our_score", 0)
        if diff >= 3 or state.get("us_scored_count", 0) >= 3:
            return self.defensive(state, rng)
        elif diff <= -5:
            return self.greedy(state, rng)
        else:
            return self.greedy(state, rng) if rng.random() < 0.5 else self.defensive(state, rng)


OPPONENT_CLASSES = {
    "random":    RandomOpponent,
    "greedy":    GreedyOpponent,
    "defensive": DefensiveOpponent,
    "mixed":     MixedOpponent,
}


def get_opponent(name: str):
    return OPPONENT_CLASSES[name]()
