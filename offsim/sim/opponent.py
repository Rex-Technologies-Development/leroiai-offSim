"""Rule-based opponent policies: random, greedy, defensive, mixed.

Opponent difficulty ramps during curriculum training to prevent
overfitting to one strategy.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sim.config import (
    Action, MAX_CARRY,
    OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL, OUR_LONG_GOAL,
)


# ---------------------------------------------------------------------------
# Opponent physical profile (capabilities) — separate from the policy/strategy
# above. The policy decides WHAT to do; the profile decides HOW capable the
# robot is. More named profiles can be added later (e.g. "fast", "weak") and
# selected per training stage.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpponentProfile:
    name:           str   = "standard"
    speed_scale:    float = 0.75    # top speed vs our robot (0.75 = 25% slower)
    score_interval: float = 2.25    # dwell (s) to deposit a load — a bit slower than
                                    # our ~1.5s back-in (env._SCORE_INTERVAL)
    capacity:       int   = MAX_CARRY   # balls it can hold (same as us, for now)


# Default profile applied to opponents until per-type profiles are introduced.
DEFAULT_OPPONENT_PROFILE = OpponentProfile()

# Registry for future opponent profiles (add entries as new types are designed).
OPPONENT_PROFILES = {
    "standard": DEFAULT_OPPONENT_PROFILE,
}


def get_opponent_profile(name: str = "standard") -> OpponentProfile:
    return OPPONENT_PROFILES.get(name, DEFAULT_OPPONENT_PROFILE)


class RandomOpponent:
    """Picks a random valid action each decision."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        return Action(rng.integers(0, len(Action)))


class GreedyOpponent:
    """Always collects nearest ball, then scores at closest goal."""

    def __call__(self, state: dict, rng: np.random.Generator) -> Action:
        if state["balls_held"] >= MAX_CARRY:
            pos   = state["position"]
            # Either long goal is a valid target (either color scores in any goal);
            # SCORE_LONG_GOAL routes to whichever long goal is nearer (see env).
            d_long = min(np.linalg.norm(pos - OPP_LONG_GOAL),
                         np.linalg.norm(pos - OUR_LONG_GOAL))
            d_mid  = np.linalg.norm(pos - CENTER_MID_GOAL)
            d_low  = np.linalg.norm(pos - CENTER_LOW_GOAL)
            best   = min(d_long, d_mid, d_low)
            if best == d_long:
                return Action.SCORE_LONG_GOAL
            elif best == d_mid:
                return Action.SCORE_CENTER_MID
            else:
                return Action.SCORE_CENTER_LOW
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
