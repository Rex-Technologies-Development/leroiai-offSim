"""Reward function — scoring, descoring, time pressure, win bonus.

Key principles:
- Win > high score: 10-8 win beats 20-25 loss
- Time-weighted: late-game scoring worth more
- Idle penalty: small so policy can wait when optimal
- No penalty for teammate failure: only own actions + outcome
"""

from sim.config import Action, MATCH_DURATION


def compute_reward(env, robot_id: int) -> float:
    """Compute per-decision reward for one robot."""
    r = 0.0

    # Scoring events (+1 per point scored this decision)
    r += env.score_events[robot_id] * 1.0

    # Descoring events (+0.8 per opponent point removed)
    r += env.descore_events[robot_id] * 0.8

    # Collection (+0.1 per ball picked up)
    r += env.collected_this_step[robot_id] * 0.1

    # Time pressure: late-game worth more
    time_frac = env.field.time_remaining / MATCH_DURATION
    time_factor = 1.0 + (1.0 - time_frac) * 0.5
    r *= time_factor

    # End-of-episode outcome
    if env.done:
        diff = env.field.my_score - env.field.opponent_score
        if diff > 0:
            r += 5.0 + diff * 0.5   # win + margin
        elif diff == 0:
            r += 1.0                  # tie
        else:
            r -= 3.0                  # loss

    # Idle penalty
    if env.current_actions[robot_id] == Action.IDLE:
        r -= 0.05

    return r
