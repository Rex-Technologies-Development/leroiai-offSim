"""Reward helpers for the Override strategy task.

The active environment uses the same formula: blue score delta minus red score
delta, normalized by one 5-point alliance half, plus a +/-10 terminal outcome.
"""
from __future__ import annotations

def strategy_reward(previous: tuple[int, int], current: tuple[int, int], done: bool = False) -> float:
    value = ((current[0]-previous[0])-(current[1]-previous[1]))/5.0
    if done: value += 10.0 if current[0] > current[1] else -10.0 if current[0] < current[1] else 0.0
    return float(value)

REWARD_DESCRIPTION = "(delta_blue - delta_red) / 5, with +10 win / -10 loss at match end"
