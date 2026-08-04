"""Scripted Override opponents used by demos, evaluation, and training."""
from __future__ import annotations
from .config import Action

class ScriptedOpponent:
    def __init__(self, style: str = "greedy"):
        if style not in {"greedy", "toggle", "mixed"}:
            raise ValueError(f"unknown opponent style: {style}")
        self.style = style

    def action(self, env, robot_index: int) -> Action:
        robot = env.field.robots[robot_index]
        if robot.held_pin is not None or robot.held_cup is not None:
            return Action.SCORE_NEAREST_GOAL
        if self.style == "toggle" or (self.style == "mixed" and int(env.field.elapsed // 6 + robot_index) % 3 == 0):
            return Action.CLAIM_TOGGLE
        if env.field.nearest_object(robot, "pin") is not None: return Action.COLLECT_PIN
        if env.field.nearest_object(robot, "cup") is not None: return Action.COLLECT_CUP
        inventory = env.field.match_loads[robot.alliance]
        if inventory["pin"] or inventory["cup"]: return Action.USE_LOADER
        return Action.DEFEND_MIDFIELD
