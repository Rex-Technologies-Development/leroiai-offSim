"""Scripted Override opponents used by demos, evaluation, and training."""
from __future__ import annotations
from .config import Action

class ScriptedOpponent:
    def __init__(self, style: str = "greedy"):
        if style not in {"greedy", "toggle", "mixed", "descore"}:
            raise ValueError(f"unknown opponent style: {style}")
        self.style = style

    def action(self, env, robot_index: int) -> Action:
        field = env.field
        robot = field.robots[robot_index]
        if robot.held_pin is not None or robot.held_cup is not None:
            return Action.SCORE_NEAREST_GOAL
        # Actively go descore the enemy's exposed Pins — this is what makes holding a
        # Goal worth defending. Aggressive for the `descore` style, periodic otherwise.
        if any(field.can_descore(robot, g) for g in field.goals):
            if self.style == "descore" or int(field.elapsed // 4 + robot_index) % 2 == 0:
                return Action.REMOVE_OWN_PIN
        if self.style == "toggle" or (self.style == "mixed" and int(field.elapsed // 6 + robot_index) % 3 == 0):
            return Action.CLAIM_TOGGLE
        if field.nearest_object(robot, "pin") is not None: return Action.COLLECT_PIN
        if field.nearest_object(robot, "cup") is not None: return Action.COLLECT_CUP
        inventory = field.match_loads[robot.alliance]
        if inventory["pin"] or inventory["cup"]: return Action.USE_LOADER
        return Action.DEFEND_MIDFIELD
