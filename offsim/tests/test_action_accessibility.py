"""Verify all 23 RL actions route and execute when preconditions are met."""
import sys
import unittest
from pathlib import Path

import numpy as np


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

from sim.config import (  # noqa: E402
    Action, NUM_ACTIONS, BALL_RED, BALL_BLUE, LONG_GOAL_POINTS, DESCORE_P0,
)
from sim.action_helpers import (  # noqa: E402
    is_stop, is_score_action, is_descore_action, is_ram_action, SCORE_ACTIONS,
)
import sim.env as E  # noqa: E402
from sim.field import Field  # noqa: E402
from sim.robot import Robot  # noqa: E402
from tests.test_descore import _seed_goal_opp_balls  # noqa: E402


class TestFullActionSpaceRouting(unittest.TestCase):
    def _field_ready_for_all_actions(self) -> tuple[Field, Robot]:
        field = Field()
        robot = Robot(x=90.0, y=72.0, heading=0.0)
        robot.balls_held = 2
        # Mixed scored balls so every descore variant has removable targets.
        for i, obj in enumerate(field.objects[:12]):
            obj.color = BALL_RED if i % 2 == 0 else BALL_BLUE
            obj.status = 2  # OBJ_SCORED_US
            gname = "opp_long" if i < 4 else ("our_long" if i < 8 else "center_mid")
            obj.scored_in_goal = gname
            field.goal_state.score_ball(gname, i, obj.color)
        return field, robot

    def test_all_actions_have_targets_or_are_stop(self):
        field, robot = self._field_ready_for_all_actions()

        for i in range(NUM_ACTIONS):
            act = Action(i)
            if is_stop(act):
                self.assertIsNone(E._action_to_target(act, field, robot))
                continue
            tgt = E._action_to_target(act, field, robot)
            self.assertIsNotNone(tgt, msg=f"{act.name} returned no nav target")
            self.assertEqual(len(tgt), 2)

    def test_greedy_score_picks_left_long_for_r0(self):
        from main import _greedy_score_action
        from sim.config import Action
        env = E.VexAIEnv(render_mode=None, num_allies=2, num_opponents=0, use_timer=False)
        env.reset(seed=1)
        robot = env.field.allies[0]
        robot.balls_held = 2
        act = Action(_greedy_score_action(robot, env))
        self.assertIn(act, (Action.SCORE_LEFT_LONG_GOAL_ALLIANCE, Action.SCORE_MID_LEFT,
                            Action.SCORE_LOW_LEFT))
        self.assertNotEqual(act, Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE)

    def test_home_long_score_action_by_lane(self):
        from sim.config import Action
        env = E.VexAIEnv(render_mode=None, num_allies=2, num_opponents=0, use_timer=False)
        env.reset(seed=1)
        self.assertEqual(
            E._home_long_score_action(env.field.allies[0]),
            Action.SCORE_LEFT_LONG_GOAL_ALLIANCE,
        )
        self.assertEqual(
            E._home_long_score_action(env.field.allies[1]),
            Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE,
        )
        env = E.VexAIEnv(render_mode=None, num_allies=1, num_opponents=0, use_timer=False)
        env.reset(seed=3)
        env.field.allies[0].balls_held = 2
        mask = env.action_masks(robot_id=0)
        for act in SCORE_ACTIONS:
            self.assertTrue(mask[int(act)], msg=f"{act.name} should be unmasked with balls held")

    def test_descore_unmasked_when_goal_in_view(self):
        env = E.VexAIEnv(render_mode=None, num_allies=1, num_opponents=0, use_timer=False)
        env.reset(seed=4)
        _seed_goal_opp_balls(env.field, "opp_long", 3)
        env.use_goal_belief = True
        env._observe_goal("opp_long")
        env._goal_in_view["opp_long"] = True
        mask = env.action_masks(robot_id=0)
        self.assertTrue(mask[int(Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD)])

    def test_empty_long_goal_has_no_slam_descore_plan(self):
        field = Field()
        robot = Robot(x=25.0, y=72.0, heading=0.0)
        plan = E._resolve_descore_plan_for_execution(
            Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL, field, robot,
        )
        self.assertIsNone(plan)

    def test_descore_masked_while_holding_balls(self):
        env = E.VexAIEnv(render_mode=None, num_allies=1, num_opponents=0, use_timer=False)
        env.reset(seed=5)
        _seed_goal_opp_balls(env.field, "opp_long", 3)
        env.field.allies[0].balls_held = 2
        mask = env.action_masks(robot_id=0)
        self.assertFalse(mask[int(Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD)])

    def test_ram_ejects_opponent_balls(self):
        field = Field()
        rng = np.random.default_rng(0)
        robot = Robot(x=25.0, y=72.0, heading=0.0)
        _seed_goal_opp_balls(field, "opp_long", 4)
        gname, approach, entry = E._ram_pose_for_action(
            Action.RAM_LEFT_LONG_GOAL_OPPONENT, robot,
        )
        robot.position = approach
        robot.speed = 30.0
        env = E.VexAIEnv(render_mode=None, num_allies=1, num_opponents=0, use_timer=False)
        env.field = field
        env.rng = rng
        env.slam_impact_speed[0] = 30.0
        env._check_ally_ram(0, robot, Action.RAM_LEFT_LONG_GOAL_OPPONENT)
        self.assertLess(len(field.goal_state.opp_long), 4)
        self.assertGreater(env.descore_events[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
