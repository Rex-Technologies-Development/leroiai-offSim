import sys
import unittest
from pathlib import Path

import numpy as np


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

import sim.env as E  # noqa: E402
from sim.config import (  # noqa: E402
    Action, BALL_BLUE, BALL_RED, OBJ_ON_FIELD, NUM_ACTIONS,
)
from sim.field import Field  # noqa: E402
from sim.robot import Robot  # noqa: E402
from training.curriculum import DEFAULT_STAGES  # noqa: E402
from training.reward import compute_team_reward  # noqa: E402


class TestTrainingReadiness(unittest.TestCase):
    def test_allied_intake_auto_rejects_red_ball(self):
        field = Field()
        robot = Robot(x=50.0, y=40.0, heading=0.0, role_id=0)
        red_idx = 0
        field.objects[red_idx].color = BALL_RED
        field.objects[red_idx].status = OBJ_ON_FIELD
        field.objects[red_idx].x = robot.x + 8.0
        field.objects[red_idx].y = robot.y

        result = field.try_collect(robot)

        self.assertEqual(result, -1)
        self.assertEqual(robot.balls_held, 0)
        self.assertEqual(robot.held_object_ids, [])
        self.assertEqual(field.objects[red_idx].status, OBJ_ON_FIELD)

    def test_allied_intake_still_collects_blue_ball(self):
        field = Field()
        robot = Robot(x=50.0, y=40.0, heading=0.0, role_id=0)
        blue_idx = 0
        field.objects[blue_idx].color = BALL_BLUE
        field.objects[blue_idx].status = OBJ_ON_FIELD
        field.objects[blue_idx].x = robot.x + 8.0
        field.objects[blue_idx].y = robot.y

        result = field.try_collect(robot)

        self.assertTrue(result)
        self.assertEqual(robot.balls_held, 1)
        self.assertEqual(robot.held_object_ids, [blue_idx])

    def test_team_action_masks_are_role_specialized(self):
        env = E.TeamAgentWrapper(num_allies=2, num_opponents=0, use_timer=False)
        env.reset(seed=3)
        inner = env.env
        inner.field.allies[0].balls_held = 1
        inner.field.allies[1].balls_held = 1

        mask = env.action_masks()
        self.assertEqual(mask.shape, (NUM_ACTIONS * 2,))
        r0 = mask[:NUM_ACTIONS]
        r1 = mask[NUM_ACTIONS:]

        self.assertTrue(r0[int(Action.SCORE_LEFT_LONG_GOAL_ALLIANCE)])
        self.assertTrue(r0[int(Action.SCORE_MID_LEFT)])
        self.assertFalse(r0[int(Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE)])
        self.assertFalse(r0[int(Action.SCORE_MID_RIGHT)])

        self.assertTrue(r1[int(Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE)])
        self.assertTrue(r1[int(Action.SCORE_MID_RIGHT)])
        self.assertFalse(r1[int(Action.SCORE_LEFT_LONG_GOAL_ALLIANCE)])
        self.assertFalse(r1[int(Action.SCORE_MID_LEFT)])

    def test_team_reward_counts_global_terms_once(self):
        env = E.VexAIEnv(num_allies=2, num_opponents=0, use_timer=False)
        env.last_reward_breakdown = [
            {"score_blue": 6.0, "ctrl_us": 2.0, "collect": 0.35},
            {"score_blue": 6.0, "ctrl_us": 2.0, "collect": 0.35},
        ]

        self.assertAlmostEqual(compute_team_reward(env), 8.70)

    def test_default_curriculum_starts_all_blue_and_reaches_2v2(self):
        thresholds = [threshold for threshold, _ in DEFAULT_STAGES]
        self.assertEqual(thresholds, [0, 200_000, 500_000, 800_000])
        self.assertTrue(DEFAULT_STAGES[0][1]["all_blue_only"])
        self.assertEqual(DEFAULT_STAGES[-1][1]["num_opponents"], 2)
        self.assertEqual(DEFAULT_STAGES[-1][1]["opponent_type"], "mixed")


if __name__ == "__main__":
    unittest.main()
