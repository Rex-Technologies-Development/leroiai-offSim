import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

from sim.config import Action, ACTION_NAMES, NUM_ACTIONS  # noqa: E402
from sim.action_helpers import (  # noqa: E402
    is_score_action, is_descore_action, is_ram_action, SCORE_ACTIONS,
)
import sim.env as E  # noqa: E402


class TestActionSpaceContract(unittest.TestCase):
    def test_enum_matches_action_names(self):
        self.assertEqual(NUM_ACTIONS, 23)
        self.assertEqual(len(Action), 23)
        for i in range(23):
            self.assertEqual(Action(i).name, ACTION_NAMES[i])

    def test_shared_yaml_matches(self):
        cfg_path = OFFSIM_DIR / "shared" / "config.yaml"
        with open(cfg_path) as f:
            shared = yaml.safe_load(f)
        self.assertEqual(shared["num_actions"], 23)
        for i in range(23):
            self.assertEqual(shared["actions"][i], ACTION_NAMES[i])

    def test_score_actions_count(self):
        self.assertEqual(len(SCORE_ACTIONS), 8)

    def test_action_masks_disable_score_when_empty(self):
        env = E.VexAIEnv(
            render_mode=None, num_allies=1, num_opponents=0, use_timer=False,
        )
        env.reset(seed=1)
        mask = env.action_masks(robot_id=0)
        self.assertEqual(mask.shape, (NUM_ACTIONS,))
        for act in SCORE_ACTIONS:
            self.assertFalse(mask[int(act)])


class TestActionRouting(unittest.TestCase):
    def test_ram_target_is_on_field(self):
        from sim.robot import Robot
        robot = Robot(90.0, 72.0, role_id=1)
        tgt = E._ram_target_for_action(Action.RAM_RIGHT_LONG_GOAL_OPPONENT, robot)
        self.assertIsNotNone(tgt)
        self.assertEqual(len(tgt), 2)

    def test_descore_field_vs_wall_modes(self):
        from sim.action_helpers import descore_mode_for_action
        from sim.config import DESCORE_MODE_SLIDE, DESCORE_MODE_SLAM

        self.assertEqual(
            descore_mode_for_action(Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD),
            DESCORE_MODE_SLIDE,
        )
        self.assertEqual(
            descore_mode_for_action(Action.DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL),
            DESCORE_MODE_SLAM,
        )


if __name__ == "__main__":
    unittest.main()
