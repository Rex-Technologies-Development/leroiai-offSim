import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

import sim.env as E  # noqa: E402
from sim.config import Action, STATE_DIM, NUM_ACTIONS  # noqa: E402


class TestTeamAgentWrapper(unittest.TestCase):
    def _make_env(self) -> E.TeamAgentWrapper:
        env = E.TeamAgentWrapper(
            render_mode=None,
            opponent_type="random",
            num_allies=2,
            num_opponents=2,
            use_timer=False,
        )
        env.reset(seed=42)
        return env

    def test_spaces(self):
        env = self._make_env()
        self.assertEqual(env.observation_space.shape, (STATE_DIM * 2,))
        self.assertIsInstance(env.action_space, spaces.MultiDiscrete)
        self.assertEqual(list(env.action_space.nvec), [NUM_ACTIONS, NUM_ACTIONS])

    def test_action_masks_length(self):
        env = self._make_env()
        masks = env.action_masks()
        self.assertEqual(masks.shape, (NUM_ACTIONS * 2,))
        self.assertTrue(masks.dtype == bool or masks.dtype == np.bool_)

    def test_both_robots_act(self):
        env = self._make_env()
        inner = env.env
        actions = np.array([int(Action.STOP), int(Action.COLLECT_BLOCKS)], dtype=np.int32)
        env.step(actions)
        self.assertEqual(int(inner.current_actions[0]), int(Action.STOP))
        self.assertEqual(int(inner.current_actions[1]), int(Action.COLLECT_BLOCKS))

    def test_make_training_wrapper_selects_team(self):
        team = E.make_training_wrapper(num_allies=2, use_timer=False)
        self.assertIsInstance(team, E.TeamAgentWrapper)
        solo = E.make_training_wrapper(num_allies=1, use_timer=False)
        self.assertIsInstance(solo, E.SingleAgentWrapper)


if __name__ == "__main__":
    unittest.main()
