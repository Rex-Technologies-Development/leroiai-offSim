import sys
import unittest
from pathlib import Path

import numpy as np


# When running from the `offsim/` directory, this file is importable as
# `tests.test_scoring`. The sim code expects `import sim.*`, so ensure that
# `offsim/` is on sys.path.
OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

import sim.env as E  # noqa: E402
from sim.config import Action, OBJ_ON_FIELD, OBJ_HELD, BALL_BLUE  # noqa: E402


def _give_robot_blue_balls(inner_env: E.VexAIEnv, robot_idx: int, n: int = 3) -> None:
    robot = inner_env.field.allies[robot_idx]

    held_ids: list[int] = []
    for i, obj in enumerate(inner_env.field.objects):
        if obj.status == OBJ_ON_FIELD and obj.color == BALL_BLUE:
            held_ids.append(i)
            if len(held_ids) >= n:
                break

    if len(held_ids) < n:
        raise RuntimeError(f"Not enough on-field blue balls to hold: needed {n}, got {len(held_ids)}")

    robot.held_object_ids = held_ids
    robot.balls_held = n
    for oid in held_ids:
        inner_env.field.objects[oid].status = OBJ_HELD


def _prime_no_action_pause(inner_env: E.VexAIEnv, action: Action) -> None:
    # Avoid the action-change pause suppressing scoring effects.
    inner_env.prev_action_for_pause[0] = int(action)
    inner_env.action_pause_ticks[0] = 0


class TestScoringInTrainingEnv(unittest.TestCase):
    def _make_env(self) -> E.SingleAgentWrapper:
        env = E.SingleAgentWrapper(
            render_mode=None,
            opponent_type="random",
            num_allies=1,
            num_opponents=0,
            use_timer=False,
        )
        env.reset(seed=123)
        return env

    def test_score_long_goal_adds_ball_to_goalstate(self):
        env = self._make_env()
        inner = env.env
        robot = inner.field.allies[0]

        # Force nearest long-goal choice to be OUR (right) top end.
        robot.position = np.array([E._RIGHT_GOAL_CX, E.LONG_GOAL_Y_MAX + 30.0], dtype=np.float64)
        score_pos, _, required_heading, gname = E._nearest_long_goal_target(robot)
        self.assertEqual(gname, "our_long")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=0, n=3)
        _prime_no_action_pause(inner, Action.SCORE_LONG_GOAL)

        env.step(int(Action.SCORE_LONG_GOAL))

        self.assertGreater(len(inner.field.goal_state.our_long), 0)
        _, color = inner.field.goal_state.our_long[0]
        self.assertEqual(color, BALL_BLUE)
        env.close()

    def test_score_center_mid_adds_ball_to_goalstate(self):
        env = self._make_env()
        inner = env.env
        robot = inner.field.allies[0]

        robot.position = E._MID_NE_POS.copy()
        score_pos, _, required_heading, gname = E._nearest_center_tip(robot, lower=False)
        self.assertEqual(gname, "center_mid")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=0, n=3)
        _prime_no_action_pause(inner, Action.SCORE_CENTER_MID)

        env.step(int(Action.SCORE_CENTER_MID))

        self.assertGreater(len(inner.field.goal_state.center_mid), 0)
        _, color = inner.field.goal_state.center_mid[0]
        self.assertEqual(color, BALL_BLUE)
        env.close()

    def test_score_center_low_adds_ball_to_goalstate(self):
        env = self._make_env()
        inner = env.env
        robot = inner.field.allies[0]

        robot.position = E._LOW_SE_POS.copy()
        score_pos, _, required_heading, gname = E._nearest_center_tip(robot, lower=True)
        self.assertEqual(gname, "center_low")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=0, n=3)
        _prime_no_action_pause(inner, Action.SCORE_CENTER_LOW)

        env.step(int(Action.SCORE_CENTER_LOW))

        self.assertGreater(len(inner.field.goal_state.center_low), 0)
        _, color = inner.field.goal_state.center_low[0]
        self.assertEqual(color, BALL_BLUE)
        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
