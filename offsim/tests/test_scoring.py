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
from sim.config import (  # noqa: E402
    Action, OBJ_ON_FIELD, OBJ_HELD, BALL_BLUE, ISOLATION_PERIOD,
)


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


def _prime_no_action_pause(inner_env: E.VexAIEnv, action: Action,
                           robot_idx: int = 0) -> None:
    # Avoid the action-change pause suppressing scoring effects.
    inner_env.prev_action_for_pause[robot_idx] = int(action)
    inner_env.action_pause_ticks[robot_idx] = 0


def _past_isolation(inner_env: E.VexAIEnv) -> None:
    """Tests that place robots across the field run in Interaction Period."""
    inner_env._match_elapsed = ISOLATION_PERIOD + 1.0


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
        inner = E.VexAIEnv(
            render_mode=None, opponent_type="random",
            num_allies=2, num_opponents=0, use_timer=False,
        )
        inner.reset(seed=123)
        robot = inner.field.allies[1]  # right lane → our long goal

        robot.position = np.array([E._RIGHT_GOAL_CX, E.LONG_GOAL_Y_MIN - 30.0], dtype=np.float64)
        score_pos, _, required_heading, gname = E._score_pose_for_action(
            Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE, robot,
        )
        self.assertEqual(gname, "our_long")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=1, n=3)
        _prime_no_action_pause(inner, Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE, robot_idx=1)
        _past_isolation(inner)

        inner.step(np.array([int(Action.STOP), int(Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE)]))

        self.assertGreater(len(inner.field.goal_state.our_long), 0)
        _, color = inner.field.goal_state.our_long[0]
        self.assertEqual(color, BALL_BLUE)
        inner.close()

    def test_score_left_long_goal_r0_adds_ball_to_goalstate(self):
        inner = E.VexAIEnv(
            render_mode=None, opponent_type="random",
            num_allies=2, num_opponents=0, use_timer=False,
        )
        inner.reset(seed=124)
        robot = inner.field.allies[0]  # left lane -> left long goal

        robot.position = np.array([E._LEFT_GOAL_CX, E.LONG_GOAL_Y_MIN - 30.0], dtype=np.float64)
        score_pos, _, required_heading, gname = E._score_pose_for_action(
            Action.SCORE_LEFT_LONG_GOAL_ALLIANCE, robot,
        )
        self.assertEqual(gname, "opp_long")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=0, n=3)
        _prime_no_action_pause(inner, Action.SCORE_LEFT_LONG_GOAL_ALLIANCE, robot_idx=0)
        _past_isolation(inner)

        inner.step(np.array([int(Action.SCORE_LEFT_LONG_GOAL_ALLIANCE), int(Action.STOP)]))

        self.assertGreater(len(inner.field.goal_state.opp_long), 0)
        _, color = inner.field.goal_state.opp_long[0]
        self.assertEqual(color, BALL_BLUE)
        inner.close()

    def test_score_center_mid_adds_ball_to_goalstate(self):
        inner = E.VexAIEnv(
            render_mode=None, opponent_type="random",
            num_allies=2, num_opponents=0, use_timer=False,
        )
        inner.reset(seed=123)
        robot = inner.field.allies[1]

        robot.position = E._MID_NE_POS.copy()
        score_pos, _, required_heading, gname = E._score_pose_for_action(
            Action.SCORE_MID_RIGHT, robot,
        )
        self.assertEqual(gname, "center_mid")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=1, n=3)
        _prime_no_action_pause(inner, Action.SCORE_MID_RIGHT, robot_idx=1)
        _past_isolation(inner)

        inner.step(np.array([int(Action.STOP), int(Action.SCORE_MID_RIGHT)]))

        self.assertGreater(len(inner.field.goal_state.center_mid), 0)
        _, color = inner.field.goal_state.center_mid[0]
        self.assertEqual(color, BALL_BLUE)
        inner.close()

    def test_score_center_low_adds_ball_to_goalstate(self):
        inner = E.VexAIEnv(
            render_mode=None, opponent_type="random",
            num_allies=2, num_opponents=0, use_timer=False,
        )
        inner.reset(seed=123)
        robot = inner.field.allies[1]  # right lane → SE center-low tip

        robot.position = E._LOW_SE_POS.copy()
        score_pos, _, required_heading, gname = E._score_pose_for_action(
            Action.SCORE_LOW_RIGHT, robot,
        )
        self.assertEqual(gname, "center_low")

        robot.position = score_pos.copy()
        robot.heading = float(required_heading)

        _give_robot_blue_balls(inner, robot_idx=1, n=3)
        _prime_no_action_pause(inner, Action.SCORE_LOW_RIGHT, robot_idx=1)
        _past_isolation(inner)

        inner.step(np.array([int(Action.STOP), int(Action.SCORE_LOW_RIGHT)]))

        self.assertGreater(len(inner.field.goal_state.center_low), 0)
        _, color = inner.field.goal_state.center_low[0]
        self.assertEqual(color, BALL_BLUE)
        inner.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
