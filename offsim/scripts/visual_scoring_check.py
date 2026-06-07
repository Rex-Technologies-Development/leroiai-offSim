"""Visual (rendered) scoring sanity check.

This is a companion to the deterministic `tests/test_scoring.py` unittest.
It opens the Pygame renderer, teleports the robot to the scoring approach pose,
forces it to hold blue balls, then issues the SCORE action and verifies that
`GoalState` updates.

Run from repo root:
    py offsim/scripts/visual_scoring_check.py

Close the Pygame window at any time to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Ensure we can import the sim package as `sim.*`.
OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

import sim.env as E  # noqa: E402
from sim.config import Action, OBJ_HELD, OBJ_ON_FIELD, BALL_BLUE  # noqa: E402


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
    # Avoid the initial action-change pause suppressing scoring effects.
    inner_env.prev_action_for_pause[0] = int(action)
    inner_env.action_pause_ticks[0] = 0


def _render_for_frames(env: E.SingleAgentWrapper, frames: int) -> None:
    for _ in range(frames):
        env.render()


def _run_case(env: E.SingleAgentWrapper, case: str, max_decisions: int) -> tuple[bool, str]:
    inner = env.env
    robot = inner.field.allies[0]

    if case == "long":
        # Force the nearest long-goal choice to be OUR (right) top end.
        robot.position = np.array([E._RIGHT_GOAL_CX, E.LONG_GOAL_Y_MAX + 30.0], dtype=np.float64)
        score_pos, _, required_heading, gname = E._nearest_long_goal_target(robot)
        if gname != "our_long":
            return False, f"expected 'our_long', got {gname!r}"
        goal_list = inner.field.goal_state.our_long
        action = Action.SCORE_RIGHT_LONG_GOAL_ALLIANCE

    elif case == "mid":
        # Bias nearest center-tip to MID goal.
        robot.position = E._MID_NE_POS.copy()
        score_pos, _, required_heading, gname = E._nearest_center_tip(robot, lower=False)
        if gname != "center_mid":
            return False, f"expected 'center_mid', got {gname!r}"
        goal_list = inner.field.goal_state.center_mid
        action = Action.SCORE_MID_RIGHT

    elif case == "low":
        # Bias nearest center-tip to LOW goal.
        robot.position = E._LOW_SE_POS.copy()
        score_pos, _, required_heading, gname = E._nearest_center_tip(robot, lower=True)
        if gname != "center_low":
            return False, f"expected 'center_low', got {gname!r}"
        goal_list = inner.field.goal_state.center_low
        action = Action.SCORE_LOW_RIGHT

    else:
        return False, f"unknown case: {case!r}"

    robot.position = score_pos.copy()
    robot.heading = float(required_heading)

    _give_robot_blue_balls(inner, robot_idx=0, n=3)
    _prime_no_action_pause(inner, action)

    before = len(goal_list)

    # Show initial setup for a brief moment.
    _render_for_frames(env, frames=30)

    for attempt in range(1, max_decisions + 1):
        env.step(int(action))

        after = len(goal_list)
        if after > before:
            # Confirm the first newly-scored ball is blue.
            _, color = goal_list[0]
            if color != BALL_BLUE:
                return False, f"scored ball had color={color}, expected BALL_BLUE"
            return True, f"scored in {attempt} decision(s)"

        # Keep window responsive between decisions.
        _render_for_frames(env, frames=5)

    return False, f"no score registered after {max_decisions} decision(s)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rendered scoring check (Pygame)")
    parser.add_argument(
        "--case",
        choices=["all", "long", "mid", "low"],
        default="all",
        help="Which scoring scenario to run",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=10,
        help="Max SCORE decisions to attempt per scenario",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=60,
        help="How many frames to keep the result on-screen after each scenario",
    )
    args = parser.parse_args()

    cases = [args.case] if args.case != "all" else ["long", "mid", "low"]

    try:
        env = E.SingleAgentWrapper(
            render_mode="human",
            opponent_type="random",
            num_allies=1,
            num_opponents=0,
            use_timer=False,
        )
    except Exception as exc:  # pygame import errors tend to show up here
        print(f"FAILED to create rendered env: {exc}")
        return 2

    ok_all = True

    try:
        env.reset(seed=123)

        # Create the renderer eagerly so `env.step()` renders tick-by-tick.
        env.render()

        for case in cases:
            print(f"\n=== CASE: {case} ===")
            passed, msg = _run_case(env, case=case, max_decisions=args.max_decisions)
            print(("PASS" if passed else "FAIL") + f": {msg}")
            ok_all = ok_all and passed

            _render_for_frames(env, frames=max(1, int(args.hold_frames)))

        print("\nOverall:", "PASS" if ok_all else "FAIL")
        return 0 if ok_all else 1

    except SystemExit:
        # Pygame QUIT raises SystemExit via the renderer.
        return 0 if ok_all else 1

    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
