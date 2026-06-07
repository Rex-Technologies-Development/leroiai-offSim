import sys
import unittest
from pathlib import Path

import numpy as np


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

from sim.config import (  # noqa: E402
    BALL_RED, BALL_BLUE, OBJ_SCORED_OPP, OBJ_SCORED_US, OBJ_ON_FIELD,
    LONG_GOAL_POINTS, CENTER_GOAL_POINTS, LONG_GOAL_CAPACITY,
    LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX,
    DESCORE_P0, DESCORE_P1, DESCORE_P2, DESCORE_P3,
    SLAM_SPEED_DIVISOR,
)
from sim.field import Field, GoalState  # noqa: E402
from sim.robot import Robot  # noqa: E402
from sim.env import _long_partition_approach  # noqa: E402


def _seed_goal_opp_balls(field: Field, gname: str, n: int) -> list[int]:
    """Place n opponent-scored red balls into a goal."""
    ids: list[int] = []
    for i, obj in enumerate(field.objects):
        if obj.status == OBJ_ON_FIELD and len(ids) < n:
            ids.append(i)
    for oid in ids:
        field.objects[oid].status = OBJ_SCORED_OPP
        field.objects[oid].scored_in_goal = gname
        field.objects[oid].color = BALL_RED
        field.goal_state.score_ball(gname, oid, BALL_RED)
    field.opponent_score = LONG_GOAL_POINTS * n if "long" in gname else CENTER_GOAL_POINTS * n
    return ids


def _seed_goal_us_balls(field: Field, gname: str, n: int) -> list[int]:
    ids: list[int] = []
    for i, obj in enumerate(field.objects):
        if obj.status == OBJ_ON_FIELD and len(ids) < n:
            ids.append(i)
    for oid in ids:
        field.objects[oid].status = OBJ_SCORED_US
        field.objects[oid].scored_in_goal = gname
        field.objects[oid].color = BALL_BLUE
        field.goal_state.score_ball(gname, oid, BALL_BLUE)
    field.my_score = LONG_GOAL_POINTS * n if "long" in gname else CENTER_GOAL_POINTS * n
    return ids


class TestGoalStateSegments(unittest.TestCase):
    def test_segment_bounds_nine(self):
        segs = GoalState.segment_bounds(9)
        self.assertEqual(segs, [(0, 2), (3, 5), (6, 8)])

    def test_slide_index_p1_to_p0(self):
        span = GoalState.slide_index_range(9, DESCORE_P1, DESCORE_P0)
        self.assertEqual(span, (0, 2))

    def test_long_ceiling_blocks_p2_to_p3(self):
        self.assertFalse(GoalState.is_slide_allowed("opp_long", DESCORE_P2, DESCORE_P3))


class TestDescoreField(unittest.TestCase):
    def setUp(self):
        self.field = Field()
        self.rng = np.random.default_rng(7)
        self.robot = Robot(x=25.0, y=72.0, heading=0.0)

    def test_long_slide_p1_to_p0_removes_first_segment_only(self):
        _seed_goal_opp_balls(self.field, "opp_long", 6)
        approach, _, _ = _long_partition_approach("opp_long", DESCORE_P1)
        self.robot.position = approach
        pts = self.field.try_descore_slide(
            self.robot, "opp_long", DESCORE_P1, DESCORE_P0,
            approach, self.rng,
        )
        self.assertEqual(pts, 2 * LONG_GOAL_POINTS)
        self.assertEqual(len(self.field.goal_state.opp_long), 4)
        self.assertEqual(self.field.opponent_score, 4 * LONG_GOAL_POINTS)

    def test_long_slide_spills_south(self):
        _seed_goal_opp_balls(self.field, "opp_long", 3)
        approach, _, _ = _long_partition_approach("opp_long", DESCORE_P1)
        self.robot.position = approach
        self.field.try_descore_slide(
            self.robot, "opp_long", DESCORE_P1, DESCORE_P0,
            approach, self.rng,
        )
        on_field = [
            o for o in self.field.objects
            if o.status == OBJ_ON_FIELD and o.color == BALL_RED
        ]
        self.assertGreater(len(on_field), 0)
        south = [o for o in on_field if o.y < LONG_GOAL_Y_MIN]
        self.assertGreater(len(south), 0)
        self.assertLess(south[0].vy, 0.0)

    def test_long_slam_ejects_floor_speed_over_k(self):
        _seed_goal_opp_balls(self.field, "opp_long", 5)
        approach = np.array([23.6, LONG_GOAL_Y_MIN - 7.5])
        self.robot.position = approach
        pts = self.field.try_descore_slam(
            self.robot, "opp_long", DESCORE_P0, 25.0,
            approach, self.rng,
        )
        expected = int(25.0 // SLAM_SPEED_DIVISOR)
        self.assertEqual(pts, expected * LONG_GOAL_POINTS)
        self.assertEqual(len(self.field.goal_state.opp_long), 5 - expected)

    def test_long_slam_spills_north_when_entering_south(self):
        _seed_goal_opp_balls(self.field, "opp_long", 4)
        approach = np.array([23.6, LONG_GOAL_Y_MIN - 7.5])
        self.robot.position = approach
        self.field.try_descore_slam(
            self.robot, "opp_long", DESCORE_P0, 25.0,
            approach, self.rng,
        )
        spilled = [o for o in self.field.objects if o.status == OBJ_ON_FIELD]
        north = [o for o in spilled if o.y > LONG_GOAL_Y_MAX]
        self.assertGreater(len(north), 0)
        self.assertGreater(north[0].vy, 0.0)

    def test_long_goal_overflow_at_capacity_spills(self):
        rng = np.random.default_rng(1)
        ball_ids = list(range(LONG_GOAL_CAPACITY + 1))
        for oid in ball_ids:
            self.field.objects[oid].color = BALL_BLUE
            self.field.objects[oid].status = OBJ_SCORED_US
        for oid in ball_ids[:LONG_GOAL_CAPACITY]:
            self.field.goal_state.score_ball("our_long", oid, BALL_BLUE)
        self.assertEqual(len(self.field.goal_state.our_long), LONG_GOAL_CAPACITY)
        new_oid = ball_ids[LONG_GOAL_CAPACITY]
        self.field.objects[new_oid].status = OBJ_SCORED_US
        ejected = self.field.goal_state.score_ball(
            "our_long", new_oid, BALL_BLUE, prepend=False,
        )
        self.assertIsNotNone(ejected)
        ej_idx, _ = ejected
        self.field.spill_overflow_ball("our_long", ej_idx, prepend=False, rng=rng)
        obj = self.field.objects[ej_idx]
        self.assertEqual(obj.status, OBJ_ON_FIELD)
        self.assertLess(obj.y, LONG_GOAL_Y_MIN)
        self.assertLess(obj.vy, 0.0)

    def test_center_mid_slide_full_span(self):
        _seed_goal_opp_balls(self.field, "center_mid", 6)
        approach = np.array([72.0, 72.0])
        self.robot.position = approach
        pts = self.field.try_descore_slide(
            self.robot, "center_mid", DESCORE_P2, DESCORE_P3,
            approach, self.rng,
        )
        span = GoalState.slide_index_range(6, DESCORE_P2, DESCORE_P3)
        self.assertIsNotNone(span)
        lo, hi = span
        self.assertEqual(pts, (hi - lo + 1) * CENTER_GOAL_POINTS)

    def test_center_low_slide_blocked_slam_works(self):
        _seed_goal_opp_balls(self.field, "center_low", 4)
        approach = np.array([72.0, 72.0])
        self.robot.position = approach
        slide_pts = self.field.try_descore_slide(
            self.robot, "center_low", DESCORE_P1, DESCORE_P0,
            approach, self.rng,
        )
        self.assertEqual(slide_pts, 0)
        slam_pts = self.field.try_descore_slam(
            self.robot, "center_low", DESCORE_P0, 20.0,
            approach, self.rng,
        )
        self.assertEqual(slam_pts, 2 * CENTER_GOAL_POINTS)

    def test_opponent_field_slide_removes_wrong_color_us_scored(self):
        """Red balls scored by blue (OBJ_SCORED_US) still count as opponent points."""
        ids = _seed_goal_opp_balls(self.field, "our_long", 6)
        for oid in ids:
            self.field.objects[oid].status = OBJ_SCORED_US
        self.field.my_score = 0
        self.field.opponent_score = 6 * LONG_GOAL_POINTS
        approach, _, _ = _long_partition_approach("our_long", DESCORE_P1)
        self.robot.position = approach
        pts = self.field.try_descore_slide(
            self.robot, "our_long", DESCORE_P1, DESCORE_P0,
            approach, self.rng, remove_ally_scored=False,
        )
        self.assertEqual(pts, 2 * LONG_GOAL_POINTS)
        self.assertEqual(len(self.field.goal_state.our_long), 4)
        self.assertEqual(self.field.opponent_score, 4 * LONG_GOAL_POINTS)

    def test_opp_descore_slam_removes_ally_balls(self):
        _seed_goal_us_balls(self.field, "our_long", 3)
        approach = np.array([120.4, 96.39 + 7.5])
        self.robot.position = approach
        pts = self.field.opp_try_descore_slam(
            self.robot, "our_long", DESCORE_P3, 30.0,
            approach, self.rng, remove_ally_scored=False,
        )
        self.assertEqual(pts, 3 * LONG_GOAL_POINTS)
        self.assertEqual(len(self.field.goal_state.our_long), 0)
        self.assertEqual(self.field.my_score, 0)

    def test_alliance_field_slide_uses_remove_ally_scored(self):
        """Alliance-field slide must pass remove_ally_scored=True to eject ally balls."""
        _seed_goal_us_balls(self.field, "opp_long", 4)
        approach, _, _ = _long_partition_approach("opp_long", DESCORE_P1)
        self.robot.position = approach
        pts = self.field.try_descore_slide(
            self.robot, "opp_long", DESCORE_P1, DESCORE_P0,
            approach, self.rng, remove_ally_scored=True,
        )
        self.assertEqual(pts, 2 * LONG_GOAL_POINTS)
        self.assertEqual(len(self.field.goal_state.opp_long), 2)
        self.assertEqual(self.field.my_score, 2 * LONG_GOAL_POINTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
