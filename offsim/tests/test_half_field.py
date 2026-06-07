import sys
import unittest
from pathlib import Path

import numpy as np


OFFSIM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFSIM_DIR))

from sim.config import (  # noqa: E402
    STATE_DIM, HALF_FIELD_Y, HALF_TOP_Y_MIN, HALF_LEFT_X_MAX, HALF_RIGHT_X_MIN,
    ROBOT_W, Action,
    ALLY_START_R0, ALLY_START_R1, OPP_START_R0, OPP_START_R1,
    ISOLATION_PERIOD,
)
import sim.env as E  # noqa: E402
from sim.field import (  # noqa: E402
    Field, field_half, cross_alliance_pair, y_bounds_for_robot,
    clamp_robot_to_half, resolve_robot_pair_collision,
    resolve_all_robot_collisions,
)
from sim.robot import Robot  # noqa: E402


class TestHalfFieldHelpers(unittest.TestCase):
    def test_half_assignment(self):
        ally0 = Robot(float(ALLY_START_R0[0]), float(ALLY_START_R0[1]), role_id=0)
        ally1 = Robot(float(ALLY_START_R1[0]), float(ALLY_START_R1[1]), role_id=1)
        opp0 = Robot(float(OPP_START_R0[0]), float(OPP_START_R0[1]), role_id=0)
        opp1 = Robot(float(OPP_START_R1[0]), float(OPP_START_R1[1]), role_id=1)
        self.assertEqual(field_half(ally0, False), "bottom")
        self.assertEqual(field_half(ally1, False), "bottom")
        self.assertEqual(field_half(opp0, True), "top")
        self.assertEqual(field_half(opp1, True), "top")

    def test_cross_alliance_pair(self):
        self.assertTrue(cross_alliance_pair(0, 0))
        self.assertTrue(cross_alliance_pair(1, 1))
        self.assertFalse(cross_alliance_pair(0, 1))
        self.assertFalse(cross_alliance_pair(1, 0))

    def test_alliance_starts_opposite_sides_of_midfield(self):
        field = Field()
        for ally in field.allies:
            self.assertLess(ally.y + ROBOT_W / 2, HALF_FIELD_Y)
        for opp in field.opponents:
            self.assertGreater(opp.y - ROBOT_W / 2, HALF_FIELD_Y)

    def test_alliance_robots_start_on_opposite_field_sides(self):
        field = Field()
        self.assertLess(field.allies[0].x, field.allies[1].x)
        self.assertLess(field.opponents[0].x, field.opponents[1].x)
        self.assertNotAlmostEqual(field.allies[0].x, field.allies[1].x)
        self.assertNotAlmostEqual(field.opponents[0].x, field.opponents[1].x)

    def test_collision_separates_overlapping_pair(self):
        a = Robot(50.0, 30.0, role_id=0)
        b = Robot(50.0, 30.0, role_id=1)
        moved = resolve_robot_pair_collision(a, b)
        self.assertTrue(moved)
        self.assertGreater(np.hypot(b.x - a.x, b.y - a.y), 0.0)

    def test_resolve_all_robot_collisions_covers_every_pair(self):
        allies = [Robot(50.0, 30.0, role_id=0), Robot(50.0, 30.0, role_id=1)]
        opps = [Robot(50.0, 30.0, role_id=0), Robot(50.0, 30.0, role_id=1)]
        resolve_all_robot_collisions(allies, opps)
        for i, ra in enumerate(allies + opps):
            for rb in (allies + opps)[i + 1:]:
                dist = np.hypot(rb.x - ra.x, rb.y - ra.y)
                self.assertGreaterEqual(dist, ROBOT_W - 0.05)


class TestHalfFieldInEnv(unittest.TestCase):
    def _make_2v2(self) -> E.TeamAgentWrapper:
        env = E.TeamAgentWrapper(
            render_mode=None,
            num_allies=2,
            num_opponents=2,
            use_timer=False,
        )
        env.reset(seed=7)
        return env

    def test_bottom_robot_stays_out_of_top_exclusive(self):
        env = self._make_2v2()
        inner = env.env
        robot = inner.field.allies[0]
        robot.y = 120.0
        clamp_robot_to_half(robot, False, Action.COLLECT_BLOCKS)
        self.assertLessEqual(robot.y, HALF_TOP_Y_MIN)

    def test_obs_excludes_robots(self):
        env = self._make_2v2()
        inner = env.env
        obs = inner._build_obs(0)
        self.assertEqual(obs.shape, (STATE_DIM,))


class TestHalfFieldStep(unittest.TestCase):
    def test_isolation_period_keeps_allies_on_alliance_side(self):
        env = E.TeamAgentWrapper(
            render_mode=None, num_allies=2, num_opponents=2, use_timer=False,
        )
        env.reset(seed=99)
        # One decision (3 s) — still inside the 15 s VAIRC Isolation Period.
        env.step(np.array([int(Action.COLLECT_BLOCKS), int(Action.COLLECT_BLOCKS)]))
        inner = env.env
        self.assertLess(inner._match_elapsed, ISOLATION_PERIOD)
        y_min, y_max = y_bounds_for_robot(inner.field.allies[0], False, Action.COLLECT_BLOCKS)
        for ally in inner.field.allies:
            self.assertGreaterEqual(ally.y, y_min)
            self.assertLessEqual(ally.y, y_max)

    def test_interaction_period_allows_crossing_midfield_y(self):
        robot = Robot(72.0, 120.0, role_id=0)
        E._set_isolation_lanes(False)
        y_min, y_max = E._half_y_bounds(robot, False, Action.COLLECT_BLOCKS)
        self.assertLess(y_min, HALF_TOP_Y_MIN)
        self.assertGreaterEqual(y_max, 120.0)
        E._set_isolation_lanes(True)
        y_min_iso, y_max_iso = E._half_y_bounds(robot, False, Action.COLLECT_BLOCKS)
        self.assertLessEqual(y_max_iso, HALF_TOP_Y_MIN)

    def test_x_lane_keeps_robots_on_own_field_half(self):
        left = Robot(72.0, 36.0, role_id=0)
        right = Robot(72.0, 36.0, role_id=1)
        left.x = 108.0
        right.x = 24.0
        clamp_robot_to_half(left, False, Action.COLLECT_BLOCKS, enforce_x_lane=True)
        clamp_robot_to_half(right, False, Action.COLLECT_BLOCKS, enforce_x_lane=True)
        self.assertLessEqual(left.x, HALF_LEFT_X_MAX)
        self.assertGreaterEqual(right.x, HALF_RIGHT_X_MIN)


if __name__ == "__main__":
    unittest.main()
