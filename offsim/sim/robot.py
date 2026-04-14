"""Robot class — position, heading, held objects, tank-drive movement.

Position is tracked with 2-decimal-inch precision matching real
VEX AI localization. The robot is 15in x 15in viewed from above.

Movement model: TANK DRIVE (differential drive).
  - Robot first turns toward target, then drives forward.
  - Turn and drive can happen simultaneously (arc) when angle error is small.
  - Each tick (DT seconds), the robot updates heading then position.

NOTE: The physical robot has a mecanum drivetrain. Strafing support can be
added by extending move_toward_point() with a lateral velocity component.
When adding strafing, the robot would be able to move sideways without
rotating, enabling more efficient paths to nearby targets.
"""

from __future__ import annotations
import numpy as np
from sim.config import (
    FIELD_W, FIELD_H, ROBOT_W, ROBOT_H, MAX_SPEED, TURN_RATE, DT, MAX_CARRY,
)


class Robot:
    """A single robot on the field."""

    def __init__(self, x: float, y: float, heading: float = 0.0, role_id: int = 0):
        self.x: float = round(x, 2)
        self.y: float = round(y, 2)
        self.heading: float = heading        # radians, 0 = facing right (+x)
        self.role_id: int = role_id
        self.balls_held: int = 0
        self.held_object_ids: list[int] = []

        # Tracking
        self.actions_attempted: int = 0
        self.actions_succeeded: int = 0

        # Movement state (for animation)
        self.target: np.ndarray | None = None   # current moveToPoint target
        self.moving: bool = False

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @position.setter
    def position(self, val: np.ndarray):
        self.x = round(float(val[0]), 2)
        self.y = round(float(val[1]), 2)

    @property
    def half_w(self) -> float:
        return ROBOT_W / 2.0

    def success_ratio(self) -> float:
        if self.actions_attempted == 0:
            return 0.0
        return self.actions_succeeded / self.actions_attempted

    # ------------------------------------------------------------------
    # Tank-drive moveToPoint  (smooth arc model)
    # ------------------------------------------------------------------
    def move_toward_point(self, target: np.ndarray) -> bool:
        """Advance one tick (DT seconds) toward target using smooth arc drive.

        Motion model (inspired by visual-tracking-demo proportional PID):
          1. Compute angle error to target.
          2. Turn proportionally to the error (P controller), capped by TURN_RATE.
             → Large errors → fast turn; small errors → gentle curve.
          3. Forward speed = MAX_SPEED * max(0.08, 1 - |angle_err| / π)
             → Robot never fully stops; it arcs smoothly instead of
               stopping to turn in place, giving fluid curved paths.

        This matches the behaviour seen in leoxie080808/visual-tracking-demo-python
        where forward_speed = speed * max(0.1, 1 - abs(angle_diff) / π).

        Returns True when within 0.5in of target.
        """
        self.target = target.copy()
        diff = target - self.position
        dist = np.linalg.norm(diff)

        if dist < 0.50:
            self.moving = False
            return True

        self.moving = True

        # Desired heading and error
        desired_heading = np.arctan2(diff[1], diff[0])
        angle_err = _wrap_angle(desired_heading - self.heading)

        # Proportional turn — gain of 4.0 rad/s per rad of error, capped by TURN_RATE
        turn_delta = np.clip(angle_err * 4.0 * DT, -TURN_RATE * DT, TURN_RATE * DT)
        self.heading = _wrap_angle(self.heading + turn_delta)

        # Recompute error after this tick's turn
        abs_err = abs(_wrap_angle(desired_heading - self.heading))

        # Continuous arc speed: full speed when aligned, 8% minimum when pointing away
        drive_factor = max(0.08, 1.0 - abs_err / np.pi)
        max_move = MAX_SPEED * DT * drive_factor

        if dist <= max_move:
            self.position = target.copy()
        else:
            forward = np.array([np.cos(self.heading), np.sin(self.heading)])
            self.position = self.position + forward * max_move

        self._clamp_to_field()
        return np.linalg.norm(target - self.position) < 0.50

    def _clamp_to_field(self):
        half = self.half_w
        self.x = round(np.clip(self.x, half, FIELD_W - half), 2)
        self.y = round(np.clip(self.y, half, FIELD_H - half), 2)

    def reset(self, x: float, y: float, heading: float = 0.0):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.heading = heading
        self.balls_held = 0
        self.held_object_ids = []
        self.actions_attempted = 0
        self.actions_succeeded = 0
        self.target = None
        self.moving = False


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
