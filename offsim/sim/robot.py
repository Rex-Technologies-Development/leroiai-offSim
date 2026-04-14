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
    # Tank-drive moveToPoint
    # ------------------------------------------------------------------
    def move_toward_point(self, target: np.ndarray) -> bool:
        """Advance one tick (DT seconds) toward target using tank drive.

        Tank drive behaviour:
          1. Compute angle to target
          2. Turn toward target (clamped by TURN_RATE * DT)
          3. Drive forward (clamped by MAX_SPEED * DT)
             - Drive speed scales down when angle error is large (realistic:
               tank drive turns in place when error > 45deg, arcs when smaller)

        Returns True if the robot has reached the target (within 0.5in).
        """
        self.target = target.copy()
        diff = target - self.position
        dist = np.linalg.norm(diff)

        if dist < 0.50:
            self.moving = False
            return True

        self.moving = True

        # Desired heading
        desired_heading = np.arctan2(diff[1], diff[0])

        # Angle error (wrapped to [-pi, pi])
        angle_err = _wrap_angle(desired_heading - self.heading)

        # Turn (clamped by turn rate)
        max_turn = TURN_RATE * DT
        if abs(angle_err) <= max_turn:
            self.heading = desired_heading
        else:
            self.heading += np.sign(angle_err) * max_turn
        self.heading = _wrap_angle(self.heading)

        # Drive speed scales with alignment:
        #   - angle_err > 45deg: turn in place (no forward movement)
        #   - angle_err < 10deg: full speed
        #   - in between: proportional
        abs_err = abs(_wrap_angle(desired_heading - self.heading))
        if abs_err > np.radians(45):
            drive_factor = 0.0   # turn in place
        elif abs_err < np.radians(10):
            drive_factor = 1.0   # full speed
        else:
            drive_factor = 1.0 - (abs_err - np.radians(10)) / np.radians(35)

        max_move = MAX_SPEED * DT * drive_factor
        if max_move > 0:
            if dist <= max_move:
                self.position = target.copy()
            else:
                forward = np.array([np.cos(self.heading), np.sin(self.heading)])
                new_pos = self.position + forward * max_move
                self.position = new_pos

        # Clamp to field bounds
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
