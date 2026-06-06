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

ACCEL = 150.0   # in/s² — ramp-up rate
DECEL = 280.0   # in/s² — braking rate (stronger so stops feel snappy)

# Treat a target as reached when we're within this distance (inches).
# 0.5" was too strict for the arc + rounding model and could cause orbiting
# around waypoints (never popping the waypoint queue).
ARRIVAL_DIST = 2.0


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
        self.speed: float = 0.0                  # current forward speed (in/s)
        self.score_timer: float = 0.0            # elapsed seconds in scoring sequence
        self.intake_active: bool = False         # True while intake is spinning (collecting)

        # Physical capabilities. Allies run at full spec; opponents may be set
        # slower / weaker via an OpponentProfile (see sim/opponent.py). These are
        # CAPABILITIES, not per-episode state, so reset() leaves them untouched.
        self.speed_scale: float = 1.0           # multiplier on MAX_SPEED top speed
        self.capacity:    int   = MAX_CARRY     # max balls this robot can hold
        self.score_interval: float = 0.0        # opponents: dwell (s) to score a load
        # Creeping exploration: consecutive COLLECT decisions with no reachable
        # ball found (the robot is cycling without seeing new blocks). Drives the
        # scan frontier progressively north in env._action_to_target(). Reset to 0
        # the moment a ball is collected or a reachable target reappears.
        self.explore_barren: int = 0

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
    def move_toward_point(self, target: np.ndarray, reverse: bool = False,
                          arrival_dist: float = ARRIVAL_DIST) -> bool:
        """Advance one tick (DT seconds) toward target with accel/decel.

        Motion model:
          1. Turn proportionally so the chosen drive axis points at the target
             (front axis when reverse=False, rear axis when reverse=True).
          2. Compute target speed — full when aligned & far; reduced when:
               a) heading error is large (arc turning penalty)
               b) close enough that braking is needed to stop cleanly
          3. Ramp self.speed toward target speed at ACCEL / DECEL rates.
          4. Drive along the chosen axis (rear-first when reverse=True).

        reverse=True drives the robot rear-first toward the target, keeping its
        BACK pointed at the destination. Used for backing into a goal so the
        scoring face feeds the goal on arrival without a 180° spin at the mouth.

        arrival_dist overrides the default ARRIVAL_DIST stop tolerance. The
        scoring approach passes a tight value so the robot noses fully onto the
        score pose (within scoring range of the goal) instead of stopping ~2"
        short — the default tolerance is wider than the scoring-range margin.

        Returns True when within arrival_dist inches of target.
        """
        self.target = target.copy()
        diff = target - self.position
        dist = float(np.linalg.norm(diff))

        if dist < arrival_dist:
            self.moving = False
            self.speed  = 0.0
            return True

        self.moving = True

        # --- Heading control ---
        # travel_heading points from the robot toward the target. Forward driving
        # aligns the front with it; reverse driving aligns the front with the
        # OPPOSITE direction so the back leads into the target.
        travel_heading  = np.arctan2(diff[1], diff[0])
        desired_heading = _wrap_angle(travel_heading + np.pi) if reverse else travel_heading
        angle_err = _wrap_angle(desired_heading - self.heading)
        turn_delta = np.clip(angle_err * 4.0 * DT, -TURN_RATE * DT, TURN_RATE * DT)
        self.heading = _wrap_angle(self.heading + turn_delta)
        abs_err = abs(_wrap_angle(desired_heading - self.heading))

        # --- Speed target ---
        # Per-robot top speed (opponents may be throttled via speed_scale).
        max_speed = MAX_SPEED * self.speed_scale
        # Turning penalty: slow when pointing away, full speed when aligned
        heading_factor = max(0.08, 1.0 - abs_err / np.pi)

        # Deceleration ramp: how far do we need to begin braking from current speed?
        # d_stop = v² / (2 * DECEL)
        stop_dist = (self.speed ** 2) / (2.0 * DECEL + 1e-6)
        if dist <= stop_dist + 1.0:
            # Inside braking window — scale down proportionally
            decel_factor = max(0.0, dist / (stop_dist + 1.0))
            target_speed = max_speed * min(heading_factor, decel_factor)
        else:
            target_speed = max_speed * heading_factor

        # --- Ramp current speed toward target ---
        if self.speed < target_speed:
            self.speed = min(self.speed + ACCEL * DT, target_speed)
        else:
            self.speed = max(self.speed - DECEL * DT, target_speed)
        self.speed = float(np.clip(self.speed, 0.0, max_speed))

        # --- Move ---
        # Drive along the rear axis when reversing, the front axis otherwise.
        drive_heading = _wrap_angle(self.heading + np.pi) if reverse else self.heading
        max_move = self.speed * DT
        if dist <= max_move:
            self.position = target.copy()
            self.speed    = 0.0
        else:
            drive = np.array([np.cos(drive_heading), np.sin(drive_heading)])
            self.position = self.position + drive * max_move

        self._clamp_to_field()
        return float(np.linalg.norm(target - self.position)) < arrival_dist

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
        self.target        = None
        self.moving        = False
        self.speed         = 0.0
        self.score_timer   = 0.0
        self.intake_active = False
        self.explore_barren = 0


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
