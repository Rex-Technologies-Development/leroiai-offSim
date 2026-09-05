"""Deterministic top-down robot and drivetrain models for Override."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np
from .config import (
    Alliance, ChassisType, FIELD_HEIGHT, FIELD_WIDTH, LINEAR_ACCEL,
    LINEAR_DECEL, MAX_FORWARD_SPEED, MAX_LATERAL_SPEED, MAX_YAW_RATE,
    ROBOT_LENGTH, ROBOT_RADIUS, ROBOT_WIDTH, YAW_ACCEL,
)

# Stuck-detection (offsim navigation fix): a robot that a controller keeps commanding forward but
# that barely translates while repeatedly colliding is pinned against a non-target obstacle. The
# reactive avoider has no global planner, so we detect this and steer along the obstacle tangent.
GRIND_WINDOW = 10     # ticks (~0.5s at dt=0.05) of position history to inspect
GRIND_EPS = 1.0       # inches: net motion below this over the window == not translating
GRIND_MIN_HITS = 4    # collisions during the window == actually in contact (not just rotating)
ESCAPE_TICKS = 25     # once grinding is detected, commit to the tangential escape this long (~1.25s)
                      # so a tank can finish turning to the tangent and drive clear without oscillating
STUCK_MIN_DIST = 16.0 # only escape when this FAR from the target: a robot pressed against its OWN
                      # target goal to score (docks at radius+goal ~12.5in) is grinding productively

def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi

def _approach(current: float, target: float, up: float, down: float, dt: float) -> float:
    rate = up if abs(target) > abs(current) and current * target >= 0 else down
    delta = max(-rate * dt, min(rate * dt, target - current))
    return current + delta

@dataclass
class Robot:
    robot_id: int
    alliance: Alliance
    chassis: ChassisType
    x: float
    y: float
    heading: float
    radius: float = ROBOT_RADIUS
    width: float = ROBOT_WIDTH        # visual footprint (inches); square when width == length
    length: float = ROBOT_LENGTH
    forward_velocity: float = 0.0
    lateral_velocity: float = 0.0
    yaw_velocity: float = 0.0
    held_pin: int | None = None
    held_cup: int | None = None
    preload_pin: int | None = None
    preload_cup: int | None = None
    collisions: int = 0
    blocked_interactions: int = 0
    telemetry: list[str] = field(default_factory=list)
    _hist: deque = field(default_factory=lambda: deque(maxlen=GRIND_WINDOW), repr=False, compare=False)
    _escape_ticks: int = 0    # >0 while committed to a tangential escape (stuck-breaker)

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float64)

    def is_grinding(self) -> bool:
        """True if, over the recent window, the robot barely translated WHILE repeatedly colliding —
        i.e. a controller is driving it into a non-target obstacle and it is pinned. Callers gate on
        'commanded forward', so this only fires as 'commanded to move but stuck against something'
        (rotation-in-place doesn't collide, so it is excluded)."""
        if len(self._hist) < GRIND_WINDOW:
            return False
        xs = [p[0] for p in self._hist]
        ys = [p[1] for p in self._hist]
        barely_moved = (max(xs) - min(xs)) < GRIND_EPS and (max(ys) - min(ys)) < GRIND_EPS
        in_contact = (self._hist[-1][2] - self._hist[0][2]) >= GRIND_MIN_HITS
        return barely_moved and in_contact

    def command(self, forward: float, lateral: float, yaw: float, dt: float) -> None:
        """Apply normalized chassis control for one tick.

        Tank ignores lateral input. Mecanum accepts body-frame forward/lateral/yaw.
        """
        self._hist.append((self.x, self.y, self.collisions))   # last tick's resolved pose, for stuck-detection
        forward = float(np.clip(forward, -1.0, 1.0)) * MAX_FORWARD_SPEED
        lateral_norm = float(np.clip(lateral, -1.0, 1.0))
        if self.chassis is ChassisType.TANK:
            lateral_norm = 0.0
        lateral = lateral_norm * MAX_LATERAL_SPEED
        yaw = float(np.clip(yaw, -1.0, 1.0)) * MAX_YAW_RATE
        self.forward_velocity = _approach(self.forward_velocity, forward, LINEAR_ACCEL, LINEAR_DECEL, dt)
        self.lateral_velocity = _approach(self.lateral_velocity, lateral, LINEAR_ACCEL, LINEAR_DECEL, dt)
        self.yaw_velocity = _approach(self.yaw_velocity, yaw, YAW_ACCEL, YAW_ACCEL, dt)
        c, s = math.cos(self.heading), math.sin(self.heading)
        self.x += (c * self.forward_velocity - s * self.lateral_velocity) * dt
        self.y += (s * self.forward_velocity + c * self.lateral_velocity) * dt
        self.heading = wrap_angle(self.heading + self.yaw_velocity * dt)

    def brake(self, dt: float) -> None:
        self.command(0.0, 0.0, 0.0, dt)

    def clamp_to_walls(self) -> bool:
        old = (self.x, self.y)
        self.x = float(np.clip(self.x, self.radius, FIELD_WIDTH - self.radius))
        self.y = float(np.clip(self.y, self.radius, FIELD_HEIGHT - self.radius))
        hit = old != (self.x, self.y)
        if hit:
            self.forward_velocity = self.lateral_velocity = 0.0
            self.collisions += 1
        return hit

    def drive_toward(self, target: tuple[float, float], dt: float) -> None:
        """Deterministic objective controller honoring the selected chassis."""
        dx, dy = target[0] - self.x, target[1] - self.y
        distance = math.hypot(dx, dy)
        if distance < 0.25:
            self.brake(dt)
            return
        desired = math.atan2(dy, dx)
        error = wrap_angle(desired - self.heading)
        yaw = float(np.clip(error / 0.7, -1.0, 1.0))
        if self.chassis is ChassisType.MECANUM:
            c, s = math.cos(self.heading), math.sin(self.heading)
            body_forward = c * dx + s * dy
            body_lateral = -s * dx + c * dy
            scale = max(distance, 1.0)
            self.command(body_forward / scale, body_lateral / scale, yaw * 0.35, dt)
        else:
            forward = max(0.0, math.cos(error)) * min(1.0, distance / 12.0)
            self.command(forward, 0.0, yaw, dt)
