"""Game object (ball) — position, color, state, and rolling physics.

VEX Push Back 2025-2026: 44 balls total (22 red, 22 blue).
Color is for display and strategic awareness; both colors score the same.
"""

from __future__ import annotations
import numpy as np
from sim.config import OBJ_ON_FIELD, BALL_BLUE

# Carpet friction: velocity multiplied by this each sim tick (0.05s).
# 0.78^60 ≈ 0.00009 — ball essentially stops within 3 seconds.
BALL_FRICTION   = 0.78
BALL_RESTITUTION = 0.45   # fraction of speed kept after a wall bounce
BALL_RADIUS     = 2.5     # inches — approximate VEX ball radius (for wall bounce)
BALL_STOP_SPEED = 0.5     # in/s threshold below which velocity is zeroed


class GameObject:
    """A single ball on the Push Back field."""

    def __init__(self, obj_id: int, x: float, y: float, color: int = BALL_BLUE):
        self.obj_id: int = obj_id
        self.x: float = round(x, 2)
        self.y: float = round(y, 2)
        self.color: int  = color           # BALL_RED=0, BALL_BLUE=1
        self.status: int = OBJ_ON_FIELD    # OBJ_ON_FIELD/OBJ_HELD/OBJ_SCORED_US/OBJ_SCORED_OPP/OBJ_REMOVED
        self.scored_in_goal: str = ""      # set when scored — "our_long"/"opp_long"/"center_mid"/"center_low"

        # Rolling physics (in/s)
        self.vx: float = 0.0
        self.vy: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @position.setter
    def position(self, val: np.ndarray):
        self.x = round(float(val[0]), 2)
        self.y = round(float(val[1]), 2)

    def apply_physics(self, dt: float, field_w: float, field_h: float):
        """Advance one sim tick: move, bounce walls, apply friction."""
        speed = np.sqrt(self.vx ** 2 + self.vy ** 2)
        if speed < BALL_STOP_SPEED:
            self.vx = 0.0
            self.vy = 0.0
            return

        self.x = round(self.x + self.vx * dt, 2)
        self.y = round(self.y + self.vy * dt, 2)

        # Wall bounce
        if self.x < BALL_RADIUS:
            self.x = BALL_RADIUS
            self.vx = abs(self.vx) * BALL_RESTITUTION
        elif self.x > field_w - BALL_RADIUS:
            self.x = field_w - BALL_RADIUS
            self.vx = -abs(self.vx) * BALL_RESTITUTION

        if self.y < BALL_RADIUS:
            self.y = BALL_RADIUS
            self.vy = abs(self.vy) * BALL_RESTITUTION
        elif self.y > field_h - BALL_RADIUS:
            self.y = field_h - BALL_RADIUS
            self.vy = -abs(self.vy) * BALL_RESTITUTION

        # Carpet friction
        self.vx *= BALL_FRICTION
        self.vy *= BALL_FRICTION

    def reset(self, x: float, y: float, color: int | None = None):
        self.x = round(x, 2)
        self.y = round(y, 2)
        if color is not None:
            self.color = color
        self.status         = OBJ_ON_FIELD
        self.scored_in_goal = ""
        self.vx             = 0.0
        self.vy             = 0.0
