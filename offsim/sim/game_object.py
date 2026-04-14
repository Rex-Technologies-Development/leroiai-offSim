"""Game object (ball/ring) — position and state tracking."""

from __future__ import annotations
import numpy as np
from sim.config import OBJ_ON_FIELD


class GameObject:
    """A single ring/ball on the field."""

    def __init__(self, obj_id: int, x: float, y: float):
        self.obj_id: int = obj_id
        self.x: float = round(x, 2)
        self.y: float = round(y, 2)
        self.status: int = OBJ_ON_FIELD   # OBJ_ON_FIELD / OBJ_HELD / OBJ_SCORED_US / OBJ_SCORED_OPP

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @position.setter
    def position(self, val: np.ndarray):
        self.x = round(float(val[0]), 2)
        self.y = round(float(val[1]), 2)

    def reset(self, x: float, y: float):
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.status = OBJ_ON_FIELD
