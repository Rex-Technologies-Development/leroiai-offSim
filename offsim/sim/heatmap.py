"""Point potential heatmap computation.

A 12x12 grid over the 144x144 inch field. Each cell holds a value
representing how many points could be gained by acting in that region.
Gaussian-blurred so nearby cells also show value.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from sim.config import (
    FIELD_W, FIELD_H, HEATMAP_W, HEATMAP_H,
    MAX_GAME_OBJECTS, OBJ_ON_FIELD,
)


def compute_heatmap(
    obj_positions: np.ndarray,
    obj_statuses: np.ndarray,
    w: int | None = None,
    h: int | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute the point-potential heatmap.

    Args:
        obj_positions: (N, 2) array of object positions in inches.
        obj_statuses:  (N,) array of status codes.
        w, h:          Grid resolution (defaults to HEATMAP_W/H from config).
        sigma:         Gaussian blur sigma in grid cells (default 1.0).

    Returns:
        (h, w) float32 array, gaussian-blurred.
    """
    gw = w if w is not None else HEATMAP_W
    gh = h if h is not None else HEATMAP_H
    sg = sigma if sigma is not None else 1.0

    cell_w = FIELD_W / gw
    cell_h = FIELD_H / gh

    grid = np.zeros((gh, gw), dtype=np.float32)
    for i in range(len(obj_positions)):
        if obj_statuses[i] == OBJ_ON_FIELD:
            gx = min(int(obj_positions[i, 0] / cell_w), gw - 1)
            gy = min(int(obj_positions[i, 1] / cell_h), gh - 1)
            grid[gy, gx] += 1.0

    # Blur so nearby cells also show value; clusters naturally peak higher
    grid = gaussian_filter(grid, sigma=sg)
    return grid
