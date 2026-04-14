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


def compute_heatmap(obj_positions: np.ndarray, obj_statuses: np.ndarray) -> np.ndarray:
    """Compute the point-potential heatmap.

    Args:
        obj_positions: (N, 2) array of object positions in inches.
        obj_statuses:  (N,) array of status codes.

    Returns:
        (HEATMAP_H, HEATMAP_W) float32 array, gaussian-blurred.
    """
    grid = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
    cell_w = FIELD_W / HEATMAP_W   # 12.0 inches per cell
    cell_h = FIELD_H / HEATMAP_H

    for i in range(len(obj_positions)):
        if obj_statuses[i] == OBJ_ON_FIELD:
            gx = min(int(obj_positions[i, 0] / cell_w), HEATMAP_W - 1)
            gy = min(int(obj_positions[i, 1] / cell_h), HEATMAP_H - 1)
            grid[gy, gx] += 1.0

    # Blur so nearby cells also show value
    grid = gaussian_filter(grid, sigma=1.0)
    return grid
