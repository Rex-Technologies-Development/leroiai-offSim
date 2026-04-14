"""Field dimensions, action IDs, state dims, timing constants.

Field: 144.00in x 144.00in (12ft x 12ft VEX AI field)
Robot: 15in x 15in footprint, tank drive
Position precision: 2 decimal places in inches (matches real localization)

NOTE: The robot base is mecanum in real life, but we model it as tank drive
for now. Strafing can be added later by modifying Robot.move_toward_point()
in robot.py to support lateral movement.
"""

import enum
import yaml
import os
import numpy as np

# ---------------------------------------------------------------------------
# Load shared config
# ---------------------------------------------------------------------------
_SHARED_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "shared", "config.yaml")
with open(_SHARED_CFG_PATH, "r") as f:
    SHARED_CFG = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Actions — the RL decides WHERE to go, not HOW to move
# ---------------------------------------------------------------------------
class Action(enum.IntEnum):
    COLLECT_NEAREST_BALL = 0
    SCORE_LONG_GOAL = 1
    SCORE_MID_GOAL = 2
    DESCORE_OPPONENT_LONG = 3
    DESCORE_OPPONENT_MID = 4
    DEFEND_ZONE = 5
    MOVE_TO_REGION_A = 6
    MOVE_TO_REGION_B = 7
    IDLE = 8

NUM_ACTIONS = len(Action)

# ---------------------------------------------------------------------------
# Field (inches, origin bottom-left, 2 decimal precision)
# ---------------------------------------------------------------------------
FIELD_W: float = SHARED_CFG["field"]["width"]    # 144.00
FIELD_H: float = SHARED_CFG["field"]["height"]   # 144.00

# Goal positions (inches, approximate VEX AI High Stakes layout)
OUR_LONG_GOAL    = np.array([6.00, 72.00])
OUR_MID_GOAL     = np.array([36.00, 6.00])
OPP_LONG_GOAL    = np.array([138.00, 72.00])
OPP_MID_GOAL     = np.array([108.00, 138.00])

# Named regions
REGION_A_CENTER  = np.array([36.00, 108.00])
REGION_B_CENTER  = np.array([108.00, 36.00])
DEFEND_ZONE_POS  = np.array([36.00, 72.00])

# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------
MATCH_DURATION: float = SHARED_CFG["game"]["match_duration"]       # 120s
DT: float             = SHARED_CFG["game"]["dt"]                   # 0.05s per sim tick
DECISION_INTERVAL: float = SHARED_CFG["game"]["decision_interval"] # 3.0s between RL queries
TICKS_PER_DECISION: int  = int(DECISION_INTERVAL / DT)             # 60 ticks per decision
MAX_CARRY: int        = SHARED_CFG["game"]["max_carry"]            # 3

# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------
MAX_GAME_OBJECTS: int = SHARED_CFG["state"]["max_game_objects"]  # 24
OBJ_FEATURES: int    = SHARED_CFG["state"]["object_features"]   # 3

# Status codes
OBJ_ON_FIELD  = 0
OBJ_HELD      = 1
OBJ_SCORED_US = 2
OBJ_SCORED_OPP = 3

# Initial ring positions — 24 objects spread across the 144x144 field
INITIAL_OBJECT_POSITIONS = np.array([
    # Row 1 (y ≈ 24)
    [24.00, 24.00], [48.00, 24.00], [72.00, 24.00], [96.00, 24.00], [120.00, 24.00],
    # Row 2 (y ≈ 48)
    [24.00, 48.00], [48.00, 48.00], [96.00, 48.00], [120.00, 48.00],
    # Row 3 (y ≈ 72)
    [24.00, 72.00], [48.00, 72.00], [96.00, 72.00], [120.00, 72.00],
    # Row 4 (y ≈ 96)
    [24.00, 96.00], [48.00, 96.00], [96.00, 96.00], [120.00, 96.00],
    # Row 5 (y ≈ 120)
    [24.00, 120.00], [48.00, 120.00], [72.00, 120.00], [96.00, 120.00], [120.00, 120.00],
    # Extra center objects
    [72.00, 48.00], [72.00, 96.00],
], dtype=np.float64)

LONG_GOAL_POINTS = 5
MID_GOAL_POINTS  = 3

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
HEATMAP_W: int = SHARED_CFG["state"]["heatmap_grid_w"]  # 12
HEATMAP_H: int = SHARED_CFG["state"]["heatmap_grid_h"]  # 12

# ---------------------------------------------------------------------------
# State vector dimension (flat)
# ---------------------------------------------------------------------------
# role_id(1) + time_remaining(1) + my_score(1) + opp_score(1)
# + my_x(1) + my_y(1) + my_heading(1) + balls_held(1) + balls_nearby(1)
# + game_objects(24 * 3) + heatmap(12 * 12)
# + expected_state_delta(1) + actions_completed_ratio(1)
STATE_DIM = 9 + (MAX_GAME_OBJECTS * OBJ_FEATURES) + (HEATMAP_W * HEATMAP_H) + 2

# ---------------------------------------------------------------------------
# Robot physics
# ---------------------------------------------------------------------------
ROBOT_W: float     = SHARED_CFG["robot"]["width"]          # 15.00 in
ROBOT_H: float     = SHARED_CFG["robot"]["height"]         # 15.00 in
MAX_SPEED: float   = SHARED_CFG["robot"]["max_speed"]      # 30.00 in/s
TURN_RATE: float   = SHARED_CFG["robot"]["turn_rate"]      # 3.14 rad/s
COLLECT_RANGE: float = SHARED_CFG["robot"]["collect_range"] # 8.00 in
SCORE_RANGE: float = SHARED_CFG["robot"]["score_range"]     # 10.00 in

MAX_SCORE = 150  # normalisation ceiling
