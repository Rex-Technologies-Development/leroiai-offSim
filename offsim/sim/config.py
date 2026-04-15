"""Field dimensions, action IDs, state dims, timing constants.

Game: VEX Push Back 2025-2026
Field: 144.00in x 144.00in (12ft x 12ft)
Robot: 15in x 15in footprint, tank drive
Position precision: 2 decimal places in inches (matches real localization)

Goals (top-down view):
  OUR_LONG_GOAL  — right wall L-bracket (blue alliance, x≈138)
  OPP_LONG_GOAL  — left wall L-bracket  (red alliance,  x≈6)
  CENTER_MID_GOAL — center field, upper half (y≈84)
  CENTER_LOW_GOAL — center field, lower half (y≈60)

Ball colors:
  BALL_RED  = 0 (22 red balls, start near left/opp side)
  BALL_BLUE = 1 (22 blue balls, start near right/our side)
  Note: color is for display and strategic awareness.
  Either color can be scored in any goal in this sim.

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
# Ball colors
# ---------------------------------------------------------------------------
BALL_RED  = 0   # red alliance balls
BALL_BLUE = 1   # blue alliance balls (our team)

# ---------------------------------------------------------------------------
# Actions — RL decides WHERE to go, not HOW to move
# ---------------------------------------------------------------------------
class Action(enum.IntEnum):
    COLLECT_NEAREST_BALL = 0
    SCORE_LONG_GOAL      = 1   # score in our long goal (right wall)
    SCORE_CENTER_GOAL    = 2   # score in nearest center goal
    DESCORE_OPP_LONG     = 3   # descore from opponent long goal (left wall)
    DESCORE_CENTER       = 4   # descore from center goals
    DEFEND_ZONE          = 5
    MOVE_TO_REGION_A     = 6
    MOVE_TO_REGION_B     = 7
    IDLE                 = 8

NUM_ACTIONS = len(Action)

# ---------------------------------------------------------------------------
# Field (inches, origin bottom-left, 2 decimal precision)
# ---------------------------------------------------------------------------
FIELD_W: float = SHARED_CFG["field"]["width"]    # 144.00
FIELD_H: float = SHARED_CFG["field"]["height"]   # 144.00

# Goal positions — center of each goal body (scoring target for robots).
# Long goals are FREE-STANDING (~14" alley between goal and nearest wall).
# Outer face at ~129.6 / ~14.4" from wall; inner face ~10.9" further in.
OUR_LONG_GOAL    = np.array([119.00, 72.00])  # center of right goal body (23" alley from wall)
OPP_LONG_GOAL    = np.array([ 25.00, 72.00])  # center of left goal body
CENTER_MID_GOAL  = np.array([ 72.00, 80.94])  # center field, upper X half
CENTER_LOW_GOAL  = np.array([ 72.00, 63.06])  # center field, lower X half

# Long goal physical extents
# Goals are FREE-STANDING with a ~23" alley between the wall and the outer goal face.
# Goal body is ~4.125" wide (tray depth).
LONG_GOAL_Y_MIN    = 47.61   # bottom of goal bar
LONG_GOAL_Y_MAX    = 96.39   # top of goal bar
LONG_GOAL_WALL_GAP = 21.50   # alley width: gap between wall and outer face of goal
LONG_GOAL_WIDTH    =  4.125  # goal tray width (outer → inner/scoring face)

# Center goal X-structure — two crossing rectangular bars forming an X.
# The upper two arms = MID goal; lower two arms = LOW goal.
# ARM_LEN is the half-length from center to each arm tip.
CENTER_GOAL_ARM_LEN = 23.18   # half-diagonal from center to arm tip (scaled from spec)
CENTER_GOAL_ARM_W   =  4.00   # arm width (bar thickness)

# Matchload tubes — stand along the TOP and BOTTOM walls at the
# tile-1/tile-2 seam (x=24" and x=120" from each side wall).
MATCHLOAD_TUBE_RADIUS = 2.00
MATCHLOAD_TUBES = np.array([
    [ 24.0,   2.0],   # bottom wall, 1-tile seam from left
    [120.0,   2.0],   # bottom wall, 1-tile seam from right
    [ 24.0, 142.0],   # top wall, 1-tile seam from left
    [120.0, 142.0],   # top wall, 1-tile seam from right
], dtype=np.float64)

# Parking zones — raised platforms centered on the top and bottom walls.
# ~25" wide (centered at x=72), 24" deep (1 tile) against each wall.
PARK_ZONE_X_MIN  = 59.5    # left edge  (72 - 12.5)
PARK_ZONE_X_MAX  = 84.5    # right edge (72 + 12.5)
PARK_ZONE_BOTTOM = 24.0    # top edge of bottom park zone
PARK_ZONE_TOP    = 120.0   # bottom edge of top park zone

# Vision cone — wide-angle camera in front of robot
VISION_HALF_ANGLE = np.radians(65.0)   # ±65° → 130° total FOV
VISION_RANGE      = 72.0               # max depth in inches

# Named zones
REGION_A_CENTER  = np.array([108.00, 108.00])  # upper-right
REGION_B_CENTER  = np.array([ 36.00,  36.00])  # lower-left
DEFEND_ZONE_POS  = np.array([108.00,  72.00])  # in front of our long goal

# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------
MATCH_DURATION: float    = SHARED_CFG["game"]["match_duration"]       # 120s
DT: float                = SHARED_CFG["game"]["dt"]                   # 0.05s
DECISION_INTERVAL: float = SHARED_CFG["game"]["decision_interval"]    # 3.0s
TICKS_PER_DECISION: int  = int(DECISION_INTERVAL / DT)                # 60
MAX_CARRY: int           = SHARED_CFG["game"]["max_carry"]            # 3

# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------
MAX_GAME_OBJECTS: int = SHARED_CFG["state"]["max_game_objects"]  # 44
OBJ_FEATURES: int    = SHARED_CFG["state"]["object_features"]   # 4

# Status codes
OBJ_ON_FIELD   = 0
OBJ_HELD       = 1
OBJ_SCORED_US  = 2
OBJ_SCORED_OPP = 3
OBJ_REMOVED    = 4   # deleted via state editor

# Scoring values
LONG_GOAL_POINTS   = 5
CENTER_GOAL_POINTS = 3

# ---------------------------------------------------------------------------
# Initial ball layout — 22 red (color=0) + 22 blue (color=1) = 44 total
# Format: [x, y, color]
# We are BLUE (right side); opponent is RED (left side)
# ---------------------------------------------------------------------------
INITIAL_OBJECTS = np.array([
    # === RED BALLS (color=0) — clustered left/center ===
    # Top-left corner cluster
    [12.00, 120.00, 0], [18.00, 126.00, 0], [24.00, 120.00, 0],
    # Bottom-left corner cluster
    [12.00,  24.00, 0], [18.00,  18.00, 0], [24.00,  24.00, 0],
    # Left wall singles
    [12.00,  96.00, 0], [12.00,  72.00, 0], [12.00,  48.00, 0],
    # Left zone column
    [30.00, 108.00, 0], [30.00,  72.00, 0], [30.00,  36.00, 0],
    # Center-left column
    [48.00, 108.00, 0], [48.00,  72.00, 0], [48.00,  36.00, 0],
    # Inner center-left
    [60.00,  96.00, 0], [60.00,  60.00, 0],
    # Upper center (red side)
    [68.00, 108.00, 0], [72.00, 120.00, 0],
    # Extra left zone
    [36.00, 108.00, 0], [36.00,  36.00, 0], [36.00,  72.00, 0],

    # === BLUE BALLS (color=1) — clustered right/center ===
    # Top-right corner cluster
    [132.00, 120.00, 1], [126.00, 126.00, 1], [120.00, 120.00, 1],
    # Bottom-right corner cluster
    [132.00,  24.00, 1], [126.00,  18.00, 1], [120.00,  24.00, 1],
    # Right wall singles
    [132.00,  96.00, 1], [132.00,  72.00, 1], [132.00,  48.00, 1],
    # Right zone column
    [114.00, 108.00, 1], [114.00,  72.00, 1], [114.00,  36.00, 1],
    # Center-right column
    [ 96.00, 108.00, 1], [ 96.00,  72.00, 1], [ 96.00,  36.00, 1],
    # Inner center-right
    [ 84.00,  96.00, 1], [ 84.00,  60.00, 1],
    # Lower center (blue side)
    [ 76.00,  36.00, 1], [ 72.00,  24.00, 1],
    # Extra right zone
    [108.00, 108.00, 1], [108.00,  36.00, 1], [108.00,  72.00, 1],
], dtype=np.float64)

assert len(INITIAL_OBJECTS) == 44, f"Expected 44 balls, got {len(INITIAL_OBJECTS)}"

# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
HEATMAP_W: int = SHARED_CFG["state"]["heatmap_grid_w"]  # 12
HEATMAP_H: int = SHARED_CFG["state"]["heatmap_grid_h"]  # 12

# ---------------------------------------------------------------------------
# State vector dimension (flat)
# role_id(1) + time(1) + my_score(1) + opp_score(1)
# + x(1) + y(1) + heading(1) + balls_held(1) + balls_nearby(1) = 9
# + game_objects(44 * 4) = 176
# + heatmap(12 * 12) = 144
# + extra(2) = 2
# Total = 331
# ---------------------------------------------------------------------------
STATE_DIM = 9 + (MAX_GAME_OBJECTS * OBJ_FEATURES) + (HEATMAP_W * HEATMAP_H) + 2

# ---------------------------------------------------------------------------
# Robot physics
# ---------------------------------------------------------------------------
ROBOT_W: float       = SHARED_CFG["robot"]["width"]           # 15.00 in
ROBOT_H: float       = SHARED_CFG["robot"]["height"]          # 15.00 in
MAX_SPEED: float     = SHARED_CFG["robot"]["max_speed"]       # 30.00 in/s
TURN_RATE: float     = SHARED_CFG["robot"]["turn_rate"]       # 3.14 rad/s
COLLECT_RANGE: float = SHARED_CFG["robot"]["collect_range"]   # 8.00 in
SCORE_RANGE: float   = SHARED_CFG["robot"]["score_range"]     # 10.00 in

MAX_SCORE = 200  # normalisation ceiling (44 balls * 5pts max)
