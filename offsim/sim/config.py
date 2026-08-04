"""Override simulator configuration loaded from the shared YAML contract."""
from __future__ import annotations
from enum import Enum, IntEnum
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "shared" / "config.yaml"
with _CONFIG_PATH.open(encoding="utf-8") as stream:
    SHARED_CONFIG = yaml.safe_load(stream)

class Alliance(str, Enum):
    BLUE = "blue"
    RED = "red"

    @property
    def opponent(self) -> "Alliance":
        return Alliance.RED if self is Alliance.BLUE else Alliance.BLUE

class ChassisType(str, Enum):
    TANK = "tank"
    MECANUM = "mecanum"

class Phase(str, Enum):
    OPENING = "opening"
    INTERACTION = "interaction"
    FINISHED = "finished"

_ACTION_MAP = {int(k): str(v) for k, v in SHARED_CONFIG["actions"].items()}
Action = IntEnum("Action", {name: idx for idx, name in _ACTION_MAP.items()})
ACTION_NAMES = dict(_ACTION_MAP)
NUM_ACTIONS = int(SHARED_CONFIG["num_actions"])
if list(_ACTION_MAP) != list(range(NUM_ACTIONS)) or [a.name for a in Action] != list(_ACTION_MAP.values()):
    raise ValueError("shared/config.yaml actions must be contiguous and match num_actions")

PROFILE = SHARED_CONFIG["profile"]
FIELD = SHARED_CONFIG["field"]
ROBOT = SHARED_CONFIG["robot"]
OBJECTS = SHARED_CONFIG["objects"]
SCORING = SHARED_CONFIG["scoring"]

FIELD_WIDTH = float(FIELD["width"])
FIELD_HEIGHT = float(FIELD["height"])
AUTONOMOUS_LINES = tuple(float(v) for v in FIELD["autonomous_lines"])
LOAD_ZONE_DEPTH = float(FIELD["load_zone_depth"])
LOAD_ZONE_SPAN = float(FIELD.get("load_zone_span", FIELD["load_zone_depth"]))
MATCH_DURATION = float(PROFILE["match_duration"])
OPENING_DURATION = float(PROFILE["opening_duration"])
DT = float(PROFILE["dt"])
DECISION_INTERVAL = float(PROFILE["decision_interval"])
ROBOT_RADIUS = float(ROBOT["radius"])
MAX_FORWARD_SPEED = float(ROBOT["max_forward_speed"])
MAX_LATERAL_SPEED = float(ROBOT["max_lateral_speed"])
MAX_YAW_RATE = float(ROBOT["max_yaw_rate"])
LINEAR_ACCEL = float(ROBOT["linear_accel"])
LINEAR_DECEL = float(ROBOT["linear_decel"])
YAW_ACCEL = float(ROBOT["yaw_accel"])
INTERACTION_RANGE = float(ROBOT["interaction_range"])
GOAL_CAPACITY = int(OBJECTS["goal_capacity"])
STATE_DIM = int(SHARED_CONFIG["state"]["per_robot_dim"])
TEAM_STATE_DIM = STATE_DIM * 2
ALLIANCE_HALF_POINTS = int(SCORING["alliance_half"])
OWNED_YELLOW_POINTS = int(SCORING["owned_yellow_half"])
MIDFIELD_ROBOT_POINTS = int(SCORING["midfield_robot"])
OPENING_BONUS_POINTS = int(SCORING["opening_bonus"])
