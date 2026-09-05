"""Override field state, symbolic scoring, interactions, and collision physics."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable
import numpy as np
from .config import (
    Alliance, ChassisType, Phase, ALLIANCE_HALF_POINTS, AUTONOMOUS_LINES,
    CONTESTED, DT, GOAL_CAPACITY, INTERACTION_RANGE, LOAD_ZONE_DEPTH, LOAD_ZONE_SPAN,
    MATCH_DURATION,
    MIDFIELD_ROBOT_POINTS, OBJECTS, OPENING_BONUS_POINTS, OPENING_DURATION,
    OWNED_YELLOW_POINTS,
)
from .robot import Robot

YELLOW = "yellow"

# Official Override coordinates are published in millimeters from field center.
# The renderer maps them into this prototype's nominal 144-inch coordinate frame.
def gps_to_field(x_mm: float, y_mm: float) -> tuple[float, float]:
    return 72.0 + x_mm / 25.4, 72.0 + y_mm / 25.4

# id, GPS x/y (mm), protected alliance, quadrant, physical type
# Quadrants follow their wall Toggle: north=0, east=1, south=2, west=3.
OFFICIAL_GOAL_SPECS = (
    (0, 0, 0, None, -1, "neutral_tall"),
    (1, -600, 1200, None, 0, "neutral_short"),
    (2, -1200, 600, None, 3, "neutral_short"),
    (3, 1200, -600, None, 1, "neutral_short"),
    (4, 600, -1200, None, 2, "neutral_short"),
    (5, -1200, -600, Alliance.RED, 3, "alliance"),
    (6, -600, -1200, Alliance.RED, 2, "alliance"),
    (7, 600, 1200, Alliance.BLUE, 0, "alliance"),
    (8, 1200, 600, Alliance.BLUE, 1, "alliance"),
)
OFFICIAL_TOGGLE_SPECS = (
    (0, 0, 1780, 0, "N"), (1, 1780, 0, 1, "E"),
    (2, 0, -1780, 2, "S"), (3, -1780, 0, 3, "W"),
)
OFFICIAL_LOADER_SPECS = (
    (0, Alliance.RED, -1740, 1490), (1, Alliance.RED, -1740, -1490),
    (2, Alliance.BLUE, 1740, 1490), (3, Alliance.BLUE, 1740, -1490),
)
# Published Pin/Cup GPS anchors; multiple objects may share one cluster anchor.
OFFICIAL_OBJECT_ANCHORS = tuple(gps_to_field(x, y) for x, y in (
    (-600,1745),(600,1745),(-1200,1200),(1200,1200),(-1745,600),
    (-600,600),(0,600),(600,600),(1745,600),(-600,0),(600,0),
    (-1745,-600),(-600,-600),(0,-600),(600,-600),(1745,-600),
    (-1200,-1200),(1200,-1200),(-600,-1745),(600,-1745),
))
MIDFIELD_DIAMOND = tuple(gps_to_field(x, y) for x, y in ((0,-600),(600,0),(0,600),(-600,0)))
# Corner → Midfield-diamond EDGE midpoints (not vertices). Forms the large
# diagonal X interrupted by the diamond; NW/SE is the Autonomous Line.
_DIAMOND_EDGE_MIDS = {
    "sw": (60.19, 60.19),
    "se": (83.81, 60.19),
    "ne": (83.81, 83.81),
    "nw": (60.19, 83.81),
}
AUTONOMOUS_TAPE_SEGMENTS = (
    ((0.0, 0.0), _DIAMOND_EDGE_MIDS["sw"]),
    ((144.0, 144.0), _DIAMOND_EDGE_MIDS["ne"]),
    ((0.0, 144.0), _DIAMOND_EDGE_MIDS["nw"]),
    ((144.0, 0.0), _DIAMOND_EDGE_MIDS["se"]),
)
# NW/SE segments are the paired Autonomous Line; SW/NE is the other diagonal.
AUTONOMOUS_LINE_SEGMENTS = AUTONOMOUS_TAPE_SEGMENTS[2:]
# Pin clusters sit on the eight diagonal anchors; Cups on wall + diamond verts.
# Placement groups mirror Appendix A / Skills figure topology under VEX U counts.
_OUTER_DIAGONAL_GPS = (
    (-1200, 1200), (1200, 1200), (-1200, -1200), (1200, -1200),
)
_INNER_DIAGONAL_GPS = (
    (-600, 600), (600, 600), (-600, -600), (600, -600),
)
_WALL_ANCHOR_GPS = (
    (-600, 1745), (600, 1745), (-1745, 600), (1745, 600),
    (-1745, -600), (1745, -600), (-600, -1745), (600, -1745),
)
_DIAMOND_VERT_GPS = ((0, 600), (0, -600), (-600, 0), (600, 0))
PIN_CLUSTER_ANCHORS = tuple(gps_to_field(x, y) for x, y in _OUTER_DIAGONAL_GPS + _INNER_DIAGONAL_GPS)
CUP_CLUSTER_ANCHORS = tuple(gps_to_field(x, y) for x, y in _WALL_ANCHOR_GPS + _DIAMOND_VERT_GPS)
WALL_CLUSTER_ANCHORS = tuple(gps_to_field(x, y) for x, y in _WALL_ANCHOR_GPS)
DIAMOND_VERT_ANCHORS = tuple(gps_to_field(x, y) for x, y in _DIAMOND_VERT_GPS)
OUTER_DIAGONAL_ANCHORS = tuple(gps_to_field(x, y) for x, y in _OUTER_DIAGONAL_GPS)
INNER_DIAGONAL_ANCHORS = tuple(gps_to_field(x, y) for x, y in _INNER_DIAGONAL_GPS)

@dataclass
class Pin:
    object_id: int
    halves: tuple[str, str]
    x: float | None
    y: float | None
    source: str = "field"
    placed_goal: int | None = None

@dataclass
class Cup:
    object_id: int
    x: float | None
    y: float | None
    source: str = "field"
    placed_goal: int | None = None

@dataclass(frozen=True)
class StackEntry:
    kind: str
    object_id: int
    nested_on: int | None = None

@dataclass
class Goal:
    goal_id: int
    x: float
    y: float
    protected_by: Alliance | None
    quadrant: int
    kind: str = "neutral_short"
    stack: list[StackEntry] = field(default_factory=list)

    def placed_pin_halves(self, pins: dict[int, Pin]) -> list[str]:
        return [half for entry in self.stack if entry.kind == "pin" for half in pins[entry.object_id].halves]

    def visible_pin_halves(self, pins: dict[int, Pin]) -> tuple[str, ...]:
        """Return the two halves of the highest Pin; Cups retain symbolic nesting."""
        for entry in reversed(self.stack):
            if entry.kind == "pin":
                return pins[entry.object_id].halves
        return ()

@dataclass
class Toggle:
    toggle_id: int
    x: float
    y: float
    quadrant: int
    compass: str
    owner: Alliance | None = None

@dataclass
class Loader:
    loader_id: int
    alliance: Alliance
    x: float
    y: float

class OverrideField:
    """Complete deterministic four-robot Override match state."""
    goal_radius = 5.0
    loader_radius = 5.0

    def __init__(self, chassis: str | ChassisType = ChassisType.TANK, seed: int | None = None):
        self.chassis = ChassisType(chassis)
        self.rng = np.random.default_rng(seed)
        self.elapsed = 0.0
        self.phase = Phase.OPENING
        self.opening_bonus: Alliance | None = None
        self.opening_raw_scores = {Alliance.BLUE: 0, Alliance.RED: 0}
        self.awp = {Alliance.BLUE: False, Alliance.RED: False}
        self.telemetry: list[str] = []
        self.pins: dict[int, Pin] = {}
        self.cups: dict[int, Cup] = {}
        self.robots: list[Robot] = []
        self.goals: list[Goal] = []
        self.toggles: list[Toggle] = []
        self.loaders = [
            Loader(loader_id, alliance, *gps_to_field(x_mm, y_mm))
            for loader_id, alliance, x_mm, y_mm in OFFICIAL_LOADER_SPECS
        ]
        amount_p = int(OBJECTS["match_load_pins_per_alliance"])
        amount_c = int(OBJECTS["match_load_cups_per_alliance"])
        yellow_p = int(OBJECTS["match_load_yellow_pins_per_alliance"])
        self.match_loads = {
            Alliance.BLUE: {"pin": amount_p, "yellow_pin": yellow_p, "cup": amount_c},
            Alliance.RED: {"pin": amount_p, "yellow_pin": yellow_p, "cup": amount_c},
        }
        self._next_id = 0
        self._setup()

        # TENURE contested mechanic (plan Section 4). OFF by default; when enabled,
        # Toggle ownership becomes dwell-based (Channel 1, territory reversal).
        self.contested = dict(CONTESTED)
        self.contested_enabled = bool(self.contested["enabled"])
        self._toggle_claim_timer: dict[int, float] = {t.toggle_id: 0.0 for t in self.toggles}
        self._toggle_pending: dict[int, Alliance | None] = {t.toggle_id: None for t in self.toggles}
        self._descore_timer: dict[int, float] = {g.goal_id: 0.0 for g in self.goals}
        self.held_value: dict[Alliance, float] = {Alliance.BLUE: 0.0, Alliance.RED: 0.0}
        self.reversal_events: list[dict] = []

    @property
    def time_remaining(self) -> float:
        return max(0.0, MATCH_DURATION - self.elapsed)

    @property
    def done(self) -> bool:
        return self.phase is Phase.FINISHED

    def _new_pin(self, halves: tuple[str, str], x: float | None, y: float | None, source: str = "field") -> int:
        oid = self._next_id; self._next_id += 1
        self.pins[oid] = Pin(oid, halves, x, y, source)
        return oid

    def _new_cup(self, x: float | None, y: float | None, source: str = "field") -> int:
        oid = self._next_id; self._next_id += 1
        self.cups[oid] = Cup(oid, x, y, source)
        return oid

    def _setup(self) -> None:
        # VEX U alliance sides are southwest (red) and northeast (blue).
        positions = [(102, 132, -math.pi/2), (132, 102, math.pi),
                     (42, 12, math.pi/2), (12, 42, 0.0)]
        for rid, (x, y, heading) in enumerate(positions):
            alliance = Alliance.BLUE if rid < 2 else Alliance.RED
            robot = Robot(rid, alliance, self.chassis, float(x), float(y), heading)
            robot.held_pin = robot.preload_pin = self._new_pin(
                (alliance.value, YELLOW), None, None, "preload"
            )
            self.robots.append(robot)

        for gid, x_mm, y_mm, protected, quadrant, kind in OFFICIAL_GOAL_SPECS:
            x, y = gps_to_field(x_mm, y_mm)
            self.goals.append(Goal(gid, x, y, protected, quadrant, kind))
        self.toggles = [
            Toggle(tid, *gps_to_field(x_mm, y_mm), quadrant, compass)
            for tid, x_mm, y_mm, quadrant, compass in OFFICIAL_TOGGLE_SPECS
        ]

        # VEX U starts only the tall Midfield Goal with a yellow/yellow Pin.
        center_pin = self._new_pin((YELLOW, YELLOW), None, None, "vexu_center_goal")
        self.pins[center_pin].placed_goal = 0
        self.goals[0].stack.append(StackEntry("pin", center_pin))

        # VEX U field objects mapped onto Appendix A anchors:
        # 4 outer diagonal clusters ×4 Pins, 4 inner ×1, 4 diamond verts ×1,
        # 8 wall yellow Pins ×1 (=32); Cups: 8 walls ×3 + 4 diamond ×3 (=36).
        assert int(OBJECTS["loose_pins"]) == 32 and int(OBJECTS["loose_cups"]) == 36
        colored = ([('red', 'blue')] * 4 + [('red', YELLOW)] * 8 + [('blue', YELLOW)] * 8)
        yellows = [(YELLOW, YELLOW)] * 12
        pin_queue = list(colored) + list(yellows)

        def _take(halves=None):
            if halves is None:
                return pin_queue.pop(0)
            for index, candidate in enumerate(pin_queue):
                if candidate == halves:
                    return pin_queue.pop(index)
            return pin_queue.pop(0)

        for anchor in OUTER_DIAGONAL_ANCHORS:
            for slot in range(4):
                angle = slot * math.pi / 2
                self._new_pin(_take(), anchor[0] + math.cos(angle) * 1.35,
                              anchor[1] + math.sin(angle) * 1.35)
        for anchor in INNER_DIAGONAL_ANCHORS:
            self._new_pin(_take(), anchor[0], anchor[1])
        for anchor in DIAMOND_VERT_ANCHORS:
            self._new_pin(_take(), anchor[0], anchor[1])
        for anchor in WALL_CLUSTER_ANCHORS:
            self._new_pin(_take((YELLOW, YELLOW)), anchor[0], anchor[1])
        assert not pin_queue

        for anchor in WALL_CLUSTER_ANCHORS:
            for slot in range(3):
                angle = slot * (2 * math.pi / 3) + math.pi / 6
                self._new_cup(anchor[0] + math.cos(angle) * 1.55,
                              anchor[1] + math.sin(angle) * 1.55)
        for anchor in DIAMOND_VERT_ANCHORS:
            for slot in range(3):
                angle = slot * (2 * math.pi / 3) + math.pi / 2
                self._new_cup(anchor[0] + math.cos(angle) * 2.0,
                              anchor[1] + math.sin(angle) * 2.0)


    def objects_on_field(self, kind: str) -> Iterable[Pin | Cup]:
        values = self.pins.values() if kind == "pin" else self.cups.values()
        return (obj for obj in values if obj.x is not None and obj.placed_goal is None)

    def nearest_object(self, robot: Robot, kind: str, alliance: Alliance | None = None) -> Pin | Cup | None:
        """Nearest loose object. For Pins with an ``alliance`` hint, prefer pins that
        carry no opponent-colored half (own-color + yellow only), so a team does not
        hand the opponent free points; falls back to any Pin when none remain."""
        pool = list(self.objects_on_field(kind))
        if kind == "pin" and alliance is not None:
            opponent = alliance.opponent.value
            friendly = [obj for obj in pool if opponent not in obj.halves]
            pool = friendly or pool
        return min(pool, key=lambda o: math.hypot(float(o.x)-robot.x, float(o.y)-robot.y), default=None)

    def collect(self, robot: Robot, kind: str, alliance: Alliance | None = None) -> bool:
        if (kind == "pin" and robot.held_pin is not None) or (kind == "cup" and robot.held_cup is not None):
            robot.blocked_interactions += 1; return False
        obj = self.nearest_object(robot, kind, alliance)
        if obj is None or math.hypot(float(obj.x)-robot.x, float(obj.y)-robot.y) > INTERACTION_RANGE:
            return False
        obj.x = obj.y = None
        if kind == "pin": robot.held_pin = obj.object_id
        else: robot.held_cup = obj.object_id
        return True

    def alliance_loaders(self, alliance: Alliance) -> list[Loader]:
        return [loader for loader in self.loaders if loader.alliance is alliance]

    def nearest_loader(self, robot: Robot) -> Loader:
        return min(self.alliance_loaders(robot.alliance),
                   key=lambda loader: math.hypot(robot.x-loader.x, robot.y-loader.y))

    def in_load_zone(self, robot: Robot) -> bool:
        # Rectangular Load Zone: shallow from the alliance wall, longer along N/S.
        on_alliance_side = (robot.x <= LOAD_ZONE_DEPTH if robot.alliance is Alliance.RED
                            else robot.x >= 144.0 - LOAD_ZONE_DEPTH)
        in_corner = robot.y <= LOAD_ZONE_SPAN or robot.y >= 144.0 - LOAD_ZONE_SPAN
        return on_alliance_side and in_corner

    def use_loader(self, robot: Robot) -> bool:
        loader = self.nearest_loader(robot)
        if not self.in_load_zone(robot) or math.hypot(robot.x-loader.x, robot.y-loader.y) > INTERACTION_RANGE+self.loader_radius: return False
        inv = self.match_loads[robot.alliance]
        if robot.held_pin is None and inv["pin"] > 0:
            yellow_only = inv["pin"] <= inv["yellow_pin"]
            halves = (YELLOW, YELLOW) if yellow_only else (robot.alliance.value, YELLOW)
            robot.held_pin = self._new_pin(halves, None, None, "match_load")
            inv["pin"] -= 1
            if yellow_only: inv["yellow_pin"] -= 1
            return True
        if robot.held_cup is None and inv["cup"] > 0:
            robot.held_cup = self._new_cup(None, None, "match_load"); inv["cup"] -= 1; return True
        robot.blocked_interactions += 1; return False

    def place(self, robot: Robot, goal: Goal) -> bool:
        if math.hypot(robot.x-goal.x, robot.y-goal.y) > INTERACTION_RANGE+self.goal_radius: return False
        if len(goal.stack) >= GOAL_CAPACITY: robot.blocked_interactions += 1; return False
        if robot.held_pin is not None:
            oid = robot.held_pin; robot.held_pin = None; self.pins[oid].placed_goal = goal.goal_id
            goal.stack.append(StackEntry("pin", oid)); return True
        if robot.held_cup is not None:
            oid = robot.held_cup; robot.held_cup = None; self.cups[oid].placed_goal = goal.goal_id
            nested = goal.stack[-1].object_id if goal.stack and goal.stack[-1].kind == "pin" else None
            goal.stack.append(StackEntry("cup", oid, nested)); return True
        return False

    def remove_own_pin(self, robot: Robot, goal: Goal) -> bool:
        if robot.held_pin is not None or math.hypot(robot.x-goal.x, robot.y-goal.y) > INTERACTION_RANGE+self.goal_radius: return False
        if goal.protected_by is robot.alliance.opponent:
            self.telemetry.append(f"protected_goal_block:r{robot.robot_id}:g{goal.goal_id}"); robot.blocked_interactions += 1; return False
        if not goal.stack or goal.stack[-1].kind != "pin": return False
        entry = goal.stack[-1]; pin = self.pins[entry.object_id]
        if pin.halves != (robot.alliance.value, robot.alliance.value):
            self.telemetry.append(f"neutral_removal_block:r{robot.robot_id}:g{goal.goal_id}"); robot.blocked_interactions += 1; return False
        goal.stack.pop(); pin.placed_goal = None; robot.held_pin = pin.object_id; return True

    def can_descore(self, robot: Robot, goal: Goal) -> bool:
        """Whether ``robot`` may remove the opponent's top Pin from ``goal``.

        You can descore the enemy's top Pin from any Goal *except their own protected
        Alliance Goal*. The top entry must be a Pin carrying the opponent's colour.
        """
        if goal.protected_by is robot.alliance.opponent:
            return False
        if not goal.stack or goal.stack[-1].kind != "pin":
            return False
        pin = self.pins[goal.stack[-1].object_id]
        return robot.alliance.opponent.value in pin.halves

    def descore_pin(self, robot: Robot, goal: Goal) -> bool:
        """Remove the opponent's top Pin from ``goal`` and drop it loose on the field
        just outside the Goal (so the credit is lost and the Pin returns to play).

        When the contested mechanic is on, descoring is dwell-based and contested
        (Channel 2) — handled by ``update_contested`` — so this instant path is
        disabled, exactly like the instant Toggle claim."""
        if self.contested_enabled:
            return False
        if math.hypot(robot.x-goal.x, robot.y-goal.y) > INTERACTION_RANGE+self.goal_radius:
            return False
        if goal.protected_by is robot.alliance.opponent:
            self.telemetry.append(f"protected_descore_block:r{robot.robot_id}:g{goal.goal_id}")
            robot.blocked_interactions += 1; return False
        if not self.can_descore(robot, goal):
            return False
        entry = goal.stack.pop(); pin = self.pins[entry.object_id]; pin.placed_goal = None
        angle = float(self.rng.uniform(0.0, 2*math.pi))
        pin.x = goal.x + math.cos(angle)*(self.goal_radius+4.0)
        pin.y = goal.y + math.sin(angle)*(self.goal_radius+4.0)
        robot.telemetry.append("descore")
        self.telemetry.append(f"descore:r{robot.robot_id}:g{goal.goal_id}")
        return True

    def claim_toggle(self, robot: Robot, toggle: Toggle) -> bool:
        # When the contested mechanic is on, ownership is dwell-based (Channel 1);
        # the instantaneous claim is disabled and update_contested() flips owners.
        if self.contested_enabled:
            return False
        if math.hypot(robot.x-toggle.x, robot.y-toggle.y) > INTERACTION_RANGE: return False
        toggle.owner = robot.alliance; return True

    def _toggle_challenger(self, near: dict, owner: Alliance | None, beta: float, mode: str) -> Alliance | None:
        """Which alliance (if any) is currently winning the dwell contest for a Toggle."""
        blue, red = near[Alliance.BLUE], near[Alliance.RED]
        if owner is None:                                   # unclaimed: majority (or sole, if suppress) takes it
            if mode == "suppress":
                if blue >= 1 and red == 0: return Alliance.BLUE
                if red >= 1 and blue == 0: return Alliance.RED
                return None
            if blue > red and blue >= 1: return Alliance.BLUE
            if red > blue and red >= 1: return Alliance.RED
            return None
        attackers, defenders = near[owner.opponent], near[owner]
        if attackers < 1:
            return None
        if mode == "none":
            return owner.opponent                           # defenders have no effect
        if mode == "suppress":
            return owner.opponent if defenders == 0 else None
        return owner.opponent if attackers > beta * defenders else None   # majority (default)

    def _toggle_quadrant_value(self, toggle: Toggle) -> int:
        """Owned-yellow value in this Toggle's quadrant — the credit that flips hands."""
        return sum(
            g.placed_pin_halves(self.pins).count(YELLOW) * OWNED_YELLOW_POINTS
            for g in self.goals if g.quadrant == toggle.quadrant
        )

    def _pin_owner(self, pin: Pin) -> Alliance | None:
        """The alliance whose colour is on a Pin (the side that would defend it)."""
        has_blue = Alliance.BLUE.value in pin.halves
        has_red = Alliance.RED.value in pin.halves
        if has_blue and not has_red:
            return Alliance.BLUE
        if has_red and not has_blue:
            return Alliance.RED
        return None                                              # neutral (yellow) or mixed

    def _update_goal_descore(self, dt: float, beta: float, mode: str) -> None:
        """Channel 2: dwell-based, contested descoring. The opponent must linger at
        your Goal to remove your top Pin, and a defender of your alliance present
        cancels it (same majority rule as Toggles). Alliance Goals are protected."""
        dwell = float(self.contested["pin_removal_dwell"]) * float(self.contested["alpha_scale"])
        for goal in self.goals:
            gid = goal.goal_id
            owner = None
            if goal.stack and goal.stack[-1].kind == "pin":
                owner = self._pin_owner(self.pins[goal.stack[-1].object_id])
            if owner is None or goal.protected_by is owner:      # nothing descorable / protected goal
                self._descore_timer[gid] = 0.0
                continue
            attacker = owner.opponent
            near = {owner: 0, attacker: 0}
            for robot in self.robots:
                if math.hypot(robot.x - goal.x, robot.y - goal.y) <= INTERACTION_RANGE + self.goal_radius:
                    near[robot.alliance] += 1
            if not self._descore_contest(near[attacker], near[owner], beta, mode):
                self._descore_timer[gid] = 0.0                   # defended (or no attacker): reset
                continue
            self._descore_timer[gid] += dt
            if self._descore_timer[gid] >= dwell:
                entry = goal.stack.pop(); pin = self.pins[entry.object_id]; pin.placed_goal = None
                angle = float(self.rng.uniform(0.0, 2 * math.pi))
                pin.x = goal.x + math.cos(angle) * (self.goal_radius + 4.0)
                pin.y = goal.y + math.sin(angle) * (self.goal_radius + 4.0)
                self.reversal_events.append({
                    "t": self.elapsed, "channel": "object", "site_id": gid,
                    "from_alliance": owner.value, "to_alliance": None, "value_delta": ALLIANCE_HALF_POINTS,
                })
                self._descore_timer[gid] = 0.0

    @staticmethod
    def _descore_contest(attackers: int, defenders: int, beta: float, mode: str) -> bool:
        if attackers < 1:
            return False
        if mode == "none":
            return True
        if mode == "suppress":
            return defenders == 0
        return attackers > beta * defenders                      # majority (default)

    def update_contested(self, dt: float) -> None:
        """Advance dwell-based Toggle (Channel 1) and Goal descore (Channel 2)
        contests, and integrate held value."""
        for alliance in (Alliance.BLUE, Alliance.RED):
            self.held_value[alliance] += self.raw_score(alliance) * dt
        scale = float(self.contested["alpha_scale"])
        claim_dwell = float(self.contested["toggle_claim_dwell"]) * scale
        beta = float(self.contested["beta"])
        mode = str(self.contested["contest_mode"])
        for toggle in self.toggles:
            near = {Alliance.BLUE: 0, Alliance.RED: 0}
            for robot in self.robots:
                if math.hypot(robot.x - toggle.x, robot.y - toggle.y) <= INTERACTION_RANGE:
                    near[robot.alliance] += 1
            tid = toggle.toggle_id
            challenger = self._toggle_challenger(near, toggle.owner, beta, mode)
            if challenger is None or challenger is toggle.owner:
                self._toggle_claim_timer[tid] = 0.0
                self._toggle_pending[tid] = None
                continue
            if self._toggle_pending.get(tid) is not challenger:         # reset dwell if the challenger changed
                self._toggle_pending[tid] = challenger
                self._toggle_claim_timer[tid] = 0.0
            self._toggle_claim_timer[tid] += dt
            if self._toggle_claim_timer[tid] >= claim_dwell:
                old = toggle.owner
                delta = self._toggle_quadrant_value(toggle)
                toggle.owner = challenger
                self.reversal_events.append({
                    "t": self.elapsed, "channel": "territory", "site_id": tid,
                    "from_alliance": None if old is None else old.value,
                    "to_alliance": challenger.value, "value_delta": delta,
                })
                self._toggle_claim_timer[tid] = 0.0
                self._toggle_pending[tid] = None

        if self.contested.get("enable_opponent_removal", True):
            self._update_goal_descore(dt, beta, mode)

    def toggle_count(self, alliance: Alliance) -> int:
        return sum(t.owner is alliance for t in self.toggles)

    def robot_in_midfield(self, robot: Robot) -> bool:
        # Center-point proxy for the official diamond's infinite vertical volume.
        radius = 600.0/25.4
        return abs(robot.x-72.0) + abs(robot.y-72.0) <= radius

    def midfield_count(self, alliance: Alliance) -> int:
        return sum(r.alliance is alliance and self.robot_in_midfield(r) for r in self.robots)

    def midfield_owner(self) -> Alliance | None:
        blue, red = self.midfield_count(Alliance.BLUE), self.midfield_count(Alliance.RED)
        return Alliance.BLUE if blue > red else Alliance.RED if red > blue else None

    def goal_owner(self, goal: Goal) -> Alliance | None:
        return self.midfield_owner() if goal.quadrant < 0 else self.toggles[goal.quadrant].owner

    def goal_status_owner(self, goal: Goal) -> Alliance | None:
        return self.goal_owner(goal) if YELLOW in goal.placed_pin_halves(self.pins) else None

    def raw_score(self, alliance: Alliance, include_midfield: bool = True) -> int:
        score = 0
        for goal in self.goals:
            halves = goal.placed_pin_halves(self.pins)
            score += halves.count(alliance.value) * ALLIANCE_HALF_POINTS
            if self.goal_owner(goal) is alliance:
                score += halves.count(YELLOW) * OWNED_YELLOW_POINTS
        if include_midfield:
            score += self.midfield_count(alliance) * MIDFIELD_ROBOT_POINTS
        return score

    def score(self, alliance: Alliance) -> int:
        return self.raw_score(alliance) + (OPENING_BONUS_POINTS if self.opening_bonus is alliance else 0)

    def _finish_opening(self) -> None:
        if self.phase is not Phase.OPENING: return
        for alliance in Alliance: self.opening_raw_scores[alliance] = self.raw_score(alliance)
        blue, red = self.opening_raw_scores[Alliance.BLUE], self.opening_raw_scores[Alliance.RED]
        self.opening_bonus = Alliance.BLUE if blue > red else Alliance.RED if red > blue else None
        for alliance in Alliance:
            placed = any(alliance.value in g.placed_pin_halves(self.pins) for g in self.goals)
            used = all(r.held_pin != r.preload_pin for r in self.robots if r.alliance is alliance)
            self.awp[alliance] = placed and self.toggle_count(alliance) >= 2 and used
        self.phase = Phase.INTERACTION
        self.telemetry.append(f"opening_end:bonus={self.opening_bonus}:awp={self.awp}")

    def advance_clock(self, dt: float) -> None:
        previous = self.elapsed
        advanced = self.elapsed + dt
        self.elapsed = MATCH_DURATION if advanced >= MATCH_DURATION-1e-9 else advanced
        if previous < OPENING_DURATION <= self.elapsed+1e-9: self._finish_opening()
        if self.elapsed >= MATCH_DURATION: self.phase = Phase.FINISHED

    def static_obstacles(self) -> list[tuple[float, float, float]]:
        """(x, y, radius) of every static obstacle — Goals and Loaders — for the
        navigation controller's forward look-ahead avoidance."""
        return ([(g.x, g.y, self.goal_radius) for g in self.goals]
                + [(l.x, l.y, self.loader_radius) for l in self.loaders])

    def _resolve_static(self, robot: Robot, old: tuple[float, float]) -> None:
        obstacles = [(g.x, g.y, self.goal_radius) for g in self.goals] + [(l.x, l.y, self.loader_radius) for l in self.loaders]
        for ox, oy, radius in obstacles:
            dx, dy = robot.x-ox, robot.y-oy; minimum = robot.radius+radius; dist = math.hypot(dx, dy)
            if dist < minimum:
                if dist < 1e-9: robot.x, robot.y = ox+minimum, oy
                else: robot.x, robot.y = ox+dx/dist*minimum, oy+dy/dist*minimum
                robot.forward_velocity = robot.lateral_velocity = 0.0; robot.collisions += 1

    def _project_robot_constraints(self, robot: Robot) -> None:
        # Static projections can touch a wall/opening line (and vice versa), so
        # use two deterministic passes to satisfy the combined hard constraints.
        for _ in range(2):
            self._constrain_robot(robot)
            self._resolve_static(robot, (robot.x, robot.y))
        self._constrain_robot(robot)

    def _constrain_robot(self, robot: Robot) -> None:
        robot.clamp_to_walls()
        if self.phase is not Phase.OPENING:
            return
        low, high = AUTONOMOUS_LINES
        diagonal = robot.x + robot.y
        if robot.alliance is Alliance.BLUE and diagonal < low:
            correction = (low-diagonal)/2
            robot.x += correction; robot.y += correction
            robot.telemetry.append("opening_line_block")
        elif robot.alliance is Alliance.RED and diagonal > high:
            correction = (diagonal-high)/2
            robot.x -= correction; robot.y -= correction
            robot.telemetry.append("opening_line_block")
        robot.clamp_to_walls()

    def _resolve_robot_collisions(self) -> None:
        for i, first in enumerate(self.robots):
            for second in self.robots[i+1:]:
                dx, dy = second.x-first.x, second.y-first.y
                dist = math.hypot(dx, dy); minimum = first.radius+second.radius
                if dist >= minimum-1e-9: continue
                if dist < 1e-9: nx, ny = 1.0, 0.0
                else: nx, ny = dx/dist, dy/dist
                overlap = minimum-dist
                first.x -= nx*overlap/2; first.y -= ny*overlap/2
                second.x += nx*overlap/2; second.y += ny*overlap/2
                self._project_robot_constraints(first); self._project_robot_constraints(second)
                # If one robot hit a wall/autonomous line, transfer the residual
                # separation to the other robot instead of leaving an overlap.
                for _ in range(2):
                    dx, dy = second.x-first.x, second.y-first.y; dist = math.hypot(dx, dy)
                    if dist >= minimum-1e-9: break
                    if dist >= 1e-9: nx, ny = dx/dist, dy/dist
                    residual = minimum-dist
                    first.x -= nx*residual; first.y -= ny*residual; self._project_robot_constraints(first)
                    dx, dy = second.x-first.x, second.y-first.y; dist = math.hypot(dx, dy)
                    if dist < minimum-1e-9:
                        if dist >= 1e-9: nx, ny = dx/dist, dy/dist
                        residual = minimum-dist
                        second.x += nx*residual; second.y += ny*residual; self._project_robot_constraints(second)
                first.forward_velocity = first.lateral_velocity = second.forward_velocity = second.lateral_velocity = 0.0
                first.collisions += 1; second.collisions += 1

    def physics_tick(self, commands: dict[int, tuple[float, float, float]], dt: float = DT) -> None:
        for robot in self.robots:
            robot.command(*commands.get(robot.robot_id, (0, 0, 0)), dt)
            self._project_robot_constraints(robot)
        # Repeated deterministic projections resolve pair/static contact chains
        # while preserving walls and autonomous lines.
        for _ in range(3):
            self._resolve_robot_collisions()
            for robot in self.robots:
                self._project_robot_constraints(robot)
        if self.contested_enabled:
            self.update_contested(dt)
        self.advance_clock(dt)
