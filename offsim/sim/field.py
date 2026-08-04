"""Override field state, symbolic scoring, interactions, and collision physics."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Iterable
import numpy as np
from .config import (
    Alliance, ChassisType, Phase, ALLIANCE_HALF_POINTS, AUTONOMOUS_LINES,
    DT, GOAL_CAPACITY, INTERACTION_RANGE, LOAD_ZONE_DEPTH, LOAD_ZONE_SPAN,
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

    def nearest_object(self, robot: Robot, kind: str) -> Pin | Cup | None:
        return min(self.objects_on_field(kind), key=lambda o: math.hypot(float(o.x)-robot.x, float(o.y)-robot.y), default=None)

    def collect(self, robot: Robot, kind: str) -> bool:
        if (kind == "pin" and robot.held_pin is not None) or (kind == "cup" and robot.held_cup is not None):
            robot.blocked_interactions += 1; return False
        obj = self.nearest_object(robot, kind)
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

    def claim_toggle(self, robot: Robot, toggle: Toggle) -> bool:
        if math.hypot(robot.x-toggle.x, robot.y-toggle.y) > INTERACTION_RANGE: return False
        toggle.owner = robot.alliance; return True

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
        self.advance_clock(dt)
