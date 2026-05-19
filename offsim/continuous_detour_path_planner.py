"""
Continuous 2D waypoint planner for a robot with known field obstacles and moving opponent robots.

This is NOT A*. It plans directly in continuous space:
1. Start with the desired route through points of interest.
2. Check each straight segment for collision with inflated static obstacles.
3. Predict moving-opponent collisions using current opponent position + velocity.
4. Insert detour waypoints around the first blocking obstacle.
5. Repeat until the route is clear or the iteration limit is reached.

Recommended use on a real robot:
- Re-run this planner every control cycle or every few cycles.
- Update opponent positions from camera / tracking / sensor fusion.
- Send the resulting waypoints to a local controller, such as pure pursuit or PID heading control.

Coordinate units can be anything, as long as you are consistent.
For example: meters, inches, field tiles, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, inf
from typing import Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Basic geometry
# ============================================================

EPS = 1e-9


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point":
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Point":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Point":
        if abs(scalar) < EPS:
            raise ZeroDivisionError("Cannot divide a Point by zero.")
        return Point(self.x / scalar, self.y / scalar)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def norm(a: Point) -> float:
    return sqrt(dot(a, a))


def distance(a: Point, b: Point) -> float:
    return norm(a - b)


def normalize(v: Point) -> Point:
    mag = norm(v)
    if mag < EPS:
        return Point(0.0, 0.0)
    return v / mag


def perp_left(v: Point) -> Point:
    return Point(-v.y, v.x)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def closest_point_on_segment(a: Point, b: Point, p: Point) -> Tuple[Point, float]:
    """
    Returns the closest point on segment AB to point P, and the segment fraction t.
    t = 0 means A, t = 1 means B.
    """
    ab = b - a
    ab_len2 = dot(ab, ab)
    if ab_len2 < EPS:
        return a, 0.0

    t = dot(p - a, ab) / ab_len2
    t = clamp(t, 0.0, 1.0)
    return a + ab * t, t


def segment_circle_distance(a: Point, b: Point, center: Point) -> Tuple[float, float, Point]:
    """
    Minimum distance from segment AB to a circle center.
    Returns: distance, segment_fraction, closest_point.
    """
    closest, t = closest_point_on_segment(a, b, center)
    return distance(closest, center), t, closest


def path_length(points: Sequence[Point]) -> float:
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))


# ============================================================
# Obstacles and planner config
# ============================================================

@dataclass(frozen=True)
class StaticObstacle:
    center: Point
    radius: float
    name: str = "static_obstacle"


@dataclass(frozen=True)
class MovingObstacle:
    position: Point
    velocity: Point
    radius: float
    name: str = "moving_obstacle"

    def position_at(self, time_s: float) -> Point:
        return self.position + self.velocity * time_s


@dataclass(frozen=True)
class FieldBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, p: Point) -> bool:
        return self.min_x <= p.x <= self.max_x and self.min_y <= p.y <= self.max_y

    def clamp_point(self, p: Point, margin: float = 0.0) -> Point:
        return Point(
            clamp(p.x, self.min_x + margin, self.max_x - margin),
            clamp(p.y, self.min_y + margin, self.max_y - margin),
        )


@dataclass(frozen=True)
class PlannerConfig:
    robot_radius: float = 0.25
    safety_clearance: float = 0.10
    robot_speed: float = 1.0

    # Extra clearance used when generating detour waypoints.
    # Larger value = wider, safer turns around obstacles.
    detour_margin: float = 0.25

    # How far before/after the obstacle, along the original line, to place detour points.
    # This is multiplied by the inflated obstacle radius.
    detour_advance_scale: float = 1.2

    # Limits how many times the planner can insert detours for one segment.
    max_insertions_per_segment: int = 20

    # Optional field boundary. Leave as None if you do not want boundary checking.
    field_bounds: Optional[FieldBounds] = None

    # If true, clamp detour waypoints to the field boundary.
    clamp_to_field: bool = True


@dataclass(frozen=True)
class CollisionHit:
    kind: str  # "static" or "moving"
    name: str
    center: Point
    inflated_radius: float
    distance_along_segment: float
    segment_fraction: float
    predicted_time_s: float
    reason: str


# ============================================================
# Planner
# ============================================================

class ContinuousDetourPlanner:
    def __init__(self, config: PlannerConfig):
        if config.robot_speed <= 0:
            raise ValueError("robot_speed must be positive.")
        if config.robot_radius < 0 or config.safety_clearance < 0:
            raise ValueError("robot_radius and safety_clearance must be non-negative.")
        self.cfg = config

    def plan_route(
        self,
        start: Point,
        points_of_interest: Sequence[Point],
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
    ) -> List[Point]:
        """
        Plans from start through every point of interest in order.

        Returns a list of waypoints:
        [start, ..., POI1, ..., POI2, ..., final POI]
        """
        if not points_of_interest:
            return [start]

        full_path: List[Point] = [start]
        current = start
        current_time_s = 0.0

        for goal in points_of_interest:
            segment_path = self._plan_one_start_to_goal(
                start=current,
                goal=goal,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
                start_time_s=current_time_s,
            )

            # Avoid duplicating the current point.
            full_path.extend(segment_path[1:])

            current_time_s += path_length(segment_path) / self.cfg.robot_speed
            current = goal

        return self._remove_redundant_collinear_points(full_path)

    def _plan_one_start_to_goal(
        self,
        start: Point,
        goal: Point,
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
        start_time_s: float,
    ) -> List[Point]:
        path = [start, goal]

        for _ in range(self.cfg.max_insertions_per_segment):
            collision = self._find_first_collision_in_path(
                path=path,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
                path_start_time_s=start_time_s,
            )

            if collision is None:
                break

            segment_index, segment_start_time_s, hit = collision
            a = path[segment_index]
            b = path[segment_index + 1]

            detour_points = self._make_detour_points(
                a=a,
                b=b,
                hit=hit,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
                segment_start_time_s=segment_start_time_s,
            )

            # Insert the detour between a and b.
            path = path[: segment_index + 1] + detour_points + path[segment_index + 1 :]
        else:
            print(
                "WARNING: planner reached max_insertions_per_segment. "
                "Returned path may still contain collisions."
            )

        return self._remove_redundant_collinear_points(path)

    def _find_first_collision_in_path(
        self,
        path: Sequence[Point],
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
        path_start_time_s: float,
    ) -> Optional[Tuple[int, float, CollisionHit]]:
        """
        Returns the first collision in time along the current path.
        Output: segment_index, segment_start_time_s, collision_hit
        """
        elapsed = 0.0

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            seg_len = distance(a, b)
            seg_start_time_s = path_start_time_s + elapsed / self.cfg.robot_speed

            hit = self._check_segment_collision(
                a=a,
                b=b,
                segment_start_time_s=seg_start_time_s,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
            )

            if hit is not None:
                return i, seg_start_time_s, hit

            elapsed += seg_len

        return None

    def _check_segment_collision(
        self,
        a: Point,
        b: Point,
        segment_start_time_s: float,
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
    ) -> Optional[CollisionHit]:
        """
        Checks whether segment AB collides with any inflated obstacle.
        Returns the earliest collision along the segment.
        """
        seg_len = distance(a, b)
        if seg_len < EPS:
            return None

        hits: List[CollisionHit] = []

        # Static obstacle checks.
        for obs in static_obstacles:
            inflated_radius = obs.radius + self.cfg.robot_radius + self.cfg.safety_clearance
            d, t_frac, _closest = segment_circle_distance(a, b, obs.center)

            if d <= inflated_radius:
                hits.append(
                    CollisionHit(
                        kind="static",
                        name=obs.name,
                        center=obs.center,
                        inflated_radius=inflated_radius,
                        distance_along_segment=t_frac * seg_len,
                        segment_fraction=t_frac,
                        predicted_time_s=segment_start_time_s + (t_frac * seg_len) / self.cfg.robot_speed,
                        reason=f"segment intersects inflated static obstacle {obs.name}",
                    )
                )

        # Moving obstacle checks.
        # Robot moves along AB at constant cfg.robot_speed.
        direction = normalize(b - a)
        robot_velocity = direction * self.cfg.robot_speed
        segment_duration = seg_len / self.cfg.robot_speed

        for obs in moving_obstacles:
            inflated_radius = obs.radius + self.cfg.robot_radius + self.cfg.safety_clearance

            # Opponent position at the beginning of this segment.
            opponent_start = obs.position_at(segment_start_time_s)

            # Relative motion: robot_position(t) - opponent_position(t)
            # t is local time since the robot starts this segment.
            relative_position = a - opponent_start
            relative_velocity = robot_velocity - obs.velocity
            rv2 = dot(relative_velocity, relative_velocity)

            if rv2 < EPS:
                closest_t = 0.0
            else:
                closest_t = -dot(relative_position, relative_velocity) / rv2
                closest_t = clamp(closest_t, 0.0, segment_duration)

            robot_closest = a + robot_velocity * closest_t
            opponent_closest = opponent_start + obs.velocity * closest_t
            min_dist = distance(robot_closest, opponent_closest)

            if min_dist <= inflated_radius:
                t_frac = closest_t / segment_duration if segment_duration > EPS else 0.0
                hits.append(
                    CollisionHit(
                        kind="moving",
                        name=obs.name,
                        center=opponent_closest,
                        inflated_radius=inflated_radius,
                        distance_along_segment=t_frac * seg_len,
                        segment_fraction=t_frac,
                        predicted_time_s=segment_start_time_s + closest_t,
                        reason=f"predicted close approach to moving obstacle {obs.name}",
                    )
                )

        if not hits:
            return None

        # Return earliest hit along this segment.
        return min(hits, key=lambda h: h.distance_along_segment)

    def _make_detour_points(
        self,
        a: Point,
        b: Point,
        hit: CollisionHit,
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
        segment_start_time_s: float,
    ) -> List[Point]:
        """
        Generates two detour points around the collision object.

        The geometry is:

              w1 -------- w2
               \          /
                \        /
        a ------- obstacle ------- b

        The planner tries both left and right side and picks the shorter valid path.
        """
        direction = normalize(b - a)
        if norm(direction) < EPS:
            return []

        left = perp_left(direction)
        inflated = hit.inflated_radius

        side_offset = inflated + self.cfg.detour_margin
        forward_offset = inflated * self.cfg.detour_advance_scale

        candidates: List[Tuple[float, List[Point]]] = []

        for side in (+1.0, -1.0):
            side_vec = left * (side * side_offset)

            w1 = hit.center - direction * forward_offset + side_vec
            w2 = hit.center + direction * forward_offset + side_vec

            if self.cfg.field_bounds is not None and self.cfg.clamp_to_field:
                # Keep the detour point inside the field while leaving margin for robot radius.
                margin = self.cfg.robot_radius + self.cfg.safety_clearance
                w1 = self.cfg.field_bounds.clamp_point(w1, margin=margin)
                w2 = self.cfg.field_bounds.clamp_point(w2, margin=margin)

            candidate_path = [a, w1, w2, b]
            score = self._score_candidate_path(
                candidate_path=candidate_path,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
                path_start_time_s=segment_start_time_s,
            )
            candidates.append((score, [w1, w2]))

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _score_candidate_path(
        self,
        candidate_path: Sequence[Point],
        static_obstacles: Sequence[StaticObstacle],
        moving_obstacles: Sequence[MovingObstacle],
        path_start_time_s: float,
    ) -> float:
        """
        Lower score is better.
        Collision-free path gets score roughly equal to length.
        Colliding or out-of-field path gets a large penalty.
        """
        score = path_length(candidate_path)

        if self.cfg.field_bounds is not None:
            for p in candidate_path:
                if not self.cfg.field_bounds.contains(p):
                    score += 10_000.0

        elapsed = 0.0
        for i in range(len(candidate_path) - 1):
            a = candidate_path[i]
            b = candidate_path[i + 1]
            seg_start_time_s = path_start_time_s + elapsed / self.cfg.robot_speed

            hit = self._check_segment_collision(
                a=a,
                b=b,
                segment_start_time_s=seg_start_time_s,
                static_obstacles=static_obstacles,
                moving_obstacles=moving_obstacles,
            )

            if hit is not None:
                # Penalize candidates that still collide.
                # The later iterative planner may still repair it, but prefer clean candidates.
                score += 1_000.0

            elapsed += distance(a, b)

        return score

    @staticmethod
    def _remove_redundant_collinear_points(points: Sequence[Point]) -> List[Point]:
        """
        Removes duplicate points and nearly-collinear middle points.
        This keeps the path cleaner without changing the main route much.
        """
        if len(points) <= 2:
            return list(points)

        cleaned: List[Point] = []
        for p in points:
            if not cleaned or distance(cleaned[-1], p) > 1e-6:
                cleaned.append(p)

        if len(cleaned) <= 2:
            return cleaned

        result = [cleaned[0]]
        for i in range(1, len(cleaned) - 1):
            prev_p = result[-1]
            curr_p = cleaned[i]
            next_p = cleaned[i + 1]

            v1 = normalize(curr_p - prev_p)
            v2 = normalize(next_p - curr_p)

            # 1.0 means same direction, so the middle point is almost unnecessary.
            if dot(v1, v2) < 0.999:
                result.append(curr_p)

        result.append(cleaned[-1])
        return result


# ============================================================
# Example usage
# ============================================================


def print_path(path: Sequence[Point]) -> None:
    print("Planned waypoints:")
    for i, p in enumerate(path):
        print(f"  {i:02d}: x={p.x:.3f}, y={p.y:.3f}")
    print(f"Total path length: {path_length(path):.3f}")


def demo() -> None:
    # Example field coordinates. You can change these to inches, tiles, meters, etc.
    field = FieldBounds(min_x=0.0, max_x=12.0, min_y=0.0, max_y=12.0)

    config = PlannerConfig(
        robot_radius=0.35,
        safety_clearance=0.15,
        robot_speed=1.2,
        detour_margin=0.35,
        detour_advance_scale=1.3,
        max_insertions_per_segment=20,
        field_bounds=field,
        clamp_to_field=True,
    )

    planner = ContinuousDetourPlanner(config)

    start = Point(1.0, 1.0)

    # These are the points your robot wants to visit in order.
    points_of_interest = [
        Point(4.0, 2.0),
        Point(9.0, 2.0),
        Point(10.5, 9.0),
    ]

    # Known fixed obstacles on the field.
    # Use larger radii for rectangular-ish obstacles, or approximate rectangles with several circles.
    static_obstacles = [
        StaticObstacle(center=Point(6.0, 2.0), radius=0.55, name="fixed_goal_zone"),
        StaticObstacle(center=Point(8.0, 6.0), radius=0.75, name="center_barrier"),
    ]

    # Opponent robots detected right now.
    # velocity is estimated from tracking. If unknown, use Point(0, 0).
    moving_obstacles = [
        MovingObstacle(
            position=Point(5.2, 1.5),
            velocity=Point(0.20, 0.10),
            radius=0.45,
            name="opponent_1",
        ),
        MovingObstacle(
            position=Point(9.5, 5.5),
            velocity=Point(-0.10, 0.05),
            radius=0.45,
            name="opponent_2",
        ),
    ]

    path = planner.plan_route(
        start=start,
        points_of_interest=points_of_interest,
        static_obstacles=static_obstacles,
        moving_obstacles=moving_obstacles,
    )

    print_path(path)

    # Optional plot. The planner itself does not depend on matplotlib.
    try:
        plot_demo(
            path=path,
            start=start,
            points_of_interest=points_of_interest,
            static_obstacles=static_obstacles,
            moving_obstacles=moving_obstacles,
            config=config,
        )
    except ImportError:
        print("matplotlib not installed, skipping plot.")


def plot_demo(
    path: Sequence[Point],
    start: Point,
    points_of_interest: Sequence[Point],
    static_obstacles: Sequence[StaticObstacle],
    moving_obstacles: Sequence[MovingObstacle],
    config: PlannerConfig,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots()

    if config.field_bounds is not None:
        fb = config.field_bounds
        ax.add_patch(
            Rectangle(
                (fb.min_x, fb.min_y),
                fb.max_x - fb.min_x,
                fb.max_y - fb.min_y,
                fill=False,
                linewidth=2,
            )
        )
        ax.set_xlim(fb.min_x - 0.5, fb.max_x + 0.5)
        ax.set_ylim(fb.min_y - 0.5, fb.max_y + 0.5)

    # Draw static obstacles and their inflated safety zones.
    for obs in static_obstacles:
        inflated = obs.radius + config.robot_radius + config.safety_clearance
        ax.add_patch(Circle(obs.center.as_tuple(), inflated, fill=False, linestyle="--"))
        ax.add_patch(Circle(obs.center.as_tuple(), obs.radius, alpha=0.25))
        ax.text(obs.center.x, obs.center.y, obs.name, ha="center", va="center")

    # Draw moving obstacles at current position and a short velocity arrow.
    for obs in moving_obstacles:
        inflated = obs.radius + config.robot_radius + config.safety_clearance
        ax.add_patch(Circle(obs.position.as_tuple(), inflated, fill=False, linestyle=":"))
        ax.add_patch(Circle(obs.position.as_tuple(), obs.radius, alpha=0.25))
        ax.arrow(
            obs.position.x,
            obs.position.y,
            obs.velocity.x,
            obs.velocity.y,
            head_width=0.10,
            length_includes_head=True,
        )
        ax.text(obs.position.x, obs.position.y + inflated + 0.15, obs.name, ha="center")

    # Draw path.
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    ax.plot(xs, ys, marker="o", linewidth=2)

    # Draw start and POIs.
    ax.scatter([start.x], [start.y], s=80)
    ax.text(start.x, start.y - 0.25, "start", ha="center")

    for i, p in enumerate(points_of_interest, start=1):
        ax.scatter([p.x], [p.y], s=80)
        ax.text(p.x, p.y + 0.25, f"POI {i}", ha="center")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("Continuous Detour Path Planner")
    plt.show()


if __name__ == "__main__":
    demo()
