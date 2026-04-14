"""Gymnasium environment for VEX AI dual-agent field simulation.

The sim runs at fine-grained ticks (DT = 0.05s) for smooth movement
animation, but the RL only makes decisions every DECISION_INTERVAL
(3.0s = 60 ticks). Each env.step() = one RL decision cycle:
  1. Receive actions for both allied robots
  2. Simulate 60 ticks of movement (robots drive toward targets)
  3. Check action effects (collect, score, descore) each tick
  4. Return observations + rewards after the full interval

This means the RL trains on strategic decisions (where to go), while
the tank-drive moveToPoint runs underneath at high resolution.
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.config import (
    Action, NUM_ACTIONS, STATE_DIM,
    FIELD_W, FIELD_H, MATCH_DURATION, DT, TICKS_PER_DECISION, MAX_CARRY,
    MAX_GAME_OBJECTS, OBJ_FEATURES, OBJ_ON_FIELD, OBJ_SCORED_US, OBJ_SCORED_OPP,
    HEATMAP_W, HEATMAP_H, MAX_SCORE,
    OUR_LONG_GOAL, OUR_MID_GOAL, OPP_LONG_GOAL, OPP_MID_GOAL,
    REGION_A_CENTER, REGION_B_CENTER, DEFEND_ZONE_POS,
    LONG_GOAL_POINTS, MID_GOAL_POINTS, COLLECT_RANGE,
)
from sim.field import Field
from sim.robot import Robot
from sim.failure import FailureConfig, FailureInjector
from sim.opponent import get_opponent


# Map actions to target positions (for moveToPoint)
def _action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    """Convert RL action to a target point the robot should moveToPoint toward."""
    if action == Action.COLLECT_NEAREST_BALL:
        return field.nearest_on_field_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        return OUR_LONG_GOAL.copy()
    elif action == Action.SCORE_MID_GOAL:
        return OUR_MID_GOAL.copy()
    elif action == Action.DESCORE_OPPONENT_LONG:
        return OPP_LONG_GOAL.copy()
    elif action == Action.DESCORE_OPPONENT_MID:
        return OPP_MID_GOAL.copy()
    elif action == Action.DEFEND_ZONE:
        return DEFEND_ZONE_POS.copy()
    elif action == Action.MOVE_TO_REGION_A:
        return REGION_A_CENTER.copy()
    elif action == Action.MOVE_TO_REGION_B:
        return REGION_B_CENTER.copy()
    elif action == Action.IDLE:
        return None
    return None


def _opp_action_to_target(action: Action, field: Field, robot: Robot) -> np.ndarray | None:
    if action == Action.COLLECT_NEAREST_BALL:
        return field.nearest_on_field_target(robot.position)
    elif action == Action.SCORE_LONG_GOAL:
        return OPP_LONG_GOAL.copy()
    elif action == Action.SCORE_MID_GOAL:
        return OPP_MID_GOAL.copy()
    elif action == Action.DESCORE_OPPONENT_LONG:
        return OUR_LONG_GOAL.copy()
    elif action == Action.DESCORE_OPPONENT_MID:
        return OUR_MID_GOAL.copy()
    elif action == Action.DEFEND_ZONE:
        return np.array([108.00, 72.00])  # opponent defend zone
    elif action == Action.MOVE_TO_REGION_A:
        return np.array([108.00, 108.00])
    elif action == Action.MOVE_TO_REGION_B:
        return np.array([36.00, 36.00])
    return None


class VexAIEnv(gym.Env):
    """Two-agent VEX AI field environment.

    Each step() = one RL decision interval (3 seconds of simulated time,
    60 ticks of fine-grained movement). Both allied robots act; opponents
    are rule-based.

    For SB3 compatibility, use SingleAgentWrapper below.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        opponent_type: str = "mixed",
        failure_config: FailureConfig | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.failure_config = failure_config or FailureConfig()

        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS, NUM_ACTIONS])
        self.observation_space = spaces.Dict({
            "robot_0": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            "robot_1": spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
        })

        self.opponent_policy = get_opponent(opponent_type)
        self.field = Field()
        self._renderer = None

        # Per-step event tracking
        self.score_events = np.zeros(2)
        self.descore_events = np.zeros(2)
        self.collected_this_step = np.zeros(2, dtype=np.int32)
        self.current_actions = np.array([Action.IDLE, Action.IDLE], dtype=np.int32)
        self.expected_state_delta = np.zeros(2)
        self.prev_predicted_score = np.zeros(2)
        self.done = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.field.reset(self.rng)
        self.field.time_remaining = MATCH_DURATION

        self.failure_injector = FailureInjector(self.failure_config, self.rng)
        self.failure_injector.reset()

        self.score_events[:] = 0
        self.descore_events[:] = 0
        self.collected_this_step[:] = 0
        self.current_actions[:] = Action.IDLE
        self.expected_state_delta[:] = 0
        self.prev_predicted_score[:] = 0
        self.done = False

        # Opponent decision targets (refreshed each decision interval)
        self.opp_targets: list[np.ndarray | None] = [None, None]

        return self._get_obs(), {}

    def step(self, actions: np.ndarray):
        """One RL decision cycle = TICKS_PER_DECISION sim ticks."""
        a0, a1 = int(actions[0]), int(actions[1])
        self.current_actions = np.array([a0, a1], dtype=np.int32)

        # Reset per-decision event counters
        self.score_events[:] = 0
        self.descore_events[:] = 0
        self.collected_this_step[:] = 0

        # --- Failure checks (per-decision) ---
        fi = self.failure_injector
        if fi.teammate_offline or fi.should_teammate_fail():
            a1 = Action.IDLE

        # --- Compute move targets for allied robots ---
        ally_targets = [
            _action_to_target(Action(a0), self.field, self.field.allies[0]),
            _action_to_target(Action(a1), self.field, self.field.allies[1]),
        ]

        # --- Compute opponent decisions (once per interval) ---
        for oi in range(2):
            opp = self.field.opponents[oi]
            opp_state = {
                "position": opp.position,
                "balls_held": opp.balls_held,
                "our_score": self.field.my_score,
                "opp_score": self.field.opponent_score,
                "us_scored_count": len(self.field.scored_by_us_indices()),
            }
            opp_action = self.opponent_policy(opp_state, self.rng)
            self.opp_targets[oi] = _opp_action_to_target(opp_action, self.field, opp)

        # --- Object steal check ---
        for idx in range(2):
            if fi.should_steal_object():
                self._steal_nearest(idx)

        # --- Simulate TICKS_PER_DECISION ticks ---
        for tick in range(TICKS_PER_DECISION):
            fi.on_tick(num_robots=2)

            # Move allied robots
            for idx in range(2):
                if fi.is_stuck(idx):
                    continue
                target = ally_targets[idx]
                if target is not None:
                    speed_mult = fi.get_speed_mult(idx)
                    # Temporarily adjust speed via robot
                    orig_speed = self.field.allies[idx]._clamp_to_field  # just use move_toward_point
                    self.field.allies[idx].move_toward_point(target)

            # Move opponents (slightly slower, no failures)
            for oi in range(2):
                if self.opp_targets[oi] is not None:
                    self.field.opponents[oi].move_toward_point(self.opp_targets[oi])

            # Check action effects each tick
            self._check_ally_effects(0, Action(a0))
            self._check_ally_effects(1, Action(a1))
            self._check_opp_effects()

            # Advance time
            self.field.time_remaining -= DT
            if self.field.time_remaining <= 0:
                break

            # Render if human mode
            if self.render_mode == "human" and self._renderer is not None:
                self._renderer.draw(self)

        self.done = self.field.time_remaining <= 0

        # Track actions
        for idx in range(2):
            self.field.allies[idx].actions_attempted += 1

        # Expected state delta
        for idx in range(2):
            actual = float(self.field.my_score)
            self.expected_state_delta[idx] = actual - self.prev_predicted_score[idx]
            self.prev_predicted_score[idx] = actual

        # Compute rewards
        from training.reward import compute_reward
        r0 = compute_reward(self, robot_id=0)
        r1 = compute_reward(self, robot_id=1)

        return self._get_obs(), (r0, r1), self.done, False, {}

    # ------------------------------------------------------------------
    # Effect checks (called each tick)
    # ------------------------------------------------------------------
    def _check_ally_effects(self, idx: int, action: Action):
        robot = self.field.allies[idx]

        if action == Action.COLLECT_NEAREST_BALL:
            if self.field.try_collect(robot):
                self.collected_this_step[idx] += 1
                robot.actions_succeeded += 1

        elif action == Action.SCORE_LONG_GOAL:
            pts = self.field.try_score(robot, OUR_LONG_GOAL, LONG_GOAL_POINTS)
            if pts > 0:
                self.score_events[idx] += pts
                robot.actions_succeeded += 1

        elif action == Action.SCORE_MID_GOAL:
            pts = self.field.try_score(robot, OUR_MID_GOAL, MID_GOAL_POINTS)
            if pts > 0:
                self.score_events[idx] += pts
                robot.actions_succeeded += 1

        elif action == Action.DESCORE_OPPONENT_LONG:
            pts = self.field.try_descore(robot, OPP_LONG_GOAL, self.rng)
            if pts > 0:
                self.descore_events[idx] += pts
                robot.actions_succeeded += 1

        elif action == Action.DESCORE_OPPONENT_MID:
            pts = self.field.try_descore(robot, OPP_MID_GOAL, self.rng)
            if pts > 0:
                self.descore_events[idx] += pts
                robot.actions_succeeded += 1

    def _check_opp_effects(self):
        for oi in range(2):
            opp = self.field.opponents[oi]
            self.field.opp_try_collect(opp, self.rng)
            self.field.opp_try_score(opp, OPP_LONG_GOAL, LONG_GOAL_POINTS)
            self.field.opp_try_score(opp, OPP_MID_GOAL, MID_GOAL_POINTS)
            self.field.opp_try_descore(opp, OUR_LONG_GOAL, self.rng)
            self.field.opp_try_descore(opp, OUR_MID_GOAL, self.rng)

    def _steal_nearest(self, robot_idx: int):
        """Opponent steals the object the ally was heading toward."""
        robot = self.field.allies[robot_idx]
        target = self.field.nearest_on_field_target(robot.position)
        if target is not None:
            for obj in self.field.objects:
                if obj.status == OBJ_ON_FIELD and np.linalg.norm(obj.position - target) < 1.0:
                    obj.status = OBJ_SCORED_OPP
                    self.field.opponent_score += MID_GOAL_POINTS
                    break

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "robot_0": self._build_obs(0),
            "robot_1": self._build_obs(1),
        }

    def _build_obs(self, role_id: int) -> np.ndarray:
        robot = self.field.allies[role_id]
        pos = robot.position

        balls_nearby = self.field.nearby_ball_count(pos)

        # Flatten game objects
        obj_flat = np.zeros(MAX_GAME_OBJECTS * OBJ_FEATURES, dtype=np.float32)
        for i, obj in enumerate(self.field.objects):
            obj_flat[i * 3] = obj.x / FIELD_W
            obj_flat[i * 3 + 1] = obj.y / FIELD_H
            obj_flat[i * 3 + 2] = float(obj.status)

        heatmap = self.field.get_heatmap()

        return np.concatenate([
            np.array([
                float(role_id),
                self.field.time_remaining / MATCH_DURATION,
                float(self.field.my_score) / MAX_SCORE,
                float(self.field.opponent_score) / MAX_SCORE,
                pos[0] / FIELD_W,
                pos[1] / FIELD_H,
                robot.heading / (2 * np.pi),
                float(robot.balls_held) / MAX_CARRY,
                float(balls_nearby) / MAX_GAME_OBJECTS,
            ], dtype=np.float32),
            obj_flat,
            heatmap.flatten(),
            np.array([
                self.expected_state_delta[role_id] / 10.0,
                robot.success_ratio(),
            ], dtype=np.float32),
        ])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from sim.renderer import PygameRenderer
                self._renderer = PygameRenderer(self)
            self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ======================================================================
# Single-agent wrapper for Stable-Baselines3 compatibility
# ======================================================================
class SingleAgentWrapper(gym.Env):
    """Wraps VexAIEnv for SB3: concatenated obs, single discrete action."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.env = VexAIEnv(render_mode=render_mode, **kwargs)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(STATE_DIM * 2,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS * NUM_ACTIONS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return np.concatenate([obs["robot_0"], obs["robot_1"]]), info

    def step(self, action: int):
        a0 = action // NUM_ACTIONS
        a1 = action % NUM_ACTIONS
        obs, rewards, done, truncated, info = self.env.step(np.array([a0, a1]))
        return np.concatenate([obs["robot_0"], obs["robot_1"]]), rewards[0] + rewards[1], done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
