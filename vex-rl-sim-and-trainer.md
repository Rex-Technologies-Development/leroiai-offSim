# VEX AI RL — Sim & Trainer Specification

This document covers the training repo (`vex-rl/`) in detail: the 2D simulation environment, the Pygame visualizer, reward shaping, training scripts, and the offline fine-tuning pipeline.

---

## Sim Environment (`sim/env.py`)

A custom Gymnasium environment that models the VEX AI field as a 2D coordinate space. No physics engine — movement is simplified to position updates per timestep.

### Field Model

The field is represented as a 2D coordinate grid matching real VEX AI field dimensions (scaled to cm or inches). Key elements modeled:

- **Robots**: 4 total (2 allied, 2 opponent). Each has position `(x, y)`, heading, and held objects count.
- **Game objects**: Balls/triballs/rings (depending on game). Each has position, state (`on_field`, `in_goal`, `held`), and which goal it's in if scored.
- **Goals**: Long goal, mid goal, and their opponent equivalents. Each tracks scored objects and point value.
- **Zones**: Defined regions for defense, loading, scoring areas.

### Step Logic

Each `env.step(action)` call for one agent:

1. Decode action (e.g., `COLLECT_NEAREST_BALL`)
2. Simulate movement toward target (clamp speed to realistic max, ~1-2 tiles/step)
3. Execute action effect if in range (pick up ball, score in goal, descore)
4. Update game object states
5. Step the other three agents (teammate + 2 opponents) using their own policies
6. Decrement time remaining
7. Compute reward
8. Return `(observation, reward, terminated, truncated, info)`

### Dual-Agent Stepping

The env manages **two allied agents internally**. On each call to `step()`, both agents act. Externally, the training loop calls `step(action_robot_0, action_robot_1)` and gets back observations and rewards for both.

```python
class VexAIEnv(gymnasium.Env):
    def step(self, actions):
        # actions = (action_robot_0, action_robot_1)
        obs_0 = self._build_obs(role_id=0)
        obs_1 = self._build_obs(role_id=1)
        
        self._execute_action(robot=0, action=actions[0])
        self._execute_action(robot=1, action=actions[1])
        self._step_opponents()
        self._update_field_state()
        self.time_remaining -= self.dt
        
        reward_0 = self._compute_reward(robot=0)
        reward_1 = self._compute_reward(robot=1)
        done = self.time_remaining <= 0
        
        return (obs_0, obs_1), (reward_0, reward_1), done, False, {}
```

### Observation Builder

Each robot gets its own observation. The `role_id` is injected so the policy can learn role-specific behavior.

```python
def _build_obs(self, role_id):
    return np.concatenate([
        [role_id],                          # 0 or 1
        [self.time_remaining / self.max_time],  # normalized
        [self.my_score / self.max_score],
        [self.opponent_score / self.max_score],
        self.robots[role_id].position,       # (x, y) normalized
        [self.robots[role_id].heading / (2 * np.pi)],
        [self.robots[role_id].balls_held / self.max_carry],
        self._nearby_ball_count(role_id),
        self._flatten_game_objects(),        # positions + states
        self._compute_heatmap().flatten(),   # point potential grid
        [self._expected_state_delta(role_id)],
        [self._actions_completed_ratio(role_id)],
    ])
```

### Heatmap

A discretized grid (e.g., 12x12) over the field. Each cell contains a value representing the "point potential" — how many points could be gained by acting in that region. Factors: number of unscouted objects, proximity to goals, defensive value. Updated every step.

```python
def _compute_heatmap(self):
    grid = np.zeros((self.grid_h, self.grid_w))
    for obj in self.game_objects:
        if obj.state == "on_field":
            gx, gy = self._pos_to_grid(obj.position)
            grid[gy, gx] += obj.point_value
    # Blur slightly so nearby cells also show value
    grid = gaussian_filter(grid, sigma=1.0)
    return grid
```

### Randomized Failures (Robustness Training)

Each step, random events can occur to train the policy against real-world noise:

- **Teammate failure**: teammate's action has a configurable chance of doing nothing (0-100%)
- **Action delay**: actions take 1-3x their normal duration randomly
- **Object stolen**: an opponent can grab a nearby object before the robot reaches it
- **Robot stuck**: robot position doesn't update for N steps (simulates being pushed/blocked)
- **Teammate offline**: teammate stops acting entirely for the rest of the match (simulates disconnect/shutdown)

These probabilities are configurable and ramped up during training.

```python
class FailureConfig:
    teammate_fail_rate: float = 0.15       # 15% chance teammate action fails
    action_delay_range: (float, float) = (1.0, 2.5)  # multiplier on action duration
    object_stolen_rate: float = 0.10       # 10% chance target object gets taken
    stuck_rate: float = 0.05               # 5% chance robot gets stuck for 2-3 steps
    teammate_offline_rate: float = 0.02    # 2% chance teammate goes offline permanently
```

### Opponent Behavior (`sim/opponent.py`)

Opponents are controlled by rule-based policies, not RL. Several difficulty tiers:

- **Random**: picks a random valid action each step
- **Greedy**: always goes for the highest-value nearby object
- **Defensive**: prioritizes descoring and blocking
- **Mixed**: switches between greedy and defensive based on score differential

During training, opponent difficulty is randomized per episode to prevent overfitting to one strategy.

---

## Pygame Visualizer (`sim/renderer.py`)

A top-down 2D render of the field. Used during training to visually verify policy behavior.

### What It Renders

- **Field**: scaled rectangle with goal zones, boundary lines, key markings
- **Robots**: colored rectangles with heading indicators. Allied = blue/green, opponents = red/orange
- **Game objects**: colored circles at their positions. Different colors for different states (on_field, held, scored)
- **Goals**: rectangles at field edges, filled proportionally to scored objects
- **Heatmap overlay**: semi-transparent color gradient showing point potential (toggle on/off)
- **HUD**: time remaining, scores, current action for each allied robot, episode count, reward

### Controls

- `Space`: pause/unpause simulation
- `S`: step one tick while paused
- `H`: toggle heatmap overlay
- `R`: reset episode
- `+`/`-`: speed up / slow down sim
- `1`/`2`: highlight robot 0 / robot 1 and show its observation details

### Rendering Mode

The renderer can run in two modes:

- **Live training**: renders every Nth frame during PPO/DQN training (configurable, e.g., every 100th episode). Slows training but lets you watch.
- **Eval mode**: after training, load a checkpoint and watch the policy play full episodes at human-readable speed.

```python
renderer = PygameRenderer(env, render_every=100)

# During training
if episode % renderer.render_every == 0:
    renderer.render(env.get_render_state())
    
# Eval mode
renderer.run_eval(policy, num_episodes=10, fps=10)
```

---

## Reward Shaping (`training/reward.py`)

The reward function is critical. It must incentivize winning, not just scoring.

### Reward Components

```python
def compute_reward(env, robot_id):
    r = 0.0
    
    # Scoring events (immediate)
    r += env.score_events[robot_id] * 1.0       # +1 per point scored this step
    
    # Descoring events (immediate)
    r += env.descore_events[robot_id] * 0.8      # +0.8 per opponent point removed
    
    # Collection (small incentive)
    r += env.collected_this_step[robot_id] * 0.1  # +0.1 per ball picked up
    
    # Time pressure (late-game scoring worth more)
    time_factor = 1.0 + (1.0 - env.time_remaining / env.max_time) * 0.5
    r *= time_factor
    
    # Score differential bonus (end of episode)
    if env.done:
        diff = env.my_score - env.opponent_score
        if diff > 0:
            r += 5.0 + diff * 0.5   # win bonus + margin
        elif diff == 0:
            r += 1.0                  # tie is okay
        else:
            r -= 3.0                  # loss penalty
    
    # Idle penalty
    if env.actions[robot_id] == IDLE:
        r -= 0.05
    
    return r
```

### Key Principles

- **Win > high score**: a 10-8 win is better than a 20-25 loss. The end-of-episode differential bonus handles this.
- **Time-weighted scoring**: scoring 2 points with 10 seconds left is more valuable than scoring 2 points with 90 seconds left (more time to recover).
- **Idle penalty**: discourages doing nothing, but kept small so the policy can choose to wait when genuinely optimal.
- **No direct penalty for teammate failure**: the policy shouldn't be punished for things outside its control. It's only rewarded/penalized for its own actions and the final outcome.

---

## Training Scripts

### Phase 1: Sim Training (`training/train_sim.py`)

Uses Stable-Baselines3 with PPO (or DQN for comparison).

```python
from stable_baselines3 import PPO
from sim.env import VexAIEnv
from sim.renderer import PygameRenderer

env = VexAIEnv(
    failure_config=FailureConfig(teammate_fail_rate=0.15),
    opponent_type="mixed",
)
renderer = PygameRenderer(env, render_every=500)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs/",
)

model.learn(total_timesteps=2_000_000, callback=renderer.callback)
model.save("models/checkpoints/ppo_v1")
```

### Curriculum Training

Difficulty ramps up over training:

1. **Stage 1** (0-500K steps): random opponents, no teammate failures
2. **Stage 2** (500K-1M steps): greedy opponents, 10% teammate failure
3. **Stage 3** (1M-2M steps): mixed opponents, 20% teammate failure, occasional teammate offline

This prevents the policy from being overwhelmed early while ensuring robustness later.

### Phase 3: Offline Fine-Tuning (`training/train_offline.py`)

Uses d3rlkit with CQL or IQL on recorded match data.

```python
import d3rlpy

# Load match data
dataset = d3rlpy.dataset.MDPDataset.load("data/matches/combined.h5")

# CQL for offline RL
cql = d3rlpy.algos.CQLConfig(
    learning_rate=1e-4,
    alpha=1.0,
).create(device="cuda:0")

cql.fit(
    dataset,
    n_steps=100_000,
    n_steps_per_epoch=1000,
    evaluators={"td_error": d3rlpy.metrics.TDErrorEvaluator(episodes=dataset.episodes[:10])},
)

cql.save_model("models/checkpoints/cql_v1.pt")
```

### Match Data Format

Each match is logged as a sequence of transitions:

```python
# Single transition
{
    "state": np.array([...]),      # flattened observation vector
    "action": int,                  # action ID
    "reward": float,                # computed reward for this step
    "next_state": np.array([...]), # observation after action
    "done": bool,                   # episode ended?
}
```

Stored as `.npz` files on the Jetson, transferred to the training PC after matches.

### Combining Sim + Real Data

```python
sim_dataset = d3rlpy.dataset.MDPDataset.load("data/sim/sim_data.h5")
match_dataset = d3rlpy.dataset.MDPDataset.load("data/matches/combined.h5")

# Concatenate
combined = d3rlpy.dataset.MDPDataset(
    observations=np.vstack([sim_dataset.observations, match_dataset.observations]),
    actions=np.hstack([sim_dataset.actions, match_dataset.actions]),
    rewards=np.hstack([sim_dataset.rewards, match_dataset.rewards]),
    terminals=np.hstack([sim_dataset.terminals, match_dataset.terminals]),
)

cql.fit(combined, n_steps=100_000)
```

---

## ONNX Export (`training/export_onnx.py`)

Converts the trained PyTorch policy into an ONNX file for Jetson deployment.

```python
import torch
import onnx

# Load trained model
model = PPO.load("models/checkpoints/ppo_v1")
policy = model.policy

# Dummy input matching state vector shape
dummy_input = torch.randn(1, env.observation_space.shape[0])

# Export
torch.onnx.export(
    policy,
    dummy_input,
    "models/onnx/model_v1.onnx",
    input_names=["state"],
    output_names=["action_logits"],
    dynamic_axes={"state": {0: "batch"}, "action_logits": {0: "batch"}},
    opset_version=17,
)

# Verify
onnx_model = onnx.load("models/onnx/model_v1.onnx")
onnx.checker.check_model(onnx_model)
print("Export successful. Input shape:", dummy_input.shape)
print("Output: action logits →", env.action_space.n, "actions")
```

---

## Config (`sim/config.py`)

Shared constants used across sim, training, and (via `config.yaml`) deployment.

```python
# Field dimensions (in cm, matching real VEX AI field)
FIELD_WIDTH = 366
FIELD_HEIGHT = 366

# Action definitions
ACTIONS = {
    0: "COLLECT_NEAREST_BALL",
    1: "SCORE_LONG_GOAL",
    2: "SCORE_MID_GOAL",
    3: "DESCORE_OPPONENT_LONG",
    4: "DESCORE_OPPONENT_MID",
    5: "DEFEND_ZONE",
    6: "MOVE_TO_REGION_A",
    7: "MOVE_TO_REGION_B",
    8: "IDLE",
}
NUM_ACTIONS = len(ACTIONS)

# State vector dimensions
HEATMAP_GRID = (12, 12)
MAX_GAME_OBJECTS = 24
STATE_DIM = 1 + 1 + 2 + 2 + 1 + 1 + 1 + (MAX_GAME_OBJECTS * 3) + (HEATMAP_GRID[0] * HEATMAP_GRID[1]) + 1 + 1

# Timing
MATCH_DURATION = 120  # seconds
DT = 0.5             # seconds per sim step (240 steps per match)

# Robot
MAX_SPEED = 50        # cm per second
MAX_CARRY = 3         # max balls held at once
```

---

## Eval & Verification Workflow

Before deploying a new model, run through this checklist:

1. **Sim eval**: load checkpoint, run 100 episodes with Pygame renderer, watch for sane behavior
2. **Role divergence check**: both robots should NOT be going to the same place. Log action distributions per role and confirm they differ.
3. **Failure recovery**: run episodes with 100% teammate failure rate. Robot 0 should adapt and try to win solo.
4. **Late-game behavior**: check that the policy prioritizes differently with 10 seconds left vs 90 seconds left.
5. **Score differential awareness**: when ahead by 10+, policy should play defensively. When behind, aggressively.
6. **Heatmap response**: place high-value clusters in different field locations and verify the policy responds to them.

```bash
# Quick eval command
python training/eval.py --model models/checkpoints/ppo_v1 --episodes 100 --render --failure-rate 0.2
```
