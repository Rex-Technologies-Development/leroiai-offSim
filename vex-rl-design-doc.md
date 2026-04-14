# VEX AI Offline RL System — Design Document

## Overview

An offline reinforcement learning system for two VEX AI robots that make strategic decisions (what to score, where to go, when to defend) without real-time communication. Trained first in simulation, then fine-tuned on real match data, and deployed on NVIDIA Jetson AGX Orin units communicating with VEX V5 brains.

---

## Software Stack

| Component | Tool |
|---|---|
| Language | Python (training/sim), Python + C++ (ROS 2 deployment) |
| Model framework | PyTorch |
| Env wrapper | Gymnasium (custom env) |
| Sim training | PPO or DQN via Stable-Baselines3 |
| Offline RL | CQL / IQL via d3rlkit |
| 2D visualization | Pygame (top-down field render) |
| Deployment runtime | ROS 2 Humble |
| Inference on Jetson | ONNX Runtime (exported from PyTorch) |

---

## Architecture

### Action Space (Discrete)

```
COLLECT_NEAREST_BALL
SCORE_LONG_GOAL
SCORE_MID_GOAL
DESCORE_OPPONENT_LONG
DESCORE_OPPONENT_MID
DEFEND_ZONE
MOVE_TO_REGION_A
MOVE_TO_REGION_B
IDLE
```

Actions are **granular and single-step**. The RL does not plan multi-step sequences like "collect 3 balls then score." It picks one action at a time and re-queries every few seconds with updated state.

### State Vector

```python
state = {
    "role_id": 0 or 1,              # fixed per robot, enables divergent behavior
    "time_remaining": float,         # seconds left in match
    "my_score": int,
    "opponent_score": int,
    "my_position": (x, y),
    "my_heading": float,
    "balls_held": int,               # what the robot currently holds
    "balls_available_nearby": int,   # quick-reach objects
    "game_objects": [...],           # positions + states of all field objects
    "heatmap": np.array(grid),       # point potential across field (discretized)
    "expected_state_delta": float,   # how far off actual state is from predicted
    "actions_completed_ratio": float # my own success rate this match
}
```

Key: **teammate position is NOT included**. Each robot only knows its own state + the field.

---

## Multi-Agent: No Communication

### Problem

Two robots run the same policy. If both see the same state and have the same role, they pick the same action and collide.

### Solution: CTDE (Centralized Training, Decentralized Execution)

- **Role conditioning**: Each robot has a fixed `role_id` (0 or 1) as part of its input state. The policy learns divergent behavior per role.
- During **training**, the sim runs both agents simultaneously. The policy learns implicit coordination: robot 0 tends to handle certain tasks/regions, robot 1 handles others.
- At **runtime**, each robot executes independently. Coordination is baked into the policy via role conditioning.

### Coordination Without Communication: Stigmergy

Robots coordinate through the **field state**, not direct messages. If robot 2 dies, robot 1 doesn't get a signal — but it sees that goals that should have been scored aren't scored, objects that should be collected are still there. The policy, trained on many failure scenarios, responds to this gap by covering what's missing.

---

## Handling Failures

### Two-Tier Architecture

```
Tier 1: RL Policy (runs every 3-5 seconds)
  → "What should I do?" → SCORE_LONG_GOAL

Tier 2: Execution Monitor (runs continuously)
  → Tracks progress toward current action
  → Detects failure conditions
  → Triggers re-query of Tier 1 if action fails
```

### Failure Detection (Hardcoded, Not Learned)

| Condition | Detection | Response |
|---|---|---|
| Timeout | Action not completed within expected duration | Abort, re-query RL |
| Stuck | Position hasn't changed >X cm in Y seconds | Abort, re-query RL |
| State invalidation | Target object scored by opponent while en route | Abort, re-query RL |
| Partial completion | Holding fewer objects than planned | Not a failure — RL re-queries with current `balls_held` and decides next best action |

### Periodic Re-Planning

Both robots re-query the RL on a **fixed interval (3-5 seconds)**, not just on failure. Each time, they feed the current observed field state. This naturally handles:

- Teammate going offline (field state diverges from expected)
- Partial completions (RL sees current holdings and adapts)
- Opponent interference (RL sees changed game state)

### Expected vs Actual State Delta

Each re-plan cycle:

1. Before executing, predict expected field state after N seconds
2. After N seconds, observe actual field state
3. Compute `delta = expected - actual`
4. Feed delta as additional input to RL

A large negative delta (less progress than expected) implicitly signals teammate failure or opponent interference. The policy learns to respond to large deltas by switching to more self-reliant strategies.

### Training for Robustness

In the sim, **randomize teammate performance heavily**:

- Teammate completes 100% of actions (ideal case)
- Teammate completes 50% of actions (partial reliability)
- Teammate completes 0% of actions (offline/disabled)
- Actions take longer than expected (random delays)
- Objects get taken by opponents mid-action

The policy learns conservative, self-sufficient strategies that are robust across these scenarios.

---

## Training Pipeline

### Phase 1: Sim Pre-Training

- Custom Gymnasium env models VEX AI field as 2D grid
- Pygame renders top-down: colored dots for objects, rectangles for robots, goals drawn to scale
- Env steps in discrete time ticks, simulates scoring/descoring with simplified physics
- Opponent behavior: random or rule-based initially
- Train with PPO or DQN to get baseline policy
- Dual-agent sim: both robots run simultaneously, role-conditioned

### Phase 2: Deploy + Collect Data

- Export trained model to ONNX
- Deploy on Jetson AGX Orin via ROS 2
- ROS 2 node subscribes to game state topics, runs ONNX inference, publishes action
- Translator node maps actions to V5 commands over serial
- **Log every timestep**: `(state, action_taken, reward, next_state, done)`
- Label matches with outcome (win/loss margin → reward signal)

### Phase 3: Offline Fine-Tuning

- Use recorded match data with offline RL (CQL or IQL from d3rlkit)
- Train on real data without needing the sim
- Can mix sim data + real data for larger dataset
- Re-export to ONNX, redeploy
- Repeat phases 2-3

---

## Deployment Architecture

### Repo Separation

Training and deployment live in **separate repositories**. The only artifact that crosses the boundary is an ONNX file. The ROS 2 workspace never imports PyTorch, Gymnasium, d3rlkit, or any training code. This keeps the Jetson image lean and builds fast.

```
vex-rl/        (Training repo — runs on PC/GPU)
  trains model → exports model.onnx

vex-ros2-ws/   (Deployment repo — runs on Jetson)
  loads model.onnx → runs inference via onnxruntime-gpu
```

### Handoff Workflow

```
1. Train on PC/GPU      →  python train_sim.py  or  python train_offline.py
2. Export                →  python export_onnx.py --output model_v3.onnx
3. Copy to Jetson        →  scp model_v3.onnx jetson:/ros2_ws/src/rl_decision/models/
4. Update config         →  ros2 param set /rl_decision model_path models/model_v3.onnx
5. Restart node          →  (or hot-reload if supported)
```

Automated via `./deploy.sh` — exports, copies over SSH, restarts the node.

### Match Data Flows Back

```
Jetson (during match) → logs to /data/match_log_001.npz
After match            → scp back to training PC
Training PC            → python train_offline.py --data match_log_001.npz
                       → exports updated model.onnx
                       → deploy.sh pushes back to Jetson
```

Data flow is circular: **model goes PC → Jetson, match data goes Jetson → PC**. They never share code, just files.

### Shared Contract

Both repos must agree on the **state vector shape** and **action ID mapping**. Maintained via a shared `config.yaml` or hardcoded in both (it changes rarely). If training defines `SCORE_LONG_GOAL = 2`, the ROS 2 translator must also map `2 → SCORE_LONG_GOAL`.

### ROS 2 Node Pipeline

```
[Camera/Sensors] → [ROS 2 State Node] → [RL Decision Node (ONNX)] → [Action Translator Node] → [Serial → V5 Brain]
```

### ROS 2 Topics

| Topic | Type | Description |
|---|---|---|
| `/game_state` | Custom msg | Field state from vision/localization |
| `/strategy/action` | String/Int | Chosen action from RL |
| `/v5/command` | Custom msg | Translated V5 command sequence |

### RL Decision Node (Minimal)

```python
import onnxruntime as ort
import rclpy
from rclpy.node import Node

class RLDecisionNode(Node):
    def __init__(self):
        super().__init__('rl_decision')
        model_path = self.declare_parameter('model_path', 'models/model.onnx').value
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        self.state_sub = self.create_subscription(GameState, '/game_state', self.on_state, 10)
        self.action_pub = self.create_publisher(Action, '/strategy/action', 10)
        self.timer = self.create_timer(3.0, self.replan)
        self.latest_state = None

    def on_state(self, msg):
        self.latest_state = msg_to_numpy(msg)

    def replan(self):
        if self.latest_state is None:
            return
        result = self.session.run(None, {'state': self.latest_state})
        action_id = result[0].argmax()
        self.action_pub.publish(Action(action=action_id))
```

No PyTorch, no training logic. Just load ONNX and run.

### Action Translation Example

```
RL output: SCORE_LONG_GOAL (action_id=2)
↓
Translator: moveToPoint(23, 56) → scoreLongGoal(timeout=5s)
↓
Serial to V5 Brain
```

---

## File Structure

### Training Repo (`vex-rl/`)

```
vex-rl/
├── sim/
│   ├── env.py              # Gymnasium env (field logic, dual-agent)
│   ├── renderer.py          # Pygame 2D top-down visualization
│   ├── config.py            # Field dimensions, game rules, action definitions
│   └── opponent.py          # Rule-based opponent behavior
├── training/
│   ├── train_sim.py         # Phase 1: PPO/DQN in sim
│   ├── train_offline.py     # Phase 3: CQL/IQL from match data
│   ├── export_onnx.py       # Export PyTorch → ONNX
│   └── reward.py            # Reward shaping functions
├── shared/
│   └── config.yaml          # Action IDs, state schema (duplicated to ROS 2 repo)
├── data/
│   ├── matches/             # Logged match data (pulled from Jetson)
│   └── sim/                 # Sim-generated training data
├── models/
│   ├── checkpoints/         # PyTorch training checkpoints
│   └── onnx/                # Exported ONNX models
└── deploy.sh                # Export + SCP + restart on Jetson
```

### Deployment Repo (`vex-ros2-ws/`)

```
vex-ros2-ws/
├── src/
│   ├── rl_decision/
│   │   ├── rl_decision/
│   │   │   └── inference_node.py   # ONNX inference + periodic re-plan
│   │   ├── models/
│   │   │   └── model.onnx          # Deployed model (copied from training repo)
│   │   ├── config/
│   │   │   └── config.yaml         # Shared action IDs + state schema
│   │   └── package.xml
│   ├── action_translator/
│   │   ├── action_translator/
│   │   │   └── translator_node.py  # Action ID → V5 serial commands
│   │   └── package.xml
│   ├── state_publisher/
│   │   ├── state_publisher/
│   │   │   └── state_node.py       # Sensor data → /game_state topic
│   │   └── package.xml
│   └── match_logger/
│       ├── match_logger/
│       │   └── logger_node.py      # Logs (s, a, r, s', done) to .npz
│       └── package.xml
└── launch/
    └── robot.launch.py             # Launches all nodes
```

---

## Key Design Decisions

1. **Granular actions over multi-step plans** — avoids partial completion ambiguity
2. **Role conditioning over separate models** — one model, two behaviors
3. **Periodic re-planning over event-driven** — handles all failure modes uniformly
4. **Field state as communication channel** — no direct robot-to-robot messaging needed
5. **Expected-state delta as input** — implicit teammate health signal
6. **2D Pygame sim over 3D** — sufficient for strategic decision-making, fast to iterate
7. **ONNX for deployment** — fast inference on Jetson, framework-agnostic
