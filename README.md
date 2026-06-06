# leroiai-offSim

A reinforcement-learning **simulator and trainer** for a VEX AI **Push Back (2025–2026)** robot.
The agent learns *high-level strategy* — which objective to pursue — while deterministic
motion controllers handle *how* to drive, collect, and score. Policies are trained in the
sim, exported to ONNX, and intended for deployment on the real robot via ROS 2 (see
[Transferring the policy to ROS 2](#transferring-the-policy-to-ros-2)).

---

## Overview

- **Game:** VEX Push Back, 144" × 144" field, 44 balls (22 red / 22 blue), two long goals
  (left/right walls) and a center "X" structure with a mid goal (NE–SW bar) and low goal
  (NW–SE bar). Either color can be scored in any goal in this sim.
- **Control hierarchy:** the RL policy picks one of **9 discrete actions** every ~3 s
  ("collect", "score long", "score mid", …). The environment then executes that intent with
  deterministic planners — vision-cone route planning, obstacle-avoiding navigation, and a
  back-in scoring approach — over fine-grained 0.05 s ticks.
- **Partial observability:** the robot only *knows* what its camera FOV has seen. Goal
  contents are tracked as a **belief** that can go stale when an opponent changes a goal
  out of view — the policy never assumes a scored ball stays forever.
- **Algorithms:** online PPO with action masking (`sb3-contrib MaskablePPO`) and offline
  RL (`d3rlpy` CQL/IQL) from logged match data.
- **Visualization:** a Pygame renderer with an interactive field editor, live training
  panel, and a heatmap overlay.

---

## How it works

### Action space (9 discrete actions)
| ID | Action | Meaning |
|----|--------|---------|
| 0 | `COLLECT_NEAREST_BALL` | Drive to and intake the best reachable blue ball |
| 1 | `SCORE_LONG_GOAL` | Back into the nearest long goal and deposit |
| 2 | `SCORE_CENTER_MID` | Back into the mid (NE–SW) center goal |
| 3 | `SCORE_CENTER_LOW` | Back into the low (NW–SE) center goal |
| 4 | `DESCORE_OPP_LONG` | Remove balls from the opponent's long goal |
| 5 | `DESCORE_CENTER` | Remove opponent balls from the center goals |
| 6 | `DEFEND_ZONE` | Hold a defensive position |
| 7 | `EJECT_WRONG_COLOR` | Dump held wrong-color balls out the back |
| 8 | `IDLE` | Do nothing this decision |

Invalid actions are masked (e.g. descore is only offered for a goal the robot *believes*
holds opponent balls).

### Observation
A flat per-robot vector (`STATE_DIM = 90` with the heatmap off; 234 with it on):
- 13 scalars — role, alliance, clock, scores, pose `(x, y, sinθ, cosθ)`, balls held/nearby,
  perceived quadrant control.
- 68 relative features — 8 nearest blue + 4 nearest red balls `(dx, dy, dist, sinβ, cosβ)`
  and 4 goal `(dx, dy)`.
- 6 **perceived** goal-state features — fill level + opponent-ball count for each goal group.
- 3 extras — expected score delta, success ratio, wrong-color held.
- (optional) 12×12 potential heatmap.

Training and eval use a **single-agent wrapper** (`SingleAgentWrapper`): the policy controls
robot 0 with a `STATE_DIM` observation and a single `Discrete(9)` action; the teammate idles.
(To drive two robots from a policy, query it once per robot.)

### Decision cycle
Every RL decision runs for `decision_interval` (3 s) of simulated time, sub-stepped at
`dt` (0.05 s). Within a decision the environment may **adaptively re-plan** (e.g. chain
COLLECT → SCORE once full), commits to a scoring goal so it doesn't thrash, and **times
out** stuck actions.

### Notable behaviors
- **Back-in scoring** — the robot drives *rear-first* onto the scoring pose so its scoring
  face feeds the goal on arrival (no 180° spin at the mouth). Applies to long, mid, and low
  goals.
- **Creeping exploration** — when it cleans out an area and stops finding balls, an
  exploration frontier sweeps up and down the field so it expands into new territory (e.g.
  the top half) instead of oscillating in place.
- **Side-lane navigation** — corridor waypoints (including lanes between each long goal and
  the center X) let the planner route the robot between the bottom and top halves.
- **Perceived goal state (FOV belief)** — goal contents update only from the robots' camera
  cone (and their own scoring/descoring); unseen goals keep their last-seen contents.
- **Park-zone soft avoidance** — the planner prefers routes that don't drive through the
  top/bottom park platforms and deprioritizes balls sitting on them, but still crosses (or
  enters for an in-park ball) as a last resort rather than stranding the robot. Tunable via
  `_PARK_AVOID` / `_PARK_MARGIN` in `route_planner.py`.
- **Randomized layouts** — each run/episode jitters and permutes ball positions/colors.
- **Failure injection** — optional stuck/offline/steal events for sim-to-real robustness.

---

## Project layout

```
.
├── README.md
├── vex-rl-design-doc.md            # design notes
├── vex-rl-sim-and-trainer.md       # sim/trainer notes
└── offsim/
    ├── main.py                     # unified CLI (demo / train / eval / export / validate / train-offline)
    ├── requirements.txt
    ├── shared/
    │   └── config.yaml             # CONTRACT shared with the ROS 2 repo (action IDs, state shape, physics)
    ├── sim/
    │   ├── env.py                  # VexAIEnv — Gymnasium env, decision loop, masks, scoring, FOV belief
    │   ├── field.py                # field state, GoalState, balls, collisions, physics
    │   ├── robot.py                # tank-drive motion model (move_toward_point, back-in/reverse)
    │   ├── route_planner.py        # vision-cone collection routing, LOS, goal FOV checks
    │   ├── renderer.py             # Pygame visualization + interactive editor + training panel
    │   ├── opponent.py             # scripted opponents (random / greedy / defensive / mixed)
    │   ├── config.py               # constants derived from shared/config.yaml
    │   ├── failure.py              # failure injection
    │   ├── heatmap.py              # potential-field heatmap
    │   ├── decision_logger.py      # per-decision CSV logging
    │   └── game_object.py          # ball model
    ├── training/
    │   ├── train_sim.py            # online PPO (MaskablePPO) + curriculum
    │   ├── train_offline.py        # offline RL (CQL / IQL) from match data
    │   ├── reward.py               # reward shaping (ground-truth based)
    │   ├── curriculum.py           # curriculum stages
    │   └── callbacks.py            # eval / stats / checkpoint callbacks
    ├── export/
    │   ├── export_onnx.py          # SB3 policy → ONNX
    │   └── validate_onnx.py        # ONNX vs SB3 parity check
    ├── scripts/                    # train.sh / deploy.sh / visual_scoring_check.py
    ├── tests/                      # pytest (e.g. test_scoring.py)
    ├── data/                       # match data for offline RL
    └── models/                     # checkpoints, best, final_model.zip, onnx/, tb_logs/
```

---

## Installation

Requires Python 3.10+.

```bash
cd offsim
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build from the official PyTorch selector
(`pip install torch` may pull a CPU-only wheel).

---

## Usage

All commands run through `offsim/main.py`:

```bash
# Visual sim — greedy auto-play; randomized layout each run
python offsim/main.py demo
python offsim/main.py demo --num-robots 2 --opponent mixed

# Train (online PPO with action masking + curriculum)
python offsim/main.py train --timesteps 1000000 --n-envs 4
python offsim/main.py train --resume latest --timesteps 200000 --render

# Evaluate a trained policy
python offsim/main.py eval --model latest --episodes 10
python offsim/main.py eval --model latest --render --opponent defensive

# Offline RL from logged matches
python offsim/main.py train-offline --data data/matches/ --algo cql

# Export to ONNX and validate parity
python offsim/main.py export   --model models/final_model --output models/onnx/model.onnx
python offsim/main.py validate --onnx models/onnx/model.onnx --model models/final_model
```

**Demo controls:** `Space` pause · `S` step · `R` reset · `H` heatmap · `+/-` speed ·
`Tab`/`WASD` manual drive · `RClick` add ball · panel buttons for auto-play / force-score /
setup mode / training.

---

## Configuration

`offsim/shared/config.yaml` is the single source of truth for the **action IDs, observation
contract, field, and robot physics**. It is intentionally the file shared with the
deployment repo so the trained policy's inputs/outputs mean the same thing on the robot.
`sim/config.py` loads it and derives the rest (e.g. `STATE_DIM`, tick counts, scoring
geometry). Change a contract value here, retrain, and re-export.

---

## Transferring the policy to ROS 2

> **Status: planned — not implemented yet.** This section is a roadmap for moving a trained
> policy onto the physical robot under ROS 2. No deployment code lives in this repo yet.

The policy itself is small and portable (action selection from a fixed-length state vector).
The hard part of sim-to-real here is that **most of the behavior is deterministic logic that
currently lives in the simulator** — perception → state vector, route/scoring planners, and
the motion model. Those must be re-created (or ported) on the robot so the *same action ID*
produces the *same behavior* it did in training.

### What already bridges to the robot
- **`shared/config.yaml`** — the agreed contract (action IDs, state shape, physics). The
  ROS 2 repo should load the *same file* so both sides never drift.
- **ONNX export** (`export/export_onnx.py`) — produces `model.onnx` (`state` float32 →
  `action_logits`), to run on the robot with `onnxruntime` (CPU or Jetson GPU);
  `validate_onnx.py` checks ONNX↔PyTorch parity. ⚠️ The export script currently assumes a
  two-robot concatenated input (`STATE_DIM × 2`), while training uses the single-agent
  wrapper (`STATE_DIM`) — reconcile the export's input shape with your trained policy before
  deploying.
- **Partial observability** — the sim already restricts goal knowledge to the camera FOV
  (the perceived `goal_belief`), so the policy is trained to act on real perception rather
  than omniscient state. This is exactly what a real camera provides.

### What needs to be built on the ROS 2 side (proposed nodes)
1. **Perception → world state.** Fuse VEX AI camera / OAK-D detections + localization into
   the same world model the sim uses: ball positions+colors, robot pose, and a **perceived
   goal-contents tracker** mirroring `env.goal_belief` (update a goal only when it's in the
   camera FOV; otherwise keep last-seen).
2. **State-vector builder.** Reproduce `env._build_obs()` *exactly* — same feature order,
   same normalization constants, same nearest-N selection, same relative encodings. This is
   the highest-risk parity surface; pin it to the shared contract and add a golden-vector
   test (sim and robot must produce identical vectors for identical world inputs).
3. **Policy inference node.** Load `model.onnx`, run at the decision interval (~3 s),
   apply the **same action mask** logic as `env.action_masks()`, and `argmax` the logits.
4. **Action → motion executor.** Port the deterministic controllers the action selects
   among — `_action_to_target` (vision-cone collection routing, nav waypoints / corridors,
   scan/explore frontier) and `robot.move_toward_point` (drive + **back-in scoring**) — onto
   the real drivetrain (likely Nav2 or a custom controller). Note the real base is
   **mecanum**; the sim models tank drive, so the executor can exploit strafing.
5. **Scoring/intake interface.** Map the scoring/eject sequences to the real intake +
   alignment routines, including the back-in approach and dwell.

### Suggested package shape
```
ros2_ws/src/leroiai_deploy/
├── config/          # symlink or copy of shared/config.yaml
├── models/          # model.onnx
├── perception/      # detections + localization → world state + goal-belief tracker
├── state_builder/   # world state → STATE_DIM observation (mirrors env._build_obs)
├── policy/          # onnxruntime inference + action masking
├── executor/        # action ID → nav goals / intake / scoring controllers
└── bringup/         # launch files, params
```

### Parity checklist before trusting the robot
- [ ] ROS 2 loads the identical `shared/config.yaml`; action IDs and `STATE_DIM` match.
- [ ] ONNX input shape reconciled with the trained policy (single-agent `STATE_DIM` vs the
      export script's `STATE_DIM × 2`).
- [ ] Golden test: identical world inputs → identical state vectors (sim vs robot), within tol.
- [ ] ONNX argmax matches SB3 `predict(deterministic=True)` on recorded observations.
- [ ] Action masking on the robot matches `env.action_masks()`.
- [ ] Each action's executor reproduces the sim's intent (collect / score long-mid-low /
      descore / eject / defend / idle), including back-in alignment.
- [ ] Goal-belief tracker goes stale/updates from FOV the same way the sim does.
- [ ] Decision cadence and commit/timeout behavior match the sim's decision loop.

---

## Notes
- `MaskablePPO` policies require action masks at inference; `eval`/export account for this.
- Changing the observation layout changes `STATE_DIM` and invalidates old checkpoints — the
  loader detects the mismatch and asks you to retrain.
