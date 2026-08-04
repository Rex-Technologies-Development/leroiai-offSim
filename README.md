# leroiai-offSim — Override 2D prototype

A deterministic, fast top-down simulator for experimenting with centralized autonomous strategy in a **provisional VEX AI Override profile**. It provides a low-level continuous-control Gymnasium environment, a high-level masked objective environment for `sb3-contrib` MaskablePPO, scripted 2v2 autoplay, and a Pygame renderer.

This repository is a pragmatic strategy prototype, **not an official rules engine**. It uses the VEX U Override field/setup concepts as a proxy where a final VEX AI format or subjective ruling is unavailable. The assumptions below are part of the simulator contract.

## Implemented match profile

- Four robots: two blue and two red; all four are autonomous and use the same globally selected `tank` or `mecanum` chassis model.
- 120-second match with no drivers: a 30-second opening/autonomous phase followed by a 90-second interaction phase.
- 144 × 144 inch nominal coordinate frame with a 6×6 tile grid. Official GPS coordinates are centered into this frame; the physical field's wall-to-wall dimension is slightly smaller.
- The white 2.5-inch tape is drawn as the official large diagonal X with paired Autonomous Line branches around the central Midfield diamond. Opening uses a coarse southwest/red versus northeast/blue diagonal-side proxy around this shared region.
- Nine GPS-positioned octagonal Goals: two red Alliance Goals in the southwest, two blue Alliance Goals in the northeast, four neutral Short Goals, and one neutral Tall Goal at center.
- Four 25.8-inch Toggles are mounted at the center of the field walls. Four Loaders and colored Load Zones occupy the corners, with two per alliance.
- The VEX U setup has 32 loose field Pins plus one yellow/yellow Pin in the Tall center Goal, 36 field Cups, and one alliance/yellow Pin Preload per Robot. Each alliance has 10 alliance/yellow plus 3 yellow/yellow Match Load Pins and 10 Match Load Cups. A Robot may possess at most one Pin and one Cup.

### Symbolic Goal and scoring proxy

Goal geometry is represented as an ordered bottom-to-top stack of Pin/Cup entries. A Cup placed directly above a Pin records `nested_on=<pin id>`. The simulator does not solve 3D contact geometry.

- A Pin has two colored halves (`blue`, `red`, or neutral `yellow`).
- **Placed halves** include both halves of every Pin in a Goal.
- **Visible halves** are the two halves of the highest Pin in that Goal. Cups preserve symbolic nesting but do not create a second Pin visibility layer.
- Every placed alliance-colored half scores **5 points** for that alliance.
- A placed yellow half in a perimeter Goal scores **10 points** for the alliance owning that Goal's wall/quadrant Toggle. Yellow halves in the Tall center Goal use the alliance with more Robots in Midfield; a tie has no owner.
- Toggles do not directly score. Each alliance Robot whose center is inside the Midfield diamond scores **8 points** for its own alliance.
- The opening winner is the alliance with the higher raw score at 30 seconds and receives the official **12-point Autonomous Bonus**; ties receive no bonus in this prototype.
- **AWP proxy:** by 30 seconds, the alliance must have placed at least one matching alliance half, own at least two Toggles, and both alliance Robots must have placed their Pin Preloads. AWP is telemetry only and does not add match points.

### Protected/removal proxy

A robot cannot remove a Pin from an opponent-protected alliance Goal. Cups and mixed/yellow Pins are treated as neutral placed objects and cannot be removed. The only supported removal is a top, fully alliance-colored Pin from a neutral or own Goal while the robot's Pin slot is empty. Prevented protected/neutral removals increment robot counters and append deterministic telemetry rather than asking for subjective referee input.

## Physics and drivetrains

### Field visualization and Goal-status key

The renderer uses the official Override GPS coordinates and Appendix A descriptions for Goal, Toggle, Loader, tile, tape, and Load Zone placement. The right panel includes a north-oriented compass and a live Goal key:

- **Base color** identifies a red/blue protected Alliance Goal or a dark neutral Goal.
- **T** marks the Tall center Goal; other dark Goals are neutral Short Goals.
- **Colored halo** identifies the alliance currently owning yellow Pin halves for that Goal.
- **Two pips** show the simulator's current visible Pin-half colors.
- **Center number** is the symbolic Pin/Cup stack-entry count; `G0`–`G8` match labels on the field.

References: [official Override manual](https://www.vexrobotics.com/override-manual) and [official VEX Override GPS coordinates](https://api.vex.com/vr/home/playgrounds/v5rc_override.html).

Simulation ticks are 0.05 seconds. Robots use circular top-down footprints with acceleration/deceleration, bounded yaw acceleration, walls, circular static Goal/Loader obstacles, and pairwise robot collision separation.

- **Tank:** normalized forward and yaw controls; lateral input is ignored.
- **Mecanum:** normalized body-frame forward, lateral, and yaw controls.
- High-level objectives use a deterministic chassis-aware point controller. This is intentionally simple and fast; it is not trajectory optimization.

## Gymnasium environments

Both classes live in `offsim/sim/env.py` and expose a centralized two-blue-robot observation of 160 floats (80 per robot). The observation includes normalized robot state, all robot relatives, Goal relative/visible ownership state, Toggle state, nearest loose Pin/Cup, score, inventories, phase, and time.

### `OverrideContinuousEnv`

- Action space: `Box(-1, 1, shape=(2, 6))`.
- Per robot: `[forward, lateral, yaw, pin_intake, cup_intake, interact]`.
- `pin_intake > 0.5` and `cup_intake > 0.5` attempt collection.
- `interact > 0.5` attempts nearby Goal placement, Toggle claim, then Loader use; `< -0.5` attempts a legal Pin removal.
- One Gym step advances one 0.05-second physics tick. Both red robots use scripted objectives.

### `OverrideStrategyEnv`

- Action space: `MultiDiscrete([10, 10])`, one objective for each allied robot.
- One Gym step executes both objectives deterministically for 2 simulated seconds while two scripted red robots act.
- `action_masks()` returns the flattened 20-entry boolean mask expected by MaskablePPO for the two branches.

| ID | Objective |
|---:|---|
| 0 | `IDLE` |
| 1 | `COLLECT_PIN` |
| 2 | `COLLECT_CUP` |
| 3 | `SCORE_NEAREST_GOAL` |
| 4 | `SCORE_ALLIANCE_GOAL` |
| 5 | `SCORE_MIDFIELD_GOAL` |
| 6 | `USE_LOADER` |
| 7 | `CLAIM_TOGGLE` |
| 8 | `DEFEND_MIDFIELD` |
| 9 | `REMOVE_OWN_PIN` |

Invalid sampled actions are defensively converted to `IDLE`; MaskablePPO normally prevents them from being sampled. Reward is `(blue score delta - red score delta) / 5`, plus `+10` for a terminal win or `-10` for a terminal loss.

## Installation

Python 3.10+ is recommended. No dependencies beyond the existing requirements are needed.

```bash
pip install -r offsim/requirements.txt
```

All commands below run from the repository root.

## Commands

```bash
# Visual deterministic 2v2 autoplay (1x is real-time)
python offsim/main.py demo --chassis tank
python offsim/main.py demo --chassis mecanum --opponent toggle
python offsim/main.py demo --chassis tank --speed 0.5  # slow motion

# Headless complete match / short smoke
python offsim/main.py demo --headless --matches 1 --seed 7
python offsim/main.py demo --headless --max-decisions 3

# Centralized MaskablePPO training
python offsim/main.py train --chassis tank --timesteps 1000000 --n-envs 4
python offsim/main.py train --chassis mecanum --timesteps 100000 --n-envs 1

# Evaluate, export, and validate a newly trained Override checkpoint
python offsim/main.py eval --model models/final_model.zip --episodes 5 --chassis tank
python offsim/main.py export --model models/final_model.zip --output models/onnx/override.onnx
python offsim/main.py validate --onnx models/onnx/override.onnx --model models/final_model.zip

# Tests
python -m pytest -q
```

Old Push Back checkpoints were removed and are incompatible with the 160-float observation and centralized two-branch action contract. ONNX export emits 20 unmasked logits; a consumer must split them into two 10-action branches and apply the corresponding mask before selection.

**Renderer controls:** `Space` pause/resume, `S` single strategy step while paused, `R` reset, `+/-` halve/double playback speed from `0.25x` through `16x`. Visual mode draws every 0.05-second physics tick; `1x` runs in real time, while headless demos and training remain uncapped. The current renderer intentionally has no field editor or manual driving mode.

## Project layout

```text
offsim/
├── main.py                 # Override CLI
├── shared/config.yaml      # action/profile/physics/scoring contract
├── sim/
│   ├── config.py           # strict shared-contract loader
│   ├── robot.py            # tank/mecanum kinematics
│   ├── field.py            # domain, stacks, scoring, collisions, phases
│   ├── env.py              # continuous and strategy Gym environments
│   ├── opponent.py         # scripted opponent objective selection
│   └── renderer.py         # Pygame renderer
├── training/
│   ├── train_sim.py        # MaskablePPO integration
│   ├── reward.py           # reward contract helper
│   └── callbacks.py        # score logging
├── export/                 # centralized actor ONNX export/parity validation
└── tests/                  # focused Override behavior and match tests
```

## Honest limitations

- Field element placement, tape, Goal colors/types, and VEX U starting counts follow Override Manual v1.0 and official GPS references. The 144-inch coordinate frame, dense cluster offsets, collision radii, opening-side enforcement, Goal protection behavior, and AWP condition remain explicit simulator proxies and must not be treated as referee-accurate.
- Pins/Cups are points or symbolic Goal entries; there is no rigid-body object physics, tipping, Pin rotation, Cup volume, stack stability, entanglement, or 3D occlusion.
- Subjective referee rules (possession nuance, incidental vs intentional contact, trapping, damage, match affecting violations, and field reset tolerances) are not adjudicated. Supported illegal removals are prevented and telemetered.
- Static navigation is a reactive point controller and can stall in congestion. Collisions are deterministic circular separation, not momentum/traction simulation.
- Scripted opponents provide full-match activity but are not competitive strategy benchmarks.
- Offline CQL/IQL and physical-robot/ROS deployment are not implemented for the centralized `MultiDiscrete` Override contract.
- The policy sees complete simulator state; perception noise, localization error, communications, actuator failures, and sim-to-real transfer are outside this prototype.
