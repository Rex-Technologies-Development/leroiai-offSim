# ONNX Action Masking — ROS 2 Deployment Contract

How to correctly use the exported policy (`model.onnx`) on the robot. The ONNX
graph emits **raw, unmasked action logits**; the ROS 2 side is responsible for
applying the action mask before choosing an action.

---

## 1. The ONNX I/O contract

| | Name | Shape | dtype | Meaning |
|---|---|---|---|---|
| **Input** | `state` | `(batch, STATE_DIM)` | `float32` | One robot's observation vector (`STATE_DIM = 90` with heatmap off). |
| **Output** | `action_logits` | `(batch, NUM_ACTIONS)` | `float32` | Raw logits. **`NUM_ACTIONS = 23`**. Unmasked. |

**Choosing an action** = apply the mask (Section 3), then `argmax` over masked logits.

Source of truth: [`sim/config.py`](../sim/config.py) (`Action`, `ACTION_NAMES`, `NUM_ACTIONS`) and [`shared/config.yaml`](../shared/config.yaml).

---

## 2. The action space (23 actions)

| Index | Tag |
|---|---|
| 0 | `STOP` |
| 1 | `MOVE_CHASSIS_ONLY` |
| 2 | `COLLECT_BLOCKS` |
| 3 | `SCORE_LEFT_LONG_GOAL_ALLIANCE` |
| 4 | `SCORE_LEFT_LONG_GOAL_OPPONENT` |
| 5 | `SCORE_RIGHT_LONG_GOAL_ALLIANCE` |
| 6 | `SCORE_RIGHT_LONG_GOAL_OPPONENT` |
| 7 | `SCORE_MID_LEFT` |
| 8 | `SCORE_MID_RIGHT` |
| 9 | `SCORE_LOW_LEFT` |
| 10 | `SCORE_LOW_RIGHT` |
| 11 | `DESCORE_LEFT_LONG_GOAL_ALLIANCE_FIELD` |
| 12 | `DESCORE_LEFT_LONG_GOAL_ALLIANCE_WALL` |
| 13 | `DESCORE_LEFT_LONG_GOAL_OPPONENT_FIELD` |
| 14 | `DESCORE_LEFT_LONG_GOAL_OPPONENT_WALL` |
| 15 | `DESCORE_RIGHT_LONG_GOAL_ALLIANCE_FIELD` |
| 16 | `DESCORE_RIGHT_LONG_GOAL_ALLIANCE_WALL` |
| 17 | `DESCORE_RIGHT_LONG_GOAL_OPPONENT_FIELD` |
| 18 | `DESCORE_RIGHT_LONG_GOAL_OPPONENT_WALL` |
| 19 | `RAM_RIGHT_LONG_GOAL_OPPONENT` |
| 20 | `RAM_RIGHT_LONG_GOAL_ALLIANCE` |
| 21 | `RAM_LEFT_LONG_GOAL_OPPONENT` |
| 22 | `RAM_LEFT_LONG_GOAL_ALLIANCE` |

Always valid: `STOP`, `MOVE_CHASSIS_ONLY`, `COLLECT_BLOCKS`.

Conditional:
- All eight `SCORE_*` actions (indices 3–10): masked when `balls_held == 0`.
- All eight `DESCORE_*` and four `RAM_*`: masked when `balls_held > 0`.
- All eight `DESCORE_*` (indices 11–18): also masked when the target long goal has no removable balls for that action (mirror `env._descore_available_for_mask()`).
- All four `RAM_*` (indices 19–22): also masked when the target long goal has no removable balls (mirror `env._ram_available_for_mask()`).

---

## 3. Masking rule (mirror of `env.action_masks()`)

```
mask = [True] * 23

if balls_held == 0:
    for i in range(3, 11):
        mask[i] = False

for each descore action a in 11..18:
    if not believed_descore_available(belief, a):
        mask[a] = False
```

Apply to ONNX output:

```
masked = logits.copy()
masked[~mask] = -inf
action = argmax(masked)
```

---

## 4. Version bump checklist

Old 9-action ONNX exports are **incompatible**. When retraining:

1. Update `NUM_ACTIONS` to 23 everywhere (robot + sim).
2. Reload action table from `shared/config.yaml`.
3. Re-implement mask rules above on the robot.
4. Re-export and run `main.py validate`.
