# ONNX Action Masking — ROS 2 Deployment Contract

How to correctly use the exported policy (`model.onnx`) on the robot. The ONNX
graph emits **raw, unmasked action logits**; the ROS 2 side is responsible for
applying the action mask before choosing an action. This document is the
contract for doing that exactly the way the simulator does.

> **Why masking lives on the ROS 2 side (not in the graph):** the mask is
> computed from game state (goal beliefs, held-ball colors) that is *not* fully
> present in the policy's input vector, so it can't be a tensor op baked into the
> model. The export is deliberately just `state -> logits`; masking is applied
> outside. See `export/export_onnx.py`.

---

## 1. The ONNX I/O contract

| | Name | Shape | dtype | Meaning |
|---|---|---|---|---|
| **Input** | `state` | `(batch, STATE_DIM)` | `float32` | One **single robot's** observation vector. `STATE_DIM = 90` (heatmap off). Batch axis is dynamic; use `batch = 1` at inference. |
| **Output** | `action_logits` | `(batch, NUM_ACTIONS)` | `float32` | One raw logit per action. `NUM_ACTIONS = 9`. **Unmasked.** Higher = more preferred, but values are not probabilities (softmax them if you want probabilities). |

**Single-robot input.** The vector describes *one* robot (the one being
controlled). Even though training runs with an opponent on the field, the
opponent only changes the *world* (which balls exist, which goals are occupied,
what the camera sees) — it never widens the observation. On the robot you build
the same 90-dim vector from your own perception; you do **not** encode the
opponent as a second robot. The state-vector builder must reproduce
`env._build_obs()` exactly (same feature order, normalization, nearest-N
selection). That is a separate parity surface from masking and is out of scope
here.

**Choosing an action** = apply the mask (Section 3), then `argmax` over the 9
masked logits. This matches `MaskablePPO.predict(deterministic=True)`.

---

## 2. The action space (current: 9 actions)

Source of truth: `sim/config.py` (`class Action`, `NUM_ACTIONS`). Index = the
position in the `action_logits` output.

| Index | Action | Always valid? | Validity condition (when conditional) |
|---|---|---|---|
| 0 | `COLLECT_NEAREST_BALL` | ✅ always | — |
| 1 | `SCORE_LONG_GOAL` | ✅ always | (auto-chains to COLLECT first if the robot is empty) |
| 2 | `SCORE_CENTER_MID` | ✅ always | (auto-chains to COLLECT first if empty) |
| 3 | `SCORE_CENTER_LOW` | ✅ always | (auto-chains to COLLECT first if empty) |
| 4 | `DESCORE_OPP_LONG` | ⚠️ conditional | Valid only if the robot **believes** the opponent's long goal holds opponent-colored balls |
| 5 | `DESCORE_CENTER` | ⚠️ conditional | Valid only if the robot **believes** a center goal (mid or low) holds opponent-colored balls |
| 6 | `DEFEND_ZONE` | ✅ always | — |
| 7 | `EJECT_WRONG_COLOR` | ⚠️ conditional | Valid only if the robot is **currently holding** at least one wrong-color (non-blue) ball |
| 8 | `IDLE` | ✅ always | — |

Only indices **4, 5, 7** are ever masked. Indices 0,1,2,3,6,8 are always legal.

---

## 3. The masking rule (mirror of `env.action_masks()`)

`mask` is a boolean array of length `NUM_ACTIONS`; `True` = action is legal this
decision. Start all-`True`, then disable the three conditional actions:

```
mask = [True] * NUM_ACTIONS   # length 9

# --- index 4: DESCORE_OPP_LONG ---
# Legal only if the opponent's long goal is BELIEVED to hold opponent-colored
# (non-blue) balls. "Believed" = from the perceived goal state / FOV belief,
# NOT ground truth. If never seen, it stays unavailable until the camera sees it.
opp_in_long = any(color != BLUE for (_, color) in belief.opp_long)
if not opp_in_long:
    mask[4] = False

# --- index 5: DESCORE_CENTER ---
# Legal only if a center goal (mid OR low) is BELIEVED to hold opponent-colored balls.
opp_in_center = (
    any(color != BLUE for (_, color) in belief.center_mid) or
    any(color != BLUE for (_, color) in belief.center_low)
)
if not opp_in_center:
    mask[5] = False

# --- index 7: EJECT_WRONG_COLOR ---
# Legal only if the robot is currently holding >=1 non-blue ball.
wrong_held = any(ball.color != BLUE for ball in robot.held_balls)
if not wrong_held:
    mask[7] = False
```

> Our alliance color is **blue** in this sim. "Wrong color" / "opponent color" =
> any ball that is not blue. If you redefine alliance color on the robot, redefine
> these comparisons consistently.

### Applying the mask to the ONNX output

```
logits = onnx_session.run(["action_logits"], {"state": obs})[0][0]  # shape (9,)
masked = logits.copy()
masked[~mask] = -inf          # force illegal actions to negative infinity
action = int(argmax(masked))  # final discrete action id
```

Use `-inf` (or a large negative like `-1e9`) — **never** drop or reorder the
logits, because the index *is* the action id.

### Inputs the mask needs (and where they come from on the robot)

| Mask input | Sim source | ROS 2 source |
|---|---|---|
| Perceived contents of opponent long goal | `env.goal_belief.opp_long` | Perceived-goal tracker (updates only when the goal is in camera FOV; else keep last-seen) |
| Perceived contents of center goals | `env.goal_belief.center_mid` / `center_low` | Same tracker |
| Balls currently held + their colors | `robot.held_object_ids` -> colors | Intake / ball-color sensing |

The descore masks use **perceived belief**, not ground truth — exactly as the
robot will experience it. A stale belief may make the robot drive to descore and
find nothing; that's expected and matches sim behavior (the planner re-routes).

---

## 4. Inference loop (reference pseudocode)

```
# once
session = onnxruntime.InferenceSession("model.onnx")

# every decision (~3 s cadence, matching decision_interval)
obs    = build_state_vector()              # 90-dim float32, mirrors env._build_obs()
logits = session.run(["action_logits"], {"state": obs[None, :]})[0][0]
mask   = compute_action_mask(world_state)  # bool[9], mirrors env.action_masks()
logits[~mask] = float("-inf")
action = int(logits.argmax())
execute(action)                            # action-id -> motion/scoring controller
```

---

## 5. Parity check before trusting the robot

- [ ] `STATE_DIM` and `NUM_ACTIONS` on the robot match `shared/config.yaml`.
- [ ] **Golden mask test:** feed identical world state to the sim's
      `env.action_masks()` and the robot's `compute_action_mask()`; assert the two
      boolean arrays are identical across a range of scenarios (empty goals, opp
      balls in long/center, holding wrong-color, etc.).
- [ ] **Golden action test:** identical `state` + identical `mask` -> identical
      `argmax` action on robot vs. `MaskablePPO.predict(deterministic=True)`.
- [ ] Descore masks driven by **perceived belief** (FOV), not ground truth.

---

## 6. ⚠️ When the action space grows to ~20 actions

The plan is to expand the training env to ~20 actions, retrain, and re-export.
When that happens, **this contract changes and everything downstream must be
updated together.** Treat the following as a single versioned bump:

1. **`NUM_ACTIONS` changes** (9 -> ~20). The ONNX `action_logits` output width
   changes to match. Any hardcoded `9` on the robot must be replaced (read it
   from `shared/config.yaml` / the model, don't hardcode).
2. **The action table in Section 2 is replaced** — new indices, new meanings.
   Index numbers may shift; do not assume old ids keep their position.
3. **The mask rules in Section 3 expand** — each new conditional action needs its
   own validity rule added to both `env.action_masks()` (sim) and
   `compute_action_mask()` (robot). New rules may need new world-state inputs
   (extend the table in Section 3).
4. **`STATE_DIM` may also change** if the observation is extended for the new
   actions — if so, re-sync the state-vector builder too (separate from masking).
5. **Re-export and re-validate:** `main.py export` then `main.py validate` must
   pass parity again on the new model before deployment.
6. **Re-run the golden mask/action tests** in Section 5 against the new action set.

> Old ONNX files and old robot mask code are **incompatible** with a retrained
> ~20-action policy. Bump them together; never mix a new model with an old mask.
> `shared/config.yaml` is the single source of truth — if the robot and sim load
> the same file, action ids and counts can't silently drift.
