# RL VEX Robot Simulation Code Review

**Project reviewed:** VEX robot reinforcement learning simulation environment  
**Focus:** architecture, environment design, policy/action design, reward shaping, training setup, route planning, and deployment readiness.

---

## 1. High-Level Summary

The project has a solid starting architecture. It separates the simulation environment, robot motion model, field state, route planner, renderer, reward function, curriculum, and training scripts. That is a good structure for an RL simulation project.

However, the current setup is not really training a low-level robot controller. It is mostly training a **high-level action selector** on top of scripted skills such as collection, scoring, descoring, and defending. This is not a problem by itself, but the code should be treated as **hierarchical RL**:

- Scripted controllers handle motion and task execution.
- The RL policy chooses which high-level skill to run.
- Rewards should clearly reflect whether that selected skill helped the robot win.

Right now, there are several places where the action selected by the policy does not cleanly match what the robot actually does. This can confuse training and make the learned policy harder to interpret.

---

## 2. Most Important Problems to Fix First

### 2.1 `COLLECT_NEAREST_BALL` can route toward red balls

The robot intake logic is designed to collect blue balls and avoid red balls. That is good. However, the route planner still gives red balls a positive strategic score in some cases. This means the collect action may send the robot toward red balls even though the robot cannot actually collect them.

This creates a mismatch:

```text
Policy chooses COLLECT
Route planner selects a red ball
Robot drives to the red ball
Intake refuses to collect it
Reward may punish low progress or wasted action
```

This makes learning slower and noisier.

#### Recommendation

During blue-alliance training, red balls should be rejected immediately in the collect planner:

```python
if obj.color != BALL_BLUE:
    return -1.0
```

If you later want the robot to strategically interact with red balls, add separate high-level skills such as:

```text
BLOCK_RED_BALL
PUSH_RED_AWAY
CLEAR_OPP_BALL
```

Do not mix opponent-ball behavior into the normal collect-blue-ball action.

---

### 2.2 Score actions are allowed even when the robot has no balls

The code currently allows score actions even when the robot is empty. In that case, the environment may silently convert the score action into a collect action.

This is a serious action-semantics issue.

The policy thinks it selected:

```text
SCORE_LONG_GOAL
```

But the robot may actually execute:

```text
COLLECT_NEAREST_BALL
```

This means the same action has two different meanings depending on internal state.

#### Why this matters

The policy may learn that `SCORE_LONG_GOAL` is a general-purpose action that sometimes collects and sometimes scores. That makes the policy harder to debug and weakens the connection between action choice and reward.

#### Recommendation A: Clean RL design

Mask score actions when the robot has no balls:

```python
if robot.balls_held == 0:
    mask[int(Action.SCORE_LONG_GOAL)] = False
    mask[int(Action.SCORE_CENTER_MID)] = False
    mask[int(Action.SCORE_CENTER_LOW)] = False
```

Then the policy must learn:

```text
collect first → score later
```

#### Recommendation B: Macro-action design

If you intentionally want a score action to include collection when empty, rename the actions to make this explicit:

```text
COLLECT_OR_SCORE_LONG
COLLECT_OR_SCORE_CENTER_MID
COLLECT_OR_SCORE_CENTER_LOW
```

This would make the project a high-level macro-action RL system.

---

### 2.3 `SCORE_LONG_GOAL` should be deterministic

Right now, the long-goal scoring target is selected based on the nearest long-goal end. This means the same action can cause different paths and different scoring approaches depending on robot position.

For RL, that makes the action less predictable.

For your intended behavior, `SCORE_LONG_GOAL` should always execute a fixed routine:

```text
1. Go to a fixed staging point
2. Move to a fixed approach point
3. Rotate to the required heading
4. Feed balls into the goal
```

#### Recommendation

Do not use a nearest-target selector for `SCORE_LONG_GOAL` if you want consistent behavior.

Use a fixed target, such as:

```text
Right long goal bottom staging point
Right long goal bottom approach point
Required scoring heading
```

Or split long-goal scoring into multiple explicit actions:

```text
SCORE_LONG_TOP
SCORE_LONG_BOTTOM
```

This makes action meanings cleaner and improves policy learning.

---

### 2.4 Descoring can remove the wrong ball

The descoring logic checks whether the robot is near a goal, but it may remove the first opponent-scored ball from anywhere instead of removing a ball from the specific goal being descored.

This can create unrealistic reward and state updates.

Example failure:

```text
Robot is near opponent long goal
try_descore() removes a ball from center goal
Reward says descoring succeeded
Policy learns incorrect cause-effect relationship
```

#### Recommendation

Filter descoring candidates by the actual goal being targeted:

```python
target_gname = _goal_name(goal_pos)

candidates = [
    i for i, obj in enumerate(self.objects)
    if obj.status == OBJ_SCORED_OPP and obj.scored_in_goal == target_gname
]

if not candidates:
    return 0
```

Also make sure the point removal matches the goal type:

```text
long goal ball → subtract long-goal points
center goal ball → subtract center-goal points
```

---

### 2.5 Possible action-count mismatch

Some comments mention 10 actions, but the visible action list appears closer to 9 actions:

```text
COLLECT_NEAREST_BALL
SCORE_LONG_GOAL
SCORE_CENTER_MID
SCORE_CENTER_LOW
DESCORE_OPP_LONG
DESCORE_CENTER
DEFEND_ZONE
EJECT_WRONG_COLOR
IDLE
```

If `NUM_ACTIONS` does not match the actual `Action` enum, the policy may sample an unused or broken action.

#### Recommendation

Add this check during startup:

```python
assert NUM_ACTIONS == len(Action), f"NUM_ACTIONS={NUM_ACTIONS}, len(Action)={len(Action)}"
```

Also update comments and documentation so they match the real action space.

---

## 3. Action and Policy Design Concerns

### 3.1 The current policy is a high-level skill selector

The RL policy is not controlling wheel velocities, path curvature, intake motors, or scoring motors directly. Instead, it chooses symbolic actions, and the environment executes scripted behavior.

This means the project should be described as:

```text
Hierarchical RL with scripted low-level skills
```

Not:

```text
End-to-end RL robot control
```

This framing is important for documentation, debugging, and future deployment.

---

### 3.2 Avoid hidden action rewriting

The environment should avoid silently replacing one action with another. This makes debugging much harder.

Bad pattern:

```text
Policy picks SCORE
Environment changes it to COLLECT
Reward is still assigned after the original decision
```

Better pattern:

```text
Invalid actions are masked
Policy only chooses valid skills
Each action has one clear meaning
```

---

### 3.3 Split “strategy” from “execution”

The RL model should answer:

```text
What should I do next?
```

The scripted skill should answer:

```text
How do I physically do that?
```

Example:

```text
RL action: SCORE_LONG_GOAL
Skill controller: move to staging point, align, score
```

This separation is good. It just needs cleaner action definitions.

---

## 4. Reward Function Concerns

### 4.1 Repeated control reward may dominate training

The reward gives ongoing reward for controlled quadrants. This can encourage the robot to preserve control, but it can also reward passive behavior too much.

If the robot receives reward just for already holding a quadrant, it may learn to stop acting after gaining control.

#### Recommendation

Use stronger one-time rewards for gaining control and weaker repeated rewards for holding control:

```python
"ctrl_us": 0.02
"ctrl_gain": 2.0
```

The terminal match result should carry more of the final outcome signal.

---

### 4.2 Terminal margin reward may be too sharp

The margin reward uses a sigmoid-like curve, but the slope appears very steep. This can make the reward jump suddenly around a specific margin threshold.

That can make value prediction harder and may contribute to high value loss during PPO training.

#### Recommendation

Make the margin curve smoother:

```python
margin_bonus = w["margin_curve"] * sigmoid((diff - 12.0) / 4.0)
```

or increase the slope denominator from something very small to something like:

```python
_MARGIN_CURVE_SLOPE = 3.0
```

The goal is to make a 5-point, 10-point, 15-point, and 20-point lead gradually more valuable instead of creating a sudden cliff.

---

### 4.3 Collection reward may be too weak early in training

Collecting a blue ball gives a small reward compared to scoring and terminal bonuses. That is logically fine for final training, but early in training the agent may need a stronger signal to discover the collect-to-score chain.

#### Recommendation

During early curriculum stages, increase collection reward:

```python
"collect": 0.5
```

Later, reduce it once the robot reliably collects and scores.

This can be done with curriculum-based reward weights.

---

### 4.4 Wasted-action penalties should be tied to action validity

Wasted-action penalties are useful, but ideally the model should not be allowed to choose impossible actions in the first place.

Use action masks for impossible actions:

```text
Cannot score with no balls
Cannot eject if holding no wrong-color balls
Cannot descore if no opponent balls are actually in reachable goals
```

Then use wasted-action penalties only for strategically poor but technically valid choices.

---

### 4.5 Keep rewards aligned with game score

The safest reward structure is:

```text
small shaping reward for useful progress
larger reward for actual scoring/control changes
largest reward for final match result
```

Avoid giving too much reward for intermediate behavior that does not directly improve match outcome.

---

## 5. Observation Space Concerns

### 5.1 Flattened object observations are not permutation-invariant

The environment appears to flatten objects into a fixed vector. This can work, but it may be inefficient because object index order does not naturally correspond to strategic importance.

If object positions and colors are randomized each episode, object index 0 does not always mean the same kind of target.

#### Better observation design

Instead of giving the policy all objects in arbitrary order, consider giving:

```text
nearest 8 blue balls relative to robot
nearest 4 red balls relative to robot
nearest goals relative to robot
held ball count
score state
quadrant control state
time remaining
current skill status
```

For each nearby object:

```text
dx, dy, distance, bearing, color/status
```

This is easier for an MLP policy to learn.

---

### 5.2 Encode heading using sine and cosine

Do not use raw heading angle directly if possible. Raw angles have a discontinuity at `pi` and `-pi`.

Use:

```python
sin_heading = math.sin(robot.heading)
cos_heading = math.cos(robot.heading)
```

This helps the policy understand orientation smoothly.

---

### 5.3 Add observation shape assertions

The observation vector is manually constructed, so it is easy for `STATE_DIM` and the actual observation length to drift apart.

Add:

```python
obs = np.concatenate([...]).astype(np.float32)
assert obs.shape == (STATE_DIM,), f"obs shape {obs.shape}, expected {STATE_DIM}"
return obs
```

This catches silent bugs immediately.

---

## 6. Environment and Simulation Concerns

### 6.1 Decision interval is long

The RL agent acts every 3 seconds, while the simulation ticks at a faster rate. This is fine for high-level skill selection, but it means the policy cannot react quickly during a skill.

This supports the interpretation that the project is macro-action RL.

#### Recommendation

Keep the long decision interval if the action space is high-level.

Reduce the decision interval only if you want the policy to do more reactive control.

---

### 6.2 Movement model is scripted, not learned

The robot uses a scripted tank-drive move-to-point controller. That is reasonable, but the documentation should make this clear.

The learned part is strategy, not drivetrain control.

---

### 6.3 Mecanum drivetrain is not currently represented

The robot comments mention that the physical robot has a mecanum drivetrain, but the sim uses a tank-drive/differential-drive movement model.

This is acceptable for early strategy training, but it creates a sim-to-real gap.

#### Recommendation

If the real VEX robot uses mecanum, eventually add a movement model with:

```text
forward velocity
lateral velocity
angular velocity
```

This will better match real robot capabilities.

---

### 6.4 Collision and route planning are mostly scripted

The route planner uses line-of-sight and fixed detour/staging waypoints. That is good for reliability, but it means RL is not learning obstacle avoidance.

This should be documented clearly.

---

## 7. Route Planner Concerns

### 7.1 Fixed detours are reliable but may be brittle

The planner uses fixed corridor points around known obstacles. This is simple and good for a structured VEX field.

However, fixed corridors can become brittle if:

```text
opponents block a corridor
balls move into the corridor
the robot starts in an unusual location
field geometry changes
```

#### Recommendation

For now, fixed detours are okay. Later, consider adding a lightweight local planner that can re-route around moving opponent robots.

---

### 7.2 Separate collection routing from scoring routing

Collection routing and scoring routing should be treated differently.

Collection routing:

```text
target nearest valuable blue ball
avoid goal structures
avoid wall/corner traps
avoid opponent-heavy clusters
```

Scoring routing:

```text
go to fixed staging point
go to fixed approach point
align to required heading
score
```

Do not make scoring target selection too dynamic unless the RL policy explicitly selects the target.

---

### 7.3 Red-ball behavior should be explicit

If red balls matter strategically, add explicit actions.

Possible actions:

```text
CLEAR_RED_FROM_OUR_GOAL_AREA
BLOCK_RED_BALL
PUSH_RED_TO_OPP_SIDE
AVOID_RED_CLUSTER
```

But do not let `COLLECT_NEAREST_BALL` select red balls.

---

## 8. Training Setup Concerns

### 8.1 Do not silently fall back to standard PPO

The training script attempts to use `MaskablePPO`, but falls back to normal PPO if `sb3-contrib` is missing.

For this project, that is dangerous. Action masking is central to the design.

#### Recommendation

Hard fail if `sb3-contrib` is missing:

```python
try:
    from sb3_contrib import MaskablePPO
except ImportError as e:
    raise ImportError("Install sb3-contrib. This project requires MaskablePPO.") from e
```

This prevents accidentally training a broken or less interpretable policy.

---

### 8.2 Multi-env episode reward logging may be inaccurate

The callback appears to keep one current episode reward accumulator. With multiple parallel environments, that can mix rewards from different envs.

This affects logging, not training, but it makes TensorBoard and console stats less trustworthy.

#### Recommendation

Track one reward accumulator per environment:

```python
self._cur_ep_rewards = np.zeros(n_envs)

self._cur_ep_rewards += rewards

for i, done in enumerate(dones):
    if done:
        self._ep_rewards.append(self._cur_ep_rewards[i])
        self._cur_ep_rewards[i] = 0.0
```

---

### 8.3 Use evaluation environments

Training reward can be noisy and shaped. You should also evaluate the model in a separate deterministic environment.

Track metrics such as:

```text
average match score
average opponent score
win rate
average blue balls scored
average red balls accidentally scored
average quadrant control
average wasted actions
average time holding balls before scoring
```

This will tell you whether the robot is actually getting better at the game.

---

### 8.4 Save videos or deterministic rollouts

TensorBoard graphs are useful, but visual rollouts are even more important for robotics RL.

Save periodic evaluation rollouts:

```text
every 50k or 100k steps
fixed random seeds
same initial ball layouts
video or rendered trace
```

This makes it much easier to detect weird learned behavior.

---

## 9. Offline RL Concerns

### 9.1 Offline RL data quality matters more than algorithm choice

The offline training script supports CQL/IQL style training on recorded data. That is useful, but offline RL will only work well if the dataset contains good examples.

A weak dataset can teach the policy bad habits.

#### Recommended dataset contents

Include successful examples of:

```text
collecting blue balls
scoring long goal
scoring center goals
avoiding red balls
recovering from blocked paths
handling stolen/disappearing balls
defending or holding quadrant control
```

Also include unsuccessful examples if you want CQL/IQL to learn what not to do.

---

### 9.2 Make sure offline data uses the same observation/action format

Before training offline RL, verify:

```text
observation shape matches current STATE_DIM
actions match current Action enum values
reward scale matches current reward function
terminal flags are correct
```

If the action enum changes, older datasets may become invalid.

#### Recommendation

Store metadata with every dataset:

```json
{
  "action_enum_version": "v1",
  "state_dim": 123,
  "reward_version": "reward_2026_05_18",
  "field_config_version": "push_back_v1"
}
```

---

## 10. Curriculum Concerns

### 10.1 Current curriculum ramps failures, not task complexity

The curriculum mainly increases failure/noise rates. That is good for robustness, but it does not necessarily help the robot learn core skills in the right order.

#### Better curriculum structure

Start with:

```text
Stage 1: only blue balls, no red balls, no obstacles, score one goal
Stage 2: blue and red balls, still no opponents
Stage 3: normal field layout, no failures
Stage 4: add object stealing/disappearing
Stage 5: add opponent robots
Stage 6: full match-like randomization
```

This helps the policy learn basic task chains before dealing with randomness.

---

### 10.2 Consider skill-specific pretraining

You asked whether training one skill at a time is useful.

Yes, but only if the action/reward design supports it.

Possible progression:

```text
1. Train collect-only behavior
2. Train score-only behavior from states where robot already holds balls
3. Train collect-then-score behavior
4. Train full match strategy
```

However, if score actions automatically collect when empty, isolated training becomes less meaningful.

---

## 11. Value Loss Interpretation

A value loss around 200 to 400 is not automatically a sign that training is broken. PPO value loss depends heavily on reward scale.

High value loss can happen when:

```text
terminal rewards are large
reward jumps sharply around win/loss outcomes
episode returns vary a lot
curriculum changes the environment
actions have inconsistent effects
```

In your case, likely contributors are:

```text
large terminal win/loss rewards
sharp margin curve
repeated control rewards
score actions sometimes behaving like collect actions
route planner sometimes targeting uncollectable red balls
```

#### Recommendation

Do not judge only by value loss. Track:

```text
episode reward
win rate
blue score
red score
score margin
wasted-action rate
successful collect rate
successful score rate
```

If those metrics improve, value loss alone is not a major concern.

---

## 12. Suggested Refactor Plan

### Phase 1: Clean action semantics

Fix:

```text
red balls selected by collect planner
score actions allowed while empty
hidden action rewriting
possible action-count mismatch
descoring wrong goal
```

This is the highest-priority phase.

---

### Phase 2: Make scoring skills deterministic

Change `SCORE_LONG_GOAL` to a fixed staged routine:

```text
fixed staging point
fixed approach point
required heading
score
```

Optional:

```text
SCORE_LONG_TOP
SCORE_LONG_BOTTOM
SCORE_CENTER_MID
SCORE_CENTER_LOW
```

---

### Phase 3: Tune reward scale

Adjust:

```text
weaker repeated control reward
stronger early collect reward
smoother terminal margin reward
clearer gain/loss rewards for quadrant control
```

Keep final match outcome as the strongest signal.

---

### Phase 4: Improve observations

Add:

```text
relative nearest-ball features
sin/cos heading
skill progress state
time remaining
held count
goal/control state
```

Avoid relying only on arbitrary flattened object order.

---

### Phase 5: Improve evaluation

Add fixed-seed evaluation rollouts and metrics:

```text
win rate
average score
average score margin
blue balls scored
red balls scored
wasted actions
successful score attempts
successful collect attempts
```

---

## 13. Concrete Code Checks to Add

### 13.1 Action count check

```python
assert NUM_ACTIONS == len(Action), f"NUM_ACTIONS={NUM_ACTIONS}, len(Action)={len(Action)}"
```

### 13.2 Observation shape check

```python
obs = np.concatenate([...]).astype(np.float32)
assert obs.shape == (STATE_DIM,), f"obs shape {obs.shape}, expected {STATE_DIM}"
return obs
```

### 13.3 Hard require MaskablePPO

```python
try:
    from sb3_contrib import MaskablePPO
except ImportError as e:
    raise ImportError("Install sb3-contrib. This project requires MaskablePPO.") from e
```

### 13.4 Mask scoring when empty

```python
if robot.balls_held == 0:
    mask[int(Action.SCORE_LONG_GOAL)] = False
    mask[int(Action.SCORE_CENTER_MID)] = False
    mask[int(Action.SCORE_CENTER_LOW)] = False
```

### 13.5 Reject red balls in collect planner

```python
if obj.color != BALL_BLUE:
    return -1.0
```

### 13.6 Filter descoring by goal

```python
target_gname = _goal_name(goal_pos)

candidates = [
    i for i, obj in enumerate(self.objects)
    if obj.status == OBJ_SCORED_OPP and obj.scored_in_goal == target_gname
]

if not candidates:
    return 0
```

---

## 14. Recommended Final Architecture

The cleanest version of this project would be:

```text
RL Policy
  ↓
High-Level Skill Selection
  ↓
Scripted Skill Controller
  ↓
Path Planner / Motion Controller
  ↓
Simulated Robot + Field State
```

Example high-level actions:

```text
COLLECT_BLUE
SCORE_LONG_FIXED
SCORE_CENTER_MID
SCORE_CENTER_LOW
DESCORE_OPP_LONG
DEFEND_CONTROL_ZONE
EJECT_WRONG_COLOR
IDLE
```

The policy should not directly manage low-level robot motion unless you specifically want to train an end-to-end controller.

---

## 15. Final Priority List

### Must fix before serious training

1. Prevent collect planner from selecting red balls.
2. Stop score actions from secretly becoming collect actions, or rename them as macro-actions.
3. Make `SCORE_LONG_GOAL` deterministic if you want consistent behavior.
4. Fix descoring so it removes balls only from the targeted goal.
5. Verify `NUM_ACTIONS == len(Action)`.
6. Hard require `MaskablePPO`.

### Should fix soon

7. Smooth the terminal margin reward.
8. Reduce repeated control reward.
9. Improve collection reward during early curriculum.
10. Add observation shape checks.
11. Track per-env episode rewards correctly.
12. Add fixed-seed evaluation rollouts.

### Longer-term improvements

13. Add mecanum-style motion to reduce sim-to-real gap.
14. Add relative nearest-object observations.
15. Add moving-opponent-aware local replanning.
16. Add dataset metadata for offline RL.
17. Build a staged curriculum from simple skills to full matches.

---

## 16. Bottom-Line Assessment

The project is close to a useful hierarchical RL training environment, but the current action semantics and reward shaping need cleanup before long training runs are worth it.

The most important design principle is:

```text
One action should mean one thing.
```

Once each action has a clear meaning, the reward function becomes easier to tune, the policy becomes easier to debug, and the learned behavior becomes more likely to transfer to the real VEX robot.
