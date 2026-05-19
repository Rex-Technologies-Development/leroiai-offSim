# RL Training Stats Explanation

This document explains the main reinforcement learning training statistics shown in the PPO training log and what good target values may look like.

The exact meaning of each value depends on the environment, reward function, and training setup. These values should be judged as trends over time rather than from one training snapshot.

---

## Current Training Snapshot

```text
[ep 1200] reward=+3.68 blue=0.0 red=0.0 steps=40,592

rollout/
  ep_len_mean          31.5
  ep_rew_mean          14.9

time/
  fps                  2
  iterations           20
  time_elapsed         16213
  total_timesteps      40960

train/
  approx_kl            0.009942967
  clip_fraction        0.0878
  clip_range           0.2
  entropy_loss         -1.88
  explained_variance   0.023
  learning_rate        0.0003
  loss                 27.3
  n_updates            152
  policy_gradient_loss -0.00655
  value_loss           97.6
```

Overall interpretation:

```text
Policy update health: okay
Task reward: somewhat promising, but unclear without success/collision data
Critic learning: poor
Training speed: very slow
```

The PPO update statistics look stable, but the value function is not learning well yet. The training speed is also very slow at only 2 FPS.

---

## Rollout Statistics

### `ep_len_mean = 31.5`

This is the average episode length over recent episodes.

In this snapshot, each episode lasts about 31.5 steps on average.

This can be good or bad depending on what ends the episode.

A short episode is good if:

```text
The robot reaches the goal quickly.
```

A short episode is bad if:

```text
The robot crashes, gets stuck, or terminates early.
```

For robot path planning, this metric should be interpreted together with success rate, collision rate, timeout rate, and average distance to the goal.

**Good target:** episode length should match the desired behavior. Shorter is only good when the robot is successfully reaching the goal.

---

### `ep_rew_mean = 14.9`

This is the average reward per episode.

It is one of the most important high-level metrics, but the absolute number depends on the reward function.

For example, if the reward system is:

```text
+100 for reaching the goal
-100 for collision
-small penalty per step
```

Then an average reward of 14.9 is still relatively weak.

However, if the maximum reward is around 20, then 14.9 may be fairly good.

**Good target:** the average reward should steadily increase over time and eventually stabilize close to the best achievable reward.

For a robot path-planning task, reward should be tracked alongside:

```text
success rate
collision rate
timeout rate
average path length
average distance from goal
average time to goal
smoothness of movement
```

---

## PPO Training Stability Statistics

### `approx_kl = 0.009942967`

This measures how much the policy changed after an update.

In PPO, the policy should improve without changing too aggressively. If the policy changes too much in one update, training can become unstable.

Rough guide:

```text
0.001 to 0.02   usually healthy
> 0.03          policy may be changing too aggressively
near 0          learning may be too slow
```

**Current value:** good.

The policy is changing, but not too aggressively.

---

### `clip_fraction = 0.0878`

This tells us what fraction of PPO updates were clipped by the PPO safety mechanism.

PPO clipping prevents the policy from moving too far away from the previous policy during one update.

With `clip_range = 0.2`, a clip fraction around 0.05 to 0.2 is usually reasonable.

**Current value:** good.

If this value becomes too high, such as 0.4 or 0.5, the learning rate may be too aggressive or the policy updates may be unstable.

---

### `clip_range = 0.2`

This is the PPO clipping limit.

A value of `0.2` is a common default for PPO.

**Current value:** normal.

---

## Exploration Statistic

### `entropy_loss = -1.88`

Entropy measures how random or exploratory the policy is.

It is shown as a negative value because it is reported as part of the loss function.

A more negative entropy loss usually means the policy is more random and exploratory. As training progresses, the policy often becomes more confident, so the magnitude of the entropy loss may decrease.

For path planning:

```text
Early training: higher entropy is useful because the agent needs to explore.
Later training: lower entropy is expected because the agent should become more confident.
```

**Current value:** the agent is still exploring quite a bit.

This is not necessarily bad, especially if the model is still early in training.

---

## Critic and Value Function Statistics

### `explained_variance = 0.023`

This measures how well the value network predicts future reward.

The value network, also called the critic, estimates how good the current state is. PPO uses this critic to improve the policy more efficiently.

Rough guide:

```text
1.0      excellent value prediction
0.8+     good
0.3-0.7  okay or improving
~0       value function is barely useful
<0       value function is worse than guessing
```

**Current value:** poor.

An explained variance of 0.023 means the critic is barely predicting future rewards better than guessing.

This can happen if:

```text
The reward is too sparse.
The reward scale is too large or inconsistent.
The environment is too random.
Episodes are too short.
The observations do not contain enough useful information.
The value network is too small.
The model has not trained long enough.
```

**Good target:** ideally above 0.5, and eventually around 0.8 or higher if the environment is learnable and the reward is well-shaped.

---

### `value_loss = 97.6`

This is the value network's prediction error.

Lower is generally better, but the absolute value depends heavily on the reward scale.

Since the average episode reward is around 14.9, a value loss of 97.6 suggests that the critic may be struggling to predict returns accurately.

**Good target:** the value loss should generally trend downward over time. More importantly, `explained_variance` should increase.

---

### `policy_gradient_loss = -0.00655`

This is the policy update loss.

The exact value is not very intuitive on its own. It is mainly useful for detecting instability when combined with other metrics.

**Current value:** nothing obviously wrong.

There is usually no specific target value for this metric.

---

### `loss = 27.3`

This is the combined training loss.

It usually includes value loss, policy loss, entropy effects, and other PPO components.

This number should not be used by itself to judge whether the model is good.

**Good target:** generally stable or decreasing over time, but task performance matters more.

---

## Time and Training Speed Statistics

### `fps = 2`

This means the training loop is running at about 2 environment steps per second.

This is very slow for RL training.

At 2 FPS, collecting 1 million timesteps would take:

```text
1,000,000 / 2 = 500,000 seconds
500,000 seconds ≈ 139 hours ≈ 5.8 days
```

For a simple 2D robot path-planning simulation, the target should likely be much faster than this.

Potential causes of low FPS:

```text
Rendering every step
Using time.sleep() inside the environment
Expensive collision checking
Expensive plotting or visualization
Heavy physics simulation
Running only one environment instance
Debug printing too frequently
Inefficient Python loops
```

**Good target:** for a simple 2D simulation, hundreds or thousands of FPS would be much better. Even tens of FPS would be a major improvement over 2 FPS.

---

## What Looks Good

The PPO update health looks reasonable:

```text
approx_kl = 0.00994      good
clip_fraction = 0.0878   good
clip_range = 0.2         normal
policy_gradient_loss     fine
```

This suggests that the policy is not exploding or changing too aggressively.

---

## What Looks Concerning

The main concerns are:

```text
explained_variance = 0.023   very low
value_loss = 97.6            high
fps = 2                      very slow
```

This suggests that the agent may be learning some behavior, but the critic is not modeling the task well yet. Also, the training speed is slow enough that improvement may take a long time.

---

## Good Target Metrics for Robot Path Planning RL

| Metric | Good Target |
|---|---:|
| Success rate | 90%+ |
| Collision rate | < 1-5% |
| Timeout rate | < 5% |
| Average reward | steadily increasing, then stable |
| Path length | close to planned or optimal path |
| Smoothness | low sudden turning or oscillation |
| `approx_kl` | around 0.005-0.02 |
| `clip_fraction` | around 0.05-0.2 |
| `explained_variance` | ideally 0.5+, eventually 0.8+ |
| FPS | much higher than 2 for simple simulation |

---

## Suggested Next Steps

### 1. Improve Training Speed

Before tuning the RL model too much, improve the FPS.

Possible fixes:

```text
Disable rendering during training.
Only render during evaluation.
Remove sleep delays.
Reduce debug printing.
Optimize collision checking.
Use vectorized environments if possible.
Simplify the environment during early training.
```

### 2. Improve Reward Shaping

The low explained variance may mean the reward is too hard for the critic to predict.

For robot path planning, the reward should provide useful feedback at every step, not only at the end.

A useful reward structure could include:

```text
positive reward for reducing distance to the goal
large bonus for reaching the goal
large penalty for collision
small penalty per step
penalty for getting too close to obstacles
penalty for sharp or unstable turns
penalty for moving away from the goal
```

### 3. Track Real Task Metrics

Do not rely only on PPO logs.

For this project, it would be useful to track:

```text
success rate
collision rate
timeout rate
average reward
average path length
average steps per episode
average final distance to goal
average minimum obstacle distance
average turn smoothness
```

---

## Feedback: Need Better Training Progress Tracking

We should add a way to track how the RL model and training are progressing over time.

Right now, a single training log snapshot gives only a limited view. It shows the current PPO metrics, but it does not clearly show whether the model is improving across training.

A logging feature would help us monitor progress across episodes and compare different training runs.

The logging system should record metrics such as:

```text
mean episode reward over time
success rate over time
collision rate over time
timeout rate over time
average episode length over time
explained variance over time
value loss over time
approx KL over time
clip fraction over time
FPS over time
```

This would make it easier to answer questions like:

```text
Is the model actually improving?
Is it reaching the goal more often?
Is it avoiding obstacles better?
Is the reward function working?
Is the critic learning properly?
Did a hyperparameter change improve training?
```

A good implementation could save training logs to CSV, TensorBoard, or another dashboard. For debugging, CSV is simple and easy to inspect. For longer experiments, TensorBoard or Weights & Biases would make it easier to visualize the training curves.

The main goal is to make training progress visible, measurable, and comparable across runs.
