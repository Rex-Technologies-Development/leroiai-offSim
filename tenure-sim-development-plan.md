# TENURE Simulation Environment Development Plan

**Audience.** Software engineer implementing the simulation and training stack.
**Prerequisite reading.** `tenure.tex` Sections III and V. This document implements that specification.
**Existing asset.** `leroiai-offSim` (VEX U Override 2D prototype, Gymnasium, sb3-contrib MaskablePPO, Pygame).
**Style note.** Symbols follow the paper. `τ_com`, `τ_rev`, `α = τ_com/τ_rev`, `β` is the contest parameter, `R_i` is retention.

---

## 0. Summary of what is being built

Two environments, one model, one baseline harness, one experiment runner.

| Component | What it is | Why it exists | New or extends |
| --- | --- | --- | --- |
| **A. Canonical env** | Batched tensor implementation of the paper's abstract dynamics | Sweeps α, β, N, M, K; where every headline claim is established | New package |
| **B. Override env** | Adversarial reversal made explicit in offSim | Shows the phenomenon is not an artifact of our own environment | Extends `offsim/` |
| **C. TENURE model** | Heterogeneous graph encoder, retention head, masked decoder | The contribution | New package |
| **D. Baselines** | Greedy, CBBA, defensive heuristic, MAPPO, QMIX, RTAW, CapAM | Comparison | New package |
| **E. Runner** | Sweep orchestration, logging, statistics | Reproducibility | New package |

**Non-goals for this phase.** Hardware deployment, perception noise, partial observability of the completion vector, ROS integration, offline RL. All are named in the paper's Limitations and are deliberately out of scope.

---

## 1. Architecture decision and rationale

### 1.1 Why two environments rather than one

offSim is the right applied environment and the wrong scientific environment, for reasons that are not defects.

- Override's reversal advantage is fixed by the rulebook. The paper's headline experiment is a continuous sweep over α. You cannot sweep α in a rules compliant match.
- Override's observation is a fixed 160 float vector for exactly two allied robots. TENURE's claims about generalization to unseen N, M, K require a variable size observation.
- Override runs one environment per process with a Pygame renderer and 0.05 s ticks. The α sweep needs thousands of parallel instances.
- Override's scoring couples pin stacking, cup nesting, toggle ownership, and midfield presence. Attributing a performance difference to retention estimation inside that coupling is very difficult.

So: establish claims in a clean environment, then demonstrate them in the real game. The paper states this division explicitly in Section V-A.

### 1.2 Why not VMAS

The paper previously specified VMAS. We are replacing it with a native batched implementation because the dynamics we need (dwell timers, contest rule, completion vector) are not a physics problem and gain nothing from a physics engine, and because a native implementation keeps the environment and the policy on the same device with the same tensor conventions. Keep VMAS in the bibliography as prior art on vectorized multi-robot simulation; do not depend on it.

### 1.3 Why dense attention rather than a graph library

At N, M, K ≤ 20 the graph is effectively complete. Dense masked attention over padded node sets is faster than PyTorch Geometric or DGL at this scale, has no additional dependency, keeps everything batched, and makes the permutation equivariance property directly testable. **Do not add a graph neural network library.**

---

## 2. Repository layout

Add to the existing repo as sibling packages. Do not modify `offsim/sim/env.py`'s existing classes.

```
leroiai-offSim/
├── offsim/                       # EXISTING, extended in Component B
│   ├── shared/config.yaml        # extend with a `contested:` block
│   ├── sim/
│   │   ├── field.py              # extend: dwell timers, reversal events
│   │   ├── opponent.py           # extend: 5 archetypes
│   │   ├── env.py                # ADD OverrideGraphEnv, leave others alone
│   │   └── telemetry.py          # NEW
│   └── tests/
├── contested/                    # NEW, Component A
│   ├── config.py                 # strict schema loader, reuse offsim pattern
│   ├── core.py                   # batched dynamics
│   ├── reference.py              # slow single-env reference for tests
│   ├── observation.py            # graph observation builder
│   ├── actions.py                # masking, greedy matching
│   ├── adversaries.py            # 5 archetypes
│   ├── labels.py                 # retention label extraction
│   ├── telemetry.py              # metric accumulation
│   ├── adapters/
│   │   ├── gym_vec.py            # Gymnasium vector API
│   │   └── pettingzoo_par.py     # ParallelEnv for BenchMARL baselines
│   └── tests/
├── tenure/                       # NEW, Component C
│   ├── encoder.py
│   ├── heads.py
│   ├── policy.py
│   ├── ppo.py
│   └── tests/
├── baselines/                    # NEW, Component D
│   ├── greedy.py
│   ├── cbba.py
│   ├── defensive_heuristic.py
│   ├── capam_wrapped.py
│   ├── rtaw_reimpl.py
│   └── marl/                     # BenchMARL configs
└── experiments/                  # NEW, Component E
    ├── sweep.py
    ├── configs/
    └── analysis/
```

---

## 3. Component A: canonical environment

### 3.1 Configuration schema

Single YAML, strict loader that rejects unknown keys (follow the existing `offsim/sim/config.py` pattern).

```yaml
canonical:
  # episode
  dt: 0.05                  # physics tick, seconds. Matches offSim.
  decision_dt: 0.5          # allocator cadence, seconds. Must be a multiple of dt.
  horizon_T: 120.0          # seconds. Matches an Override match.

  # counts
  n_robots: 4
  n_tasks: 12
  n_adversaries: 4
  max_robots: 8             # padding bound for the graph
  max_tasks: 20
  max_adversaries: 8

  # field
  field_size: [3.66, 3.66]  # metres. 144 inches, matches Override.
  service_radius: 0.20      # metres

  # dynamics
  v_max: 1.5                # m/s
  a_max: 3.0                # m/s^2
  adv_v_max: 1.5
  adv_a_max: 3.0

  # the contested mechanic
  tau_com: 2.0              # seconds
  alpha: 1.0                # tau_rev is DERIVED: tau_rev = tau_com / alpha
  beta: 1.0                 # contest parameter, see paper Section III-B
  contest_mode: majority    # majority | suppress | none

  # task layout
  weight_dist: lognormal    # uniform | lognormal | bimodal
  weight_range: [1.0, 5.0]
  layout: clustered         # uniform | clustered | polarized
  n_clusters: 3

  # allocation
  allow_multi_defend: true
  allow_multi_acquire: false

  # adversaries
  adversary_population: [greedy_nearest, value_targeting, feinter]
  holdout_population: [camper, learned_selfplay]

  seed: 0
```

**Critical:** `alpha` is the knob; `tau_rev` is derived. `alpha: 0.0` must map to `tau_rev = inf` and must be handled without division by zero. Add an explicit branch, not a large finite number.

### 3.2 State tensors

All state lives in one dataclass of tensors on one device. `B` is the batch of parallel environments.

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `robot_pos` | `(B, N, 2)` | f32 | metres |
| `robot_vel` | `(B, N, 2)` | f32 | m/s |
| `robot_action` | `(B, N)` | i64 | index into action layout, §3.5 |
| `adv_pos` | `(B, K, 2)` | f32 | |
| `adv_vel` | `(B, K, 2)` | f32 | |
| `adv_target` | `(B, K)` | i64 | task index or -1 |
| `adv_archetype` | `(B, K)` | i64 | |
| `task_pos` | `(B, M, 2)` | f32 | fixed within an episode |
| `task_w` | `(B, M)` | f32 | |
| `task_c` | `(B, M)` | bool | the completion vector |
| `sigma` | `(B, M)` | f32 | service timer |
| `eta` | `(B, M)` | f32 | reversal timer |
| `t_since_change` | `(B, M)` | f32 | |
| `t` | `(B,)` | f32 | elapsed seconds |
| `held_integral` | `(B,)` | f32 | running Σ w_i ∫ c_i dt, for J_H |

Masks `robot_valid (B,N)`, `task_valid (B,M)`, `adv_valid (B,K)` support padding to `max_*` so a single set of model weights runs across configurations.

### 3.3 Physics tick

One `tick()` call advances `dt`. Order of operations matters and must be fixed and documented, because timer semantics depend on it.

```
1. integrate motion       (both teams, accel-clipped then velocity-clipped, then walls)
2. compute occupancy      n_team[b,i], n_adv[b,i] from distances < service_radius
3. update service timers  sigma += dt where (c==0 & n_team>=1), else sigma = 0
4. update reversal timers phi = n_adv - beta*n_team
                          eta += dt where (c==1 & phi>0), else eta = 0
5. apply transitions      c: 0->1 where sigma >= tau_com  (then sigma = 0)
                          c: 1->0 where eta   >= tau_rev  (then eta = 0)
6. accumulate             held_integral += (task_w * c).sum(-1) * dt
7. advance clock          t += dt
```

Motion model is first order with acceleration limit, not a full rigid body. Robots are points with a collision radius used only for wall clamping. **Do not add inter-robot collision to the canonical environment.** It is not in the formalism and it would silently make `intercept` meaningful, which the paper explicitly removes.

`contest_mode` controls step 4:

| Mode | `eta` accumulates when | Meaning |
| --- | --- | --- |
| `majority` (default) | `n_adv > beta * n_team` | defenders cancel attackers one for one |
| `suppress` | `n_adv >= 1 and n_team == 0` | any single defender holds indefinitely |
| `none` | `n_adv >= 1` | defenders have no effect |

`none` and `suppress` are the degenerate limits analyzed in the paper. They exist for experiment E6 and must both be reachable from config.

### 3.4 Decision step

`step(actions)` runs `decision_dt / dt` ticks (default 10) with the assignment held fixed, then returns a new observation. Between decisions, each robot runs a proportional navigation law toward its assigned target position, clipped by `a_max` and `v_max`. Keep this controller trivial and in one function; the paper commits to that.

### 3.5 Action layout and masking

Flat index layout, fixed forever:

```
0        .. M-1     ACQUIRE task i
M        .. 2M-1    DEFEND  task (i - M)
2M                  IDLE
```
Total `A = 2M + 1`.

Mask rules, returned as `(B, N, A)` bool:

- `ACQUIRE i` valid iff `task_valid[i] and not task_c[i]`
- `DEFEND i` valid iff `task_valid[i] and task_c[i]`
- `IDLE` always valid
- rows for invalid robots are all False except IDLE

An `INTERCEPT k` block may be appended after `IDLE` behind a config flag, for the canonical-only ablation named in the paper. Default off.

### 3.6 Greedy conflict resolution

Input `scores (B, N, A)`, output `assignment (B, N)`.

Vectorize as `N` sequential argmax picks rather than a sort, since `N ≤ 8`:

```
for _ in range(N):
    masked = scores.masked_fill(~available, -inf)          # (B, N, A)
    flat_idx = masked.view(B, -1).argmax(-1)               # (B,)
    robot, action = divmod(flat_idx, A)
    assign[b, robot] = action
    available[b, robot, :] = False                          # robot consumed
    if action is ACQUIRE and not allow_multi_acquire:
        available[b, :, action] = False                     # target consumed
    if action is DEFEND and not allow_multi_defend:
        available[b, :, action] = False
```

Note `IDLE` is never consumed. Add an autoregressive variant behind a flag for the ablation.

### 3.7 Observation

Return a dict of tensors, never a flat vector. Flattening is what makes the existing Override env unable to generalize across sizes.

| Key | Shape | Contents |
| --- | --- | --- |
| `robot_feat` | `(B, N, 9)` | pos/field_size (2), vel/v_max (2), has_assignment (1), assignment type one-hot (3), normalized distance to current target (1) |
| `task_feat` | `(B, M, 7)` | pos/field_size (2), w/w_max (1), c (1), sigma/tau_com (1), eta/tau_rev (1), t_since_change/T (1) |
| `adv_feat` | `(B, K, 6)` | pos/field_size (2), vel/v_max (2), speed (1), valid (1) |
| `e_rt` | `(B, N, M, 2)` | travel time / T, horizon factor `(T - t_hat_ji)/T` from paper Eq. 11 |
| `e_at` | `(B, K, M, 2)` | threat time `(d(y_k,p_i)/v_adv + tau_rev)/T`, closing indicator |
| `e_tt` | `(B, M, M, 1)` | `exp(-d(p_i,p_j)/service_radius_scale)` |
| `scalar` | `(B, 4)` | t/T, fraction complete, held_integral normalized, alpha |
| `action_mask` | `(B, N, A)` | §3.5 |
| `*_valid` | `(B, N)` etc. | padding masks |

`e_rt`'s second channel is deliberately the exact horizon factor from the paper. Precomputing it in the environment means the decoder does not have to rediscover Eq. 12 from raw distances.

### 3.8 Reward

Dense, and equal to the normalized held value increment so that the undiscounted return of an episode equals `J_H` normalized to `[0,1]`:

```
r_t = (task_w * task_c).sum(-1) * dt / (horizon_T * task_w.sum(-1))
```

Report `J_T` as telemetry only, never as reward, unless running the terminal-objective ablation.

### 3.9 Retention labels

This is the subtlest part of the build. Read carefully.

**Definition to implement.** For any pair `(decision step d, task i)` with `c_i = 1` at that step, the label is the realized held fraction over the remaining episode:

```
R_label[d, i] = sum_{u = t_d}^{T} c_i(u) * dt / (T - t_d)
```

**Compute it by reverse scan.** After a full episode, walk backwards accumulating `c_i` to get `∫_t^T c_i du` for every `t` in one pass. Do not recompute per label.

**Two label sources, both needed:**

1. **Dense.** Every `(d, i)` with `c_i = 1`. Cheap and plentiful.
2. **Completion events.** For each `0 -> 1` transition, additionally record the decision step at which the `ACQUIRE` that caused it was issued, and attach the same forward-looking label. These are the only labels that teach the head about tasks that are *not yet* complete.

**Why source 2 matters.** The score in Eq. 10 applies `R̂_i` to `ACQUIRE` candidates, which are incomplete at decision time. If the head is trained only on complete tasks it is extrapolating at exactly the point where it is used. Record both, and report estimator error separately for the two regimes in E7. If the gap is large, that is a finding, not a bug.

**Non-stationarity.** Labels are realized under the current behavior policy. As the policy improves, true retention changes. **Discard the label buffer at every PPO update.** Never accumulate across updates.

**Truncation.** Labels require the full remainder of the episode. Collect whole episodes per rollout. If you must truncate, drop labels from the truncated tail rather than bootstrapping them.

### 3.10 Adversary archetypes

Five, all scripted except the last, all implemented batched, all selectable per environment instance so a batch can contain a mixture.

| Name | Target selection | Notes |
| --- | --- | --- |
| `greedy_nearest` | nearest task with `c_i = 1` | baseline |
| `value_targeting` | `argmax w_i / (d + eps)` over `c_i = 1` | punishes high-value undefended work |
| `camper` | picks a spatial cluster at reset, cycles within it | spatially concentrated; behaviourally distinct |
| `feinter` | commits, then re-samples target with prob `p` when within `d_switch` | breaks naive threat-time extrapolation |
| `learned_selfplay` | PPO policy trained against a frozen scripted defender | strategically distinct |

**Holdout choice.** Default holdout is `camper` and `learned_selfplay`. Rationale: one is distinct in spatial distribution, the other in strategy, so the holdout tests two different kinds of generalization. Make it config driven; the paper reports whichever is used.

All archetypes must be deterministic given seed.

### 3.11 Telemetry

Accumulate per episode, return in `info` on termination:

`J_H`, `J_T`, `retention_rate`, `defense_fraction`, `n_reversals`, `n_recompletions`, `assignment_churn` (reassignments per robot per minute), `decision_latency_ms`, `mean_R_error`, `R_calibration_bins`, `time_to_first_completion`, `per_task_held_fraction`.

`retention_rate` definition, to avoid ambiguity: `Σ_i ∫_{t_i^first}^{T} c_i dt / Σ_i (T - t_i^first)` over tasks completed at least once.

### 3.12 Adapters

Core is framework agnostic. Two thin wrappers:

- `adapters/gym_vec.py`: Gymnasium vector API, dict observation space, for TENURE's own PPO and for SB3 comparisons.
- `adapters/pettingzoo_par.py`: `ParallelEnv` for BenchMARL, which is how MAPPO, QMIX, and MASAC are run.

Both must be pure views over the same core state. No duplicated dynamics.

### 3.13 Determinism and testing

`reference.py` is a slow, obvious, single-environment NumPy implementation written independently from the batched core. It exists only to test. **This is the highest-value test in the project.**

Required tests:

| Test | Asserts |
| --- | --- |
| `test_batched_matches_reference` | 500 random seeds, batched core and `reference.py` produce identical trajectories |
| `test_determinism` | same seed gives bit-identical rollout across two runs and across `B` values |
| `test_alpha_zero_is_monotone` | at `alpha = 0`, no `1 -> 0` transition ever occurs |
| `test_timer_reset` | leaving the service region resets `sigma` and `eta` to zero |
| `test_contest_modes` | each of the three modes produces the documented accumulation |
| `test_mask_correctness` | never `ACQUIRE` a complete task, never `DEFEND` an incomplete one |
| `test_greedy_conflict_free` | no two robots share an `ACQUIRE` target when `allow_multi_acquire` is false |
| `test_permutation_equivariance` | permuting robot indices permutes the assignment identically |
| `test_retention_label_handbuilt` | hand-constructed `c_i` trajectories give known labels |
| `test_padding_invariance` | padding to `max_*` does not change results |

Throughput target: **a full E1 cell (one method, one α, one seed) must train in under two GPU-hours.** If it does not, profile before proceeding. Suggested starting point `B = 2048`.

---

## 4. Component B: Override environment extension

### 4.1 Verify before building

These cannot be determined from the README and must be checked in code first. Report findings before writing anything in this component.

1. **Toggle claim.** Is `CLAIM_TOGGLE` instantaneous or does it require dwell? If instantaneous, `τ_rev` for the territory channel is zero and α is unbounded. It must become dwell-based.
2. **Opponent removal.** The README says the only supported removal is a top, fully alliance-coloured Pin from a neutral or own Goal. Determine in `field.py` whether an opponent robot may execute this against a pin the other alliance placed, or whether `REMOVE_OWN_PIN` is the only exposed path. This decides how much work channel two needs.
3. **Match Load replenishment.** Do Loaders refill? This decides whether the effective task set is fixed or growing.
4. **Midfield geometry.** How is "centre inside the Midfield diamond" computed, and is the 8-point award applied per tick or per second?
5. **Reward semantics.** Is the existing `(blue delta - red delta)/5` reward already effectively held-value, or purely terminal?

### 4.2 Mapping to the formalism

| Paper concept | Override realization |
| --- | --- |
| Task site `i` | A goal, or a wall toggle |
| `c_i = 1` | Alliance-credited value present at that site |
| `τ_com` | Travel plus scoring dwell, or toggle claim dwell |
| `τ_rev` | Opponent removal dwell, or opponent toggle claim dwell |
| **Reversal channel 1, territory** | Opponent claims a wall toggle, flipping credit for the neutral halves in that quadrant. **Value moves, objects do not.** |
| **Reversal channel 2, object** | Opponent removes a top scored pin from an unprotected goal |
| `DEFEND` | Occupying the midfield region, or holding position at a toggle or contested goal |
| Contest rule at `β = 1` | Tall-goal neutral ownership decided by which alliance has more robots in midfield |
| Protected sites | Alliance goals, which cannot be reversed. These give `R_i ≈ 1` and create the spatial variation the retention head must learn |

Channel one is the cleanest instance of the formalism in the whole project. Prioritize it.

### 4.3 Changes to `shared/config.yaml`

```yaml
contested:
  enabled: false            # default off so existing behaviour is untouched
  objective_set: v1         # v1 = shipped 10 objectives, v2 = extended
  toggle_claim_dwell: 1.0   # seconds
  pin_removal_dwell: 1.5    # seconds
  enable_opponent_removal: true
  alpha_scale: 1.0          # multiplies both dwell times, for the sensitivity study
  contest_mode: majority
  beta: 1.0
  opponent_archetype: greedy_nearest
```

`alpha_scale: 1.0` is the rules operating point of record. Any other value is explicitly non rules compliant and must be labelled as such in output metadata.

### 4.4 Changes to `sim/field.py`

- Add dwell timers for toggle claim and pin removal, mirroring the `sigma` / `eta` semantics of §3.3 including the reset-on-leave rule.
- Add a held-value accumulator per alliance so `J_H` is directly measurable.
- Emit structured reversal events `{t, channel, site_id, from_alliance, to_alliance, value_delta}`.
- Gate all of the above behind `contested.enabled` so existing tests and checkpoints are unaffected.

### 4.5 Objective set v2

Extend `MultiDiscrete([10, 10])` to `[13, 13]` **only when `objective_set: v2`**:

| ID | Objective |
| --- | --- |
| 10 | `DEFEND_GOAL` |
| 11 | `DEFEND_TOGGLE` |
| 12 | `REMOVE_OPPONENT_PIN` |

**Compatibility warning.** This changes the ONNX export from 20 logits to 26 and invalidates existing checkpoints. That is why it is behind a flag. `v1` must remain the default and must remain byte-compatible with `models/final_model.zip`.

### 4.6 `OverrideGraphEnv`

New class in `sim/env.py`. Leave `OverrideContinuousEnv` and `OverrideStrategyEnv` untouched.

- Dict observation matching §3.7's schema, with Override-specific task features appended (goal type, protected flag, stack height, current credited alliance).
- Task nodes: 9 goals plus 4 toggles = 13 sites. Adversary nodes: 2 opposing robots.
- Same flat action layout as §3.5 so TENURE's decoder is shared unchanged between environments.

### 4.7 Opponent archetypes

Port the five archetypes from §3.10 into `sim/opponent.py`, expressed over Override objectives. The `camper` maps naturally onto a robot that parks in midfield and cycles nearby toggles.

### 4.8 The free baseline

The shipped MaskablePPO policy already has `DEFEND_MIDFIELD` in its action space but has no retention estimate and no threat geometry in its observation. **It is a naturally occurring instance of the "defend action without retention head" ablation.** Do not build a synthetic version of that ablation for Override. Retrain the shipped policy under `contested.enabled: true` with `objective_set: v1` and report it directly. This is called out in the paper's baselines section.

---

## 5. Component C: TENURE model

```python
class HeteroEncoder(nn.Module):
    """Type-specific multi-head attention over padded, masked node sets."""
    def forward(self, obs: dict) -> tuple[Tensor, Tensor, Tensor]:
        # returns h_robot (B,N,D), h_task (B,M,D), h_adv (B,K,D)

class RetentionHead(nn.Module):
    """Predicts R_hat per task from task embedding + pooled adversary context."""
    def forward(self, h_task, h_adv, adv_mask) -> Tensor:  # (B, M) in [0,1]

class Decoder(nn.Module):
    """Scores every (robot, action) pair. Retention enters MULTIPLICATIVELY."""
    def forward(self, h_robot, h_task, R_hat, e_rt, mask) -> Tensor:  # (B, N, A)
```

**Design requirement on the decoder.** Retention must enter the `ACQUIRE` and `DEFEND` scores as an explicit multiplicative factor on `w_i`, structured after paper Eq. 10, not merely as an extra input feature concatenated into an MLP. Two reasons. First, it makes the ablation causal rather than correlational; if `R̂` only shapes shared embeddings through an auxiliary loss, the encoder can learn the same geometry implicitly and the ablation will under-report the effect. Second, it makes `R̂ ≡ 1` an exact recovery of the conventional score, which is the paper's central rhetorical move and should be literally true in code. Provide a `retention_mode: multiplicative | feature | off` switch so all three can be compared.

Loss follows paper Eq. 14. Suggested starting weights `λ_r = 0.5`, `λ_v = 0.5`, `λ_e = 0.01`. Tune `λ_r` first; it is the one that matters.

Tests: shape and mask propagation; `retention_mode: off` reproduces a conventional allocator; permutation equivariance end to end; ONNX export parity following the existing `export/` pattern.

---

## 6. Component D: baselines

| Baseline | Source | Effort | Risk |
| --- | --- | --- | --- |
| Greedy | write | 0.5 d | none |
| Defensive heuristic | write | 1 d | none |
| CBBA | write from Choi et al. 2009 | 3 d | low, well specified |
| MAPPO, QMIX, MASAC | BenchMARL via PettingZoo adapter | 3 d | integration only |
| CapAM | public repo, wrap as receding-horizon replanner | 4 d | moderate |
| RTAW | **verify public code exists** | 5-8 d | **high, may need reimplementation** |
| DC-MRTA | **verify public code exists** | 5-8 d | **high, may need reimplementation** |

**Action item.** Confirm code availability for RTAW and DC-MRTA in week one. If unavailable, budget reimplementation and mark those rows with a dagger in the results tables, as the paper's tables already provide for.

Fair adaptation, per the paper, is non-negotiable and must be enforced in the harness rather than left to per-baseline scripts: every baseline receives adversary features, every baseline is retrained from scratch in the contested environment, sample budget and wall clock are matched and logged.

---

## 7. Component E: experiment runner

- Config-driven sweeps producing a cartesian product over `{method} × {α} × {seed}`, five seeds minimum.
- One run directory per cell, containing resolved config, git SHA, metric time series, final checkpoint, and a metadata flag recording whether the cell is rules compliant.
- Aggregation and confidence intervals following the BenchMARL protocol, since the paper cites it for statistical reporting.
- A single command must regenerate every figure in the paper from run directories.

Experiment to config mapping: E1 sweeps `alpha`, E2 sweeps `n_adversaries/n_robots`, E3 evaluates checkpoints on held-out `N, M, K, alpha`, E4 sweeps model ablation flags, E5 reads `defense_fraction` from E1 runs (no new runs), E6 sweeps `beta`, E7 reads estimator error from E1 and Override runs, E8 and E9 run in Override.

---

## 8. Milestones and acceptance criteria

| # | Milestone | Duration | Acceptance criterion |
| --- | --- | --- | --- |
| M0 | Spec freeze, config schema, Override verification (§4.1) | 1 wk | Five verification questions answered in writing |
| M1 | Canonical core plus reference implementation | 2 wk | `test_batched_matches_reference` passes on 500 seeds |
| M2 | Observation, masking, greedy matching, adapters | 1.5 wk | Equivariance and padding tests pass; BenchMARL runs a random policy |
| M3 | TENURE model and PPO integration | 2 wk | Beats greedy at α = 1; retention MAE below 0.15 |
| M4 | Baselines tier 1 and 2 | 2 wk | Parity with TENURE at α = 0, which is the fairness check |
| M5 | Override extension, channels one and two | 2.5 wk | Reversal events emitted; shipped v1 policy retrains under `contested.enabled` |
| M6 | Runner, logging, statistics | 1 wk | One command regenerates Figure 1 from run directories |
| M7 | Full sweeps and ablations | 2 wk | All tables populated with five seeds |

Roughly thirteen weeks of engineering plus compute. M1 and M5 are the two that historically slip.

---

## 9. Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| **Defense degeneracy.** At `contest_mode: suppress`, optimal play is to complete N cheap tasks and park, retention becomes trivially 1, and the paper's contribution evaporates. | **Critical** | `majority` is the default. E6 sweeps `β` and reports the non-monotone advantage curve honestly. This is already written into the paper. |
| **Retention labels for incomplete tasks.** The head is applied to `ACQUIRE` candidates but is most easily trained on complete tasks. | **High** | Dual label sources, §3.9. Report the two error regimes separately in E7. |
| **Ablation under-reports.** If retention only enters as a feature, the encoder learns it implicitly and "no retention head" looks fine. | **High** | Multiplicative decoder, §5. Compare all three `retention_mode` settings. |
| RTAW and DC-MRTA code unavailable | Medium | Verify week one; budget reimplementation; dagger the rows |
| Throughput insufficient for the sweep | Medium | Profile at M1, not M7. Two GPU-hours per cell is the gate. |
| Framework split between SB3 and BenchMARL | Medium | Core is framework agnostic; two thin adapters over shared state |
| v2 objective set breaks checkpoints and ONNX | Low | Behind a flag; v1 remains default and byte-compatible |
| Override rules proxy criticized by reviewers | Low | Paper states the operating point precisely and disclaims referee accuracy |

---

## 10. Open questions for the PI

1. Holdout archetype choice. Default proposed is `camper` plus `learned_selfplay`. Confirm or override.
2. Should the canonical environment's task weights be observable to adversaries? Currently yes, which makes `value_targeting` strong. An unobservable variant is a cheap extra ablation.
3. Is a hardware demonstration in scope for this paper, or deferred? The Limitations section currently defers it. This affects whether the ONNX export path needs to stay live for the graph environment.
4. Override channel priority. Territory reversal is cleaner and cheaper to build than object reversal. Confirm it goes first.
