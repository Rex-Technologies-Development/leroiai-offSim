# Component B (§4.1) Override Verification Findings

**Milestone M0 acceptance artifact.** The development plan requires the five §4.1
questions answered *in writing, from code*, before any Override extension is
written. Answers below cite the current `offsim/` source.

---

## 1. Toggle claim — instantaneous or dwell-based?

**Instantaneous. Must be converted to dwell-based.**

`OverrideField.claim_toggle` ([offsim/sim/field.py:333-335](offsim/sim/field.py#L333-L335)):

```python
def claim_toggle(self, robot: Robot, toggle: Toggle) -> bool:
    if math.hypot(robot.x-toggle.x, robot.y-toggle.y) > INTERACTION_RANGE: return False
    toggle.owner = robot.alliance; return True
```

Ownership flips in a single tick the instant a robot is within `INTERACTION_RANGE`.
There is no timer. Consequence per the plan: **τ_rev for the territory channel is
zero, so α = τ_com/τ_rev is unbounded** and the reversal mechanic is not
representable. Channel 1 (territory) therefore requires a `toggle_claim_dwell`
timer with the reset-on-leave semantics of §3.3 (mirror `sigma`/`eta`).

Aggravating detail: a toggle can be re-claimed by the opponent instantly and
repeatedly — claim/counter-claim has no cost today.

## 2. Opponent removal — can an opponent remove the other alliance's pin?

**No. `REMOVE_OWN_PIN` (action id 9) is the only exposed removal path, and it
only removes your *own* fully-alliance-coloured top pin from a neutral/own goal.**

`OverrideField.remove_own_pin` ([offsim/sim/field.py:323-331](offsim/sim/field.py#L323-L331))
blocks two cases and telemeters them:
- `goal.protected_by is robot.alliance.opponent` → `protected_goal_block`
- `pin.halves != (robot.alliance.value, robot.alliance.value)` → `neutral_removal_block`

The env-level legality helper `_legal_removal_goal`
([offsim/sim/env.py:35-41](offsim/sim/env.py#L35-L41)) enforces the same rule for
masking and the continuous controller. So a robot can **only** pick up a pin whose
two halves are both its own colour, from an unprotected goal.

Consequence per the plan: **Channel 2 (object reversal) needs substantial new
work.** There is currently no mechanism for an opponent to reverse value an
alliance placed. This confirms the plan's guidance to prioritise Channel 1
(territory), which the codebase can express with a dwell timer alone.

## 3. Match-Load replenishment — fixed or growing task set?

**Fixed. Loaders do not refill.**

Inventories are seeded once in `OverrideField.__init__`
([offsim/sim/field.py:170-176](offsim/sim/field.py#L170-L176)) — 13 pins (3 of them
yellow) + 10 cups per alliance — and `use_loader`
([offsim/sim/field.py:296-309](offsim/sim/field.py#L296-L309)) only ever
decrements them (`inv["pin"] -= 1`, `inv["cup"] -= 1`). There is no refill event.

Consequence: **the effective scoring-object supply is finite and monotically
depleting.** For the formalism this means the task set (sites that can hold value)
is fixed within an episode — consistent with §3.2's "`task_pos` fixed within an
episode". No growing-task-set handling is required.

## 4. Midfield geometry — how is "centre in the diamond" computed, per-tick or per-second?

**L1 (Manhattan) diamond test; the 8-point award is a stateless snapshot of
current occupancy — neither per-tick nor per-second accumulated.**

`robot_in_midfield` ([offsim/sim/field.py:340-343](offsim/sim/field.py#L340-L343)):

```python
radius = 600.0/25.4          # ≈ 23.62 inches
return abs(robot.x-72.0) + abs(robot.y-72.0) <= radius
```

An L1 ball of radius ≈23.62 in centred on the field centre (72, 72).
`midfield_count` counts an alliance's robots inside it. The 8 points enter
`raw_score` ([offsim/sim/field.py:358-367](offsim/sim/field.py#L358-L367)) as
`midfield_count(alliance) * MIDFIELD_ROBOT_POINTS` — recomputed from *current*
state every time the score is read. It is **not** integrated over time; it is the
instantaneous standing value of holding the midfield right now.

This is directly usable as a `DEFEND`/contest signal: midfield presence maps onto
the plan's β=1 contest rule ("tall-goal neutral ownership decided by which
alliance has more robots in midfield", §4.2), and `midfield_owner`
([offsim/sim/field.py:348-350](offsim/sim/field.py#L348-L350)) already implements
the majority rule.

## 5. Reward semantics — held-value or terminal?

**Predominantly held-value (dense standing-score delta), with an added terminal
win/loss bonus.**

`_OverrideGymBase._reward` ([offsim/sim/env.py:108-114](offsim/sim/env.py#L108-L114)):

```python
now = (score(BLUE), score(RED))
reward = ((now[0]-last[0]) - (now[1]-last[1])) / 5.0
if done: reward += 10 if now[0]>now[1] else -10 if now[0]<now[1] else 0.0
```

Because `score()` is a pure function of the *current* field state (placed halves ×5
+ owned-yellow ×10 + midfield ×8; see `raw_score`), the per-step reward is the
change in current standing value — i.e. it already behaves like a dense held-value
(J_H-like) signal, and the telescoping sum equals final standing score minus
initial. It is **not** purely terminal. The one non-held term is the ±10 terminal
win/loss bonus (a J_T-like component) added at match end.

Consequence: §4.4's "held-value accumulator per alliance so J_H is directly
measurable" is a small addition — the standing score already is held value; we
need to *integrate* it over time (Σ score·dt) rather than just diff it, and to
separate the terminal bonus out for the `J_H`/`J_T` telemetry split.

---

## Summary table

| # | Question | Finding | Work implied for Component B |
| - | --- | --- | --- |
| 1 | Toggle claim dwell? | Instantaneous | Add `toggle_claim_dwell` timer (τ_rev>0) — **Channel 1** |
| 2 | Opponent pin removal? | Only `REMOVE_OWN_PIN`; no cross-alliance removal | Build opponent-removal dwell — **Channel 2**, larger effort |
| 3 | Loaders refill? | No, finite depleting supply | None — task set is fixed, matches §3.2 |
| 4 | Midfield award timing | L1 diamond, stateless snapshot | Reusable as β=1 contest signal directly |
| 5 | Reward held vs terminal | Held-value delta + terminal ±10 bonus | Add time-integrated `J_H` accumulator; split out `J_T` |

**Recommendation (matches §4.2/§10.4):** build **Channel 1 (territory / toggle
dwell) first.** It is the cleanest instance of the formalism, needs only a dwell
timer with reset-on-leave, and the midfield majority contest rule already exists.
Channel 2 (object reversal) is a genuinely new mechanic and should follow.

All Component B changes remain gated behind `contested.enabled` (default `false`)
and `objective_set: v1` so `models/final_model.zip` stays byte-compatible.
