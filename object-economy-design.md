# Object-economy model for the canonical env (design, for review)

## Why

Today a canonical "task" is completed by **presence** — a robot dwells for `τ_com` and it's done.
There is **no limited pool of game objects and no fetch cost**. That makes "grabbing" free and instant,
so a greedy grab-everything policy looks far stronger than it should (it can own 12 tasks without
sourcing 12 objects), and it's why coverage keeps competing with concentration.

Real Override is the opposite: **32 pins + 36 cups**, a robot carries **one pin + one cup at a time**,
and every object must be **fetched** (loose on the floor or from a loader) before it can be scored.
Picking is the rate-limiting, non-trivial step — exactly what the abstraction removes.

**The key consequence for the thesis:** once each placement costs a fetch trip, a *placed* object is
worth the trip you spent on it. Losing it means **repeating that trip**. So **holding beats churning
organically** — churn re-pays the fetch cost every time, retention pays it once. Retention stops
being something we inject via a toggle multiplier or a convexity knob and becomes a property of the
game's object economy. And greedy's coverage stops being free.

## Minimal model (batched-tensor, keeps the env fast)

Gated behind a new `object_economy: bool = False` (default off ⇒ byte-identical to today).

1. **Carry state** `robot_carry: (B,R) i8` ∈ {0,1} — is the robot holding an object.
2. **Supply.** A small number of **loader** points (reuse the layout; e.g. 2 per side) plus a global
   remaining-supply count `supply: (B,) i16` (start = `n_objects`, Override ≈ 68 → scale to task count).
   A robot within `service_radius` of a loader with `supply>0` and `carry==0` picks up: `carry=1`,
   `supply-=1` (one pickup per `τ_load`, a short dwell).
3. **Completion gated by an object.** The existing service/build transition fires **only if the
   dominating robot has `carry==1`**; completing/placing **consumes** it (`carry=0`). So a build now
   costs: fetch trip → carry → dwell `τ_com` → place. Height still needs one object per level.
4. **Reversal returns the object to the world.** A reversed/dismantled piece drops as **loose supply**
   (`supply+=1`) rather than vanishing — so the pool is conserved and the opponent's descore literally
   hands you back a re-fetchable object (matches Override's descore-drops-loose).
5. **Obs:** add `robot_carry` to robot features and `supply/n_objects` + nearest-loader edge, all gated.

## Strategic effect we expect (and will measure)

- **Greedy weakens sharply.** Every completion needs a trip; grabbing 12 tasks needs 12 trips + 12
  objects. Coverage is no longer cheap; greedy runs out of objects/trips.
- **Retention pays without any injected structure.** A held task = a banked trip; defending it avoids
  re-fetching. Hold-and-defend should beat churn on plain equal-value tasks — the null regime that was
  decorative before — *because objects are now scarce*.
- **Stacking gets a second reason.** "Worth more to add to an existing tall stack" is partly that the
  stack is already sourced and defended — each lost piece is a wasted trip, so a tall held stack is a
  big banked-trip investment.

## Implementation plan (mirrors how toggles/stacking landed)

- `config.py`: `object_economy`, `n_objects`, `tau_load`, `n_loaders` (+ validation).
- `core.py`: `robot_carry`/`supply` state, loader pickup in `tick`, gate `_transitions_*` builds on
  `carry`, drop-on-reverse; `sample_initial_arrays` places loaders + inits supply.
- `reference.py`: mirror; extend the 400-seed match.
- `observation.py`: carry + supply + loader-edge features (gated); `observation_spec` bump.
- Tests: pickup consumes supply; completion requires carry; reverse returns supply; reference match.
- Experiments: re-run **greedy vs defensive vs TENURE** on plain equal-value tasks *with* the economy
  (does hold-and-defend now beat churn with no toggles/convexity?), then re-run the toggle and combined
  comparisons under it (expect greedy to fall, margins to widen).

## Status: HOLD (next phase, not now) — consultant steer 2026-09

**Hold the build, do not abandon it.** The idea is right and fixes a real artifact (free completion ⇒
greedy owns 12 tasks without sourcing 12 objects), but it is a large, design-dependent build that
touches core/reference/observation/every experiment and would **invalidate the comparability** of
everything already run. With three findings still at 1–3 training seeds, an unexplained `mult−off`
gap, and no run yet on the real Override sim, building this now risks *four half-validated
environments instead of one well-validated result.* It runs **after** the validation plan:
(1) seed the M9 hardening headline to 5 seeds; (2) the **isolation ablation** (leverage-in-obs vs
blind, head off) — the causal claim the reframe rests on; (3) lead-time logging (preemptive vs
reactive); (4) **Override E9** on the real game. Then this, as its own phase.

## Design choices — RESOLVED (consultant)

1. **Supply model:** ✅ global count + a few loader points (as proposed).
2. **Carry capacity:** ✅ one object (as proposed, simplest).
3. **Drop-on-reverse:** ✅ return to global supply (as proposed).
4. **Scale:** **do not pick a value — SWEEP it.** Scarcity (`n_objects` relative to tasks) is the
   parameter and the sweep *is* the result: retention should cross from decorative to decisive as
   objects get scarcer, exactly as it does across the toggle multiplier `M` and the stack-value power
   `p`. This makes scarcity the **third independent instance of the one scope condition** —
   *retention pays when losing an objective is an outsized loss* — which is what turns the paper's
   claim from "a pair of coincidences" into a general law. The scarcity sweep is the phase's headline
   figure; report it in the same crossover form as the `M` and `p` tables.
