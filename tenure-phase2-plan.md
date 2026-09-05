# TENURE Phase 2 — Symmetric env + REVERSE action + self-play

**Status:** design spec / build plan. **P2-0, P2-A, and the two round-6 pre-P2-B fixes
(differential objective + first-arrival tie-break) are LANDED and test-green.** Everything
is gated behind `cfg.symmetric` (default `False`) so the existing env — and the 500-seed
reference match, and Phase 1's eval — stay byte-identical until we deliberately turn it on.

**Consultant round-6 review incorporated:** (1) objective was wrong (blue-only rewards
pure denial, doesn't match Override's score *difference*) → **differential J_H** landed;
(2) model 3A stalemate trap → **first-arrival priority + standoff telemetry** landed;
(3) reverse-to-neutral helps the defensive ρ path but not obviously the offensive one →
**hypothesis split** (§7); (4) collapse instrumentation too weak → **reverse-usage floor +
per-team action/holdings logging** (§6); (5) all scripted red are reversers → **add a
`builder` archetype** (§6/§8); (6) one R̂ can't price REVERSE → **two-head / owner-conditioned
R̂** (§5). Items 3–6 are P2-B+ and captured below.

**Consultant round-8 (post-ρ-curve; "Go on P2-B" WITH a gate).** The ρ curve is a clean null —
oracle-R ties weight-only at every ρ, so there is NO headroom in acquire/defend; the gate can only
*suppress* acquisition, never *encourage*, because without a REVERSE channel there's nothing to
encourage. Risk for Phase 2: if owner-conditioned R̂ is *also* compressed near a constant, REVERSE-
in-the-action-space yields another tie with a new label = Phase 2 repeating Phase 1. So: (a) the two
heads must be **genuinely separate** (shared trunk, two output heads — one head + owner flag
collapses them, since *retention-if-I-hold* [how fast red reaches you] and *retention-if-I-take* [how
fast red comes back after you flip it + whether they already sit on it] are different functions);
(b) a **HARD acceptance gate before self-play** — std of R̂-take across red-held tasks > ~0.15
(ideally bimodal: a red task deep in their half with two red bots ≈ untakeable/low; one just grabbed
on our side ≈ takeable/high), measured under a scripted red; if flat, the head is a constant → fix
the architecture, do NOT spend GPU; (c) **multiple TRAINING seeds** (not just eval seeds) before any
magnitude enters a table — Phase 1's one-seed-per-cell is the record's weak point (direction is safe
via the monotone 3-point trend; magnitudes are not). Framing: state the null as a property of the
SETTING so Phase 2 reads as the theory's natural consequence, not a rescue.

**Consultant round-9 (the R̂-take label spec + a reorder; "oracle ceiling first").**
*Label definition* — for a red-held task i at decision t, **R̂-take = the fraction of the
REMAINING HORIZON that BLUE HOLDS task i, given blue reverts it and then re-acquires it** (NOT
the fraction it stays neutral — blue only scores when blue owns it). The re-acquisition at
**ρ·τ_com must be INSIDE the label**, not bolted on — that is what puts ρ on the offensive path
(the whole reason for the symmetric build). Three consequences: (1) **it is a counterfactual**
(Phase 1's were observed) — it needs blue to *actually* have reverted the task, which early in
training almost never happens, so it is **undefined for most red-held tasks** → **check label
coverage FIRST**; if <~few % of red-held tasks are reverted per rollout the head trains on
nothing → fix with **forced-exploration rollouts** (a random subset of robots assigned REVERSE
regardless of policy, purely to generate labels — standard remedy). (2) **Normalize the horizon
FROM THE FLIP, not from now** — integrate blue ownership from the moment the flip *completes* to
T over that interval; integrating from decision time buries the per-task travel+revert+reacquire
delay in the label and reintroduces the horizon-factor confusion the score eq. already fixed.
(3) **Residual spread, not raw spread, is the gate** — take-retention varies strongly with
*distance* (a far red task takes forever to reach+flip), but that is already priced by the score's
travel term; if the spread is mostly distance the head duplicates the decoder's info → misranking
again. So regress R̂-take on travel time and report the **RESIDUAL** spread: **residual std >
0.15 is the acceptance gate** (the sharpened version — it would have caught Phase 1 early).

*Reorder.* (a) The **`builder` archetype moves ahead of the gate** — R̂-take is undefined against
a red that holds nothing, so the spread check depends on it. (b) **Run a scripted ORACLE CEILING
in the symmetric env** — oracle R-take from hindsight vs weight-only, with REVERSE available,
exactly as the oracle ceiling settled Phase 1. **If oracle ties again, there is no headroom in
the symmetric env either → STOP before self-play.** It is hours of CPU and the single
highest-value thing available now: it can end the line cleanly or justify everything after it.
**Order: `builder` archetype → two-head R̂ + counterfactual label → label-coverage check →
symmetric oracle ceiling → residual-spread gate → self-play.**

**Consultant round-10 (oracle-ceiling build spec — "push straight in").** Five requirements:
(1) **THREE arms, not two** — both reverse arms must have REVERSE in the action space, else you
compare "retention + a new action" vs "neither" and a win credits the *action*, not retention.
Arms: (a) WO no-reverse, (b) WO+reverse, (c) oracle+reverse; headline = (c)−(b), and (b)−(a) is
the reverse action's own value. (2) **Decompose the gap by action type** — report acquire-channel
vs reverse-channel separately; if a gap is all on acquire, the offensive channel isn't doing the
work and the story differs from the one being told. (3) **Sweep ρ∈{1,2,4} inside the ceiling** —
does offensive headroom appear and GROW with ρ (it didn't defensively)? That is the crossover
curve for free, before any GPU. (4) **Coverage** is a property of the *label-generating* policy;
scripted ⇒ we control the reversal rate ⇒ not at risk here (don't build forced-exploration
prematurely — it's for the learned phase). (5) **PRE-REGISTERED STOP RULE, decided NOW:** if the
oracle ties weight-only at *every* ρ, the symmetric env has no headroom either and the
retention-ESTIMATION line ENDS. The honest negative that survives (retention-aware allocation
works when retention is KNOWN; the estimator reads geometry at corr ~0.5; a measured scope
condition that the multiplicative gate cannot pay in EITHER action space) is an **acceptable
outcome** — committed in advance, so a marginal number can't pull us into chasing noise.
Built as `experiments/symmetric_oracle_ceiling.py`.

---

## 1. Why (the asymmetry bug — consultant round 5)

The canonical env today is **one-sided**:

| capability            | blue (our robots) | red (adversaries) |
|-----------------------|:-----------------:|:-----------------:|
| ACQUIRE a neutral task| ✅                | ❌                |
| DEFEND a held task    | ✅                | —                 |
| REVERSE the *other* team's held task | ❌ | ✅ (this is all they do) |

Red only *reverts* our completions; red never *holds* anything; we can never
*take back* anything. So "retention" only ever means "keep what I grabbed," and
at ρ=1 keeping is worthless (re-grabbing is as cheap as the first grab). That is
**why every retention arm ties weight-only** — not a decoder failure, a missing
half of the game.

Override is **symmetric**: both alliances score, descore, and re-score the same
goals. The value of estimating retention lives mostly on the **opponent's** held
tasks — "which of their goals can I flip and keep?" — a decision our allocators
**cannot even represent** today. Restoring that symmetry is the real fix, and it
is *sharper* than the defend argument: reversing is offensive, so no reviewer can
dismiss it as passive risk-aversion.

---

## 2. Target env contract (what changes)

Per-task ownership becomes three-valued: **neutral / blue-owned / red-owned**.
Both teams ACQUIRE neutral tasks, DEFEND their own, and REVERSE the opponent's
(revert → neutral, then someone must re-acquire at the ρ cost).

**Objective (round-6 fix — LANDED).** The primary score is the **differential**
`J_H = Σ w_i · ([owner_i==blue] − [owner_i==red])` integrated over the episode, and
the per-decision reward follows it. Override is decided by score *difference*: a
blue-only objective silently rewards pure denial and rates "both hold 40%" the same
as "you hold 40%, they hold 5%." The differential also fixes the reward — it credits
a reversal that neutralizes a high-value red task *at the moment it happens* (red's
held value drops), which a blue-only increment would score as zero. The blue-only
`held_integral` is retained as telemetry (`J_H`, `J_H_red`, `J_H_diff` in the summary).
In the single-team env red owns nothing, so the differential **reduces exactly to
blue-only** — Phase 1's numbers and its ρ eval are unaffected.

Representation (chosen for **minimum blast radius** — keeps every `task_c`
reference working):

- keep `task_c` = **"blue owns this task"** (unchanged meaning).
- add `task_c_red` = **"red owns this task"**. Invariant: `not (task_c & task_c_red)`.
- add `held_integral_red` (red's held-value integral) for the differential objective.
- neutral ⇔ `~task_c & ~task_c_red`.
- when `symmetric=False`, `task_c_red` is all-False forever ⇒ every existing code
  path is unchanged ⇒ reference match holds trivially.

---

## 3. Ownership & servicing model  — **DECIDED + built (model 3A + first-arrival)**

When a **neutral** task has both teams present, who makes progress? Built rule:
**model 3A (contested-neutral) with a first-arrival tie-break.**

- **Per-team service timers** (`sigma` = blue, `sigma_red` = red); ownership on
  completion goes to whoever was servicing. Reversal stays a single `eta` per task
  (only the current owner can be reverted, by the opponent dominating).
- **The sole dominator services** a neutral task (majority per `contest_mode`).
- **First-arrival tie-break (round-6 fix):** a per-task `neutral_claim` ∈ {none,blue,red}.
  The sole dominator (re)claims; when both teams are matched (neither dominates) the
  **incumbent claimant keeps progressing**, so two matched robots never freeze a task
  forever. The residual true deadlock — matched presence with *no* incumbent — is
  flagged as `standoff` and counted in telemetry (`standoff_fraction`, 0 ⇒ no deadlock).
  Losing the claim resets that team's service timer.

**REVERSE = ACQUIRE-of-an-opponent-task, mechanically**, but a **distinct action
index** so the decoder can *choose and value* it (the whole representability claim).
Reverting an opponent task sends it **neutral** (`eta ≥ tau_rev`) — confirmed over
direct-flip, so re-acquisition costs `tau_com · ρ` and **ρ stays in the offensive
loop**. But note the round-6 caveat (§7): revert-to-neutral makes re-acquisition
cheaper for *both* sides as ρ rises, so ρ's benefit to the *offensive* path is not a
given — measure it.

---

## 4. Action layout change (REVERSE)

Today (`config.action_dim = 2T + 1`):

```
[0, T)     ACQUIRE task i
[T, 2T)    DEFEND  task i-T
2T         IDLE
```

Symmetric (`3T + 1`, only when `cfg.symmetric`):

```
[0, T)     ACQUIRE  task i          (neutral only)
[T, 2T)    DEFEND   task i-T        (blue-owned only)
[2T, 3T)   REVERSE  task i-2T       (red-owned only)   ← new
3T         IDLE
```

Touch list (all gated on `symmetric`; non-symmetric path unchanged):
`config.action_dim` / `idle_action` / new `reverse_base`; `actions.action_masks`
(REVERSE valid iff `task_valid & task_c_red`); `actions.greedy_assign` (consume
REVERSE target column too); `core._robot_targets` (REVERSE → task pos);
`observation` robot one-hot (4-way: acquire/defend/reverse/idle).

Old (non-symmetric) checkpoints keep the `2T+1` head and load fine — the head
width is config-derived, so symmetric runs are simply a different, wider policy.

---

## 5. Observation change (opponent-owned tasks)

Add the info the head needs to value reversing:

- task feature `task_c_red` (owner==red) — gated, appended like `expose_protected`.
- the `e_at` adversary→task edge already encodes threat geometry; symmetric play
  reuses it as "how fast can red re-take a task I just flipped."
- global scalar: fraction red-owned (mirror of `frac_complete`).

Keep `expose_protected` orthogonal so the killer-control 4th arm still composes.

**Two-headed / owner-conditioned R̂ (round-6 fix — P2-B).** The head today predicts one
R̂ per task: *how long will I hold this if I own it*. Valuing a REVERSE needs a **different
quantity** — the retention of a task I would *take* (prospective owner = blue, sitting in
red's geometry), which is not the same as the retention of one I already hold. As-is, the
decoder would get the REVERSE *action* but not the *estimate that prices it*, so it would
default to reversing-is-dominated. Fix: either predict **two heads** (hold-R̂, take-R̂) or
**condition R̂ on the prospective owner**. This lands with P2-B, before any self-play spend.

---

## 6. Self-play loop (the deep part)

Red stops being a script and becomes a **policy** choosing acquire/defend/reverse
from a red-centric observation (swap the blue/red roles in `build_observation`).
Staged:

1. **Frozen self-play against the Phase 1 checkpoint specifically (round-6 fix)** —
   red = a frozen copy of *the Phase 1 blue policy*, role-swapped. This gives a direct
   read on whether symmetry changed **the policy** or just **the environment**: same
   weights, new game, measure the delta. Cheapest, and no moving target.
2. **Co-trained self-play** — red updates too (shared weights, or a slowly-updated
   past-self pool to avoid chasing).

**Collapse instrumentation (round-6 — the per-task reversal count is too weak).** Three
failure modes, only one of which the naive metric catches:

- both converge to **pure denial** (low absolute holdings both sides) — the *differential*
  rates this a tie and it looks fine in aggregate. → log **absolute held fraction for both
  sides**, not just the difference.
- both converge to **pure acquisition, never reverse** — REVERSE sits in the action space
  unused and we've learned nothing. → log the **per-team action distribution over
  acquire/defend/reverse per update**.
- endless **trading of the same neutral goal** → per-task reversal counts (the original).

**Hard floor: if reverse-usage is under a few percent, the experiment is uninformative
regardless of J_H** — do not report a ρ curve off a policy that never reverses.

Scripted archetypes stay as **fixed eval opponents**, but they are all *reversers* today.
Add a **`builder`** (acquires + holds only) and a **`balanced`** (splits) archetype so
"how good is blue at reversing" is tested against a red that actually holds something worth
taking (§8).

---

## 7. ρ interaction + hypothesis

Phase 2 doesn't replace the ρ sweep — it **completes** it. Prediction: the head stays
~flat at **ρ=1** even in the symmetric env (flipping a task you can't keep, that they
re-take for free, is worthless) and **earns its keep as ρ rises**.

**But split the hypothesis into two paths and measure them separately (round-6 fix):**

- **Defensive path** — ρ↑ makes a task *you lose* expensive to regain, so retention-aware
  holding pays. **Well-founded.**
- **Offensive path** — because revert sends the task to *neutral*, ρ↑ makes re-acquisition
  more expensive **symmetrically for both sides**, so it does *not* obviously favor
  retention-awareness on reversing. **Not established — an open question, not an assumption.**

So the deliverable is not one crossover curve but a **decomposition**: the gap attributable
to **defend** decisions vs the gap attributable to **reverse** decisions, each vs ρ, for
`retention_mode ∈ {multiplicative, off}`. If ρ only helps the defensive path, **that is still
a result** — a narrower, honest scope condition, sharper than "retention helps." Reporting a
single blended curve would hide which half of the game is doing the work.

---

## 8. Incremental build plan (each step ends test-green)

Consultant's ordering: **fix objective → P2-A + standoff instrumentation → P2-B → verify
reverse-usage is nonzero → only then spend on P2-C/D.** Run the ρ eval the moment Phase 1
frees the GPU; that result decides whether P2-C/D are worth the compute at all.

- **P2-0 (DONE):** `symmetric` config flag, inert. All tests green.
- **Objective + tie-break (DONE — round-6, pre-P2-B):** differential J_H + reward;
  first-arrival priority + `standoff_fraction` telemetry. Single-team path byte-identical.
- **P2-A (DONE):** ownership state (`task_c_red`, `sigma_red`, `held_integral_red`,
  `neutral_claim`), symmetric tick (red completes; both reversal directions; first-arrival).
  Mirrored in `reference.py`. **`test_symmetric_batched_matches_reference` (400 seeds)** +
  red-completes + blue-reverts-red + first-arrival/standoff. Scripted red.
- **P2-B (next):** REVERSE action end-to-end (layout `3T+1`, masks, greedy_assign, targets,
  obs one-hot) **+ the two-head / owner-conditioned R̂ (§5)** so the decoder can price it.
  Test: a blue policy *can* select REVERSE on a red-owned task and it reverts.
- **P2-B.5:** add `builder` + `balanced` scripted archetypes (§6) so eval red holds tasks.
- **Gate:** train once in symmetric mode and confirm **reverse-usage > a few %** (§6) before
  spending on self-play. If blue never reverses, stop and diagnose.
- **P2-C:** self-play — frozen vs the **Phase 1 checkpoint** first, then co-trained. Full
  collapse instrumentation (§6).
- **P2-D:** ρ ∈ {1,2,4} sweep in symmetric env, mult vs off, 15-seed eval, **decomposed into
  defend-gap vs reverse-gap** (§7). Compare to Phase 1's acquire/defend numbers.

## 9. Paper framing

Claim 2 ("assigning to an already-completed task is worthless") gets **restated
over opponent-completed tasks and strengthened**: current allocators cannot
*represent* the decision to revert opponent-held work, so they leave the entire
offensive-retention lever on the table. The headline result becomes a **scope
condition** — retention-aware allocation pays when re-acquisition is expensive
relative to acquisition (ρ > ~1), with the crossover measured — rather than a
flat win/loss. Structural, swept, and predictive of Override.
