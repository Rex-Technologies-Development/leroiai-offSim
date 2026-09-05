# Dynamic cluster exposure — contested value as a continuous, state-dependent spectrum

## Why (consultant, 2026-09)

The env today is a **step function**: a toggle multiplies its cluster (×M), an ordinary goal does not.
Real Override is a **spectrum**, and the value of a contested point is **dynamic** — it changes minute to
minute with what is currently underneath it:

- a toggle governing 4 goals is worth more than one governing 2 (cluster **size**);
- a goal already holding a 5-stack is worth more than an empty one (**contents**);
- **the same toggle** over a cluster full of *your* scored objects is worth far more *right now* than the
  identical toggle over an empty cluster (**dynamic exposure**).

Early in a match a toggle governs nothing and is nearly worthless; late, over a loaded cluster, losing it
is catastrophic. **A fixed rule cannot track that; a policy reasoning about current exposure can.** This is
the graded prediction problem the retention head never had — every earlier head failure came from the
target being a **constant** (ρ=1) or a **binary lookup** (protected flag). Cluster exposure varies
continuously, changes with board state, and **cannot be read off a flag**.

## The gap in the current implementation

The **reward** is already dynamic (`core.py::_held_values` multiplies *currently-held* cluster goals by M
while you hold the toggle). But the **observation** value channel is static: `_effective_value` gives a
toggle a flat `w_max` (`observation.py`), independent of how loaded its cluster is. So the policy is scored
on dynamic exposure but cannot *see* it. Closing that gap is the whole idea.

## Two distinct experiments (keep them separate)

**A. Representation (the safe positive).** Define, per contested point i, its **current exposure**
`E_i(t) = base_i + (M-1) · Σ_{j∈cluster(i)} held_value_j(t)` — computed from the *current* contents of its
cluster each step. Put `E_i(t)` in the observation (replacing the static toggle premium) and score by it.
Predicted result: dynamic-exposure scoring **beats both plain weight and the retention gate**. This is a
representation win (E_i is observable from current state), consistent with the reframe, and the positive
result that lets the head negative sit beside it as *the thing that motivated the fix*.

**B. Head redemption (the first fair test).** The defend/concede decision hinges on **future** exposure —
whether a currently-loaded cluster will *stay* loaded and whether you will *hold* the toggle under
contention. That is **not** directly observable; it must be predicted. A head estimating *expected retained
exposure* now faces a target that is graded, state-dependent, and not flag-readable — unlike every prior
case. Test: does gating by the head's exposure estimate beat computing current exposure alone? If **yes**,
we get a *scope condition on the head* (decorative for static/binary value, valuable for dynamic exposure).
If **no**, the head negative is far deeper (it fails even on its ideal graded target). Either is publishable.

## The decision it instantiates (the compelling version)

Several contested points of **differing, time-varying exposure** and **not enough robots to hold them all**:
which do you defend and which do you concede? A retention/allocation policy that prices each point by its
*current* exposure and commits scarce defenders to the highest-exposure ones — updating as the board
changes — is exactly the interesting problem a reviewer will find compelling, and no fixed heuristic tracks
it.

## Status: ENV BUILT ✅ (2026-09, during the 370-extension window; gated, default off = byte-identical)

- `config.py::dynamic_exposure` flag (+ validation: requires toggle_regions>0).
- `observation.py::_effective_value`: toggle value = base + (M-1)·(blue-held cluster value now); verified
  it rises 0.47→4.8 as a cluster fills, other toggles unaffected; `task_dim` unchanged (12).
- `core.py::sample_initial_arrays`: varied cluster sizes via the shared rng (reference-match preserved).
- `tenure/train.py::--dynamic-exposure`.
- Test `test_dynamic_exposure_prices_toggle_by_current_cluster_load` added; **all 63 contested tests pass**.
- **Next (needs GPU, after the 370 extension):** train the 3 arms (off_aware / off_blind / mult_aware)
  with `--dynamic-exposure` and run `experiments.eval_isolation` — the same pipeline, on the env where
  representation should now separate from noise (blind can't read continuous state-dependent value) and
  the head faces its first graded target.

## REGISTERED PREDICTION (write-it-down-first, consultant 2026-09-04 — before any dynamic-exposure run)

Dynamic cluster exposure is the **first graded, state-dependent, non-flag target the head has ever faced**
(every prior head failure had a *constant* target, ρ=1, or a *binary* one, the protected flag). So the
outcome is worth predicting on the record:

- **Prediction: the retention HEAD will still NOT help (decorative → mildly harmful).** Reason: current
  exposure is **observable** — it is computed from the current cluster contents and now sits in the obs —
  so a *learned estimate* of it adds nothing over *reading* it, and gating a clean exposure ordering by a
  noisy head estimate still misranks (the same mechanism as every prior null). The head could only add
  value on the **unobservable** part — *future* exposure (will a loaded toggle stay loaded under
  contention) — which is a smaller, second-order signal.
- **If the head FAILS here too → the negative is airtight across *every* target structure** (constant,
  binary, and graded-observable). That is the strongest form of the claim.
- **If the head HELPS → surprising and a much more interesting paper.** It would mean *future*-exposure
  prediction is the operative signal and the head captures it; the claim would become "retention
  estimation pays iff the value is graded AND its future is unobservable." Must then understand *why*
  before believing it.

Registered now so the outcome is honest either way.

### PRE-REGISTRATION — committed BEFORE the run (2026-09-04), after being burned twice reading a gap generously

- **Seed count: 10** env-seeds (0–9), all three arms (off_aware / off_blind / mult_aware), trained FRESH
  to the fixed unbiased budget (420, verified past the plateau for this env post-hoc), with the
  outcome-blind restart rule from `convergence-protocol.md` (restart any cell < 50% of its ARM median).
  8 is the floor; we run 10 for power given the observed per-seed spreads (~±0.11).
- **Separation threshold (what counts as a real effect), fixed now:** a paired gap is declared a REAL
  effect **iff** (a) its 95% CI over training seeds — `mean ± 1.96 · std/√n` — **excludes 0**, AND
  (b) `|mean| ≥ 0.05` (a pre-committed minimum effect size). Anything else is reported as
  **"within noise / not established"**, no matter how suggestive the point estimate. At n=10 with
  std ≈ 0.11 (SE ≈ 0.035), this means a gap must reach `|mean| ≳ 0.07` to be called real — so a
  sub-0.07 effect will be reported as null even if positive.
- **Committed predictions (both gaps):**
  - **ISOLATION (off_aware − off_blind): predict POSITIVE and SEPARABLE.** Dynamic exposure is the env
    *designed* for representation to matter — blind cannot reconstruct continuous, state-dependent
    cluster value from geometry. If it comes back **within noise**, that is a *strong* negative:
    representation fails even where it structurally should help ⇒ the step-env null was not a fluke and
    representation does not help in this task family; the contribution is "learned allocator beats
    scripted." (The step-toggle env gave isolation +0.050 ± 0.115 — null — as the contrast baseline.)
  - **HEAD (mult − off_aware): predict DECORATIVE (within noise).** Current exposure is observable, so
    estimating it adds nothing over reading it; the head could only pay on *future* exposure.

## Keep the exposure term EXPLICIT in the decoder (positive contribution, not a workaround) — consultant

Today `off_aware` wins by having leverage/exposure in the **observation** and letting the network figure
out what to do with it — which reads as "we removed the broken part (the head)." The sharper, positive
framing is to **score tasks by exposure as a NAMED value term**: `value_i = exposure_i` = *what collapses
if you lose task i* (`base + (M-1)·current held-cluster value`). In the dynamic-exposure env `task_value`
IS exposure and the multiplicative decoder's value channel is `w_norm = obs["task_value"]`, so off_aware
already scores by exposure — but **name it and verify that wiring**, and consider a dedicated
"collapse-value" decoder term in the retention-head's slot. This converts the story from *"we removed the
broken retention gate"* into *"we replaced it with an exposure-weighted allocator (score by
collapse-value),"* i.e. a positive contribution rather than a negative + workaround.

## CONVERGENCE-CURVE experiment (the 220-vs-370 hazard fix — STEP env, do BEFORE dynamic-exposure training)

Two points (220 noise, 370 separable) is *consistent* with convergence but also with "370 happened to
cooperate" (esp. seed 2 flipping −0.126→+0.018). Prove it: **report the gap as a function of training
length.** Re-run a few seeds (0,1,2) × 3 arms with `--save-every` (from scratch to ~400, or `--load` 220
→ 400), eval the isolation/head gap at each saved length, and plot gap(L).
- **Monotone rise → plateau ⇒ convergence proven**, 370 is a principled stop, compelling figure.
- **Non-monotone ⇒ pick the stopping rule on a *convergence criterion* (e.g. J_H/entropy plateau), not on
  the gap**, and say so. Runs AFTER the 5-seed 370 headline; do not jump ahead.

## Implementation sketch (gated, default off = byte-identical)

- `config.py`: `dynamic_exposure: bool = False`; `cluster_sizes` (vary k per region instead of uniform, e.g.
  sample sizes so exposure range is wide).
- `core.py`: exposure already computable from `_held_values`; expose a helper `cluster_exposure(state)`
  returning `E_i(t)` per task; **vary cluster sizes** in `sample_initial_arrays` (currently uniform `per`).
- `observation.py::_effective_value`: when `dynamic_exposure`, set the toggle/goal value channel to
  `E_i(t)/norm` (dynamic) instead of the static `w_max` premium; add a normalized `cluster_load` feature.
- `reference.py`: mirror `cluster_exposure` + varied sizes; extend the reference-match test.
- Labels (`labels.py`): optionally add a **future-exposure** regression target for experiment B (expected
  held cluster value over the remaining horizon), so the head has the graded target to estimate.
- Experiments: `exposure_sweep.py` — score by {plain weight, retention gate, current exposure, head-predicted
  exposure} across a range of cluster-size/load distributions; show the exposure arms dominate and locate
  where (if anywhere) the head adds over current-exposure representation.

## Sequencing

This is the **most important env axis proposed** (it directly tests head redemption and makes value a
spectrum), and it should likely precede the object economy. It is a real build (core+reference+obs+labels+
tests), so it runs as its own phase after the current step-function validation (isolation ablation, 5-seed
headline, lead-time, E9) completes — those results stand on their own and motivate this.
