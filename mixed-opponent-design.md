# Mixed-opponent test for lead-time — is anticipation probabilistic, or a habit that happens to pay?

## Why (consultant)

Preemptive toggle defense is **free** against an opponent that *always* raids toggles: if a raider is
guaranteed, pre-positioning a defender is never wasted, so "anticipation" may just be a reflex tuned to
a fixed threat. The honest version makes the threat **uncertain and unobservable at t=0**: some episodes
contain a toggle-raider, some do not, and the policy cannot tell which it faces until the opponent acts.
If lead-time (preemptive occupancy + low capture) **survives** that, the anticipation is genuinely
**probabilistic insurance** — the policy defends the leverage because it *might* be attacked, and eats a
small guarding cost in the episodes where no raider comes. That is a much stronger claim than "it learned
to defend because it's always attacked."

## What the current regime already does (and why it's not enough)

`_resolve_archetypes` draws each of the `nk` adversaries **independently uniformly** from the population.
With `[toggle_raider, greedy_nearest]` and `nk=4`, the raider count is `Binomial(4, 0.5)` — so ~6% of
episodes already have **zero** raiders, unobservable at start. Good, but raiders are present ~94% of the
time, so anticipation still pays in almost every episode. We want a regime where preemptive defense is
**wasted often enough to punish a blind habit** (e.g. ~50% peaceful episodes).

## Minimal implementation (gated, default off = byte-identical)

Add per-episode **population coupling**: one latent Bernoulli per episode selects the whole roster.

- `config.py`: `raider_episode_prob: float | None = None` (default None ⇒ current independent draw).
- `core.py::sample_initial_arrays`: if set, draw `is_raider_world ~ Bernoulli(p)` **once per episode**;
  then `_resolve_archetypes` uses `[toggle_raider, greedy_nearest]` in a raider-world and a **peaceful**
  roster `[greedy_nearest]` (goal-seeker that ignores toggles) otherwise. All `nk` adversaries follow the
  same world, so raider **presence** — not just count — is the per-episode latent.
- `reference.py`: mirror; extend the reference-match test (the coin is drawn from the same shared rng, so
  parity holds).
- Test: with `p=1` every episode has raiders; with `p=0` none; the policy obs is unchanged (the latent is
  not exposed — that's the point).

## Protocol (the decisive version needs RE-TRAINING, hence "later")

1. **Train** the M9 arms (`off_aware`, `off_blind`, `mult_aware`) under `raider_episode_prob≈0.5`.
2. **Lead-time eval**, split by realized world:
   - *raider episodes*: does it still pre-position + hold (occupancy lead, low capture)?
   - *peaceful episodes*: what does it cost? (guard-waste on toggles that were never threatened.)
3. **Verdict:** anticipation is genuine probabilistic insurance iff it keeps low capture in raider
   episodes **without** blowing up guard-waste in peaceful ones — i.e. it defends the *expected*-value
   leverage, sized to the threat probability, not a reflex.

## Cheap eval-only precursor (no env change, runs now on a banked checkpoint)

Override the eval population to **peaceful only** (`--adversary greedy_nearest`) and re-run
`eval_leadtime` on the trained winner: if it *still* pours defense onto toggles no one attacks (high
toggle def-effort / guard-waste), the current always-raider policy is a **habit**; the decisive
retraining above is then required to get a risk-sized policy. (This only characterises the existing
policy's generalisation; it does not substitute for training under the unobservable mix.)
