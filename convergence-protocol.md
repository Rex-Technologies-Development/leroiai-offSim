# Training protocol for the isolation/head decomposition — PRE-REGISTERED (2026-09-04, before the run)

Path (b), chosen after the diagnostic showed the 220→370 "seed variance" is **training collapse**, not
under-training: ~2 of 13 extension runs plateaued far below their siblings (`off_aware_s3` = 0.212 vs
~0.47; `mult_aware_s1` = 0.115 vs ~0.38), i.e. a **~15% PPO-collapse rate concentrated in the arms with
extra machinery** (toggle-value channel / retention head; `off_blind` never collapses, only trains
lower). Because each collapse swings its gap by ~0.25 and the true effects are ~0.15, the decomposition
cannot be measured without excluding collapsed runs. This is standard practice for unstable RL training;
pre-registering the rule below is what keeps it from being cherry-picking.

## The rule (fixed before seeing any new result)

1. **Fixed budget past the empirical plateau + post-hoc per-seed convergence (NOT an early-stop
   criterion).** Train every cell to a **fixed 420 updates** (`--updates 420`, no early-stop), which the
   370 extension showed is past every good run's J_H plateau (~380) and far past every collapse's. Then
   **demonstrate convergence per seed from the J_H trajectories** (all plateau before 420).
   *Why not an early-stop criterion, which we first tried:* a diminishing-returns early-stop
   (gain-over-window < ε) **differentially under-trains slow-converging arms**. `off_blind` is a slow,
   low-J_H climber, so its window-gain dips below ε during a still-productive slow climb and it stops
   early (observed: `off_blind_s0` stopped at update 200 at J_H 0.066, vs its true ~0.19 plateau), while
   `off_aware` climbs fast and never triggers — which would **artificially inflate the isolation gap
   toward our own hypothesis**. A fixed budget past the plateau removes that bias entirely; we report
   "trained 420 updates, verified past every per-seed plateau."

2. **Collapse = restart.** After all cells of an arm converge, compute that **arm's median** converged
   J_H. Any cell whose converged J_H is **< 50% of its arm median** is declared a training collapse and
   **restarted with a new policy-init seed** (`--init-seed = seed + 100·k` for restart k), keeping the
   **env `--seed` fixed** so the paired comparison's conditions are unchanged. Restart up to **k = 3**
   times; if still collapsed, keep the run and flag it (do not silently drop).

3. **Report the restart rate.** The number of restarts triggered is reported as a **training-stability
   result**, not hidden — "the aware/mult arms collapsed on X% of inits and were restarted," including
   the observation that `off_blind` never collapses. This is honest and is itself a finding (the extra
   machinery destabilises PPO).

4. **Threshold justification.** 50%-of-median is unambiguous here: collapses sit at 0.11–0.21 while
   siblings cluster at 0.38–0.49 — a clean bimodal gap, no borderline calls. The threshold is set now,
   before the run, and will not be tuned to the outcome.

## FINAL COMMITTED RULES — apply verbatim to the dynamic-exposure batch (pre-registered 2026-09-04)

Restated crisply so a reviewer cannot read it as dropping an inconvenient seed:

- **Restart criterion (outcome-blind).** After a batch finishes, for each ARM compute the median of its
  cells' final held-out J_H. **Any** cell whose final held-out J_H `< 0.50 × (its arm median)` is a
  training collapse and is restarted — decided mechanically by the number, **before** and independent of
  which arm/seed it is or which direction the restart moves any gap. Restart = same env `--seed`, new
  `--init-seed = seed + 100·k`, up to **k = 3**; if still below threshold at k=3, keep and flag it (never
  silently drop). The count of restarts is reported.
- **Why per-ARM median, not a global median.** `off_blind` is *legitimately* lower (it cannot see the
  leverage), so a global-median rule would flag genuine blind runs as "collapses" and restart them
  upward — biasing the blind arm the opposite way. Per-arm median flags only *within-arm* collapses
  (a run far below its own siblings), which is the actual failure mode observed (0.11–0.21 vs 0.38–0.52).
- **Threshold (0.50) and seed count are fixed before the run and not tuned to the outcome.** The 0.50
  cut is unambiguous here (collapses sit at ~¼–½ of median; healthy runs at ~0.9–1.1×), no borderline
  calls near it.

## What this does and does not claim

- It measures the isolation/head decomposition **on converged policies**, which is the intended
  estimand (does representing leverage help / does the head hurt, when training succeeds).
- It does **not** claim the head/machinery is free of cost — the collapse rate is part of the ledger and
  is reported alongside.
- The robust result stands regardless: `off_aware` beats defensive/greedy on **every** seed including
  collapsed ones; only the fine-grained decomposition needs converged runs.
