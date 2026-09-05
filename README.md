# leroiai-offSim — Override 2D prototype

A deterministic, fast top-down simulator for experimenting with centralized autonomous strategy in a **provisional VEX AI Override profile**. It provides a low-level continuous-control Gymnasium environment, a high-level masked objective environment for `sb3-contrib` MaskablePPO, scripted 2v2 autoplay, and a Pygame renderer.

This repository is a pragmatic strategy prototype, **not an official rules engine**. It uses the VEX U Override field/setup concepts as a proxy where a final VEX AI format or subjective ruling is unavailable. The assumptions below are part of the simulator contract.

---

## TENURE research stack (in progress)

The repo now hosts **two** things: the original Override 2D prototype (`offsim/`, documented below), and a new research stack implementing **TENURE** — retention-aware task allocation, i.e. holding contested objectives against adversarial reversal. See [`tenure-sim-development-plan.md`](tenure-sim-development-plan.md).

### New packages

- **`contested/`** — Component A, the *canonical* environment. A native, batched-tensor (PyTorch, CUDA-ready) implementation of the paper's abstract dynamics for sweeping α, β, N, M, K. Contains: a strict config loader (α is the knob, `τ_rev = τ_com/α`, `α=0 → ∞`); a batched core plus an **independent NumPy reference** (500-seed equality test); a graph *dict* observation with `e_rt`/`e_at`/`e_tt` edge features; action masking + greedy conflict resolution; **five adversary archetypes**; retention-label extraction (dense + completion-event, reverse scan); per-episode telemetry; Gymnasium-vector and PettingZoo adapters; and a Pygame visualiser (`python -m contested.demo`).
- **`tenure/`** — Component C, the model. A heterogeneous graph encoder, a retention head, and a **multiplicative decoder** in which retention scales task value so that `R̂ ≡ 1` *exactly* recovers a conventional allocator (`retention_mode: multiplicative | feature | off`). Trained with PPO plus a retention-regression auxiliary (`python -m tenure.train`).

### Two-environment split (decided)

Baselines, sweeps, and TENURE **training all run on the canonical env** — the seven
comparators (greedy, CBBA, defensive, MAPPO, QMIX, RTAW, CapAM) were designed for
that abstraction, and only batched-on-GPU is fast enough for the 160-cell sweeps.
The **Override sim is the real-game demonstration**, reached through `OverrideGraphEnv`
(same §3.7 observation + §3.5 actions), so a canonical-trained TENURE policy transfers
unchanged. Override can't sweep α, but it contains **three α regimes at once, in space**:
alliance goals are protected (α=0 → retention≈1), neutral goals are removable
(intermediate), toggles flip easily (α≥1 → low) — the E9 site-class retention result.

### Retention validation status (honest — retention-aware allocation pays when value is concentrated in defensible control points; a scope condition with a swept parameter, see RESOLUTION below)

Does retention-aware allocation actually buy anything here? The go/no-go is three
measurements, **none of which involve defending**: (1) is realized held-fraction variance
non-trivial; (2) is it predictable from structure/geometry; (3) does *acting* on it
(`w×R`) beat *ignoring* it (`w`). The key structural ingredient is `protected_fraction`
— a per-task irreversible flag (Override's protected alliance-goal analogue). Without it
every task is equally reversible and **retention is provably decorative** (the gap is
*exactly* 0.000).

**1. Scripted CEILING — the value of *knowing* protection.** `experiments/retention_probe.py`
sweeps `protected_fraction` with `w×R_oracle` (R = 1 protected / low contested) vs
weight-only, paired over 5 seeds. It is a clean inverted-U — zero at both ends, large in
between — and the effect is **maximal at Override's real operating point**:

| protected_fraction | weight-only | w×R_oracle | gap (95% CI) |
| ---: | ---: | ---: | :--- |
| 0.00 | 0.128 | 0.128 | **+0.000** — decorative |
| **0.15** (Override) | 0.358 | 0.545 | **+0.186** [.175, .197] |
| **0.20** (Override) | 0.438 | 0.620 | **+0.182** [.167, .196] |
| 0.30 | 0.557 | 0.678 | +0.121 [.114, .129] |
| 0.50 | 0.647 | 0.716 | +0.069 [.063, .075] |
| 1.00 | 0.793 | 0.793 | **+0.000** — decorative |

**2. The honest caveat (why the ceiling is not the claim).** In the probe, `R` is handed
in as the protected-flag **oracle** — a field that is **not in the observation**. So the
ceiling measures *the value of knowing which tasks are protected*, not *the value of a
learned retention estimate*. (Burglary analogy: a model that "predicts" break-ins by
reading a `has_security_system` column already in the data — correct and useful, but it
demonstrates no prediction skill.) Any allocator handed that flag gets the ceiling for free.

**3. The *learned* test — and why it reframes the question around capacity.** The §3.7
observation exposes geometry/dynamics but **not** `task_protected`, so a trained head must
*estimate* retention; `retention_mode: off` severs it into a plain learned allocator. The
learned results are more nuanced than the ceiling and turn on the paper's **capacity
heuristic** (Eq. 10 — you can hold ground only when `N/K ≳ (d+τ_com)/(d_adv+τ_rev)`):

| α (4 robots vs 4 adv) | capacity | learned killer control (deterministic, 15 seeds) |
| --- | --- | --- |
| 0 | greedy near-optimal | learned ≈ greedy (0.896 vs 0.900) — nothing to beat (hold-and-keep is trivial) |
| 1 | holdable (at the boundary) | learned **beats** greedy (0.850 vs 0.802); **mult = off** (tie) → head decorative |
| 3 † | unholdable (far below threshold) | greedy **beats** learned; **mult < off** — Eq. 10, see below |

† that run also used 8 adversaries (`N/K=0.5`), unholdable even at α=0.5 — a doubly-bad regime.

**α=3 is an Eq. 10 confirmation, not a counterexample.** Far below the capacity threshold you
*cannot* hold anything, defense is dominated, and **churning is correct** — so greedy winning
there is the paper's own heuristic being confirmed (it belongs in E2 as such), not evidence
against retention. A trained RL policy losing to a one-line heuristic there also means the
`mult` vs `off` gap is between two policies neither of which is the right strategy.

**Two rigor items still open (consultant):**
- **The 4th arm / Claim 4** — built: `--expose-protected --retention-mode off` is a weight-only
  learned allocator *handed the protected flag* (+ the adversary geometry it already had). If it
  matches `multiplicative`, retention is just a **relabel** of "which goals are protected"; if
  `multiplicative` (geometry estimation, no flag) beats it, the head extracts something real. It
  is now a fourth arm in the α-sweep, answering Claim 4 directly.
- **R is not marginal** (`contested/labels.py`) — `R` integrates *future re-completions*, so
  `w·R` over-values tasks under churn and may itself depress `multiplicative` in the α≈1.5–2
  zone. **A flatter-than-predicted inverted-U ⇒ suspect R first, not the method.**

**What the learned test found — and the design correction (consultant).** First-pass killer
control (`off+flag` had the flag, `mult` did not) showed being *handed* protection beats
estimating it. Two design errors made that verdict on the **estimator** invalid:
1. **Unequal inputs** — `off+flag` (8-dim, has the flag) vs `mult` (7-dim, no flag) tests
   *"more information beats less,"* not architecture. Fix: **give the flag to every arm**
   (`mult+flag` vs `off+flag`), so the head is judged only on **discrimination among contested
   tasks**.
2. **Wrong retention variance** — `protected_fraction` makes retention a **static binary rule**
   (protected 0.925 / contested churn), which geometry *cannot* reveal (protection is a rule,
   not a spatial fact) and which the flag gives for free. The real, geometric question is:
   *among the **contested** tasks, which will I hold, given where the campers are this episode?*
3. **Wrong regime** — every α tested (1, 1.5, 3) is *at/above* the capacity boundary (~α=1);
   the head should matter where holding is **feasible but not free** (**α≈0.5–0.75**), which was
   never run. (The no-revisit-R backfire confirms the standard R is a **low-churn approximation**,
   valid *below* the boundary and degrading above — a paper scope condition that predicts the
   observed *tie at the boundary, harm above it*.)

**Banked positive:** `off+flag` (0.788, retention 0.925, 9 reversals) **beats greedy** (0.775,
35 reversals) — retention-aware allocation demonstrably works; Claim 4's *concept* stands.

**Corrected-experiment result (both arms have the flag, holdable α, standard R, 15 seeds).**
`mult+flag` vs `off+flag` is a **TIE** at both operating points — α=0.5: +0.000 [−0.001, +0.001];
α=0.75: +0.002 [+0.001, +0.003] (statistically separated but **below the pre-committed 0.01
"real-effect" threshold**, so a tie). Both learned arms beat greedy by ~0.05. So the retention
**head does not improve allocation.** This *looked* like a decoder problem — **an error, now
corrected.** An **oracle-ceiling** test (`experiments/oracle_ceiling.py`) settles it: even a
*perfect hindsight oracle* (`w × realized-R`) **ties weight-only** (α=0.5: +0.001; α=0.75: +0.001),
so there is **no headroom** — which *exonerates* the decoder (no decoder can extract value that
isn't there). And this is **not** because variance is absent: the std of realized contested
retention is **0.117–0.144** (real, and the head's corr-0.5 is genuine signal). The variance is
simply **worthless at ρ=1**: the payoff to knowing retention ≈ **variance × cost-of-losing-a-task**,
and here losing a task costs only τ_com to re-occupy the spot (**re-acquisition ratio ρ = 1**), so
even perfect knowledge buys nothing — you re-grab the "unretainable" tasks anyway. That is why
*every* α ties or harms: it is **structural**, not a knife-edge and not a decoder failure.

**Claim 4, correctly stated:** (1) a head *does* estimate contested retention from geometry
(corr ~0.5 on real std ~0.13); (2) but at **ρ=1 that estimate is unrealizable** — perfect knowledge
ties weight-only; (3) so the value can appear only when **losing a task is expensive**. **The
missing variable is ρ, the re-acquisition cost — not α, and not the decoder.** The canonical env
pins ρ=1 (retention provably cannot pay); Override has ρ ≫ 1 (re-scoring a descored goal is a full
fetch-and-place). The real experiment is therefore a **ρ sweep**: retention-aware allocation should
cross from *tie* to *win* as ρ rises — a scope condition with a swept parameter that explains every
result in this thread (including the ties) and predicts Override's behaviour before running it.
(Keep the scripted ceiling — value of *knowing* protection — as a separate experiment.)

**RESULT — the ρ sweep *refutes* that prediction (`experiments/rho_curve.py`, 15 seeds; all
regimes identical except ρ).** We predicted `mult` would cross from *tie* to *win* as ρ rises.
It does the **opposite, monotonically**:

| ρ | `mult+flag` | `off+flag` | greedy | mult − off | verdict |
| ---: | ---: | ---: | ---: | :---: | :--- |
| 1 | 0.862 ±.002 | 0.860 ±.002 | 0.818 | **+0.002** | tie |
| 2 | 0.801 ±.004 | 0.817 ±.005 | 0.770 | **−0.016** | off (weight-only) wins |
| 4 | 0.754 ±.006 | 0.783 ±.003 | 0.693 | **−0.029** | off (weight-only) wins |

**Read this precisely — the learned system is NOT worse than greedy.** Both learned arms *beat*
the scripted greedy baseline at every ρ, and the lead **grows** with ρ (weight-only `off` over
greedy: +0.042 → +0.047 → +0.090 — greedy re-grabs constantly, which gets punishing at high ρ).
What loses is the **retention gate itself**: `mult` (task value × R̂, the paper's Eq. 10
contribution) is beaten by `off` — the *same* learned graph allocator with the gate *removed* —
and that gap **widens** with ρ too. So the finding is narrow and specific: our headline mechanism
is worse than its own ablation, *not* worse than the baseline.

**Why the gate actively hurts (and more so as ρ rises).** The oracle ceiling already showed R̂
has **no headroom** (perfect hindsight R ties weight-only). R̂ is therefore a **noisy
near-constant**: labels in a holdable acquire/defend world barely vary (you hold what you
complete), so R̂ compresses near its 0.95 warm-start. Multiplying a clean weight ordering by a
noisy ≈constant can only **misrank** tasks. A misrank is **cheap at ρ=1** (re-grabbing a lost
task is free) but **expensive at ρ=4** (re-grabbing costs 4×), so the damage **grows with ρ**.
The gate had nothing to gain and something to lose, and ρ raises the stakes on the losing.

**The correct framing — a property of the SETTING, not a decoder failure.** In an **acquire +
defend** action space, retention-weighting can only ever *suppress* acquisitions (down-weight
tasks you'll lose); it has **nothing to encourage**, because you cannot act on the one thing
retention makes valuable — **taking back opponent-held work**. The null is exactly what the
theory predicts once the action space is seen to be one-sided. Retention-aware allocation
becomes an argument for *acting* only when **reversing opponent-held tasks is available** — i.e.
**Phase 2** (symmetric env + a REVERSE action; [`tenure-phase2-plan.md`](tenure-phase2-plan.md)).
Phase 2 is now **the decisive test of the whole thesis**; the ρ null is its motivation, not a
rescue attempt.

**RESULT — Phase 2 comes back NEGATIVE too (`experiments/symmetric_oracle_ceiling.py`, 8 seeds).**
We built the symmetric take mechanic the consultant specified: a successful revert **transfers
ownership to the reverter** (not neutral — Override's descore+rescore is one continuous piece of
work; removes the "both stand on an unowned task forever" deadlock), two robots may **commit to one
reverse target** to out-number a defender (1v1 can't take, 2v1 can), and R-take labels come from
**forced exploration** (committed robot pairs sampling the takeable→untakeable range). Coverage was
confirmed *before* the arms — a weight-only reverse policy takes-and-holds **nothing** (targets
defended tasks 1v1); forced exploration restored the signal (blue-took-and-held 0→890, R-take
**0.585 ± 0.294** among held takes). Three arms (no-reverse / weight-only reverse / **oracle**
reverse with *perfect hindsight* R-take); the decisive test is (c−a): does oracle-reverse beat
*not* reversing? — not (c−b), which only measures self-harm recovery and misled the first runs.

| red | no-rev (a) | WO-rev (b) | oracle-rev (c) | reverse action (b−a) | retention-on-reverse (c−b) | NET (c−a) |
| --- | --- | --- | --- | --- | --- | --- |
| contesting (builder+value) | 0.089 | **0.108** | 0.081 | **+0.020** | **−0.027** | −0.008 |
| builder only | 0.048 | −0.004 | 0.014 | −0.052 | +0.018 | −0.034 |

Against a **contesting** opponent the REVERSE *action* is genuinely valuable (+0.020 over no-reverse)
— "stealing is worth doing." But the retention **oracle** *degrades* it (−0.027 vs weight-only),
landing below even no-reverse. The decomposition shows the oracle helps **neither** channel: steal
(c−b = −0.027) nor acquire (dcmp−a = −0.056). **This replicates Phase 1's exact misranking mechanism
on an independent channel — with *perfect* knowledge** — so a learned two-head (bounded by its own
oracle) cannot help. The reverse action's own value is a separate Phase-2 positive, orthogonal to
retention.

**ρ made LIVE (airtight).** The take cost was then made to bite (`take_thresh = ρ·τ_com`, a defended
task needs ρ·τ_com of pressure to flip) and the ceiling re-run sweeping ρ∈{1,2,4}. Oracle-reverse
**loses at every ρ and by MORE as ρ rises** (NET c−a = −0.005/−0.008/−0.011). Even with a live cost
parameter and a *perfect* R-take oracle, retention-**weighting** on selection does not pay — the
selection/estimation channel is closed with no gap.

### RESOLUTION — retention pays when value is concentrated in a defensible control point

The scope condition, stated positively: **retention-aware allocation pays exactly when losing an
objective is an *outsized* loss** — when value is concentrated in a point whose loss collapses more
than itself. The earlier null results all sat on the other side of that line: **equally-valuable,
equally-reversible objectives**, where you win by grabbing whatever is cheapest and losing one task
costs one task. That is not Override.
Real Override retention is **multiplicative and scarce**: owning a quadrant's **toggle** turns its
goals from 5 to 15 points (a 3× on a whole cluster; `offsim/tests/test_domain_scoring.py`), and
alliance goals are **protected** (irreversible). Losing a toggle collapses a cluster — an *outsized*
loss. Modelling that (`experiments/toggle_retention.py`: regions whose goals score `M×` while you hold
their toggle) flips the result cleanly:

| opponent | M=1 | M=2 | M=3 | M=5 | reading |
| --- | --- | --- | --- | --- | --- |
| **smart** (attacks toggles) | +0.000 | +0.029 | +0.045 | **+0.133** | retention-aware **wins**, gap grows with M |
| **smart, equal speed** | +0.000 | — | +0.046 | **+0.136** | does not need a build-edge |
| **dumb** (ignores toggles) | +0.000 | −0.164 | −0.169 | −0.196 | over-defending a safe point is wasteful |

A retention-aware allocator that **prices a toggle by the cluster it unlocks and DEFENDS it** beats a
retention-blind one **iff (1)** the loss is multiplicative (M>1; a perfect tie at M=1 confirms the
policies are otherwise identical) **and (2)** a smart opponent actually contests the leverage points
(the dumb-red control proves the edge is defence of *contested* value, not merely knowing the
objective). This **reconciles all the negatives**: they were right about the regime they tested, and
retention wins precisely in the regime the real game inhabits. The multiplier is now **productionized
in the env** (`toggle_multiplier`/`toggle_regions`, height- and toggle-weighted held value in both the
J_H integral and the PPO reward; reference-matched, 400-seed; default off = byte-identical), so TENURE
can **train** on it. **And it does (M8, learned):** with the toggle leverage in the observation and a
new REVERSE head, a trained TENURE policy learns to defend its toggles while holding their goals and
beats greedy/defensive ~2.5× — *against a red that ignores goals*.

**Hardened, and the honest decomposition (M9).** Against a **goal-contesting** opponent (a red that
steals toggles *and* contests goals, removing the slack), trained 3 seeds × {multiplicative, off} to
370 updates: **the win survives** — the learned **TENURE(off)** reaches **+0.392 ± .016 and beats the
scripted `defensive` hold-and-defend policy** (+0.229; greedy −0.081), robustly and with the best
behaviour (holds the most toggles, denies red almost entirely). The margin shrinks from 2.5× to ~1.7×,
as expected. **But the retention *head* is decorative — here actively harmful** (`mult − off = −0.294`):
gating an already value-structure-aware observation by a noisy retention estimate misranks and costs
you, the same mechanism as every earlier null. **So the payoff is from *representing* the concentrated
value in the observation/decoder, not from *estimating* retention with the multiplicative head** —
retention-aware allocation wins, retention-*estimation* does not.

**The same scope condition on a second axis — convex stack value.** The multiplier is not the only way
value concentrates. A stack's held value is `w · hᵖ` (`stack_value_power`, default 1). At **p=1 it is
linear** — a height-h stack is worth exactly h height-1 tasks, the *interchangeable* world where
concentration is pointless. At **p>1 it is convex** — a tall stack is worth disproportionately more
than the same pieces spread flat, and its marginal piece is worth more than starting a new one. A
concentrate-vs-flat sweep (`experiments/stack_value_sweep.py`) crosses over exactly as the toggle
M-sweep does: spreading wins at p=1 (gap −0.38) and concentration wins as convexity rises (+0.53 at
p=2, +5.64 at p=3). So the unifying claim is **retention is decorative when objectives are
interchangeable (equal weight, linear height) and pays when value is concentrated in a defensible point
whose loss is outsized — a toggle (multiplier) or a tall stack (convex height).**

**Caveat (rigor).** The oracle-ceiling tables above have one *training* seed per cell — the monotone
trends make the **direction** safe, but **magnitudes** need multiple training seeds. The toggle result
is a **scripted-policy ceiling** (8–12 seeds); the learned demonstration is M8.

### Milestone status

| # | Milestone | Status |
| --- | --- | --- |
| M0 | Spec + §4.1 Override verification | ✅ done — [`tenure-override-verification.md`](tenure-override-verification.md) |
| M1 | Canonical core + reference | ✅ done — 500-seed reference-match |
| M2 | Obs, masking, matching, adversaries, telemetry, labels, adapters | ✅ done |
| M2.5 | `OverrideGraphEnv` adapter (de-risk: TENURE architecture represents Override) | ✅ done — obs matches §3.7, `TENURE.act` runs on the real game |
| M3 | TENURE model + PPO | ✅ built & training on GPU (retention MAE ↓ ~0.14 in a short run) |
| M4 | Baselines (greedy, defensive, CBBA + eval harness; MARL/CapAM/RTAW stubs) | ✅ self-contained ones done; external-code ones stubbed |
| M5 | Override contested extension | ✅ Channel 1 (dwell Toggle reversal + held value + events, gated); v2 objective set + policy retrain pending |
| M6 | Experiment runner / sweeps (`experiments/`) | ✅ done — run dirs, git SHA, CIs, `python -m experiments.sweep` |
| M7 | Full sweeps + ablations | ⏳ infra done; scripted ceiling validated (inverted-U, +0.186 at Override's pf); **learned ρ sweep NEGATIVE** on acquire/defend; **Phase 2 (symmetric + REVERSE) NEGATIVE** even with a perfect oracle and live ρ (loses more as ρ rises); **stacking/bait NEGATIVE** in the full game (a shadowing opponent freezes the build; slack rewards churn) — but the bait *mechanism* is real (serial teardown, validated to closed form). **All negatives share one condition: equal-value tasks.** See *Retention validation status* |
| M7.5 | **Retention RESOLVED — the multiplier regime** | ✅ **POSITIVE**: on Override's real value structure (a **toggle** that multiplies its cluster), a retention-aware allocator that prices+defends the leverage points beats a blind one, gap grows with M (+0.029→+0.133), needs a *contesting* opponent (dumb-red control loses). **Productionized** in the env (`toggle_multiplier`/`toggle_regions`, multiplied held value in J_H + PPO reward; 116 tests, 400-seed reference-match). `experiments/toggle_retention.py` |
| M8 | **Learned toggle win** | ✅ **DONE** — obs exposes toggle leverage (`is_toggle`, who-holds-it, effective value); decoder gained a **REVERSE head** (first-ever two-team training); `toggle_raider` opponent. Trained TENURE beats greedy and defensive ~2.5× vs the raider-only red — but that red ignores goals (slack). `experiments/eval_toggle_behavior.py` |
| M9 | **Hardened + no-head ablation** | ✅ **DONE** — 3 seeds × {mult, off}, 370 updates, vs a **goal-contesting** opponent (`toggle_raider` + `greedy_nearest`). **The win SURVIVES**: TENURE(off) **+0.392 ± .016 beats `defensive` +0.229** (~1.7×, robust) and greedy −0.081. **The retention HEAD is decorative/harmful** (`mult − off = −0.294`) — the gate degrades a policy that already sees the leverage in its obs. **⇒ the payoff is from *representing* the value structure in the observation, not *estimating* retention.** `experiments/eval_harden.py` |

**116 tests pass.** CUDA is available in `.venv313` (RTX 5070 Ti); use it for tests and training:

```bash
# tests — offsim + contested + tenure + baselines + experiments (98 pass)
.venv313/Scripts/python.exe -m pytest -q

# visualise the canonical env (abstract) — filmstrip PNG, animated GIF, or live window
.venv313/Scripts/python.exe -m contested.demo --mode filmstrip --policy greedy --layout uniform --alpha 1.0
.venv313/Scripts/python.exe -m contested.demo --mode gif --policy defensive --out recordings/defensive.gif

# record the REAL Override game (square robots, goal contents, pickup animation)
.venv313/Scripts/python.exe -m offsim.record --out recordings/override_match.gif --seconds 45

# train TENURE on the canonical env (GPU) — full steps in the CUDA section below
.venv313/Scripts/python.exe -m tenure.train --updates 200 --alpha 1.0

# evaluate a baseline; run an experiment sweep
.venv313/Scripts/python.exe -m baselines.evaluate --method greedy --alpha 1.0
.venv313/Scripts/python.exe -m experiments.sweep --methods tenure greedy --alphas 0 1 2 --seeds 0 1 2 --out runs/mini
```

---

## Training TENURE on a CUDA GPU

The canonical environment and the model are both batched tensors, so training runs
entirely on the GPU (the environment steps on-device — there is no CPU/GPU copy per
step). These are the exact steps.

**1. Use the CUDA interpreter.** This repo has two virtualenvs. `.venv313`
(Python 3.13) has the CUDA build of PyTorch (`torch ...+cuXXX`); `.venv` is CPU-only.
Always train with `.venv313`:

```bash
# Windows paths shown; the interpreter is .venv313\Scripts\python.exe
.venv313/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x.x+cuXXX  True  NVIDIA GeForce RTX 5070 Ti
```

If `torch.cuda.is_available()` prints `False`, the CUDA wheel/driver isn't active —
fix that first (install the CUDA build of torch for your driver); the trainer auto-
selects `cuda` when available and falls back to CPU otherwise, so it will *silently*
train on CPU if CUDA is missing.

**2. Smoke test (a few updates) to confirm it runs on the GPU:**

```bash
.venv313/Scripts/python.exe -m tenure.train --updates 5 --horizon 20 --n-tasks 10 --batch-size 128
# first line prints e.g.:  device=cuda params=... mode=multiplicative alpha=1.0 D=40
```

**3. A real training run.** Larger batch = better GPU utilisation. Defaults are
`--batch-size 256 --d-model 128`:

```bash
.venv313/Scripts/python.exe -m tenure.train --updates 500 --alpha 1.0 --batch-size 512
```

**4. The retention-mode ablation** (the paper's causal comparison — run all three and
compare `ret_MAE` and `J_H`):

```bash
.venv313/Scripts/python.exe -m tenure.train --updates 500 --alpha 1.0 --retention-mode multiplicative
.venv313/Scripts/python.exe -m tenure.train --updates 500 --alpha 1.0 --retention-mode feature
.venv313/Scripts/python.exe -m tenure.train --updates 500 --alpha 1.0 --retention-mode off
```

**5. Reading the output.** Each update prints one line:

```
upd  12  J_H=0.318  ret_MAE=0.121  pol=-0.004  val=0.002  ret=0.031  ent=7.90
```

- `J_H` — normalized held value this rollout (the objective; higher is better, ∈[0,1]).
- `ret_MAE` — retention-estimate mean-abs-error vs realized labels (target **< 0.15**).
- `pol` / `val` / `ret` / `ent` — policy, value, retention-regression, and entropy terms.

**Useful flags:** `--alpha` (the contested knob, `0` = no reversal), `--contest-mode
{majority,suppress,none}`, `--n-tasks`, `--n-adversaries`, `--horizon`, `--lr`,
`--lam-r` (retention-loss weight — tune this first), `--seed`. See `tenure/train.py`.

**Throughput note:** reference point is `B≈512`, `D=n_decisions`; scale `--batch-size`
to fill GPU memory. On the RTX 5070 Ti a short run trains ~2.4 s/update at B=128, D=40.

---

## Evaluation, experiments & recordings

**Baselines** (canonical env, `baselines/`). Greedy is acquisition-only; the defensive
heuristic over-commits to holding; CBBA is a time-discounted consensus auction. All are
evaluated in the same env for a fair table; MARL (MAPPO/QMIX/MASAC) runs via the
PettingZoo adapter, and CapAM/RTAW/DC-MRTA are typed stubs pending upstream code.

```bash
.venv313/Scripts/python.exe -m baselines.evaluate --method greedy    --alpha 1.0
.venv313/Scripts/python.exe -m baselines.evaluate --method defensive  --alpha 1.0
```

**Sweeps** (`experiments/`). Each `{method}×{alpha}×{seed}` cell gets a run directory
with resolved config, git SHA, metric series, checkpoint, and a rules-compliance flag;
aggregation reports means with 95% CIs and an optional `J_H`-vs-`alpha` figure.

```bash
.venv313/Scripts/python.exe -m experiments.sweep --methods tenure greedy --alphas 0 1 2 --seeds 0 1 2 --out runs/mini
.venv313/Scripts/python.exe -m experiments.analysis runs/mini --figure runs/mini/jh.png
.venv313/Scripts/python.exe -m experiments.sweep --experiment E1 --updates 500 --seeds 0 1 2 3 4   # headline (compute-heavy)
```

**Retention validation** (`experiments/`, see *Retention validation status* above). The
probe is a fast scripted *ceiling* (oracle flag, not learned); the learned test trains real
policies whose head must estimate retention from geometry (the honest killer control):

```bash
# scripted CEILING — inverted-U over protected_fraction, w×R_oracle vs weight-only (5 seeds, CIs)
.venv313/Scripts/python.exe -m experiments.retention_probe --horizon 90 --seeds 5
# learned KILLER CONTROL — multiplicative vs off at Override's pf, paired, held-out J_H (GPU)
.venv313/Scripts/python.exe -m experiments.retention_learned --protected-fraction 0.15 --seeds 5 --updates 100 --out runs/retention_learned_pf015.json
```

**TENURE on the real game.** `offsim/sim/graph_env.py` (`OverrideGraphEnv`) exposes
Override behind TENURE's §3.7 observation and §3.5 actions, so a canonical-trained
policy plays it unchanged (de-risked: `TENURE.act(obs)` runs on the real match).
`retention_by_site_class(policy, env)` is the E9 tool — mean predicted retention per site
class (alliance goal / neutral goal / toggle) in one match, the head never told the class.

**Recordings** live in [`recordings/`](recordings/):

| File | Env | Shows |
| --- | --- | --- |
| `override_match.gif` | Override (real game) | square robots, goal contents, pickup pulse, color-aware scoring |
| `baseline_greedy.gif` | canonical | greedy churns and re-acquires reverted tasks |
| `baseline_defensive.gif` | canonical | defensive over-parks on held tasks (few reversals, low J_H) |

```bash
# regenerate: real Override match, or a canonical baseline
.venv313/Scripts/python.exe -m offsim.record --out recordings/override_match.gif --seconds 45
.venv313/Scripts/python.exe -m contested.demo --mode gif --policy greedy --out recordings/baseline_greedy.gif
```

---

## Next steps

Ordered, and mostly *compute* now that the infrastructure is in place:

0. **Retention killer control (the current scientific priority).** Finish
   `experiments/retention_learned.py` — `multiplicative` vs `off` at Override's `pf=0.15`,
   ≥5 seeds, paired, deterministic held-out J_H — and confirm the learned head beats a plain
   learned allocator (not just the oracle ceiling). Then repeat at `pf` ∈ {0.10, 0.20, 0.30}
   and add the `feature` mode. Only if this lands is the central claim established; until then
   it is *promising, not proven*. See **Retention validation status** above.
1. **Train TENURE checkpoints** at α=1 to convergence — the `multiplicative` model plus
   its `off` and `feature` ablations (`python -m tenure.train`, now with
   `--protected-fraction` / `--tau-com` / `--adversary`).
2. **Run the E1 α-sweep** (the headline result): `python -m experiments.sweep --experiment E1`,
   then `experiments.analysis`. Confirms TENURE beats greedy and the advantage-vs-α curve.
3. **E9 on Override** with a trained checkpoint: show `R̂ ≈ 1` on alliance goals,
   intermediate on neutral goals, low on toggles — *without telling the head which is which*.
   Then **E10**: predict the Override TENURE-vs-baseline gap from the canonical α curve and
   check it matches (turns two environments into one theory with a confirmed prediction).
4. **M5 remainder** (Override): `objective_set: v2` (adds `DEFEND_GOAL` / `DEFEND_TOGGLE` /
   `REMOVE_OPPONENT_PIN` behind a flag), Channel-2 object reversal, and retrain the shipped
   policy under `contested.enabled`.
5. **Baseline availability**: confirm public code for RTAW / DC-MRTA (else reimplement and
   dagger those rows); wrap CapAM; wire MAPPO/QMIX/MASAC through BenchMARL.
6. **Future direction**: migrate game objects to real 3D physics (stacking / tipping /
   rolling) while keeping the top-down-only renderer.

---

## Implemented match profile

- Four robots: two blue and two red; all four are autonomous and use the same globally selected `tank` or `mecanum` chassis model.
- 120-second match with no drivers: a 30-second opening/autonomous phase followed by a 90-second interaction phase.
- 144 × 144 inch nominal coordinate frame with a 6×6 tile grid. Official GPS coordinates are centered into this frame; the physical field's wall-to-wall dimension is slightly smaller.
- The white 2.5-inch tape is drawn as the official large diagonal X with paired Autonomous Line branches around the central Midfield diamond. Opening uses a coarse southwest/red versus northeast/blue diagonal-side proxy around this shared region.
- Nine GPS-positioned octagonal Goals: two red Alliance Goals in the southwest, two blue Alliance Goals in the northeast, four neutral Short Goals, and one neutral Tall Goal at center.
- Four 25.8-inch Toggles are mounted at the center of the field walls. Four Loaders and colored Load Zones occupy the corners, with two per alliance.
- The VEX U setup has 32 loose field Pins plus one yellow/yellow Pin in the Tall center Goal, 36 field Cups, and one alliance/yellow Pin Preload per Robot. Each alliance has 10 alliance/yellow plus 3 yellow/yellow Match Load Pins and 10 Match Load Cups. A Robot may possess at most one Pin and one Cup.

### Symbolic Goal and scoring proxy

Goal geometry is represented as an ordered bottom-to-top stack of Pin/Cup entries. A Cup placed directly above a Pin records `nested_on=<pin id>`. The simulator does not solve 3D contact geometry.

- A Pin has two colored halves (`blue`, `red`, or neutral `yellow`).
- **Placed halves** include both halves of every Pin in a Goal.
- **Visible halves** are the two halves of the highest Pin in that Goal. Cups preserve symbolic nesting but do not create a second Pin visibility layer.
- Every placed alliance-colored half scores **5 points** for that alliance.
- A placed yellow half in a perimeter Goal scores **10 points** for the alliance owning that Goal's wall/quadrant Toggle. Yellow halves in the Tall center Goal use the alliance with more Robots in Midfield; a tie has no owner.
- Toggles do not directly score. Each alliance Robot whose center is inside the Midfield diamond scores **8 points** for its own alliance.
- The opening winner is the alliance with the higher raw score at 30 seconds and receives the official **12-point Autonomous Bonus**; ties receive no bonus in this prototype.
- **AWP proxy:** by 30 seconds, the alliance must have placed at least one matching alliance half, own at least two Toggles, and both alliance Robots must have placed their Pin Preloads. AWP is telemetry only and does not add match points.

### Protected/removal proxy

A robot cannot remove a Pin from an opponent-protected alliance Goal. Cups and mixed/yellow Pins are treated as neutral placed objects and cannot be removed. The only supported removal is a top, fully alliance-colored Pin from a neutral or own Goal while the robot's Pin slot is empty. Prevented protected/neutral removals increment robot counters and append deterministic telemetry rather than asking for subjective referee input.

## Physics and drivetrains

### Field visualization and Goal-status key

The renderer uses the official Override GPS coordinates and Appendix A descriptions for Goal, Toggle, Loader, tile, tape, and Load Zone placement. The right panel includes a north-oriented compass and a live Goal key:

- **Base color** identifies a red/blue protected Alliance Goal or a dark neutral Goal.
- **T** marks the Tall center Goal; other dark Goals are neutral Short Goals.
- **Colored halo** identifies the alliance currently owning yellow Pin halves for that Goal.
- **Two pips** show the simulator's current visible Pin-half colors.
- **Center number** is the symbolic Pin/Cup stack-entry count; `G0`–`G8` match labels on the field.

On the field itself, a Goal holding scored objects now shows a **two-tone core** (both visible Pin-half colours) sized by stack height, with the entry count over it — empty Goals stay hollow, so you can read contents at a glance without the side panel.

References: [official Override manual](https://www.vexrobotics.com/override-manual) and [official VEX Override GPS coordinates](https://api.vex.com/vr/home/playgrounds/v5rc_override.html).

Simulation ticks are 0.05 seconds. Robots are drawn as rotated **rectangular footprints** oriented along heading (configurable `width`/`length` in `shared/config.yaml`; **square by default**, set them differently for a rectangular chassis). Motion has acceleration/deceleration, bounded yaw acceleration, walls, circular static Goal/Loader obstacles, and pairwise separation — collision itself still uses the circular `radius`. When a robot acquires a Pin or Cup, a brief pulse ring animates the pickup and the carried object is drawn in the robot's gripper (heading direction); it disappears when scored. Scripted objectives collect **color-aware**: a team prefers Pins carrying no opponent-colored half (own-color + yellow), so teams score their own colours instead of feeding the opponent.

- **Tank:** normalized forward and yaw controls; lateral input is ignored.
- **Mecanum:** normalized body-frame forward, lateral, and yaw controls.
- High-level objectives use a deterministic chassis-aware point controller. This is intentionally simple and fast; it is not trajectory optimization.

## Gymnasium environments

Both classes live in `offsim/sim/env.py` and expose a centralized two-blue-robot observation of 160 floats (80 per robot). The observation includes normalized robot state, all robot relatives, Goal relative/visible ownership state, Toggle state, nearest loose Pin/Cup, score, inventories, phase, and time.

### `OverrideContinuousEnv`

- Action space: `Box(-1, 1, shape=(2, 6))`.
- Per robot: `[forward, lateral, yaw, pin_intake, cup_intake, interact]`.
- `pin_intake > 0.5` and `cup_intake > 0.5` attempt collection.
- `interact > 0.5` attempts nearby Goal placement, Toggle claim, then Loader use; `< -0.5` attempts a legal Pin removal.
- One Gym step advances one 0.05-second physics tick. Both red robots use scripted objectives.

### `OverrideStrategyEnv`

- Action space: `MultiDiscrete([10, 10])`, one objective for each allied robot.
- One Gym step executes both objectives deterministically for 2 simulated seconds while two scripted red robots act.
- `action_masks()` returns the flattened 20-entry boolean mask expected by MaskablePPO for the two branches.

| ID | Objective |
|---:|---|
| 0 | `IDLE` |
| 1 | `COLLECT_PIN` |
| 2 | `COLLECT_CUP` |
| 3 | `SCORE_NEAREST_GOAL` |
| 4 | `SCORE_ALLIANCE_GOAL` |
| 5 | `SCORE_MIDFIELD_GOAL` |
| 6 | `USE_LOADER` |
| 7 | `CLAIM_TOGGLE` |
| 8 | `DEFEND_MIDFIELD` |
| 9 | `REMOVE_OWN_PIN` |

Invalid sampled actions are defensively converted to `IDLE`; MaskablePPO normally prevents them from being sampled. Reward is `(blue score delta - red score delta) / 5`, plus `+10` for a terminal win or `-10` for a terminal loss.

## Installation

Python 3.10+ is recommended. No dependencies beyond the existing requirements are needed.

```bash
pip install -r offsim/requirements.txt
```

All commands below run from the repository root.

## Commands

```bash
# Visual deterministic 2v2 autoplay (1x is real-time)
python offsim/main.py demo --chassis tank
python offsim/main.py demo --chassis mecanum --opponent toggle
python offsim/main.py demo --chassis tank --speed 0.5  # slow motion

# Headless complete match / short smoke
python offsim/main.py demo --headless --matches 1 --seed 7
python offsim/main.py demo --headless --max-decisions 3

# Centralized MaskablePPO training
python offsim/main.py train --chassis tank --timesteps 1000000 --n-envs 4
python offsim/main.py train --chassis mecanum --timesteps 100000 --n-envs 1

# Evaluate, export, and validate a newly trained Override checkpoint
python offsim/main.py eval --model models/final_model.zip --episodes 5 --chassis tank
python offsim/main.py export --model models/final_model.zip --output models/onnx/override.onnx
python offsim/main.py validate --onnx models/onnx/override.onnx --model models/final_model.zip

# Tests
python -m pytest -q
```

Old Push Back checkpoints were removed and are incompatible with the 160-float observation and centralized two-branch action contract. ONNX export emits 20 unmasked logits; a consumer must split them into two 10-action branches and apply the corresponding mask before selection.

**Renderer controls:** `Space` pause/resume, `S` single strategy step while paused, `R` reset, `+/-` halve/double playback speed from `0.25x` through `16x`. Visual mode draws every 0.05-second physics tick; `1x` runs in real time, while headless demos and training remain uncapped. The current renderer intentionally has no field editor or manual driving mode.

## Project layout

```text
offsim/                     # Override game simulator (below) — the applied env
├── main.py                 # Override CLI
├── record.py               # record an Override match as a GIF
├── shared/config.yaml      # action/profile/physics/scoring + contested block
├── sim/
│   ├── config.py           # strict shared-contract loader
│   ├── robot.py            # tank/mecanum kinematics + footprint dims
│   ├── field.py            # domain, stacks, scoring, collisions, phases, contested Channel 1
│   ├── env.py              # continuous and strategy Gym environments
│   ├── graph_env.py        # OverrideGraphEnv — Override behind TENURE's §3.7 interface
│   ├── opponent.py         # scripted opponent objective selection
│   └── renderer.py         # Pygame renderer (rect robots, goal contents, pickup anim)
├── training/ · export/ · tests/

contested/                  # Component A — canonical env (α/β/N/M/K sweeps, training)
├── config.py core.py reference.py observation.py actions.py adversaries.py
├── labels.py telemetry.py render.py demo.py
├── adapters/{gym_vec,pettingzoo_par}.py   └── tests/
tenure/                     # Component C — model: encoder, heads, policy, ppo, train
baselines/                  # Component D — greedy, defensive, cbba, evaluate, external stubs, marl/
experiments/                # Component E — runner, sweep, analysis, E1–E10 map
                            #   retention_probe.py (scripted ceiling) · retention_learned.py (learned killer control)
recordings/                 # sample GIFs (canonical baselines + real Override match)
```

## Honest limitations

- Field element placement, tape, Goal colors/types, and VEX U starting counts follow Override Manual v1.0 and official GPS references. The 144-inch coordinate frame, dense cluster offsets, collision radii, opening-side enforcement, Goal protection behavior, and AWP condition remain explicit simulator proxies and must not be treated as referee-accurate.
- Pins/Cups are points or symbolic Goal entries; there is no rigid-body object physics, tipping, Pin rotation, Cup volume, stack stability, entanglement, or 3D occlusion.
- Subjective referee rules (possession nuance, incidental vs intentional contact, trapping, damage, match affecting violations, and field reset tolerances) are not adjudicated. Supported illegal removals are prevented and telemetered.
- Static navigation is a reactive point controller and can stall in congestion. Collisions are deterministic circular separation, not momentum/traction simulation.
- Scripted opponents provide full-match activity but are not competitive strategy benchmarks.
- Offline CQL/IQL and physical-robot/ROS deployment are not implemented for the centralized `MultiDiscrete` Override contract.
- The policy sees complete simulator state; perception noise, localization error, communications, actuator failures, and sim-to-real transfer are outside this prototype.
