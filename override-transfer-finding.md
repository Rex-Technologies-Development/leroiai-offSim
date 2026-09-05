# Override transfer: TENURE plays AND wins the real game once converged (2026-09-04)

**Question (user):** the trained policies must actually play the real Override game — "the whole
point is to have TENURE play the override game."

**Answer: yes — it plays it, and once converged it is COMPETITIVE, winning the majority of matches
against the scripted opponent, primarily by controlling toggles (the contested-value mechanic).**

## Setup
`e9_vanilla.pt` = a BASE-interface checkpoint (task_dim=7, non-symmetric), trained on the canonical
contested env with `--horizon 30` (D=60, matched to Override's 60 decisions/match), transferred
zero-shot into `OverrideGraphEnv` vs the scripted `mixed` opponent. Blue = TENURE, red = scripted.
Recordings in `recordings/override_sim/`.

## Scores (6-match no-render sweep, `scratchpad/override_score.py`)

| Blue controller | J_H(canon) | mean blue | mean red | blue wins |
|---|---|---|---|---|
| Untrained TENURE (floor) | – | 78 ± 43 | 285 | 0/6 |
| TENURE @120 (under-trained) | 0.38 | 107 ± 2 | 279 | 0/6 |
| **TENURE @260 (converged)** | **0.47** | **202 ± 36** | **182** | **4/6** |
| Scripted, Override-native (ceiling) | – | 242 ± 115 | 196 | 3/6 |

**Converged TENURE is competitive with the scripted ceiling** — fewer points (202 vs 242) but MORE
match wins (4/6 vs 3/6), because it holds red down more (182 vs 196).

## The under-trained trap (corrected honestly)
The J_H≈0.38 reading at update 120 looked like a plateau (0.385/0.386/0.376 over updates 130–150)
and scored a rigid 107 / 0-of-6 on Override. It was NOT converged — J_H resumed climbing to ~0.47 by
update 260 and Override competitiveness came with it (107→202, 0/6→4/6). **Lesson (again): do not
judge from a non-converged checkpoint.** The earlier "legal but not competitive" conclusion was that
mistake pointing pessimistic.

## Why it wins = TOGGLE CONTROL (the thesis on the real game), NOT the loader
Diagnostic on @260 (`scratchpad/override_diag.py`, seed 3): action mix over 120 ally-robot-decisions
= `ACQ-GOAL 50, DEF-TOG 21, ACQ-TOG 14, IDLE 34, DEF-GOAL 1`. Vs the under-trained policy it roughly
**doubles toggle work (35 vs 20 toggle actions, DEF-TOG 12→21)** — it claims and then DEFENDS the
toggles that multiply cluster value. Score climbs monotonically 30→167 (vs the earlier oscillation
from being descored), and red is suppressed to 20–45 mid-game. This is the retention-aware /
concentrated-contested-value story executing on the real Override field: hold the high-leverage points.

- **Env is fair** (`scratchpad/scripted_vs_scripted.py`): both sides scripted → blue 242, wins 3/6.
- **Loader-reload adapter fix is INERT** (still 13 pin + 10 cup unused at match end): `nearest_object`
  almost always returns a field object, so the field-empty branch never fires. Kept as a faithful
  correctness improvement (6/6 graph-env tests pass) but it is NOT load-bearing; the gain is pure
  policy convergence.

## E9 — retention HEAD by native site class (8 matches, `experiments.eval_e9_override`)
`alliance_goal 0.812 ± 0.005 | neutral_goal 0.764 ± 0.004 | toggle 0.793 ± 0.005`. Ordering
alliance ≥ neutral ≥ toggle **does NOT hold** (toggle isn't lowest); spread is ~0.05 (near-flat).
**The head does not cleanly read retention structure from geometry** — decorative, consistent with the
step-env decomposition null (head within noise). The clean split: the POLICY exploits contested
structure well (toggle control → 4/6 wins) while the explicit retention-ESTIMATION head is flat. That
is exactly the paper's thesis — representation/allocation of concentrated contested value matters; the
retention-estimation head does not.

## Verified artifacts
- `recordings/override_sim/tenure_override_final_s3.gif` (blue 167–140 WIN),
  `tenure_override_final_s5.gif` (blue 177–160 WIN), `tenure_override_e40.gif` (under-trained, for contrast)
- `scratchpad/override_score.py`, `scripted_vs_scripted.py`, `override_diag.py`
