"""Does NON-LINEAR (convex) stack value make retention/concentration pay? -- the user's idea.

Held value of a height-h stack is w * h ** power. At power=1 (linear) a height-H stack is worth
exactly H height-1 tasks -- objects are INTERCHANGEABLE, the equal-value world where every retention
experiment came back decorative. power>1 makes value CONVEX: a tall stack is worth H^power >> H, and
the marginal piece on a height-h stack ((h+1)^p - h^p) is worth more than starting a new one. That
turns a tall stack into a defensible, outsized-loss control point -- the SAME mechanism as the toggle
multiplier, applied to height. So concentration should cross from decorative (tie at power=1) to
winning (gap grows with power), exactly like the toggle M-sweep.

Two committed policies in the real mechanic, vs a beatable opponent (few `greedy_nearest` red that
chase blue's completed stacks but cannot out-number a 4-robot pack, so the build is not frozen --
the "we out-build the baseline" premise):
  * FLAT: acquire the nearest neutral task, never stack -> many height-1 tasks (wide, cheap, linear).
  * CONCENTRATE (k=1): pile all robots on ONE task, build it to the cap and hold -> one tall stack.

PRE-REGISTERED: CONCENTRATE ties FLAT at power=1 (linear = interchangeable) and pulls ahead as power
rises (convex = a tall stack is worth disproportionately more and is expensive to tear down).

    python -m experiments.stack_value_sweep --powers 1 1.5 2 3 --cap 5 --seeds 8
"""
from __future__ import annotations

import argparse
import statistics

import torch

from contested.config import load_config
from contested.core import CanonicalCore, default_device
from experiments.stacking_ceiling import FlatBuilder, Concentrator, _rollout


def _diag_height(cfg, mk, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    pol = mk()
    for _ in range(cfg.n_decisions):
        core.step(pol.act(core.state, cfg))
    s = core.state
    tallest = (s.task_height * s.task_c).max().item()
    bt = s.task_c.sum(-1).float().mean().item()
    return bt, tallest


def run_power(power, cap, seeds, batch, dev, n_tasks, n_adv, alpha, k, diagnose):
    over = {"symmetric": True, "stack_cap": cap, "stack_value_power": power, "alpha": alpha,
            "n_tasks": n_tasks, "n_robots": 4, "n_adversaries": n_adv,
            "adversary_population": ["greedy_nearest"], "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    flat = [_rollout(cfg, FlatBuilder(), s, batch, dev) for s in range(seeds)]
    conc = [_rollout(cfg, Concentrator(k=k), s, batch, dev) for s in range(seeds)]
    m = statistics.fmean
    gap = m(conc) - m(flat)
    tag = "CONCENTRATE wins" if gap > 0.01 else ("flat wins" if gap < -0.01 else "tie")
    print(f"power={power:<4} | FLAT={m(flat):+.3f}  CONCENTRATE={m(conc):+.3f}  gap={gap:+.3f}  [{tag}]")
    if diagnose:
        for name, mk in [("flat", lambda: FlatBuilder()), ("conc", lambda: Concentrator(k=k))]:
            bt, tall = _diag_height(cfg, mk, 0, batch, dev)
            print(f"        [{name}] blue holds {bt:.1f} tasks, tallest stack {tall:.0f}")
    return power, gap


def main() -> None:
    ap = argparse.ArgumentParser(description="Non-linear stack value sweep (does convex value make concentration pay)")
    ap.add_argument("--powers", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    ap.add_argument("--n-tasks", dest="n_tasks", type=int, default=8)
    ap.add_argument("--n-adversaries", dest="n_adv", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=1, help="stacks the concentrator builds (1 = one tall stack)")
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    dev = default_device()
    print(f"STACK VALUE SWEEP (device={dev}, cap={args.cap}, {args.seeds} seeds, n_tasks={args.n_tasks}, "
          f"n_adv={args.n_adv}, k={args.k}) -- does CONVEX stack value make concentration pay?")
    print("Pre-registered: tie at power=1 (linear=interchangeable), CONCENTRATE pulls ahead as power rises.\n")
    gaps = []
    for p in args.powers:
        _, g = run_power(p, args.cap, args.seeds, args.batch, dev, args.n_tasks, args.n_adv,
                         args.alpha, args.k, args.diagnose)
        gaps.append((p, g))
    print()
    lo, hi = gaps[0], gaps[-1]
    grows = hi[1] > lo[1] + 0.01
    if grows:
        print(f"CONFIRMED: concentration's edge GROWS with convexity (power {lo[0]}: {lo[1]:+.3f} -> "
              f"power {hi[0]}: {hi[1]:+.3f}). Retention pays on stacks once value is non-linear.")
    else:
        print(f"Not confirmed: gap {lo[0]}->{hi[0]} = {lo[1]:+.3f}->{hi[1]:+.3f}. Inspect above.")


if __name__ == "__main__":
    main()
