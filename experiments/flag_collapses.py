"""Path-(b) collapse flagging: eval each hd2conv cell (arm x seed) deterministically, compute each ARM's
median J_H, and flag cells below 50% of it as training collapses (per the pre-registered rule in
convergence-protocol.md). Prints the per-cell table, the flagged cells, and the exact restart commands
(same env --seed, new --init-seed = seed+100). Run after the first pass completes.

    python -m experiments.flag_collapses --prefix tenure/checkpoints/hd2conv
"""
from __future__ import annotations

import argparse
import os
import statistics

from experiments.eval_isolation import _load, ARMS
from experiments.eval_toggle_behavior import _rollout
from experiments.eval_harden import _obs

ARM_FLAGS = {"off_aware": ("off", "--expose-toggle"), "off_blind": ("off", ""),
             "mult_aware": ("multiplicative", "--expose-toggle")}
REGIME = ("--updates 420 --symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 "
          "--horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --adversary toggle_raider greedy_nearest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag (b) training collapses (<50% of arm median)")
    ap.add_argument("--prefix", default="tenure/checkpoints/hd2conv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=6)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    ap.add_argument("--init-offset", dest="init_offset", type=int, default=100,
                    help="restart init-seed = seed + init_offset*restart_k")
    args = ap.parse_args()

    jh = {}                                                    # (arm, seed) -> J_H
    for arm in ARMS:
        for s in args.seeds:
            p = f"{args.prefix}_{arm}_s{s}.pt"
            if not os.path.exists(p):
                continue
            cfg, pol = _load(p)
            rs = [_rollout(cfg, lambda c: pol.act(_obs(c, cfg), deterministic=True)["action"], es, args.batch, "cpu")
                  for es in range(args.eval_seeds)]
            jh[(arm, s)] = statistics.fmean(r[0] for r in rs)

    print(f"\n{'arm':>12} | median | per-seed J_H (<STUCK = below 50% of arm median)")
    print("-" * 78)
    flagged = []
    for arm in ARMS:
        vals = {s: jh[(arm, s)] for s in args.seeds if (arm, s) in jh}
        if not vals:
            continue
        med = statistics.median(vals.values())
        cells = []
        for s, v in vals.items():
            stuck = v < 0.5 * med
            cells.append(f"s{s}:{v:+.3f}{'<STUCK' if stuck else ''}")
            if stuck:
                flagged.append((arm, s))
        print(f"{arm:>12} | {med:6.3f} | " + "  ".join(cells))

    print(f"\n{len(flagged)} collapse(s) flagged: {flagged if flagged else 'NONE -- clean, proceed to decomposition'}")
    if flagged:
        print("\nrestart commands (same env --seed, new --init-seed = seed+100; overwrites the collapsed ckpt):")
        py = ".venv313/Scripts/python.exe"
        for arm, s in flagged:
            mode, extra = ARM_FLAGS[arm]
            out = f"tenure/checkpoints/{os.path.basename(args.prefix)}_{arm}_s{s}.pt"
            print(f"  {py} -m tenure.train {REGIME} --seed {s} --init-seed {s + args.init_offset} "
                  f"--retention-mode {mode} {extra} --save {out}")


if __name__ == "__main__":
    main()
