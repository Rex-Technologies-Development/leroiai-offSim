"""Graph the path-(b) decomposition (step-toggle env, unbiased fixed-420, collapses restarted).

Left: per-seed J_H for each policy (5 learned/scripted), mean +/- 95% CI -- shows the three LEARNED arms
clustered together and well above the scripted baselines. Right: the two paired gaps (isolation, head)
with 95% CIs -- both spanning zero (within noise). This is the honest 'learned >> scripted, but
representation & head not separable' result.

    python -m experiments.graph_decomposition --prefix tenure/checkpoints/hd2conv --out recordings/decomposition.png
"""
from __future__ import annotations

import argparse
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import REGISTRY
from experiments.eval_isolation import _load, ARMS
from experiments.eval_toggle_behavior import _rollout
from experiments.eval_harden import _obs


def _ci95(vals):
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, 1.96 * s / (len(vals) ** 0.5 if vals else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="tenure/checkpoints/hd2conv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=6)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    ap.add_argument("--out", default="recordings/decomposition.png")
    args = ap.parse_args()

    per = {a: {} for a in ARMS}                                   # arm -> seed -> J_H
    cfg = None
    for arm in ARMS:
        for s in args.seeds:
            p = f"{args.prefix}_{arm}_s{s}.pt"
            if not os.path.exists(p):
                continue
            cfg, pol = _load(p)
            rs = [_rollout(cfg, lambda c: pol.act(_obs(c, cfg), deterministic=True)["action"], es, args.batch, "cpu")
                  for es in range(args.eval_seeds)]
            per[arm][s] = statistics.fmean(r[0] for r in rs)
    base = {}
    for b in ("defensive", "greedy"):
        pol_b = REGISTRY[b]()
        rs = [_rollout(cfg, lambda c: pol_b.act(c.state, cfg), s, args.batch, "cpu") for s in args.seeds]
        base[b] = statistics.fmean(r[0] for r in rs)

    labels = ["off_aware\n(repr)", "mult_aware\n(repr+head)", "off_blind\n(no repr)", "defensive", "greedy"]
    arms_order = ["off_aware", "mult_aware", "off_blind"]
    colors = ["#2a7", "#27a", "#a72", "#999", "#bbb"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 2]})

    # left: per-seed points + mean +/- 95% CI
    for i, arm in enumerate(arms_order):
        vals = list(per[arm].values())
        m, ci = _ci95(vals)
        axL.scatter([i] * len(vals), vals, color=colors[i], alpha=0.5, s=28, zorder=3)
        axL.errorbar(i, m, yerr=ci, fmt="o", color=colors[i], capsize=5, ms=9, zorder=4)
    for j, b in enumerate(("defensive", "greedy")):
        axL.scatter([3 + j], [base[b]], color=colors[3 + j], marker="s", s=80, zorder=4)
    axL.axhline(base["defensive"], color="#999", ls="--", lw=0.8, alpha=0.7)
    axL.set_xticks(range(5)); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("J_H (held value differential)")
    axL.set_title("Learned arms cluster together, all >> scripted", fontsize=10)
    axL.grid(axis="y", alpha=0.25)

    # right: the two gaps with 95% CI (paired by seed)
    common = sorted(set(per["off_aware"]) & set(per["off_blind"]) & set(per["mult_aware"]))
    iso = [per["off_aware"][s] - per["off_blind"][s] for s in common]
    head = [per["mult_aware"][s] - per["off_aware"][s] for s in common]
    for k, (name, d) in enumerate([("ISOLATION\naware - blind", iso), ("HEAD\nmult - aware", head)]):
        m, ci = _ci95(d)
        col = "#2a7" if k == 0 else "#c44"
        axR.errorbar(k, m, yerr=ci, fmt="o", color=col, capsize=6, ms=10)
        axR.annotate(f"{m:+.3f}\n±{ci:.3f}", (k, m), textcoords="offset points", xytext=(14, -4), fontsize=9)
    axR.axhline(0, color="k", lw=1)
    axR.set_xlim(-0.5, 1.8); axR.set_xticks([0, 1]); axR.set_xticklabels(["ISOLATION\naware-blind", "HEAD\nmult-aware"], fontsize=8)
    axR.set_ylabel("paired gap (95% CI)")
    axR.set_title("Both gaps span 0 -> within noise", fontsize=10)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Step-toggle env, path-(b) unbiased (n={len(common)}): learned allocation wins; "
                 f"representation & head not separable", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}  (n={len(common)} seeds)")
    print(f"  isolation {_ci95(iso)[0]:+.3f} ± {_ci95(iso)[1]:.3f} | head {_ci95(head)[0]:+.3f} ± {_ci95(head)[1]:.3f}")


if __name__ == "__main__":
    main()
