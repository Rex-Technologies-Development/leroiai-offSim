"""Task 1 — find the environment regime where retention matters.

TENURE only has room to win where *holding* a task beats *churning* (re-grabbing it
after it's reverted). We probe that cheaply, without training, by racing two scripted
policies across the environment knobs:

- ``greedy``    — pure churn: grab the nearest valuable incomplete task, never defend.
- ``defensive`` — pure hold: defend held tasks that are under threat.

For each knob setting we report each policy's held value ``J_H``. The regime we want
is where ``defensive`` is at least competitive with ``greedy`` (gap >= ~0): holding
pays, so *selective* defense (what TENURE learns) can beat both. Where greedy
dominates everywhere, the environment rewards churning and needs reshaping (more
adversaries, higher alpha, fewer tasks, or slower completion) before the headline
sweep is worth running.
"""
from __future__ import annotations

import argparse

from baselines import DefensiveHeuristic, GreedyAllocator
from baselines.evaluate import evaluate_policy
from contested.config import load_config


def sweep(alphas, n_advs, n_tasks, n_robots, horizon, batch, seeds, device=None):
    greedy, defensive = GreedyAllocator(), DefensiveHeuristic()
    print(f"n_robots={n_robots} n_tasks={n_tasks} horizon={horizon}s  "
          f"(batch={batch} x {seeds} seeds; J_H in [0,1], higher is better)\n")
    header = f"{'alpha':>6} {'n_adv':>6} | {'greedy':>8} {'defensive':>10} {'gap(d-g)':>9} | {'reads':>18}"
    print(header)
    print("-" * len(header))
    hits = []
    for alpha in alphas:
        for n_adv in n_advs:
            cfg = load_config(overrides=dict(
                alpha=alpha, n_adversaries=n_adv, n_tasks=n_tasks,
                n_robots=n_robots, horizon_T=horizon))
            g = evaluate_policy(greedy.act, cfg, batch_size=batch, n_seeds=seeds, device=device)["J_H"]
            d = evaluate_policy(defensive.act, cfg, batch_size=batch, n_seeds=seeds, device=device)["J_H"]
            gap = d - g
            if gap > 0.03:
                verdict = "HOLDING pays *"
            elif gap > -0.05:
                verdict = "close (selective)"
            else:
                verdict = "churn dominates"
            if gap > -0.05:
                hits.append((alpha, n_adv, gap))
            print(f"{alpha:6.2f} {n_adv:6d} | {g:8.3f} {d:10.3f} {gap:+9.3f} | {verdict:>18}")

    print()
    if hits:
        best = max(hits, key=lambda h: h[2])
        print(f"=> retention-relevant regimes found (gap >= -0.05): {len(hits)}.")
        print(f"   strongest at alpha={best[0]}, n_adversaries={best[1]} (gap {best[2]:+.3f}).")
        print("   Use a setting from this band for M3 training and the E1 sweep.")
    else:
        print("=> greedy dominates everywhere here — the environment rewards churning.")
        print("   Reshape it: raise alpha / n_adversaries, cut n_tasks, or lengthen tau_com.")


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1: locate the regime where retention matters")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--n-advs", dest="n_advs", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=12)
    p.add_argument("--n-robots", dest="n_robots", type=int, default=4)
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--batch-size", dest="batch", type=int, default=128)
    p.add_argument("--seeds", type=int, default=2)
    args = p.parse_args()
    sweep(args.alphas, args.n_advs, args.n_tasks, args.n_robots, args.horizon, args.batch, args.seeds)


if __name__ == "__main__":
    main()
