"""E9 on the REAL Override game: does the retention head read a site's class from geometry alone?

Runs a canonical-trained TENURE policy on OverrideGraphEnv (the real match behind the Section 3.7
interface) and averages the head's predicted retention R_hat per NATIVE site class over N matches. The
head is never told a site's class, so a class-ordered R_hat is a genuine geometry read, not a memorised
layout. Override folds three alpha regimes into space at once:

  alliance_goal  protected (tau_rev = inf, alpha = 0)   -> expect R_hat ~ high
  neutral_goal   removable (intermediate alpha)          -> expect R_hat intermediate
  toggle         flips both ways (alpha >= 1)            -> expect R_hat low

This is the ESTIMATION half of the retention-head story: per the isolation ablation the head does not
improve ALLOCATION (acting on R_hat ties/loses), but E9 asks the separate question of whether it can
ESTIMATE retention structure at all. A clean class ordering here + 'acting on it doesn't help' there is
the complete, honest account of the head. Needs a vanilla checkpoint matching the base interface
(task_dim=7, non-symmetric, retention head active); train it with:

    python -m tenure.train --updates 220 --alpha 1.0 --protected-fraction 0.15 \
        --retention-mode multiplicative --save tenure/checkpoints/e9_vanilla.pt
    python -m experiments.eval_e9_override tenure/checkpoints/e9_vanilla.pt --matches 8
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

import torch

from offsim.sim.graph_env import OverrideGraphEnv, retention_by_site_class
from tenure.policy import TenurePolicy


def main() -> None:
    ap = argparse.ArgumentParser(description="E9: retention by native site class on real Override")
    ap.add_argument("checkpoint")
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--chassis", default="tank")
    ap.add_argument("--opponent", default="mixed")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    if ck.get("symmetric") or ck.get("task_dim", 7) != 7:
        print(f"WARNING: checkpoint task_dim={ck.get('task_dim')} symmetric={ck.get('symmetric')} -- "
              f"E9 needs the base interface (task_dim=7, non-symmetric). It will likely error/mismatch.")
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"]); pol.eval()

    per_class = defaultdict(list)
    for seed in range(args.matches):
        env = OverrideGraphEnv(chassis=args.chassis, opponent=args.opponent, contested=True, seed=seed)
        out = retention_by_site_class(pol, env)
        for cls, v in out.items():
            per_class[cls].append(v)

    print(f"\nE9 -- retention by native site class, {args.matches} Override matches "
          f"(chassis={args.chassis}, red={args.opponent})\n")
    print(f"{'site class':>14} | {'R_hat mean+/-std':>18} | expected")
    print("-" * 56)
    order = ["alliance_goal", "neutral_goal", "toggle"]
    exp = {"alliance_goal": "high (protected)", "neutral_goal": "mid (removable)", "toggle": "low (flips)"}
    means = {}
    for cls in order:
        if cls not in per_class:
            continue
        m = statistics.fmean(per_class[cls]); s = statistics.pstdev(per_class[cls]) if len(per_class[cls]) > 1 else 0.0
        means[cls] = m
        print(f"{cls:>14} | {m:7.3f} +/- {s:5.3f}     | {exp[cls]}")
    if {"alliance_goal", "toggle"} <= set(means):
        gap = means["alliance_goal"] - means["toggle"]
        ok = means.get("alliance_goal", 0) >= means.get("neutral_goal", 0) >= means.get("toggle", 1)
        print(f"\nordering alliance >= neutral >= toggle: {'HOLDS' if ok else 'does NOT hold'} "
              f"(alliance - toggle = {gap:+.3f})")
        print("Read: a positive, class-ordered spread = the head reads retention from geometry on the real "
              "game\n(estimation works). Per the isolation ablation, acting on it still does not improve "
              "allocation.")


if __name__ == "__main__":
    main()
