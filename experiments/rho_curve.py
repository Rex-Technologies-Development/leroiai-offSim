"""The Phase 1 headline: the rho crossover curve.

TENURE **multiplicative** (retention gate, Eq. 10) vs **off** (weight-only allocator),
both WITH the protected flag, across re-acquisition cost rho in {1, 2, 4} at alpha=0.75.
Each checkpoint is evaluated at ITS OWN training regime (rho included) on held-out seeds,
so the comparison at each rho is fair. Prints each regime so comparability is visible.

The hypothesis (consultant + us): the multiplicative gate should pull AHEAD of weight-only
as rho rises, because that is when losing a task is expensive enough that predicting
retention pays. A flat-or-negative curve says rho alone does NOT rescue the gate in the
asymmetric acquire/defend env -> the payoff must live on the offensive/reverse side (Phase 2).

    python -m experiments.rho_curve --seeds 15
"""
from __future__ import annotations

import argparse
import os

import torch

from baselines import REGISTRY
from baselines.evaluate import evaluate_policy, tenure_policy_fn
from contested.config import load_config
from contested.core import default_device
from tenure.policy import TenurePolicy

POINTS = [(1.0, ""), (2.0, "_rho2"), (4.0, "_rho4")]
TIE = 0.01  # pre-committed real-effect threshold (prior real gaps were 0.018-0.031)


def eval_ckpt(path: str, seeds: int, batch: int, device) -> tuple[dict, float, float]:
    ck = torch.load(path, map_location="cpu")
    overrides = dict(ck.get("overrides") or {})
    cfg = load_config(overrides=overrides or None)
    pol = TenurePolicy(d_model=ck.get("d_model", 128),
                       retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7),
                       retention_head=ck.get("retention_head", "regression"))
    pol.load_state_dict(ck["state_dict"]); pol.to(device).eval()
    r = evaluate_policy(tenure_policy_fn(pol), cfg, batch_size=batch, n_seeds=seeds, device=device)
    return overrides, r["J_H"], r["J_H_std"]


def main() -> None:
    ap = argparse.ArgumentParser(description="rho crossover curve: mult vs off across rho")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--batch-size", dest="batch", type=int, default=256)
    ap.add_argument("--prefix", default="runs/tenure_")
    args = ap.parse_args()
    device = default_device()
    print(f"device={device} seeds={args.seeds} batch={args.batch}\n")

    res: dict[tuple[float, str], tuple[float, float]] = {}
    for rho, suffix in POINTS:
        ov_rho = None
        for mode in ("mult", "off"):
            path = f"{args.prefix}{mode}flag_a075{suffix}.pt"
            if not os.path.exists(path):
                print(f"  MISSING {path}"); continue
            ov, jh, std = eval_ckpt(path, args.seeds, args.batch, device)
            res[(rho, mode)] = (jh, std); ov_rho = ov
            print(f"  {mode:>4} rho={rho:.0f}: J_H={jh:.3f} +/-{std:.3f}  regime={ov}")
        if ov_rho is not None:  # scripted-greedy baseline at this same regime
            cfg = load_config(overrides=ov_rho or None)
            g = evaluate_policy(REGISTRY["greedy"]().act, cfg, batch_size=args.batch,
                                n_seeds=args.seeds, device=device)
            res[(rho, "greedy")] = (g["J_H"], g["J_H_std"])
            print(f"  grdy rho={rho:.0f}: J_H={g['J_H']:.3f} +/-{g['J_H_std']:.3f}")

    print(f"\n{'=' * 78}\nrho crossover curve (alpha=0.75, +flag, {args.seeds} seeds)")
    print(f"{'rho':>4} | {'mult':>8} | {'off':>8} | {'greedy':>8} | {'mult-off':>9} | {'mult-grdy':>9} | verdict")
    print("-" * 78)
    for rho, _ in POINTS:
        if all((rho, k) in res for k in ("mult", "off", "greedy")):
            m = res[(rho, "mult")][0]; o = res[(rho, "off")][0]; g = res[(rho, "greedy")][0]
            gap = m - o
            v = "mult WINS" if gap > TIE else ("off wins" if gap < -TIE else "TIE")
            print(f"{rho:>4.0f} | {m:8.3f} | {o:8.3f} | {g:8.3f} | {gap:+9.3f} | {m - g:+9.3f} | {v}")
    print("\nhypothesis: the gate (mult) should pull AHEAD of weight-only (off) as rho rises.")
    print("both mult and off are the LEARNED system; greedy is the scripted baseline.")


if __name__ == "__main__":
    main()
