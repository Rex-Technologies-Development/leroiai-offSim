"""Evaluate a trained TENURE checkpoint vs baselines at its own training regime.

Loads the policy saved by ``tenure.train --save`` (which stores the env ``overrides``
it trained under), then deterministically evaluates it and the named baselines in the
SAME regime, printing J_H and the retention-relevant telemetry so the learned policy
is judged by what it actually does, not the noisy on-policy training reward.

    python -m experiments.eval_checkpoint runs/tenure_demo.pt --baselines greedy defensive
"""
from __future__ import annotations

import argparse

import torch

from baselines import REGISTRY
from baselines.evaluate import evaluate_policy, tenure_policy_fn
from contested.config import load_config
from contested.core import default_device
from tenure.policy import TenurePolicy


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval a TENURE checkpoint vs baselines")
    ap.add_argument("checkpoint")
    ap.add_argument("--baselines", nargs="+", default=["greedy", "defensive"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", dest="batch", type=int, default=256)
    args = ap.parse_args()

    device = default_device()
    ck = torch.load(args.checkpoint, map_location="cpu")
    overrides = dict(ck.get("overrides") or {})
    cfg = load_config(overrides=overrides or None)
    policy = TenurePolicy(d_model=ck.get("d_model", 128),
                          retention_mode=ck.get("retention_mode", "multiplicative"),
                          task_dim=ck.get("task_dim", 7),
                          retention_head=ck.get("retention_head", "regression"),
                          symmetric=ck.get("symmetric", False))
    policy.load_state_dict(ck["state_dict"])
    policy.to(device).eval()

    rows = [(f"TENURE({ck.get('retention_mode', '?')})",
             evaluate_policy(tenure_policy_fn(policy), cfg, batch_size=args.batch,
                             n_seeds=args.seeds, device=device))]
    for b in args.baselines:
        rows.append((b, evaluate_policy(REGISTRY[b]().act, cfg, batch_size=args.batch,
                                        n_seeds=args.seeds, device=device)))

    print("checkpoint:", args.checkpoint)
    print("regime:", overrides)
    print(f"\n{'method':>16} | {'J_H':>16} | {'retain':>7} {'revrsl':>7} {'churn':>7} {'ttfc':>6}")
    print("-" * 72)
    for name, r in rows:
        print(f"{name:>16} | {r['J_H']:7.3f} +/-{r['J_H_std']:.3f} | {r['retention_rate']:7.3f} "
              f"{r['n_reversals']:7.2f} {r['assignment_churn']:7.2f} {r['time_to_first_completion']:6.2f}")
    jh = {name: r["J_H"] for name, r in rows}
    tname = rows[0][0]
    for b in args.baselines:
        d = jh[tname] - jh[b]
        print(f"\n{tname} - {b}: {d:+.3f}  ({'TENURE better' if d > 0 else 'baseline better/tie'})")


if __name__ == "__main__":
    main()
