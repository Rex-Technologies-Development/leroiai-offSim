"""Evaluate any allocation policy in the canonical environment (plan Section 6).

A policy is any ``callable(state, cfg) -> actions (B, max_robots)`` — the baselines'
``.act`` methods, or a wrapped TENURE policy. Runs whole episodes and aggregates the
telemetry (mean over the batch, mean/std over seeds).
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from contested.config import CanonicalConfig, load_config
from contested.core import CanonicalCore, default_device

_METRICS = ("J_H", "J_T", "retention_rate", "defense_fraction",
            "n_reversals", "assignment_churn", "time_to_first_completion")

PolicyFn = Callable[[object, CanonicalConfig], torch.Tensor]


@torch.no_grad()
def evaluate_policy(policy_fn: PolicyFn, cfg: CanonicalConfig, batch_size: int = 256,
                    n_seeds: int = 3, base_seed: int = 0, device=None) -> dict:
    device = device or default_device()
    core = CanonicalCore(cfg, batch_size=batch_size, device=device)
    per_seed = {k: [] for k in _METRICS}
    for s in range(n_seeds):
        core.reset(base_seed + s)
        for _ in range(cfg.n_decisions):
            core.step(policy_fn(core.state, cfg))
        tel = core.telemetry.summary(core.state)
        for k in _METRICS:
            per_seed[k].append(tel[k].float().mean().item())

    out = {"method_batch": batch_size, "n_seeds": n_seeds, "alpha": cfg.alpha, "contest_mode": cfg.contest_mode}
    for k in _METRICS:
        out[k] = statistics.fmean(per_seed[k])
        out[k + "_std"] = statistics.pstdev(per_seed[k]) if n_seeds > 1 else 0.0
    return out


def tenure_policy_fn(policy, deterministic: bool = True) -> PolicyFn:
    """Wrap a trained :class:`tenure.policy.TenurePolicy` as a policy_fn."""
    from contested.observation import build_observation

    def fn(state, cfg):
        obs = build_observation(state, cfg)
        return policy.act(obs, deterministic=deterministic)["action"]
    return fn


def main() -> None:
    from . import REGISTRY
    p = argparse.ArgumentParser(description="Evaluate a baseline in the canonical env")
    p.add_argument("--method", choices=sorted(REGISTRY), default="greedy")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--contest-mode", dest="contest_mode", default=None)
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=None)
    p.add_argument("--n-adversaries", dest="n_adversaries", type=int, default=None)
    p.add_argument("--horizon", dest="horizon_T", type=float, default=None)
    args = p.parse_args()

    overrides = {k: getattr(args, k) for k in
                 ("alpha", "contest_mode", "n_tasks", "n_adversaries", "horizon_T")
                 if getattr(args, k) is not None}
    cfg = load_config(overrides=overrides or None)
    policy = REGISTRY[args.method]()
    res = evaluate_policy(policy.act, cfg, batch_size=args.batch_size, n_seeds=args.seeds)
    print(f"method={args.method} alpha={res['alpha']} contest={res['contest_mode']}")
    for k in _METRICS:
        print(f"  {k:26s} {res[k]:8.4f} ± {res[k + '_std']:.4f}")


if __name__ == "__main__":
    main()
