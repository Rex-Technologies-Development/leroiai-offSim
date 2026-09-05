"""Retention-value probes (consultant's framing).

IMPORTANT SCOPE. These are SCRIPTED probes that measure the *ceiling* — the value of
knowing retention (``w x R_oracle`` vs weight-only). Here ``R`` is the structural
oracle: ``1`` for protected tasks, low for contested. A positive gap is necessary
(retention CAN buy value) but NOT sufficient: it does not show a *learned* head beats
a plain allocator that simply has the protected flag + adversary geometry in its
observation. That "killer control" is the ``retention_mode: off`` learned ablation and
needs training (``tenure.train``). See NEXT-STEPS in the repo README.

What this tool answers rigorously:
- the protected_fraction SWEEP (expect an inverted-U: ~0 at 0.0 and 1.0, peak between),
  which is much stronger mechanism evidence than a single gap number;
- confidence intervals over >=5 paired seeds (same layout for both policies per seed);
- calibration at Override's real protected fraction (~0.15-0.20).
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from baselines.evaluate import evaluate_policy
from contested.config import load_config


class WeightOnly(BaselinePolicy):
    name = "weight"

    def scores(self, state, cfg):
        _, travel = robot_task_geometry(state, cfg)
        inc = (~state.task_c & state.task_valid).unsqueeze(1).to(travel.dtype)
        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = inc * state.task_w.unsqueeze(1) / (travel + 0.1)
        return s


class RetentionOracle(BaselinePolicy):
    """w x R with the STRUCTURAL oracle R (1 protected, low contested)."""
    name = "w*R_oracle"

    def __init__(self, contested_r: float = 0.25):
        self.contested_r = contested_r

    def scores(self, state, cfg):
        _, travel = robot_task_geometry(state, cfg)
        r = torch.where(state.task_protected, torch.ones_like(state.task_w),
                        torch.full_like(state.task_w, self.contested_r))
        inc = (~state.task_c & state.task_valid).to(travel.dtype)
        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = (inc * state.task_w * r).unsqueeze(1) / (travel + 0.1)
        return s


def _ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def paired_gaps(base_overrides, protected_fraction, seeds, batch, device=None):
    """Per-seed paired gap (oracle - weight-only) at one protected_fraction."""
    wo, rw = WeightOnly(), RetentionOracle()
    gaps, wos, rws = [], [], []
    for s in range(seeds):
        cfg = load_config(overrides={**base_overrides, "protected_fraction": protected_fraction})
        a = evaluate_policy(wo.act, cfg, batch_size=batch, n_seeds=1, base_seed=s, device=device)["J_H"]
        b = evaluate_policy(rw.act, cfg, batch_size=batch, n_seeds=1, base_seed=s, device=device)["J_H"]
        wos.append(a); rws.append(b); gaps.append(b - a)
    return wos, rws, gaps


def sweep(fractions, base_overrides, seeds, batch, device=None):
    print("SCRIPTED CEILING (value of KNOWING protection), not the learned-head result.")
    print("regime:", {k: base_overrides[k] for k in base_overrides})
    print(f"\n{'protected':>9} | {'weight':>7} {'w*R_oracle':>10} | {'gap':>7} {'95% CI':>16} | {'seeds':>5}")
    print("-" * 66)
    for pf in fractions:
        wos, rws, gaps = paired_gaps(base_overrides, pf, seeds, batch, device)
        g, ci = statistics.fmean(gaps), _ci95(gaps)
        sig = "" if (g - ci) > 0 else "   (CI includes 0)"
        print(f"{pf:9.2f} | {statistics.fmean(wos):7.3f} {statistics.fmean(rws):10.3f} | "
              f"{g:+7.3f} [{g - ci:+.3f},{g + ci:+.3f}]{sig}")
    print("\nExpect ~0 at 0.0 (nothing safe) and 0.0/1.0 (everything safe); a peak between")
    print("is the mechanism. Override's real protected fraction is ~0.15-0.20 (2 alliance")
    print("goals / ~11 sites) — the effect must be present THERE, not only at 0.4.")


def main() -> None:
    p = argparse.ArgumentParser(description="Retention ceiling: protected_fraction sweep with CIs")
    p.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--horizon", type=float, default=90.0)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=16)
    p.add_argument("--n-robots", dest="n_robots", type=int, default=2)
    p.add_argument("--n-adversaries", dest="n_adversaries", type=int, default=6)
    p.add_argument("--tau-com", dest="tau_com", type=float, default=5.0)
    p.add_argument("--adversary", default="camper")
    p.add_argument("--batch-size", dest="batch", type=int, default=192)
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    base = dict(alpha=args.alpha, n_tasks=args.n_tasks, n_robots=args.n_robots,
                n_adversaries=args.n_adversaries, tau_com=args.tau_com,
                adversary_population=[args.adversary], horizon_T=args.horizon)
    sweep(args.fractions, base, args.seeds, args.batch)


if __name__ == "__main__":
    main()
