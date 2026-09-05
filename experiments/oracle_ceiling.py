"""Oracle-ceiling headroom test (consultant): is there ANY value in perfect retention
knowledge in the holdable region, or is the tie structural (re-acquisition cost rho=1)?

Per seed, two passes on the SAME layout:
  pass 1  weight-only allocator; record the realized held fraction R per (decision, task);
  pass 2  a hindsight ORACLE that allocates by ``w * R_realized`` (the pass-1 R as the
          per-task priority; campers are near-static so pass-1 R is a fair proxy).

Readout:
  * oracle ties weight-only  -> NO headroom: perfect retention knowledge buys nothing, so
    the multiplicative decoder is exonerated and the earlier tie is STRUCTURAL. Payoff to
    retention ~ variance(R) x cost-of-losing, and cost-of-losing = tau_com (rho=1) is cheap.
  * oracle wins clearly      -> headroom exists and the learned head/decoder is failing.

Also logs the std of realized CONTESTED retention: a corr of 0.5 on std 0.03 is noise.

    python -m experiments.oracle_ceiling --alphas 0.5 0.75 --seeds 8
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from contested.config import load_config
from contested.core import CanonicalCore, default_device
from contested.labels import RetentionLabelRecorder
from experiments.retention_probe import WeightOnly


class HindsightOracle(BaselinePolicy):
    """``w * R_ref[decision]`` — allocate by the realized retention recorded in pass 1."""
    name = "oracle_realizedR"

    def __init__(self, R_ref: torch.Tensor):
        self.R_ref = R_ref                       # (B, D, T)
        self.d = 0

    def scores(self, state, cfg):
        _, travel = robot_task_geometry(state, cfg)
        r = self.R_ref[:, min(self.d, self.R_ref.shape[1] - 1), :]      # (B, T)
        self.d += 1
        inc = (~state.task_c & state.task_valid).to(travel.dtype)
        s = blank_scores(state, cfg)
        s[:, :, 0:cfg.max_tasks] = (inc * state.task_w * r).unsqueeze(1) / (travel + 0.1)
        return s


@torch.no_grad()
def _held(cfg, policy, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    total = torch.zeros(batch, device=dev, dtype=core.dtype)
    for _ in range(cfg.n_decisions):
        r, _, _ = core.step(policy.act(core.state, cfg))
        total += r
    return total.mean().item()


@torch.no_grad()
def _held_and_labels(cfg, policy, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    rec = RetentionLabelRecorder(); core.reset(seed)
    total = torch.zeros(batch, device=dev, dtype=core.dtype)
    for _ in range(cfg.n_decisions):
        rec.before_decision(core)
        r, _, _ = core.step(policy.act(core.state, cfg))
        total += r
        rec.after_decision(core)
    return total.mean().item(), rec.finalize(core), core.state.task_protected.clone()


def _ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def run(alpha, seeds, batch, dev):
    over = {"alpha": alpha, "n_tasks": 16, "n_robots": 4, "n_adversaries": 4,
            "protected_fraction": 0.15, "adversary_population": ["camper"], "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    wo_jh, or_jh, cstd = [], [], []
    for s in range(seeds):
        jh, lab, prot = _held_and_labels(cfg, WeightOnly(), s, batch, dev)
        wo_jh.append(jh)
        or_jh.append(_held(cfg, HindsightOracle(lab["R"]), s, batch, dev))
        m = (lab["dense_mask"] | lab["event_mask"]) & ~prot.unsqueeze(1)   # contested labeled
        cstd.append(lab["R"][m].std().item())
    g = statistics.fmean(or_jh) - statistics.fmean(wo_jh)
    ci = _ci95([o - w for o, w in zip(or_jh, wo_jh)])
    verdict = "HEADROOM (oracle wins)" if g - ci > 0.01 else "NO HEADROOM (tie) -> structural"
    print(f"alpha={alpha:<4}  weight-only J_H={statistics.fmean(wo_jh):.3f}   "
          f"oracle(w*realizedR) J_H={statistics.fmean(or_jh):.3f}   "
          f"gap={g:+.3f} [{g - ci:+.3f},{g + ci:+.3f}]  ->  {verdict}")
    print(f"           std of realized CONTESTED retention = {statistics.fmean(cstd):.3f}  "
          f"(corr on this std is ~noise if small)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle-ceiling headroom test")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.75])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=128)
    args = ap.parse_args()
    dev = default_device()
    print("ORACLE CEILING (perfect retention knowledge vs weight-only). tie => no headroom => "
          "the holdable-region tie is STRUCTURAL (rho=1), not a decoder failure.")
    for a in args.alphas:
        run(a, args.seeds, args.batch, dev)


if __name__ == "__main__":
    main()
