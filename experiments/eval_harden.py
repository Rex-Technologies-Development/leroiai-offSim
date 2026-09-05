"""Aggregate multi-seed eval for the hardened toggle result + the no-retention-head ablation.

Loads harden_{multiplicative,off}_s{0..N}.pt (trained vs a GOAL-CONTESTING opponent), evaluates each
on its own regime, and reports mean +/- std across seeds for:
  * TENURE(multiplicative) -- the retention gate ON,
  * TENURE(off)            -- same obs incl. toggle leverage, retention head IGNORED (the ablation:
                             does ESTIMATING retention beat merely KNOWING toggles are valuable?),
  * greedy / defensive baselines (the win must survive a goal-contesting opponent).

    python -m experiments.eval_harden --prefix tenure/checkpoints/harden --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import glob
import statistics

import torch

from baselines import REGISTRY
from contested.config import load_config
from tenure.policy import TenurePolicy
from experiments.eval_toggle_behavior import _rollout


def _load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = load_config(overrides=dict(ck.get("overrides") or {}))
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return cfg, pol


def _stats(vals):
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _eval_ckpts(paths, seeds, batch):
    """Mean over (checkpoint x eval-seed) for J_H and behaviour; returns per-metric (mean,std over ckpts)."""
    per_ckpt = []
    cfg = None
    for p in paths:
        cfg, pol = _load(p)
        rs = [_rollout(cfg, lambda c: pol.act(_obs(c, cfg), deterministic=True)["action"], s, batch, "cpu")
              for s in range(seeds)]
        per_ckpt.append([statistics.fmean(r[i] for r in rs) for i in range(5)])
    cols = list(zip(*per_ckpt)) if per_ckpt else [[]] * 5
    return cfg, [_stats(c) for c in cols]


def _obs(core, cfg):
    from contested.observation import build_observation
    return build_observation(core.state, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed hardened toggle eval + no-head ablation")
    ap.add_argument("--prefix", default="tenure/checkpoints/harden")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=4)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    args = ap.parse_args()

    rows = []
    cfg = None
    for mode in ("multiplicative", "off"):
        paths = [f"{args.prefix}_{mode}_s{s}.pt" for s in args.seeds]
        paths = [p for p in paths if glob.glob(p)]
        if not paths:
            print(f"(no checkpoints for mode={mode})")
            continue
        cfg, stats = _eval_ckpts(paths, args.eval_seeds, args.batch)
        rows.append((f"TENURE({mode})", len(paths), stats))

    # baselines on the same regime
    if cfg is not None:
        for b in ("greedy", "defensive"):
            pol_b = REGISTRY[b]()
            rs = [_rollout(cfg, lambda c: pol_b.act(c.state, cfg), s, args.batch, "cpu")
                  for s in range(args.eval_seeds)]
            stats = [_stats([statistics.fmean(r[i] for r in rs)]) for i in range(5)]
            rows.append((b, 1, stats))

    print(f"\nregime: n_tasks={cfg.n_tasks} toggles={cfg.toggle_regions} M={cfg.toggle_multiplier} "
          f"red={cfg.adversary_population}  ({args.seeds} training seeds)\n")
    print(f"{'policy':>20} | {'J_H (mean+/-std)':>18} | blue_tog red_tog | goals_cashed/held")
    print("-" * 78)
    for name, n, st in rows:
        (jh, jhs), (bt, _), (rt, _), (gh, _), (gc, _) = st
        print(f"{name:>20} | {jh:+7.3f} +/- {jhs:5.3f} ({n}) | {bt:7.2f} {rt:6.2f} | {gc:6.2f}/{gh:.2f}")
    # the ablation verdict
    jh = {name: st[0][0] for name, n, st in rows}
    if "TENURE(multiplicative)" in jh and "TENURE(off)" in jh:
        d = jh["TENURE(multiplicative)"] - jh["TENURE(off)"]
        print(f"\nno-head ablation  mult - off = {d:+.3f}  "
              f"({'head ESTIMATE adds value' if d > 0.01 else 'head decorative -- win is KNOWING toggles, not estimating retention'})")


if __name__ == "__main__":
    main()
