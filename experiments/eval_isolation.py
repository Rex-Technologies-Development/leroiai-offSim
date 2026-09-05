"""Isolation ablation -- the load-bearing test for the 'representing concentrated value' reframe.

Every prior result showed off >= mult: gating by a retention ESTIMATE doesn't help. That kills
retention-*estimation* but does NOT, by itself, establish the positive claim -- that REPRESENTING the
concentrated value (toggle leverage) in the observation is what wins. This tests that directly, with
the retention head OFF in both arms so only the OBSERVATION differs:

  off_aware  : retention head OFF, toggle leverage VISIBLE  (is_toggle / who-holds-it / effective value)
  off_blind  : retention head OFF, leverage HIDDEN          (identical game + reward multiplier; obs cannot
                                                             see which tasks are toggles) -- task_dim 8
  mult_aware : retention head ON,  toggle leverage VISIBLE  (the head-decorative check)

Both aware and blind play the SAME game (toggle_regions>0, multiplier live in the reward); expose_toggle
only changes the observation, so J_H is directly comparable. The two decisive gaps:

  ISOLATION  off_aware - off_blind   > 0  => representing the value structure is what wins (reframe earned)
  HEAD       mult_aware - off_aware  <= 0 => estimating retention with the head does not help (as before)

    python -m experiments.eval_isolation --prefix tenure/checkpoints/hd2 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch

from baselines import REGISTRY
from tenure.policy import TenurePolicy
from contested.config import load_config
from experiments.eval_toggle_behavior import _rollout
from experiments.eval_harden import _obs

ARMS = ("off_aware", "off_blind", "mult_aware")
METRICS = ("J_H", "blue_tog", "red_tog", "goals_held", "goals_cashed")


def _load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = load_config(overrides=dict(ck.get("overrides") or {}))
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    return cfg, pol


def _arm(prefix, arm, seeds, eval_seeds, batch):
    """Returns (cfg, {seed: [J_H, blue_tog, red_tog, goals_held, goals_cashed] averaged over eval seeds})."""
    cfg, res = None, {}
    for s in seeds:
        p = f"{prefix}_{arm}_s{s}.pt"
        if not os.path.exists(p):
            continue
        cfg, pol = _load(p)
        rs = [_rollout(cfg, lambda c: pol.act(_obs(c, cfg), deterministic=True)["action"], es, batch, "cpu")
              for es in range(eval_seeds)]
        res[s] = [statistics.fmean(r[i] for r in rs) for i in range(5)]
    return cfg, res


def _ms(vals):
    m = statistics.fmean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _paired_gap(a: dict, b: dict):
    """mean +/- std of (a - b) over seeds present in BOTH (paired by training seed)."""
    common = sorted(set(a) & set(b))
    if not common:
        return float("nan"), float("nan"), 0
    diffs = [a[s][0] - b[s][0] for s in common]         # index 0 = J_H
    return statistics.fmean(diffs), (statistics.pstdev(diffs) if len(diffs) > 1 else 0.0), len(common)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolation ablation: representing leverage vs not (head off)")
    ap.add_argument("--prefix", default="tenure/checkpoints/hd2")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=4)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    args = ap.parse_args()

    arms, cfg_common = {}, None
    for arm in ARMS:
        cfg, res = _arm(args.prefix, arm, args.seeds, args.eval_seeds, args.batch)
        arms[arm] = res
        if arm == "off_aware" and cfg is not None:
            cfg_common = cfg
        elif cfg_common is None and cfg is not None:
            cfg_common = cfg

    if cfg_common is None:
        print("(no checkpoints found for any arm -- training may still be running)")
        return

    n_tog = cfg_common.toggle_regions
    print(f"\nregime: n_tasks={cfg_common.n_tasks} toggles={n_tog} M={cfg_common.toggle_multiplier} "
          f"red={cfg_common.adversary_population}  (eval {args.eval_seeds} seeds x training seeds present)\n")
    print(f"{'arm':>12} | {'n':>1} | {'J_H (mean+/-std)':>18} | blue_tog red_tog | cashed/held")
    print("-" * 74)
    for arm in ARMS:
        res = arms[arm]
        if not res:
            print(f"{arm:>12} |   (none yet)")
            continue
        cols = list(zip(*res.values()))                 # per-metric across seeds
        jh_m, jh_s = _ms(cols[0])
        bt, rt, gh, gc = (statistics.fmean(c) for c in cols[1:])
        print(f"{arm:>12} | {len(res)} | {jh_m:+7.3f} +/- {jh_s:5.3f}    | {bt:7.2f} {rt:6.2f} | {gc:5.2f}/{gh:.2f}")

    # baselines on the common game
    print("-" * 74)
    base_jh = {}
    for b in ("greedy", "defensive"):
        pol_b = REGISTRY[b]()
        rs = [_rollout(cfg_common, lambda c: pol_b.act(c.state, cfg_common), s, args.batch, "cpu")
              for s in range(args.eval_seeds)]
        m = [statistics.fmean(r[i] for r in rs) for i in range(5)]
        base_jh[b] = m[0]
        print(f"{b:>12} |   | {m[0]:+7.3f}            | {m[1]:7.2f} {m[2]:6.2f} | {m[4]:5.2f}/{m[3]:.2f}")

    # the decisive gaps (paired by training seed)
    print("\n=== decisive gaps (paired by training seed) ===")
    iso_m, iso_s, iso_n = _paired_gap(arms["off_aware"], arms["off_blind"])
    head_m, head_s, head_n = _paired_gap(arms["mult_aware"], arms["off_aware"])
    def _verdict(m, s, pos, neg, null):
        if m != m:                                  # nan
            return "n/a (no paired seeds)"
        if m - s > 0.01:                             # lower bound clears zero
            return pos
        if m + s < -0.01:                            # upper bound below zero
            return neg
        return f"{null} (gap {m:+.3f} within its own +/-{s:.3f} spread -> not separable from 0)"
    verdict_iso = _verdict(iso_m, iso_s, "representation WINS (gap exceeds its spread) -> reframe earned",
                           "blind WINS (representation hurts!)",
                           "WITHIN NOISE -- representation not established at this variance")
    verdict_head = _verdict(head_m, head_s, "head HELPS (unexpected)",
                            "head HARMFUL (robust negative)", "within noise")
    print(f"ISOLATION  off_aware - off_blind  = {iso_m:+.3f} +/- {iso_s:.3f}  (n={iso_n})  -> {verdict_iso}")
    print(f"HEAD       mult_aware - off_aware = {head_m:+.3f} +/- {head_s:.3f}  (n={head_n})  -> {verdict_head}")
    if arms["off_aware"]:
        aw = statistics.fmean(v[0] for v in arms["off_aware"].values())
        print(f"HEADLINE   off_aware - defensive  = {aw - base_jh['defensive']:+.3f}   "
              f"off_aware - greedy = {aw - base_jh['greedy']:+.3f}")


if __name__ == "__main__":
    main()
