"""WHY does the retention head hurt (mult - off = -0.294)? A mechanistic diagnosis on banked checkpoints.

Hypothesis: toggles are the CONTESTED control points -- red raids them, so their realized retention is
LOWER than protected/uncontested goals. A head trained on those labels learns R_hat(toggle) < R_hat(goal).
The multiplicative gate then scores a task by (effective value x R_hat), so it DEMOTES exactly the toggles
whose leverage you most want to hold -- because they look 'unretainable'. Net: the mult policy holds fewer
toggles than the identical off policy, and loses the multiplier. This is the same misranking mechanism as
every earlier null, now shown to act ON the leverage points.

Measures, on the hardening regime, mult (head ON) vs off (head OFF):
  * R_hat on toggles vs on goals (is the head demoting toggles?)
  * behaviour: toggles held, goals cashed (does the demotion cost toggles?)

    python -m experiments.diag_head_harm --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import statistics

import torch

from contested.config import load_config
from contested.core import CanonicalCore
from contested.observation import build_observation
from tenure.policy import TenurePolicy
from experiments.eval_toggle_behavior import _rollout


def _load(path):
    ck = torch.load(path, map_location="cpu")
    cfg = load_config(overrides=dict(ck.get("overrides") or {}))
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    return cfg, pol


@torch.no_grad()
def _rhat_split(cfg, pol, seed, batch):
    """Mean R_hat on toggles vs on non-toggle goals, over a rollout (policy acts on its own R_hat)."""
    core = CanonicalCore(cfg, batch_size=batch, device="cpu"); core.reset(seed)
    tog, goal = [], []
    for _ in range(cfg.n_decisions):
        obs = build_observation(core.state, cfg)
        out = pol(obs)
        r = out["r_hat"] if "r_hat" in out else None
        s = core.state
        if r is not None:
            istog = s.task_is_toggle & s.task_valid
            isgoal = (~s.task_is_toggle) & s.task_valid
            tog.append(r[istog]); goal.append(r[isgoal])
        core.step(pol.act(obs, deterministic=True)["action"])
    if not tog:
        return None
    return torch.cat(tog), torch.cat(goal)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mechanistic diagnosis of retention-head harm")
    ap.add_argument("--mult-prefix", default="tenure/checkpoints/harden_ext_multiplicative")
    ap.add_argument("--off-prefix", default="tenure/checkpoints/harden_ext_off")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=4)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    args = ap.parse_args()

    print("\n== 1. Is the head demoting toggles? R_hat(toggle) vs R_hat(goal) on the MULT policy ==\n")
    tog_means, goal_means = [], []
    cfg = None
    for s in args.seeds:
        cfg, pol = _load(f"{args.mult_prefix}_s{s}.pt")
        out = _rhat_split(cfg, pol, 0, args.batch)
        if out is None:
            print(f"  seed {s}: policy exposes no r_hat"); continue
        tog, goal = out
        tm, gm = tog.mean().item(), goal.mean().item()
        tog_means.append(tm); goal_means.append(gm)
        print(f"  seed {s}:  R_hat(toggle)={tm:.3f} (std {tog.std().item():.3f})   "
              f"R_hat(goal)={gm:.3f} (std {goal.std().item():.3f})   demotion={gm - tm:+.3f}")
    if tog_means:
        dt = statistics.fmean(goal_means) - statistics.fmean(tog_means)
        print(f"\n  mean demotion R_hat(goal) - R_hat(toggle) = {dt:+.3f}  "
              f"-> {'toggles DEMOTED by the gate (mechanism confirmed)' if dt > 0.02 else 'no consistent demotion'}")

    print("\n== 2. Does the demotion cost toggles? behaviour: mult vs off ==\n")
    print(f"{'policy':>16} | {'J_H':>7} | blue_tog red_tog | cashed/held")
    print("-" * 60)
    beh = {}
    for name, prefix in (("mult (head ON)", args.mult_prefix), ("off (head OFF)", args.off_prefix)):
        rows = []
        for s in args.seeds:
            cfg, pol = _load(f"{prefix}_s{s}.pt")
            rs = [_rollout(cfg, lambda c: pol.act(build_observation(c.state, cfg), deterministic=True)["action"],
                           es, args.batch, "cpu") for es in range(args.eval_seeds)]
            rows.append([statistics.fmean(r[i] for r in rs) for i in range(5)])
        m = [statistics.fmean(c) for c in zip(*rows)]
        beh[name] = m
        print(f"{name:>16} | {m[0]:+7.3f} | {m[1]:7.2f} {m[2]:6.2f} | {m[4]:5.2f}/{m[3]:.2f}")
    if len(beh) == 2:
        a, b = beh["mult (head ON)"], beh["off (head OFF)"]
        print(f"\n  mult - off:  J_H {a[0] - b[0]:+.3f}   blue_tog {a[1] - b[1]:+.2f}   "
              f"goals_cashed {a[4] - b[4]:+.2f}")
        print("  Read: if the head demotes toggles (part 1) AND mult holds fewer toggles / cashes fewer "
              "goals\n  than off (part 2), the -0.294 is the retention gate down-ranking the contested "
              "leverage points\n  precisely because they are contested -- representation's opposite.")


if __name__ == "__main__":
    main()
