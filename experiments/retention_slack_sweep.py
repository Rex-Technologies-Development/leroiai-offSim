"""Does caring about retention beat not caring? -- and WHEN (the slack sweep).

Every stacking experiment failed for ONE reason: SLACK. With many more tasks than the opponent can
cover, you win by grabbing UNCONTESTED value and moving on -- so a policy that HOLDS what it has is
strictly dominated by one that CHURNS (grab, abandon, grab the next free task). That is not a defeat
for retention; it is retention being irrelevant because nothing is contested. Real Override has no
slack: the few multiplier objectives (a toggle 3x's its quadrant; protected alliance goals) are all
contested at once, and holding them IS the game.

This isolates the core thesis with NO stacking (stack_cap=1, the plain symmetric take model):
  * CHURN (does NOT care about retention): acquire the nearest neutral high-value task; once it is
    yours it scores 0 to you, so you leave for the next neutral -- and the opponent reverses what you
    abandoned. This is the FLAT baseline.
  * HOLD (cares about retention): acquire, then DEFEND -- sit on your objective so the opponent never
    gets domination to reverse it. You cover FEWER tasks but you keep them.

We sweep tasks-per-robot (slack). PREDICTION: HOLD beats CHURN when the field is SATURATED
(tasks ~ robots) and loses when there is slack (tasks >> robots). The crossover is the point:
retention pays exactly in the regime it was meant for -- a contested, saturated field.

    python -m experiments.retention_slack_sweep --tasks 4 6 8 12 --seeds 8
"""
from __future__ import annotations

import argparse
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from contested.config import load_config
from contested.core import CanonicalCore, CanonicalState, default_device
from experiments.stacking_ceiling import FlatBuilder                     # CHURN baseline


class HoldDefender(BaselinePolicy):
    """Cares about retention: acquire the nearest neutral high-value task, then DEFEND it forever
    (never abandon). Covers fewer tasks than CHURN but the opponent can never out-dominate a task a
    robot is sitting on (beta=1: 1v1 is a standoff, so blue keeps ownership)."""
    name = "hold"

    def scores(self, state: CanonicalState, cfg) -> torch.Tensor:
        _, travel = robot_task_geometry(state, cfg)                      # (B, R, T)
        T = cfg.max_tasks
        s = blank_scores(state, cfg)
        neutral = (~state.task_c & ~state.task_c_red & state.task_valid).to(travel.dtype)
        blue = (state.task_c & state.task_valid).to(travel.dtype)
        prox = 1.0 / (travel + 0.1)
        s[:, :, 0:T] = (neutral * state.task_w).unsqueeze(1) * prox       # ACQUIRE a neutral task
        s[:, :, T:2 * T] = (blue * state.task_w).unsqueeze(1) * prox * 5.0  # DEFEND own (preferred)
        return s


@torch.no_grad()
def _rollout(cfg, policy, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    for _ in range(cfg.n_decisions):
        core.step(policy.act(core.state, cfg))
    return core.telemetry.summary(core.state)["J_H_diff"].mean().item()


def run_tasks(n_tasks, seeds, batch, dev, n_robots, n_adv, alpha):
    over = {"symmetric": True, "stack_cap": 1, "alpha": alpha, "n_tasks": n_tasks, "n_robots": n_robots,
            "n_adversaries": n_adv, "adversary_population": ["value_targeting"], "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    churn = [_rollout(cfg, FlatBuilder(), s, batch, dev) for s in range(seeds)]
    hold = [_rollout(cfg, HoldDefender(), s, batch, dev) for s in range(seeds)]
    m = statistics.fmean
    gap = m(hold) - m(churn)
    slack = n_tasks / n_robots
    tag = "HOLD wins" if gap > 0.005 else ("churn wins" if gap < -0.005 else "tie")
    print(f"tasks={n_tasks:<3}(slack {slack:.2f}) | CHURN={m(churn):+.3f}  HOLD={m(hold):+.3f}  "
          f"gap={gap:+.3f}  [{tag}]")
    return n_tasks, gap


def main() -> None:
    ap = argparse.ArgumentParser(description="Retention slack sweep (does holding beat churning, and when)")
    ap.add_argument("--tasks", nargs="+", type=int, default=[4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=128)
    ap.add_argument("--n-robots", dest="n_robots", type=int, default=4)
    ap.add_argument("--n-adversaries", dest="n_adv", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()
    dev = default_device()
    print(f"RETENTION SLACK SWEEP (device={dev}, {args.seeds} seeds, {args.n_robots} robots, "
          f"{args.n_adv} red, alpha={args.alpha}) -- HOLD (cares) vs CHURN (doesn't).")
    print("PREDICTION: HOLD wins when saturated (tasks ~ robots), loses under slack (tasks >> robots).\n")
    gaps = []
    for nt in args.tasks:
        _, g = run_tasks(nt, args.seeds, args.batch, dev, args.n_robots, args.n_adv, args.alpha)
        gaps.append((nt, g))
    print()
    lo, hi = gaps[0], gaps[-1]
    if lo[1] > 0.005 and hi[1] < lo[1]:
        print(f"CONFIRMED: retention (HOLD) pays when the field is saturated (tasks={lo[0]}: {lo[1]:+.3f}) "
              f"and the edge shrinks/flips with slack (tasks={hi[0]}: {hi[1]:+.3f}).")
        print("Caring about retention beats not caring -- in the contested regime it was meant for.")
    else:
        print(f"No clean crossover: saturated gap={lo[1]:+.3f}, slack gap={hi[1]:+.3f}. Inspect above.")


if __name__ == "__main__":
    main()
