"""Full-match bait vs flat (Phase 3) -- does the tempo weapon translate into a match win?

stacking_bait.py proved the MECHANISM in isolation: an abandoned height-H stack ties the opponent up
for tau_rev*H(H+1)/2 (serial -- extra attackers wasted) and banks ~tau_rev*H(H+1)(2H+1)/6 held value
while it comes down. Here we ask whether a policy built on that idea beats a flat baseline over a
whole match against a live (navigating) value_targeting red.

  * FLAT (weight-only WIDE): spread, acquire many neutral height-1 tasks, never stack.
  * BAIT-PACK (the user's tactic): move as ONE pack, build the target stack to the cap FAST (four
    robots => red cannot freeze it in a standoff), then ABANDON it and move to the next neutral --
    leaving a trail of tall stacks the value_targeting red must chase and tear down one piece at a
    time (serial) while they keep scoring for blue.

Retention should win when dismantling is SLOW relative to building -- tau_rev = tau_com/alpha, so
LOW alpha (alpha<1 => tau_rev>tau_com) is the retention-friendly regime the mechanic predicts. We
sweep alpha and report differential J_H for both.

    python -m experiments.stacking_bait_match --alphas 0.5 0.75 1.0 1.5 --cap 5 --seeds 6
"""
from __future__ import annotations

import argparse
import statistics

import torch

from contested.config import load_config
from contested.core import CanonicalCore, CanonicalState, default_device
from experiments.stacking_ceiling import FlatBuilder


class BaitPack:
    """One pack of robots that builds a stack to the cap then abandons it for the next neutral task.
    Fixed pack target (recomputed only when the current stack is finished or lost) => no thrashing;
    the pack out-numbers red locally so it is never frozen mid-build, and the abandoned tall stacks
    are the bait."""
    name = "bait"

    def __init__(self):
        self.target: torch.Tensor | None = None                          # (B,) current stack

    @torch.no_grad()
    def act(self, state: CanonicalState, cfg) -> torch.Tensor:
        B, T = state.task_valid.shape
        R = state.robot_valid.shape[1]
        dev = state.task_valid.device
        pack = torch.where(state.robot_valid.unsqueeze(-1), state.robot_pos,
                           torch.zeros_like(state.robot_pos)).sum(1) / \
            state.robot_valid.sum(1, keepdim=True).clamp_min(1)          # (B,2) pack centroid
        d = torch.linalg.vector_norm(pack.unsqueeze(1) - state.task_pos, dim=-1)   # (B,T)
        neutral = ~state.task_c & ~state.task_c_red & state.task_valid
        pick = torch.where(neutral, state.task_w / (d + 0.1), torch.full_like(d, -1.0))

        if self.target is None:
            self.target = pick.argmax(1)
        h = state.task_height.gather(1, self.target[:, None])[:, 0]
        lost = state.task_c_red.gather(1, self.target[:, None])[:, 0]
        done = (h >= cfg.stack_cap) | lost                               # finished or stolen -> move on
        cur = self.target
        alt = pick.clone().scatter(1, cur[:, None], torch.full((B, 1), -1.0, device=dev))
        self.target = torch.where(done & (alt.amax(1) > 0), alt.argmax(1), cur)

        tgt = self.target[:, None].expand(B, R)
        blue = state.task_c.gather(1, tgt)
        red = state.task_c_red.gather(1, tgt)
        act = torch.where(blue, cfg.max_tasks + tgt,
              torch.where(red, cfg.reverse_base + tgt, tgt))
        return torch.where(state.robot_valid, act, torch.full_like(act, cfg.idle_action))


@torch.no_grad()
def _rollout(cfg, policy, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    for _ in range(cfg.n_decisions):
        core.step(policy.act(core.state, cfg))
    return core.telemetry.summary(core.state)["J_H_diff"].mean().item()


def _diag(cfg, mk, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    pol = mk()
    for _ in range(cfg.n_decisions):
        core.step(pol.act(core.state, cfg))
    s = core.state
    bt = s.task_c.sum(-1).float().mean().item()
    bh = (s.task_height * s.task_c).sum(-1).float().mean().item()
    rh = (s.task_height * s.task_c_red).sum(-1).float().mean().item()
    return bt, bh, rh


def run_alpha(alpha, cap, seeds, batch, dev, n_tasks, n_adv, diagnose):
    over = {"symmetric": True, "stack_cap": cap, "alpha": alpha, "n_tasks": n_tasks, "n_robots": 4,
            "n_adversaries": n_adv, "adversary_population": ["value_targeting"], "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    flat = [_rollout(cfg, FlatBuilder(), s, batch, dev) for s in range(seeds)]
    bait = [_rollout(cfg, BaitPack(), s, batch, dev) for s in range(seeds)]
    m = statistics.fmean
    gap = m(bait) - m(flat)
    tag = "BAIT WINS" if gap > 0.005 else ("flat wins" if gap < -0.005 else "tie")
    print(f"alpha={alpha:<4} (tau_rev={cfg.tau_rev:.2f}) | FLAT={m(flat):+.3f}  BAIT={m(bait):+.3f}  "
          f"gap={gap:+.3f}  [{tag}]")
    if diagnose:
        for name, mk in [("flat", lambda: FlatBuilder()), ("bait", lambda: BaitPack())]:
            bt, bh, rh = _diag(cfg, mk, 0, batch, dev)
            print(f"        [{name}] blue {bt:.1f} tasks tot-h {bh:.1f} | red tot-h {rh:.1f}")
    return alpha, gap


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-match bait vs flat (Phase 3)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--cap", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    ap.add_argument("--n-tasks", dest="n_tasks", type=int, default=8)
    ap.add_argument("--n-adversaries", dest="n_adv", type=int, default=4)
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    dev = default_device()
    print(f"FULL-MATCH BAIT vs FLAT (device={dev}, cap={args.cap}, {args.seeds} seeds, "
          f"n_tasks={args.n_tasks}, n_adv={args.n_adv})")
    print("Retention-friendly regime is LOW alpha (tau_rev>tau_com => slow to undo).\n")
    for a in args.alphas:
        run_alpha(a, args.cap, args.seeds, args.batch, dev, args.n_tasks, args.n_adv, args.diagnose)


if __name__ == "__main__":
    main()
