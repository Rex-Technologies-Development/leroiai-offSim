"""Stacking ceiling (Phase 3 -- retention as something you BUILD, not predict).

Placing the piece at level L costs tau_com * L and dismantling it costs tau_rev * L -- reaching
higher is slower on BOTH sides. Two consequences drive the experiment:

  1. Building a stack of height h costs tau_com * h(h+1)/2 (quadratic), so the SAME total pieces are
     far more EXPENSIVE to raise when concentrated than when spread. Concentration is a big up-front
     investment.
  2. Dismantling is SERIAL and per-piece: eta accrues one piece at a time and does NOT speed up with
     more attackers (core: eta += dt while red dominates, one drop per tau_rev * h). So a tall stack
     forces the opponent to tear it down ONE robot at a time over tau_rev * h(h+1)/2 -- extra
     attackers are wasted -- whereas flat work is torn down by many attackers in PARALLEL.

So concentration trades a quadratic build cost for immunity to being out-numbered on defence. The
question is whether the retention (2) outweighs the build handicap (1).

Two committed policies (no thrashing -- targets are fixed for the whole match) against a red that
dismantles blue's work (value_targeting):
  * FLAT (weight-only WIDE): acquire the nearest neutral high-value task, never stack -> builds many
    height-1 tasks, each cheaply flattened (tau_rev * 1) in parallel and re-acquired.
  * CONCENTRATE (retention PRODUCED): pick the top-k value*defensible tasks ONCE and pile all robots
    onto them, building TALL and holding -> a serial teardown the opponent cannot parallelise.
  * ORACLE: concentrate on the k tasks that held the most value in HINDSIGHT (the ceiling: perfect
    choice of which stacks to build).

PRE-REGISTERED PREDICTION: CONCENTRATE beats FLAT on differential J_H, and the gap WIDENS with the
height cap. Caution (not rigged): a tall stack is a single point of failure and costs quadratically
to raise, so concentration must trade persistence against exposure and build-tempo.

    python -m experiments.stacking_ceiling --caps 1 2 3 5 --seeds 8
"""
from __future__ import annotations

import argparse
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from contested.config import load_config
from contested.core import CanonicalCore, CanonicalState, default_device


class FlatBuilder(BaselinePolicy):
    """Weight-only WIDE build: acquire the nearest neutral high-value task, never stack. Single-assign
    (default) so robots spread; each task ends up height 1 and, once red flattens it back to neutral,
    is re-acquired. In a contested regime red keeps making fresh neutrals, so it stays genuinely flat."""
    name = "flat"

    def scores(self, state: CanonicalState, cfg) -> torch.Tensor:
        _, travel = robot_task_geometry(state, cfg)                      # (B, R, T)
        T = cfg.max_tasks
        neutral = (~state.task_c & ~state.task_c_red & state.task_valid).to(travel.dtype)
        s = blank_scores(state, cfg)
        s[:, :, 0:T] = (neutral * state.task_w).unsqueeze(1) / (travel + 0.1)   # ACQUIRE neutral only
        return s


class Concentrator:
    """Retention PRODUCED. COMMIT once (at t=0) to the top-k tasks by value * initial defensibility
    (distance to the nearest red), split all robots across them, and hold FOREVER -- ACQUIRE a neutral
    target, DEFEND a blue-owned one (stack + protect), REVERSE a red-owned one (dislodge). Fixed
    targets => robots arrive and build instead of thrashing as red moves. ``commit_scores`` (B,T)
    overrides the t=0 heuristic with an arbitrary ranking (the oracle passes hindsight held value)."""
    name = "concentrate"

    def __init__(self, k: int = 1, commit_scores: torch.Tensor | None = None):
        self.k = k
        self.commit_scores = commit_scores
        self.committed: torch.Tensor | None = None                       # (B, k) chosen task indices

    @torch.no_grad()
    def act(self, state: CanonicalState, cfg) -> torch.Tensor:
        B, T = state.task_valid.shape
        R = state.robot_valid.shape[1]
        dev = state.task_valid.device
        if self.committed is None:                                       # one-time strategic choice
            if self.commit_scores is not None:
                score = self.commit_scores.to(dev)
            else:
                d_red = torch.linalg.vector_norm(
                    state.adv_pos.unsqueeze(2) - state.task_pos.unsqueeze(1), dim=-1)   # (B,K,T)
                d_red = torch.where(state.adv_valid.unsqueeze(-1), d_red,
                                    torch.full_like(d_red, 1e9)).min(1).values          # (B,T)
                d_norm = d_red / (d_red.amax(1, keepdim=True) + 1e-9)
                score = state.task_w * (0.5 + d_norm)                    # value, favouring defensible
            score = torch.where(state.task_valid, score, torch.full_like(score, -1.0))
            self.committed = score.topk(min(self.k, T), dim=1).indices   # (B, k)
        k = self.committed.shape[1]
        per = max(1, R // k)
        pair = torch.tensor([min(r // per, k - 1) for r in range(R)], device=dev)
        tgt = self.committed[:, pair]                                    # (B, R) robot -> a chosen stack
        blue = state.task_c.gather(1, tgt)
        red = state.task_c_red.gather(1, tgt)
        act = torch.where(blue, cfg.max_tasks + tgt,                     # DEFEND (own -> stack/hold)
              torch.where(red, cfg.reverse_base + tgt,                   # REVERSE (red owns -> dislodge)
                          tgt))                                          # ACQUIRE (neutral -> claim)
        return torch.where(state.robot_valid, act, torch.full_like(act, cfg.idle_action))


@torch.no_grad()
def _rollout(cfg, policy, seed, batch, dev, want_pertask=False):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    per_task = torch.zeros(batch, cfg.max_tasks, device=dev) if want_pertask else None
    for _ in range(cfg.n_decisions):
        core.step(policy.act(core.state, cfg))
        if want_pertask:                                                 # accrue hindsight held value
            s = core.state
            per_task += s.task_w * s.task_height.to(torch.float32) * s.task_c.to(torch.float32)
    jh = core.telemetry.summary(core.state)["J_H_diff"].mean().item()
    if not want_pertask:
        return jh
    return jh, per_task                                                  # (B, T) per-task held value


def _final_heights(cfg, policy, seed, batch, dev):
    """Return (blue_tasks, blue_total_height, red_tasks, red_total_height) means for diagnostics."""
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    for _ in range(cfg.n_decisions):
        core.step(policy.act(core.state, cfg))
    s = core.state
    bt = s.task_c.sum(-1).float().mean().item()
    bh = (s.task_height * s.task_c).sum(-1).float().mean().item()
    rt = s.task_c_red.sum(-1).float().mean().item()
    rh = (s.task_height * s.task_c_red).sum(-1).float().mean().item()
    return bt, bh, rt, rh


def run_cap(cap, seeds, batch, dev, n_tasks=8, n_adv=4, alpha=1.5, k=1, diagnose=False):
    over = {"symmetric": True, "stack_cap": cap, "alpha": alpha, "n_tasks": n_tasks, "n_robots": 4,
            "n_adversaries": n_adv, "adversary_population": ["value_targeting"], "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    flat, conc, orac = [], [], []
    for s in range(seeds):
        flat.append(_rollout(cfg, FlatBuilder(), s, batch, dev))
        jc, per_task = _rollout(cfg, Concentrator(k=k), s, batch, dev, want_pertask=True)
        conc.append(jc)
        orac.append(_rollout(cfg, Concentrator(k=k, commit_scores=per_task), s, batch, dev))
    m = statistics.fmean
    gap, ogap = m(conc) - m(flat), m(orac) - m(flat)
    print(f"cap={cap:<2} | FLAT J_H_diff={m(flat):+.3f}   CONCENTRATE={m(conc):+.3f}   ORACLE={m(orac):+.3f}")
    print(f"        concentrate-minus-flat = {gap:+.3f}   (oracle-minus-flat = {ogap:+.3f})")
    if diagnose:
        for name, mk in [("flat", lambda: FlatBuilder()), ("conc", lambda: Concentrator(k=k))]:
            bt, bh, rt, rh = _final_heights(cfg, mk(), 0, batch, dev)
            mh = bh / bt if bt > 0.05 else 0.0
            print(f"        [{name:4}] blue {bt:.1f} tasks tot-h {bh:.1f} (mean {mh:.1f}/task) | "
                  f"red {rt:.1f} tasks tot-h {rh:.1f}")
    return cap, gap, ogap


def main() -> None:
    ap = argparse.ArgumentParser(description="Stacking ceiling (Phase 3)")
    ap.add_argument("--caps", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=128)
    ap.add_argument("--n-tasks", dest="n_tasks", type=int, default=8)
    ap.add_argument("--n-adversaries", dest="n_adv", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.5,
                    help="dismantle speed: alpha>1 => tau_rev<tau_com, red flattens faster than blue rebuilds")
    ap.add_argument("--k", type=int, default=1, help="how many stacks the concentrator builds")
    ap.add_argument("--diagnose", action="store_true", help="print per-team final task counts + heights")
    args = ap.parse_args()
    dev = default_device()
    print(f"STACKING CEILING (device={dev}, {args.seeds} seeds, n_tasks={args.n_tasks}, "
          f"n_adv={args.n_adv}, alpha={args.alpha}, k={args.k}) -- does CONCENTRATING on defensible "
          "tasks beat\nbuilding FLAT, and does the gap WIDEN with the height cap?\n")
    gaps = []
    for cap in args.caps:
        _, g, _ = run_cap(cap, args.seeds, args.batch, dev, args.n_tasks, args.n_adv, args.alpha,
                          args.k, diagnose=args.diagnose)
        gaps.append((cap, g))
    print()
    widens = len(gaps) > 1 and gaps[-1][1] > gaps[0][1] + 1e-6
    positive = all(g > 0.005 for c, g in gaps if c > 1)
    if positive and widens:
        print("PREDICTION CONFIRMED: retention-aware CONCENTRATION beats flat AND the gap widens with "
              "the cap.\nRetention wins on the axis that matters -- the allocator PRODUCES retention.")
    else:
        print(f"Prediction not (yet) confirmed: positive={positive} widens={widens}. "
              "Inspect the per-cap gaps above.")


if __name__ == "__main__":
    main()
