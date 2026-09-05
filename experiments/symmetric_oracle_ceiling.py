"""Symmetric oracle ceiling (Phase 2, consultant round-9): is there HEADROOM for
retention-aware REVERSE selection in the symmetric env? Scripted, CPU-runnable — the same
test that settled Phase 1, now with a REVERSE action against a builder red that holds tasks.

THREE arms (both reverse arms have REVERSE in the action space, so a gap isolates RETENTION,
not the action itself):
  (a) WO no-reverse   — ACQUIRE/DEFEND only, weight-scored ("neither")
  (b) WO reverse      — ACQUIRE/DEFEND/REVERSE, all weight-scored (REVERSE, no retention)
  (c) ORACLE reverse  — REVERSE scored by w * R-take (hindsight R_ref from a pass-1 rollout)
Headline = (c) - (b): the value of KNOWING retention on the offensive channel. (b) - (a) is the
REVERSE action's own value. A decomposition arm (oracle on ACQUIRE) separates the offensive
channel from the acquire channel — if a gap is all on acquire, the offensive story isn't the
work being done. Swept over rho in {1,2,4}: does offensive headroom appear and GROW with rho
(it did not on the defensive path)?

PRE-REGISTERED STOP RULE: if the oracle ties weight-only at EVERY rho, the symmetric env has no
headroom either and the retention-ESTIMATION line ends. The honest negative that survives —
retention-aware allocation works when retention is KNOWN, the estimator reads geometry at
corr ~0.5, and a measured scope condition says the multiplicative gate cannot pay in EITHER
action space — is an ACCEPTABLE outcome, decided now (not after seeing a marginal number).

Coverage note (consultant): coverage is a property of the LABEL-GENERATING policy. The first
build learned the hard way that a WEIGHT-ONLY reverse policy takes-and-holds NOTHING (R-take was
identically zero) -- it targets high-value DEFENDED red tasks 1v1, which never flip. So R_ref now
comes from FORCED EXPLORATION (ForcedReverseExplorer): committed robot PAIRS that out-number a
defender and hold what they take, sampling diverse red tasks. This runs ON TOP of the mechanic fix
(a successful revert now transfers ownership to the reverter, no neutral deadlock).

    python -m experiments.symmetric_oracle_ceiling --rhos 1 2 4 --seeds 8
"""
from __future__ import annotations

import argparse
import math
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from contested.config import load_config
from contested.core import CanonicalCore, CanonicalState, default_device
from contested.labels import RetentionLabelRecorder


class SymmetricAllocator(BaselinePolicy):
    """ACQUIRE neutral / DEFEND blue-owned / REVERSE red-owned, scored by w/(travel+eps).
    ``R_ref`` (B,D,T) supplies hindsight retention; ``on_acquire``/``on_reverse`` multiply that
    block's per-task weight by ``R_ref[d]``; ``allow_reverse=False`` removes REVERSE entirely."""
    name = "symmetric_alloc"

    def __init__(self, R_ref: torch.Tensor | None = None, on_acquire: bool = False,
                 on_reverse: bool = False, allow_reverse: bool = True):
        self.R_ref = R_ref
        self.on_acquire = on_acquire
        self.on_reverse = on_reverse
        self.allow_reverse = allow_reverse
        self.d = 0

    def scores(self, state: CanonicalState, cfg) -> torch.Tensor:
        _, travel = robot_task_geometry(state, cfg)                 # (B, R, T)
        T = cfg.max_tasks
        inv = 1.0 / (travel + 0.1)
        w = state.task_w                                            # (B, T)
        neutral = (~state.task_c & ~state.task_c_red & state.task_valid).to(travel.dtype)
        blue = (state.task_c & state.task_valid).to(travel.dtype)
        red = (state.task_c_red & state.task_valid).to(travel.dtype)
        r = None if self.R_ref is None else self.R_ref[:, min(self.d, self.R_ref.shape[1] - 1), :]
        self.d += 1
        acq_w = w * r if (self.on_acquire and r is not None) else w
        rev_w = w * r if (self.on_reverse and r is not None) else w
        s = blank_scores(state, cfg)
        s[:, :, 0:T] = (neutral * acq_w).unsqueeze(1) * inv
        s[:, :, T:2 * T] = (blue * w).unsqueeze(1) * inv
        if self.allow_reverse:
            s[:, :, cfg.reverse_base:cfg.reverse_base + T] = (red * rev_w).unsqueeze(1) * inv
        return s


class ForcedReverseExplorer:
    """Pass-1 label generator (consultant option A, ON TOP of the mechanic fix). To give R-take
    COVERAGE it FORCES blue to take-and-hold a diverse set of red tasks: each decision it commits
    PAIRS of robots to REVERSE red-held tasks (two robots out-number a lone defender, so under the
    fixed take-transfers-ownership mechanic the flip lands), holds a target until it is taken (or a
    timeout), then rotates to fresh red tasks chosen by a rotating random key. Purely to GENERATE
    labels -- coverage is a property of the label-generating policy (consultant)."""
    name = "forced_reverse_explorer"

    def __init__(self, seed: int = 0, hold: int = 28):   # hold long enough to complete a rho=4 take
        self.seed, self.hold = seed, hold                #   (rho*tau_com = 8s = 16 decisions + travel)
        self.gen = self.tgt = self.age = None

    @torch.no_grad()
    def act(self, state: CanonicalState, cfg) -> torch.Tensor:
        red = state.task_c_red & state.task_valid                    # (B, T) red-held tasks
        B, T = red.shape
        R = state.robot_valid.shape[1]
        dev = red.device
        if self.tgt is None:
            self.gen = torch.Generator(device=dev).manual_seed(self.seed)
            self.tgt = torch.full((B, R), -1, dtype=torch.long, device=dev)
            self.age = torch.zeros((B, R), dtype=torch.long, device=dev)
        red_f = red.to(torch.float32)
        key = torch.rand((B, T), generator=self.gen, device=dev) * red_f - (1.0 - red_f)  # ~-1 for non-red
        k = min(max(1, R // 2), T)
        top = key.topk(k, dim=1).indices                             # (B, k) target red tasks; pairs share one
        pair = torch.tensor([(r // 2) % k for r in range(R)], device=dev)
        newtgt = top[:, pair]                                        # (B, R): robots 2j,2j+1 -> same target
        cur = self.tgt.clamp_min(0)
        still_red = red.gather(1, cur) & (self.tgt >= 0)             # target still red-held (not yet taken)?
        refresh = (self.age >= self.hold) | ~still_red              # rotate on timeout or once taken/gone
        self.tgt = torch.where(refresh, newtgt, self.tgt)
        self.age = torch.where(refresh, torch.zeros_like(self.age), self.age + 1)
        cur = self.tgt.clamp_min(0)
        ok = red.gather(1, cur) & (self.tgt >= 0) & state.robot_valid
        return torch.where(ok, cfg.reverse_base + cur, torch.full_like(cur, cfg.idle_action))


@torch.no_grad()
def _rollout(cfg, policy, seed, batch, dev, want_labels=False):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    rec = RetentionLabelRecorder() if want_labels else None
    core.reset(seed)
    total = torch.zeros(batch, device=dev, dtype=core.dtype)     # sum of differential reward = J_H
    for _ in range(cfg.n_decisions):
        if rec:
            rec.before_decision(core)
        r, _, _ = core.step(policy.act(core.state, cfg))
        total += r
        if rec:
            rec.after_decision(core)
    jh = total.mean().item()
    return (jh, rec.finalize(core)) if want_labels else jh


def _ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


@torch.no_grad()
def diagnose(rho, alpha, seed, batch, dev) -> dict:
    """Is the ceiling even testing take-and-hold? Report: how often blue picks REVERSE, whether
    blue ever OWNS a task red held (take-and-hold vs revert-and-leave), and the spread of R-take
    over red-held tasks (the oracle's signal). If take-and-hold ~0 or take-signal std ~0, the
    scripted allocator reverts-and-leaves and the ceiling is UNINFORMATIVE, not a clean null."""
    over = {"symmetric": True, "alpha": alpha, "n_tasks": 16, "n_robots": 4, "n_adversaries": 4,
            "protected_fraction": 0.15, "reacquire_cost": rho, "adversary_population": ["builder"],
            "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    rec = RetentionLabelRecorder(); core.reset(seed)
    pol = ForcedReverseExplorer(seed=seed)          # forced exploration generates the R-take labels
    T = cfg.max_tasks
    rev_acts = nonidle = 0
    red_ever = torch.zeros(batch, cfg.max_tasks, dtype=torch.bool, device=dev)
    blue_took = torch.zeros_like(red_ever)
    red_held = []
    for _ in range(cfg.n_decisions):
        rec.before_decision(core)
        red_held.append(core.state.task_c_red.clone())
        act = pol.act(core.state, cfg)
        rev_acts += ((act >= 2 * T) & (act < 3 * T)).sum().item()
        nonidle += (act != cfg.idle_action).sum().item()
        core.step(act)
        rec.after_decision(core)
        red_ever |= core.state.task_c_red
        blue_took |= core.state.task_c & red_ever
    lab = rec.finalize(core)
    take = lab["R"][torch.stack(red_held, dim=1)]                     # R-take over red-held task-decisions
    nz = take[take > 1e-6]                                            # the takes that actually held
    valid = core.state.task_valid
    return {
        "rev_action_frac": rev_acts / max(1, nonidle),
        "red_held_tasks": (red_ever & valid).sum().item(),
        "blue_took_and_held": (blue_took & valid).sum().item(),
        "take_signal_mean": take.mean().item() if take.numel() else 0.0,
        "take_signal_std": take.std().item() if take.numel() > 1 else 0.0,
        "take_frac_positive": (take > 1e-6).float().mean().item() if take.numel() else 0.0,
        "take_pos_mean": nz.mean().item() if nz.numel() else 0.0,     # mean R-take among tasks blue held
        "take_pos_std": nz.std().item() if nz.numel() > 1 else 0.0,   # spread among held takes (the signal)
    }


def run_rho(rho, alpha, seeds, batch, dev, red=("builder",)) -> tuple[float, float]:
    over = {"symmetric": True, "alpha": alpha, "n_tasks": 16, "n_robots": 4, "n_adversaries": 4,
            "protected_fraction": 0.15, "reacquire_cost": rho, "adversary_population": list(red),
            "horizon_T": 60.0}
    cfg = load_config(overrides=over)
    a, b, c, dcmp = [], [], [], []
    for s in range(seeds):
        # R-take labels come from FORCED EXPLORATION (the coverage fix), not the weight-only policy
        # which never takes-and-holds -> its R_ref was identically zero. Same seed as the arms.
        _, lab = _rollout(cfg, ForcedReverseExplorer(seed=s), s, batch, dev, want_labels=True)
        R_ref = lab["R"]
        a.append(_rollout(cfg, SymmetricAllocator(allow_reverse=False), s, batch, dev))          # (a) no reverse
        b.append(_rollout(cfg, SymmetricAllocator(allow_reverse=True), s, batch, dev))           # (b) WO + reverse
        c.append(_rollout(cfg, SymmetricAllocator(R_ref=R_ref, on_reverse=True), s, batch, dev)) # (c) oracle-rev
        dcmp.append(_rollout(cfg, SymmetricAllocator(R_ref=R_ref, on_acquire=True), s, batch, dev))  # decomp
    m = statistics.fmean
    # THE decisive test: is reversing worth it WITH PERFECT R-take knowledge -- does oracle-rev beat
    # NOT reversing? (c)-(a). If even a perfect oracle can't beat no-reverse, a learned (imperfect)
    # R-take never will. (c)-(b) only measures self-harm recovery and misled the first runs.
    net = m(c) - m(a)
    net_ci = _ci95([x - y for x, y in zip(c, a)])
    recover = m(c) - m(b)                                     # knowing R-take recovers this much self-harm
    rev_val = m(b) - m(a)                                     # cost of the weight-only reverse action
    acq = m(dcmp) - m(a)                                      # acquire-channel oracle vs baseline
    verdict = ("NET HEADROOM: oracle-reverse BEATS no-reverse -> offensive channel pays" if net - net_ci > 0.005
               else "NO NET HEADROOM: reverse loses even with a PERFECT R-take oracle (only self-harm recovered)")
    print(f"rho={rho:<3} | WO-norev={m(a):.3f}  WO-rev={m(b):.3f}  oracle-rev={m(c):.3f}  oracle-acq={m(dcmp):.3f}")
    print(f"         NET offensive value (oracle-rev - NO-reverse, c-a)={net:+.3f} "
          f"[{net - net_ci:+.3f},{net + net_ci:+.3f}] -> {verdict}")
    print(f"         breakdown: reverse-action cost (b-a)={rev_val:+.3f}; knowing-R-take recovers (c-b)={recover:+.3f}; "
          f"acquire-oracle (dcmp-a)={acq:+.3f}")
    return net, net_ci


def main() -> None:
    ap = argparse.ArgumentParser(description="Symmetric oracle ceiling (round-9)")
    ap.add_argument("--rhos", nargs="+", type=float, default=[1, 2, 4])
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=128)
    ap.add_argument("--diagnose", action="store_true",
                    help="is the ceiling testing take-and-hold? report reverse rate + take-signal spread")
    ap.add_argument("--red", nargs="+", default=["builder"],
                    help="red population, e.g. builder (holds only) or 'builder value_targeting' (also contests)")
    args = ap.parse_args()
    dev = default_device()
    if args.diagnose:
        print("DIAGNOSTIC — forced-exploration label-gen: does blue take-and-hold, and does R-take vary?")
        for rho in args.rhos:
            d = diagnose(rho, args.alpha, 0, args.batch, dev)
            print(f"rho={rho:<3} rev-frac={d['rev_action_frac']:.2f}  red-held={d['red_held_tasks']}  "
                  f"blue-took&held={d['blue_took_and_held']}  R-take mean={d['take_signal_mean']:.3f} "
                  f"std={d['take_signal_std']:.3f}  | held-takes: frac>0={d['take_frac_positive']:.3f} "
                  f"mean={d['take_pos_mean']:.3f} std={d['take_pos_std']:.3f}")
        return
    print(f"SYMMETRIC ORACLE CEILING (device={dev}, {args.seeds} seeds, red={args.red}) — hindsight "
          "R-take vs weight-only.\nDECISIVE test = NET offensive value (oracle-rev - NO-reverse, c-a): "
          "is reversing worth it even with a PERFECT R-take oracle?\nStop rule: no net headroom at "
          "every rho => the offensive-retention line ends.\n")
    nets = []
    for rho in args.rhos:
        h, ci = run_rho(rho, args.alpha, args.seeds, args.batch, dev, red=args.red)
        nets.append((rho, h, ci))
    print()
    if all(h - ci <= 0.005 for _, h, ci in nets):
        print("STOP RULE MET: oracle-reverse does NOT beat no-reverse at any rho -> the offensive")
        print("channel does not pay even with perfect retention knowledge. A learned two-head cannot")
        print("do better than the oracle, so the retention-ESTIMATION line ends here (honest negative).")
    else:
        grows = len(nets) > 1 and nets[-1][1] > nets[0][1] + 1e-6
        print(f"NET HEADROOM present{' and GROWING with rho' if grows else ''} -> reversing pays with "
              "perfect R-take -> proceed to the learned two-head + residual-spread gate.")


if __name__ == "__main__":
    main()
