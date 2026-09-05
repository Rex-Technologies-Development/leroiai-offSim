"""Toggle / multiplier retention (Phase 3, Override-grounded) -- does caring about retention win when
losing an objective has OUTSIZED consequences?

Real Override retention is not equal-value stacking; it is MULTIPLIER CONTROL. Owning a quadrant's
toggle turns its goals from 5 to 15 points (offsim/tests/test_domain_scoring.py) -- a 3x on a whole
cluster. So a toggle is a SCARCE, HIGH-LEVERAGE point: holding it multiplies your work in its region,
losing it collapses the cluster back to 1x. That is exactly the regime retention was meant for, and
one the equal-value experiments never had.

Premise (the user's, and the project's): OUR coordinated system out-builds a BASELINE opponent -- so
we can actually win and hold the scarce toggles (red is the slower archetype; ``--red-speed`` < 1).

Layout: n_tasks split into Q regions; task 0 of each region is its TOGGLE, the rest are GOALS.
Owning a region's toggle multiplies the held value of YOUR goals there by M. The env dynamics are the
plain symmetric take model (stack_cap=1) -- the multiplier lives in the OBJECTIVE we score, so this
tests whether an allocator that PRICES the multiplier (values a toggle by the cluster it unlocks and
DEFENDS it) beats one blind to it.

  * BLIND (does not care): every task scored by base weight -- a toggle is just another task.
  * RETENTION-AWARE (cares): a toggle is priced at its LEVERAGE = base + (M-1)*cluster_value, taken
    early and DEFENDED; goals are worth more once you hold their toggle.

PREDICTION: they tie at M=1 (no multiplier, retention decorative) and RETENTION-AWARE pulls ahead as
M grows -- the gap should widen with the multiplier. If it appears, retention wins on Override's own
value structure.

    python -m experiments.toggle_retention --mults 1 2 3 5 --seeds 8
"""
from __future__ import annotations

import argparse
import statistics

import torch

from baselines.base import BaselinePolicy, blank_scores, robot_task_geometry
from contested.config import load_config
from contested.core import CanonicalCore, CanonicalState, default_device

DEFEND_PREF = 4.0                                                        # hold what you own


def _layout(cfg, dev, Q):
    """Fixed index-based region/toggle layout: Q regions of `per` tasks, task 0 of each is the toggle."""
    T = cfg.n_tasks
    per = T // Q
    region_of = torch.arange(T, device=dev) // per
    region_of = region_of.clamp(max=Q - 1)                              # remainder tasks fold into last region
    is_toggle = torch.zeros(T, dtype=torch.bool, device=dev)
    for q in range(Q):
        is_toggle[q * per] = True                                       # first task in the region block
    toggle_of_region = torch.tensor([q * per for q in range(Q)], device=dev)
    return region_of, is_toggle, toggle_of_region


def multiplied_value(state, region_of, is_toggle, toggle_of_region, M, team_c):
    """Instantaneous multiplied held value for the team whose ownership mask is `team_c` (B,T)."""
    w = state.task_w[:, :region_of.shape[0]]                            # (B,T) base weights (unpadded)
    own = team_c[:, :region_of.shape[0]].to(w.dtype)
    holds_toggle = own.gather(1, toggle_of_region.unsqueeze(0).expand(w.shape[0], -1))  # (B,Q)
    mult_per_task = torch.where(holds_toggle[:, region_of] > 0.5, M, 1.0)               # (B,T)
    mult_per_task = torch.where(is_toggle.unsqueeze(0), torch.ones_like(mult_per_task), mult_per_task)
    return (w * mult_per_task * own).sum(-1)                            # (B,)


class TogglePolicy(BaselinePolicy):
    """Scores acquire/defend/reverse by an effective weight. BLIND uses base weight; AWARE prices the
    toggle at the cluster value it unlocks and defends it. Only the weight vector differs."""

    def __init__(self, aware, region_of, is_toggle, toggle_of_region, M):
        self.aware = aware
        self.region_of = region_of
        self.is_toggle = is_toggle
        self.toggle_of_region = toggle_of_region
        self.M = M
        self.name = "aware" if aware else "blind"

    def _w_eff(self, state, cfg):
        T = self.region_of.shape[0]
        w = state.task_w[:, :T].clone()                                 # (B,T)
        if not self.aware:
            return w
        Q = self.toggle_of_region.shape[0]
        # cluster base value = sum of GOAL base weights in each region
        goal = (~self.is_toggle).to(w.dtype)
        cluster = torch.zeros(w.shape[0], Q, device=w.device, dtype=w.dtype)
        cluster.index_add_(1, self.region_of, w * goal)                 # (B,Q)
        w_eff = w.clone()
        lev = w + (self.M - 1.0) * cluster[:, self.region_of]           # toggle leverage
        w_eff = torch.where(self.is_toggle.unsqueeze(0), lev, w_eff)
        # a goal is worth M x once you hold its toggle (cash in the multiplier)
        own_tog = state.task_c[:, :T].to(w.dtype).gather(
            1, self.toggle_of_region.unsqueeze(0).expand(w.shape[0], -1))
        goalmult = torch.where(own_tog[:, self.region_of] > 0.5, self.M, 1.0)
        w_eff = torch.where(self.is_toggle.unsqueeze(0), w_eff, w * goalmult)
        return w_eff

    def scores(self, state: CanonicalState, cfg) -> torch.Tensor:
        _, travel = robot_task_geometry(state, cfg)                      # (B,R,T)
        T = cfg.max_tasks
        nt = self.region_of.shape[0]
        w_eff = torch.zeros(state.task_w.shape, device=state.device, dtype=state.task_w.dtype)
        w_eff[:, :nt] = self._w_eff(state, cfg)
        prox = 1.0 / (travel + 0.1)
        s = blank_scores(state, cfg)
        neutral = (~state.task_c & ~state.task_c_red & state.task_valid).to(w_eff.dtype)
        blue = (state.task_c & state.task_valid).to(w_eff.dtype)
        red = (state.task_c_red & state.task_valid).to(w_eff.dtype)
        s[:, :, 0:T] = (neutral * w_eff).unsqueeze(1) * prox                          # ACQUIRE
        s[:, :, T:2 * T] = (blue * w_eff * DEFEND_PREF).unsqueeze(1) * prox           # DEFEND (hold)
        s[:, :, cfg.reverse_base:cfg.reverse_base + T] = (red * w_eff).unsqueeze(1) * prox  # REVERSE (steal)
        return s


def _set_weights(core, is_toggle, toggle_w):
    """Toggles are CHEAP to place but HIGH-LEVERAGE: give them a low base weight so a retention-blind
    policy ignores them, and their value comes only from the multiplier they unlock."""
    T = is_toggle.shape[0]
    w = core.state.task_w
    w[:, :T] = torch.where(is_toggle.unsqueeze(0), torch.full_like(w[:, :T], toggle_w), 1.0)


@torch.no_grad()
def _toggle_aware_red(state, region_of, is_toggle):
    """A smart opponent that goes for the MULTIPLIER: each red robot targets the nearest blue-owned
    toggle (to flip it and collapse the cluster), else the nearest blue-owned goal."""
    T = region_of.shape[0]
    K = state.adv_pos.shape[1]
    d = torch.linalg.vector_norm(state.adv_pos.unsqueeze(2) - state.task_pos[:, :T].unsqueeze(1), dim=-1)  # (B,K,T)
    bc = state.task_c[:, :T]
    tog = (is_toggle.unsqueeze(0) & bc).unsqueeze(1).expand(-1, K, -1)         # (B,K,T)
    goal = ((~is_toggle).unsqueeze(0) & bc).unsqueeze(1).expand(-1, K, -1)
    neg = torch.finfo(d.dtype).min
    tgt_tog = torch.where(tog, -d, torch.full_like(d, neg)).argmax(-1)         # (B,K)
    tgt_goal = torch.where(goal, -d, torch.full_like(d, neg)).argmax(-1)
    has_tog = tog.any(-1)
    has_goal = goal.any(-1)
    tgt = torch.where(has_tog, tgt_tog, torch.where(has_goal, tgt_goal, torch.full_like(tgt_tog, -1)))
    full = torch.full((state.adv_pos.shape[0], state.adv_valid.shape[1]), -1, dtype=torch.int64, device=d.device)
    full[:, :K] = tgt
    return torch.where(state.adv_valid, full, torch.full_like(full, -1))


@torch.no_grad()
def _rollout(cfg, policy, seed, batch, dev, region_of, is_toggle, tog_of, M, toggle_w, red_mode):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    _set_weights(core, is_toggle, toggle_w)
    if red_mode == "toggle":
        core.adversaries = None                                                # drive red ourselves
    acc = torch.zeros(batch, device=dev)
    for _ in range(cfg.n_decisions):
        if red_mode == "toggle":
            core.state.adv_target = _toggle_aware_red(core.state, region_of, is_toggle)
        core.step(policy.act(core.state, cfg))
        vb = multiplied_value(core.state, region_of, is_toggle, tog_of, M, core.state.task_c)
        vr = multiplied_value(core.state, region_of, is_toggle, tog_of, M, core.state.task_c_red)
        acc += (vb - vr)
    denom = cfg.n_decisions * core.state.task_w[:, :region_of.shape[0]].sum(-1).clamp_min(1e-9)
    return (acc / denom).mean().item()


def run_mult(M, seeds, batch, dev, n_tasks, Q, n_adv, red_speed, alpha, toggle_w, red_mode):
    over = {"symmetric": True, "stack_cap": 1, "alpha": alpha, "n_tasks": n_tasks, "n_robots": 4,
            "n_adversaries": n_adv, "adversary_population": ["value_targeting"], "horizon_T": 60.0,
            "adv_v_max": 1.5 * red_speed, "adv_a_max": 3.0 * red_speed}
    cfg = load_config(overrides=over)
    region_of, is_toggle, tog_of = _layout(cfg, dev, Q)
    blind = TogglePolicy(False, region_of, is_toggle, tog_of, float(M))
    aware = TogglePolicy(True, region_of, is_toggle, tog_of, float(M))
    b = [_rollout(cfg, blind, s, batch, dev, region_of, is_toggle, tog_of, float(M), toggle_w, red_mode)
         for s in range(seeds)]
    a = [_rollout(cfg, aware, s, batch, dev, region_of, is_toggle, tog_of, float(M), toggle_w, red_mode)
         for s in range(seeds)]
    m = statistics.fmean
    gap = m(a) - m(b)
    tag = "AWARE wins" if gap > 0.005 else ("blind wins" if gap < -0.005 else "tie")
    print(f"M={M:<2} | BLIND={m(b):+.3f}  RETENTION-AWARE={m(a):+.3f}  gap={gap:+.3f}  [{tag}]")
    return M, gap


def main() -> None:
    ap = argparse.ArgumentParser(description="Toggle/multiplier retention (Override-grounded)")
    ap.add_argument("--mults", nargs="+", type=float, default=[1, 2, 3, 5])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=128)
    ap.add_argument("--n-tasks", dest="n_tasks", type=int, default=12)
    ap.add_argument("--regions", dest="Q", type=int, default=4)
    ap.add_argument("--n-adversaries", dest="n_adv", type=int, default=4)
    ap.add_argument("--red-speed", type=float, default=0.8, help="opponent speed factor (<1 = we out-build)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--toggle-weight", dest="toggle_w", type=float, default=0.15,
                    help="toggle base weight (low => a blind policy ignores it; value is in the multiplier)")
    ap.add_argument("--red-mode", choices=["value", "toggle"], default="toggle",
                    help="'toggle' = a smart opponent that attacks your toggles to steal the multiplier")
    args = ap.parse_args()
    dev = default_device()
    print(f"TOGGLE/MULTIPLIER RETENTION (device={dev}, {args.seeds} seeds, n_tasks={args.n_tasks}, "
          f"regions={args.Q}, red_speed={args.red_speed}, red_mode={args.red_mode}, toggle_w={args.toggle_w})")
    print("Does pricing the toggle multiplier + DEFENDING it beat being blind to it? Gap should grow with M.\n")
    gaps = []
    for M in args.mults:
        _, g = run_mult(M, args.seeds, args.batch, dev, args.n_tasks, args.Q, args.n_adv,
                        args.red_speed, args.alpha, args.toggle_w, args.red_mode)
        gaps.append((M, g))
    print()
    grows = len(gaps) > 1 and gaps[-1][1] > gaps[0][1] + 0.01
    wins = all(g > 0.005 for M, g in gaps if M > 1)
    if wins and grows:
        print("CONFIRMED: retention-aware beats blind AND the gap widens with the multiplier.")
        print("On Override's own value structure (toggle leverage), caring about retention WINS.")
    else:
        print(f"Not (yet) confirmed: wins={wins} grows={grows}. Inspect the per-M gaps above.")


if __name__ == "__main__":
    main()
