"""Lead-time behavioural eval: does the policy defend toggles PREEMPTIVELY or only REACTIVELY?

J_H says a policy holds toggles; it does not say *when* it commits a defender relative to the
threat. This races, for every contested toggle, blue's commitment against red's arrival:

  t_red   = first decision step a red adversary comes within R of the toggle (threat onset)
  t_blue  = first step a blue robot ASSIGNED to that toggle (acquire/defend/reverse) is within R
  lead    = (t_red - t_blue) * dt   -> POSITIVE means blue got there first (anticipation)

A purely reactive policy (defend only what is being lost) cannot get positive lead; greedy churns
in *after* the flip. A learned retention policy that prices the leverage should station a defender
*before* red arrives -- and only on the toggles red actually contests (guarding a safe toggle is the
defensive baseline's waste). Reports, per policy over N seeds:

  * %preemptive / %reactive / %undefended  among CONTESTED toggles
  * mean signed lead (s)                    over defended contested toggles
  * capture rate                            fraction of contested toggles red ever flips (lower=better)
  * guard-waste                             blue robot-steps spent committed to UNcontested toggles

    python -m experiments.eval_leadtime tenure/checkpoints/harden_ext_off_s0.pt --radius 0.5
"""
from __future__ import annotations

import argparse
import statistics

import torch

from baselines import REGISTRY
from contested.config import load_config
from contested.core import CanonicalCore
from contested.observation import build_observation
from tenure.policy import TenurePolicy


@torch.no_grad()
def _rollout_trace(cfg, actfn, seed, batch, dev):
    """Run one rollout, capturing per-step positions/assignments. Returns dict of stacked tensors."""
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    rp, ra, ap, rcred = [], [], [], []
    for _ in range(cfg.n_decisions):
        core.step(actfn(core))
        s = core.state
        rp.append(s.robot_pos.clone()); ra.append(s.robot_action.clone())
        ap.append(s.adv_pos.clone());   rcred.append(s.task_c_red.clone())
    s = core.state
    return {
        "rp": torch.stack(rp),           # (S,B,R,2)
        "ra": torch.stack(ra),           # (S,B,R)
        "ap": torch.stack(ap),           # (S,B,K,2)
        "rc_red": torch.stack(rcred),    # (S,B,T)
        "task_pos": s.task_pos,          # (B,T,2) static
        "is_tog": s.task_is_toggle,      # (B,T)
        "rvalid": s.robot_valid, "avalid": s.adv_valid, "tvalid": s.task_valid,
    }


def _leadtime_stats(tr, cfg, R, dt):
    T = cfg.max_tasks
    rp, ra, ap = tr["rp"], tr["ra"], tr["ap"]
    S, B = rp.shape[0], rp.shape[1]
    tpos = tr["task_pos"]                                    # (B,T,2)
    is_tog = tr["is_tog"] & tr["tvalid"]                     # (B,T)

    # committed target of each robot this step (acquire<T, defend<2T, reverse<3T, idle=3T)
    a = ra
    tgt = torch.where(a < T, a, torch.where(a < 2 * T, a - T, torch.where(a < 3 * T, a - 2 * T,
          torch.full_like(a, -1))))                          # (S,B,R) task idx or -1
    is_defend_mode = (a >= T) & (a < 3 * T)                  # defend or reverse (vs plain acquire)

    # distances: robot->task and adv->task, per step
    # rp (S,B,R,2), tpos (B,T,2) -> (S,B,R,T)
    d_rt = torch.linalg.vector_norm(rp.unsqueeze(3) - tpos.unsqueeze(0).unsqueeze(2), dim=-1)
    d_at = torch.linalg.vector_norm(ap.unsqueeze(3) - tpos.unsqueeze(0).unsqueeze(2), dim=-1)  # (S,B,K,T)

    rvalid = tr["rvalid"].unsqueeze(0).unsqueeze(-1)         # (1,B,R,1)
    avalid = tr["avalid"].unsqueeze(0).unsqueeze(-1)         # (1,B,K,1)
    d_at = torch.where(avalid.bool(), d_at, torch.full_like(d_at, 1e9))
    red_here = (d_at < R).any(dim=2)                         # (S,B,T) any red within R this step

    # blue committed to task t AND within R this step: (S,B,R,T)
    committed = (tgt.unsqueeze(-1) == torch.arange(T, device=a.device)) & rvalid.bool()
    blue_here = (committed & (d_rt < R)).any(dim=2)          # (S,B,T)
    blue_def_here = (committed & is_defend_mode.unsqueeze(-1) & (d_rt < R)).any(dim=2)

    # first-step-True helper along S (returns S if never)
    def first_true(x):                                       # x: (S,B,T) bool
        idx = torch.where(x.any(0), x.float().argmax(0), torch.full(x.shape[1:], S, device=x.device))
        return idx                                           # (B,T) int
    t_red = first_true(red_here)
    # Two distinct races, reported separately because they answer different halves of "preemptive?":
    #   OCCUPANCY   t_occ  = first step a blue robot committed to the toggle is within R (pre-positioning)
    #   DEFENSE-ACT t_def  = first step that commitment is a DEFEND/REVERSE action (intent label)
    # The anticipation lives in occupancy (blue is already ON the toggle before red), not in the action
    # label (which flips to "defend" reactively as red arrives). Greedy also occupies early -- but to
    # GRAB, then leaves; capture_rate is the ungameable outcome that separates hold-and-defend from
    # grab-and-go.
    t_occ = first_true(blue_here)
    t_def = first_true(blue_def_here)
    captured = tr["rc_red"].any(0)                           # (B,T) red ever owned

    tog = is_tog                                             # (B,T) bool
    contested = tog & (t_red < S)                            # red arrived at a toggle
    committed_ever = t_occ < S

    pre = contested & committed_ever & (t_occ <= t_red)      # blue occupying BEFORE red arrives
    rea = contested & committed_ever & (t_occ > t_red)
    und = contested & ~committed_ever
    lead_s = (t_red.float() - t_occ.float()) * dt            # occupancy lead (signed)
    def_lead_s = (t_red.float() - t_def.float()) * dt        # defend-action lead (signed)
    defended = contested & committed_ever
    def_engaged = contested & (t_def < S)
    n_contest = contested.sum().item()
    # guard-waste: blue robot-steps committed to an UNcontested toggle (safe point)
    uncontested_tog = tog & (t_red >= S)                     # (B,T)
    commit_on_tog = (committed & (d_rt < R)).any(2)          # (S,B,T) blue present-committed
    waste = (commit_on_tog & uncontested_tog.unsqueeze(0)).sum().item() / B
    # defend-action share among toggle commitments
    tog_commit_steps = (committed & tog.unsqueeze(0).unsqueeze(2)).any(2)      # present or not
    tog_commit_def = (committed & is_defend_mode.unsqueeze(-1) & tog.unsqueeze(0).unsqueeze(2)).any(2)
    def_share = tog_commit_def.sum().item() / max(1, tog_commit_steps.sum().item())

    def frac(x):
        return (x.sum().item() / n_contest) if n_contest else float("nan")
    mean_lead = lead_s[defended].mean().item() if defended.any() else float("nan")
    mean_def_lead = def_lead_s[def_engaged].mean().item() if def_engaged.any() else float("nan")
    return {
        "n_contest_per_ep": n_contest / B,
        "pct_pre": frac(pre), "pct_rea": frac(rea), "pct_und": frac(und),
        "mean_lead_s": mean_lead, "mean_def_lead_s": mean_def_lead,
        "capture_rate": frac(contested & captured),
        "guard_waste": waste, "def_share": def_share,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lead-time (preemptive vs reactive) toggle-defense eval")
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    ap.add_argument("--radius", type=float, default=0.5, help="contest radius R (field units)")
    ap.add_argument("--adversary", nargs="+", default=None,
                    help="override the eval opponent population (e.g. 'greedy_nearest' for a PEACEFUL, "
                         "non-toggle-raiding world -> tests whether toggle defense is a wasteful habit)")
    args = ap.parse_args()
    dev = "cpu"

    ck = torch.load(args.checkpoint, map_location="cpu")
    ov = dict(ck.get("overrides") or {})
    if args.adversary is not None:
        ov["adversary_population"] = list(args.adversary)
    cfg = load_config(overrides=ov)
    dt = cfg.horizon_T / cfg.n_decisions
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"]); pol.eval()

    def tenure_act(core):
        return pol.act(build_observation(core.state, cfg), deterministic=True)["action"]

    rows = [("AWARE-TENURE", tenure_act)]
    for b in ("greedy", "defensive"):
        pol_b = REGISTRY[b]()
        rows.append((b, (lambda c, p=pol_b: p.act(c.state, cfg))))

    print(f"regime: toggles={cfg.toggle_regions} M={cfg.toggle_multiplier} red={cfg.adversary_population} "
          f"R={args.radius} dt={dt}s  ({args.seeds} seeds)\n")
    hdr = (f"{'policy':>14} | {'capture':>7} | {'%occ<red':>8} {'occ-lead':>8} | {'defact-lead':>11} "
           f"{'def%':>4} | {'waste':>5} {'contest/ep':>10}")
    print(hdr); print("-" * len(hdr))
    keys = ("n_contest_per_ep", "pct_pre", "pct_rea", "pct_und", "mean_lead_s", "mean_def_lead_s",
            "capture_rate", "guard_waste", "def_share")
    for name, fn in rows:
        agg = {k: [] for k in keys}
        for s in range(args.seeds):
            tr = _rollout_trace(cfg, fn, s, args.batch, dev)
            st = _leadtime_stats(tr, cfg, args.radius, dt)
            for k in keys:
                agg[k].append(st[k])
        m = {k: statistics.fmean(v) for k, v in agg.items()}
        print(f"{name:>14} | {m['capture_rate']*100:6.0f}% | {m['pct_pre']*100:6.0f}% {m['mean_lead_s']:+8.2f} "
              f"| {m['mean_def_lead_s']:+11.2f} {m['def_share']*100:3.0f}% | {m['guard_waste']:5.1f} "
              f"{m['n_contest_per_ep']:10.2f}")
    print("\nRead: the anticipation is in OCCUPANCY, not the action label. %occ<red / occ-lead = blue is "
          "already\nON the toggle before red arrives (pre-positioned). defact-lead ~0/negative = it flips to "
          "the\nDEFEND action reactively, on contact. capture = the ungameable outcome: TENURE pre-positions "
          "AND\nHOLDS (low capture); greedy also occupies early but to GRAB then leaves (capture ~1.0); "
          "defensive\nholds but over-guards SAFE toggles (high waste). => preemptive positioning, reactive "
          "labelling,\nselective targeting -- none of which a reactive baseline reproduces.")


if __name__ == "__main__":
    main()
