"""Bait / tempo ceiling (Phase 3) -- the retention win the user described.

The point of a tall stack is NOT to hold it (defending ties up robots and the opponent just freezes
it in a standoff -- see stacking_ceiling.py). The point is that a stack is EXPENSIVE-TO-UNDO and
ownership PERSISTS while it is being undone. You build it, ABANDON it, and the opponent must pay to
flatten it -- one piece at a time -- while it keeps scoring for you the whole time.

Two facts of the mechanic make this a real weapon (both verified below against closed form):
  * Dismantling is SERIAL and per-piece and does NOT speed up with more attackers: eta accrues at dt
    per tick while red dominates, and the piece at level L comes off only after tau_rev * L. So a
    height-H stack takes tau_rev * H(H+1)/2 to flatten NO MATTER how many red robots pile on -- extra
    attackers are wasted. H flat height-1 tasks, by contrast, fall to A attackers in PARALLEL.
  * An abandoned stack stays blue-owned (task_c) until its LAST piece comes off, scoring w * height
    every tick. So while red spends tau_rev * H(H+1)/2 tearing it down, blue banks
    w * tau_rev * sum_{h=1..H} h^2 = w * tau_rev * H(H+1)(2H+1)/6 of held value -- CUBIC in H.

This script measures, in the REAL mechanic with pinned occupancy (no navigation confound):
  (1) time-to-flatten and bait held-value vs the height H, and that both are INDEPENDENT of the
      attacker count A (the serialisation), and
  (2) for a MATCHED number of pieces P, tall (one height-P bait) vs flat (P height-1 tasks) under A
      attackers: how much longer red is tied up and how much more value blue banks.

    python -m experiments.stacking_bait
"""
from __future__ import annotations

import argparse

import torch

from contested.config import load_config
from contested.core import CanonicalCore, default_device

_FAR = 99.0                                                              # park idle agents off-field


def _make(cfg, dev, batch=1):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(0)
    s = core.state
    T = cfg.max_tasks
    # spread tasks along a line, unit weight, all invalid except the ones we use later
    xs = torch.linspace(0.3, cfg.field_size[0] - 0.3, T, device=dev)
    s.task_pos[:] = torch.stack([xs, torch.full_like(xs, cfg.field_size[1] / 2)], dim=-1)
    s.task_w[:] = 1.0
    s.task_valid[:] = False
    s.task_c[:] = False
    s.task_c_red[:] = False
    s.task_height[:] = 0
    s.robot_pos[:] = _FAR
    s.robot_vel[:] = 0.0
    s.adv_pos[:] = _FAR
    s.adv_vel[:] = 0.0
    s.robot_valid[:] = False
    s.adv_valid[:] = False
    s.held_integral[:] = 0.0
    s.held_integral_red[:] = 0.0
    return core


def _pin(core, blue_on, red_on):
    """Pin blue robots and red advs onto given task indices (one agent per (slot, task)); the rest
    are parked far away and idle. Returns after setting positions/actions so a tick keeps them put."""
    s = core.state
    cfg = core.cfg
    dev = s.task_pos.device
    T = cfg.max_tasks
    s.robot_pos[:] = _FAR
    s.robot_vel[:] = 0.0
    s.robot_valid[:] = False
    s.robot_action[:] = cfg.idle_action
    for r, t in enumerate(blue_on):
        s.robot_pos[0, r] = s.task_pos[0, t]
        s.robot_valid[0, r] = True
        owned = bool(s.task_c[0, t])
        s.robot_action[0, r] = (cfg.max_tasks + t) if owned else t       # DEFEND if owned else ACQUIRE
    s.adv_pos[:] = _FAR
    s.adv_vel[:] = 0.0
    s.adv_valid[:] = False
    s.adv_target[:] = -1
    for k, t in enumerate(red_on):
        s.adv_pos[0, k] = s.task_pos[0, t]
        s.adv_valid[0, k] = True
        s.adv_target[0, k] = t


def _run(core, steps, blue_on_fn, red_on_fn):
    """Step the real mechanic `steps` ticks, re-pinning each tick per the schedule functions."""
    for i in range(steps):
        _pin(core, blue_on_fn(i, core), red_on_fn(i, core))
        core.tick()


def teardown(cfg, dev, height, n_red, task=0):
    """Pre-build one blue stack of `height` on `task`, abandon it, and let `n_red` attack until it is
    flat. Return (seconds_to_flatten, bait_held_value_during_teardown)."""
    core = _make(cfg, dev)
    s = core.state
    s.task_valid[0, task] = True
    s.task_c[0, task] = True
    s.task_height[0, task] = height
    s.held_integral[:] = 0.0
    reds = [task] * n_red
    t_flat, dt = None, cfg.dt
    for i in range(20000):
        _pin(core, [], reds)                                            # blue absent (abandoned)
        core.tick()
        if int(s.task_height[0, task]) == 0:
            t_flat = (i + 1) * dt
            break
    return t_flat, float(s.held_integral[0])


def build_time(cfg, dev, height, n_blue=4, task=0):
    """Time for `n_blue` robots to raise one stack from neutral to `height` (no opponent)."""
    core = _make(cfg, dev)
    core.state.task_valid[0, task] = True
    blues = [task] * n_blue
    dt = cfg.dt
    for i in range(20000):
        _pin(core, blues, [])
        core.tick()
        if int(core.state.task_height[0, task]) >= height:
            return (i + 1) * dt
    return None


def teardown_flat(cfg, dev, pieces, n_red):
    """Pre-build `pieces` separate height-1 blue tasks, abandon them, and let `n_red` attack them in
    PARALLEL (round-robin) until all flat. Return (seconds, total_held_value)."""
    core = _make(cfg, dev)
    s = core.state
    tasks = list(range(pieces))
    for t in tasks:
        s.task_valid[0, t] = True
        s.task_c[0, t] = True
        s.task_height[0, t] = 1
    s.held_integral[:] = 0.0
    dt = cfg.dt
    for i in range(20000):
        alive = [t for t in tasks if int(s.task_height[0, t]) > 0]
        if not alive:
            return (i) * dt, float(s.held_integral[0])
        reds = [alive[j % len(alive)] for j in range(n_red)]           # spread attackers in parallel
        _pin(core, [], reds)
        core.tick()
    return None, float(s.held_integral[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Bait / tempo ceiling (Phase 3)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--heights", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    dev = default_device()
    cfg = load_config(overrides={"symmetric": True, "stack_cap": max(args.heights), "alpha": args.alpha,
                                 "n_tasks": 1, "max_tasks": 8, "n_robots": 4, "n_adversaries": 4,
                                 "horizon_T": 120.0})
    tr, tc = cfg.tau_rev, cfg.tau_com
    print(f"BAIT / TEMPO CEILING (alpha={args.alpha}, tau_com={tc}, tau_rev={tr:.3f}, dt={cfg.dt})\n")

    print("(1) One abandoned stack of height H, attacked by A red. Flatten time and bait held-value.")
    print("    Closed form: flatten = tau_rev*H(H+1)/2 (INDEPENDENT of A); "
          "held = w*tau_rev*sum h^2 = tau_rev*H(H+1)(2H+1)/6\n")
    print(f"    {'H':>2} {'A':>2} | {'flatten_s':>9} {'predict':>8} | {'held':>7} {'predict':>8}")
    for H in args.heights:
        for A in (1, 2, 3):
            t_flat, held = teardown(cfg, dev, H, A)
            pred_t = tr * H * (H + 1) / 2
            pred_h = tr * H * (H + 1) * (2 * H + 1) / 6
            print(f"    {H:>2} {A:>2} | {t_flat:>9.2f} {pred_t:>8.2f} | {held:>7.2f} {pred_h:>8.2f}")

    print("\n(2) MATCHED pieces P: one height-P bait vs P flat height-1 tasks, under A=3 attackers.")
    print(f"    {'P':>2} | {'tall_flatten':>12} {'flat_flatten':>12} | {'tall_held':>9} {'flat_held':>9} "
          f"| {'build_tall':>10} {'build_flat':>10}")
    for P in args.heights:
        if P < 2:
            continue
        tt, th = teardown(cfg, dev, P, 3)
        ft, fh = teardown_flat(cfg, dev, P, 3)
        bt = build_time(cfg, dev, P, n_blue=4)
        bf = P * tc                                                     # flat: P pieces at tau_com each
        print(f"    {P:>2} | {tt:>12.2f} {ft:>12.2f} | {th:>9.2f} {fh:>9.2f} | {bt:>10.2f} {bf:>10.2f}")

    print("\nRead: tall flatten-time is flat in A (serial) and >> flat's parallel teardown; tall banks")
    print("cubically more held value while dying. Build cost is the tax you pay for that tempo weapon.")


if __name__ == "__main__":
    main()
