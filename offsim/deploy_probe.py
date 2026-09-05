"""Deploy a trained TENURE policy on the REAL Override game, INSTRUMENTED for stuck-detection.

Empirically settles the question the static audit could only reason about: does any robot ever
get *stuck* — commanded to drive but its position not updating — at any point in a full match?

For every physics tick and every robot it records (x, y, heading, forward/lateral/yaw command,
per-tick displacement, cumulative collisions), flags each "commanded-to-move but did-not-move"
tick, and reports sustained stuck STREAKS with where and when they happened. It writes:
  * a watchable GIF with a red "STUCK r<id>" overlay on any frame where a robot is stuck,
  * a CSV trace (tick, robot, x, y, heading, fwd/lat/yaw cmd, dpos, collisions) for inspection,
  * a printed per-robot verdict.

A robot legitimately stops when it ARRIVES (throttle -> 0 within 0.5in) or is ROTATING in place
(forward throttle ~0 while yawing to face its target); both are excluded from stuck accounting.
Only "forward throttle > MOVE_CMD and net displacement < MOVE_EPS" counts, and only a run of
>= STUCK_STREAK such ticks (0.5s) is reported as an event.

    python -m offsim.deploy_probe runs/tenure_demo.pt --out recordings/probe.gif
"""
from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

MOVE_CMD = 0.05     # forward throttle above this = intent to translate (not just rotate/arrived)
MOVE_EPS = 0.05     # net displacement (inches) below this in a tick = "did not move"
STUCK_STREAK = 10   # consecutive PINNED ticks (@dt=0.05 => 0.5s) to call it a real stuck EVENT
DOCK_DIST = 16.0    # <= this to target-center = docked at target (radius+obj ~12.5, interact <=15):
                    # commanded-but-not-moving HERE is productive docking, NOT a navigation pin


class _RobotProbe:
    """Splits commanded-but-not-moving into DOCKED (at its target, i.e. legitimately parked at the
    goal it is scoring) vs PINNED (stuck FAR from its target, i.e. a genuine navigation stall)."""

    def __init__(self, rid: int):
        self.rid = rid
        self.commanded = 0          # ticks with forward throttle > MOVE_CMD
        self.moved_ok = 0           # of those, ticks that actually translated
        self.docked = 0             # stuck, but AT its target (benign)
        self.pinned = 0             # stuck, and FAR from its target (real nav pin)
        self.streak = 0             # current consecutive PINNED ticks
        self.max_streak = 0
        self.max_streak_at = None   # (tick, x, y) of the worst PINNED streak
        self.events = 0             # number of PINNED streaks that reached STUCK_STREAK
        self._in_event = False

    def update(self, tick, fwd_cmd, dpos, x, y, dist_to_target):
        if fwd_cmd <= MOVE_CMD or dpos >= MOVE_EPS:     # arrived/rotating, or actually moved
            if fwd_cmd > MOVE_CMD:
                self.commanded += 1
                self.moved_ok += 1
            self.streak = 0
            self._in_event = False
            return False
        # commanded to drive, but did not move
        self.commanded += 1
        at_target = dist_to_target is not None and dist_to_target <= DOCK_DIST
        if at_target:                                   # parked at its own target goal => benign
            self.docked += 1
            self.streak = 0
            self._in_event = False
            return False
        self.pinned += 1                                # stuck far from target => genuine pin
        self.streak += 1
        if self.streak > self.max_streak:
            self.max_streak = self.streak
            self.max_streak_at = (tick, x, y)
        if self.streak >= STUCK_STREAK and not self._in_event:
            self.events += 1
            self._in_event = True
        return self.streak >= STUCK_STREAK              # currently in a real pin (for the overlay)


def _overlay(frame, msgs):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    if msgs:
        d = ImageDraw.Draw(img)
        for i, m in enumerate(msgs):
            d.text((6, 6 + 14 * i), m, fill=(255, 40, 40))
    return img


def probe(checkpoint: str, out: str, seed: int = 7, opponent: str = "mixed", contested: bool = True,
          seconds: float | None = None, target_frames: int = 240, fps: int = 20, size: int = 560,
          deterministic: bool = True, csv_path: str | None = None) -> dict:
    import torch

    from offsim.sim.config import DECISION_INTERVAL, DT, MATCH_DURATION
    from offsim.sim.graph_env import OverrideGraphEnv
    from offsim.sim.renderer import PygameRenderer
    from tenure.policy import TenurePolicy

    ck = torch.load(checkpoint, map_location="cpu")
    mode = ck.get("retention_mode", "multiplicative")
    policy = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=mode,
                          task_dim=ck.get("task_dim", 7),
                          retention_head=ck.get("retention_head", "regression"))
    policy.load_state_dict(ck["state_dict"])
    policy.eval()

    env = OverrideGraphEnv(opponent=opponent, contested=contested, seed=seed)
    renderer = PygameRenderer("rgb_array", size=size)
    seconds = seconds or MATCH_DURATION
    ticks_per_decision = int(round(DECISION_INTERVAL / DT))
    total_ticks = int(round(seconds / DT))
    stride = max(1, total_ticks // target_frames)

    probes: dict[int, _RobotProbe] = {r.robot_id: _RobotProbe(r.robot_id) for r in env.field.robots}
    rows = []
    frames, tick = [], 0
    while tick < total_ticks and not env.field.done:
        obs = env.observation()
        with torch.no_grad():
            action = policy.act(obs, deterministic=deterministic)["action"][0].tolist()
        env.last_actions = action[: env.max_robots]
        reds = {i: env.opponent.action(env, i) for i in (2, 3)}
        for _ in range(ticks_per_decision):
            if tick >= total_ticks or env.field.done:
                break
            commands = {0: env._ally_command(0, action[0]), 1: env._ally_command(1, action[1])}
            commands.update({i: env.controller.command(i, a) for i, a in reds.items()})

            pre = {r.robot_id: (r.x, r.y, r.collisions) for r in env.field.robots}
            # target of each BLUE robot this decision (for docked-vs-pinned classification)
            btgt = {}
            for rid in (0, 1):
                t = env._ally_target(rid, action[rid])
                btgt[rid] = (t.x, t.y) if t is not None else None
            env.field.physics_tick(commands, DT)
            env._ally_interact(0, action[0])
            env._ally_interact(1, action[1])
            for i, a in reds.items():
                env.controller.interact(i, a)

            stuck_now = []
            for r in env.field.robots:
                px, py, pcol = pre[r.robot_id]
                dpos = ((r.x - px) ** 2 + (r.y - py) ** 2) ** 0.5
                cmd = commands.get(r.robot_id, (0.0, 0.0, 0.0))
                tgt = btgt.get(r.robot_id)
                dtt = (((r.x - tgt[0]) ** 2 + (r.y - tgt[1]) ** 2) ** 0.5) if tgt else None
                is_stuck = probes[r.robot_id].update(tick, abs(cmd[0]), dpos, r.x, r.y, dtt)
                if is_stuck:
                    stuck_now.append(f"PINNED r{r.robot_id} @({r.x:.0f},{r.y:.0f})")
                if csv_path:
                    rows.append((tick, r.robot_id, round(r.x, 2), round(r.y, 2), round(r.heading, 3),
                                 round(cmd[0], 3), round(cmd[1], 3), round(cmd[2], 3),
                                 round(dpos, 3), r.collisions - pcol,
                                 round(dtt, 1) if dtt is not None else ""))
            if tick % stride == 0:
                frames.append(_overlay(renderer.draw(env.field), stuck_now))
            tick += 1
        env._update_change_times()

    if frames:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                       loop=0, disposal=2)   # disposal=2 -> clean per-frame replace (no ghosting)
    renderer.close()
    if csv_path and rows:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick", "robot", "x", "y", "heading", "fwd_cmd", "lat_cmd", "yaw_cmd",
                        "dpos", "d_collisions", "dist_to_target"])
            w.writerows(rows)

    b, r = env.field.score(env.ally), env.field.score(env.foe)
    return {"probes": probes, "frames": len(frames), "ticks": tick, "mode": mode,
            "blue": b, "red": r, "reversals": len(env.field.reversal_events), "out": out, "csv": csv_path}


def _report(res: dict) -> None:
    dt = 0.05
    print(f"\nwrote {res['out']}  ({res['frames']} frames)  policy={res['mode']}  "
          f"blue={res['blue']} red={res['red']}  reversals={res['reversals']}  ticks={res['ticks']}")
    if res.get("csv"):
        print(f"trace: {res['csv']}")
    print("\nSTUCK PROBE  (commanded to drive but net displacement < "
          f"{MOVE_EPS}in). DOCKED = parked at own target goal (<= {DOCK_DIST:.0f}in, benign scoring); "
          f"PINNED = stuck far from target (real nav stall). event = >= {STUCK_STREAK} pinned ticks "
          f"/ {STUCK_STREAK*dt:.1f}s in a row. (dist-to-target tracked for BLUE only.)")
    print(f"{'robot':>5} {'drive-ticks':>11} {'moved-ok':>8} {'docked':>7} {'pinned':>7} "
          f"{'pinned%':>8} {'max-pin':>8} {'events':>6}   worst-pin-at")
    worst = 0
    for rid in sorted(res["probes"]):
        p = res["probes"][rid]
        frac = 100.0 * p.pinned / max(1, p.commanded)
        at = f"tick {p.max_streak_at[0]} ({p.max_streak_at[1]:.0f},{p.max_streak_at[2]:.0f})" if p.max_streak_at else "-"
        team = "blue" if rid in (0, 1) else "red"
        print(f"{rid:>3}{team:>3} {p.commanded:>11} {p.moved_ok:>8} {p.docked:>7} {p.pinned:>7} "
              f"{frac:>7.1f}% {p.max_streak:>6}t {p.events:>6}   {at}")
        worst = max(worst, p.max_streak)
    print()
    blue_worst = max((res["probes"][r].max_streak for r in (0, 1) if r in res["probes"]), default=0)
    if blue_worst < STUCK_STREAK:
        print(f"VERDICT (blue/TENURE): no genuine navigation pin >= {STUCK_STREAK*dt:.1f}s. "
              f"Longest real pin was {blue_worst} tick(s) ({blue_worst*dt:.2f}s); the rest of the "
              "commanded-but-not-moving time is legitimate docking at target goals. Navigation is clean.")
    else:
        print(f"VERDICT (blue/TENURE): a genuine navigation pin FAR from target lasted "
              f"{blue_worst} ticks / {blue_worst*dt:.2f}s — the policy's robot could not reach where "
              "it was steering. This is a real offsim navigation bug (see flagged frames/CSV), and it "
              "corrupts any offsim-based evaluation of the policy.")


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy a TENURE checkpoint on real Override with stuck-detection")
    p.add_argument("checkpoint")
    p.add_argument("--out", default="recordings/override_probe.gif")
    p.add_argument("--csv", default="recordings/override_probe.csv")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--opponent", default="mixed", choices=["greedy", "toggle", "mixed", "descore"])
    p.add_argument("--no-contested", dest="contested", action="store_false")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--frames", dest="target_frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--size", type=int, default=560)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--no-csv", dest="want_csv", action="store_false")
    args = p.parse_args()
    res = probe(args.checkpoint, args.out, args.seed, args.opponent, args.contested,
                args.seconds, args.target_frames, args.fps, args.size, args.deterministic,
                args.csv if args.want_csv else None)
    _report(res)


if __name__ == "__main__":
    main()
