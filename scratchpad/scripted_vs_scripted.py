"""Fairness probe: drive BOTH alliances (blue 0,1 and red 2,3) with the IDENTICAL scripted
pipeline (ScriptedOpponent -> ObjectiveController) and compare scores. If blue ~= red, the blue
side of OverrideGraphEnv is fair and TENURE's loss is a genuine weak-transfer result. If scripted
blue also loses ~110 vs ~280, the blue side is structurally handicapped (env asymmetry to fix).

    python -m scratchpad.scripted_vs_scripted --seeds 6
"""
from __future__ import annotations
import argparse, os, statistics
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--opponent", default="mixed")
    a = ap.parse_args()
    from offsim.sim.graph_env import OverrideGraphEnv
    from offsim.sim.config import DT, DECISION_INTERVAL
    ticks = int(round(DECISION_INTERVAL / DT))
    blues, reds = [], []
    for s in range(a.seeds):
        env = OverrideGraphEnv(opponent=a.opponent, contested=True, seed=s)
        while not env.field.done:
            acts = {i: env.opponent.action(env, i) for i in (0, 1, 2, 3)}   # SAME scripts, all 4 robots
            for _ in range(ticks):
                if env.field.done:
                    break
                commands = {i: env.controller.command(i, act) for i, act in acts.items()}
                env.field.physics_tick(commands, DT)
                for i, act in acts.items():
                    env.controller.interact(i, act)
        blues.append(env.field.score(env.ally)); reds.append(env.field.score(env.foe))
    bwins = sum(1 for x, y in zip(blues, reds) if x > y)
    print(f"SCRIPTED (both sides identical)  vs red={a.opponent}  ({a.seeds} matches)")
    print(f"  blue {statistics.fmean(blues):6.1f} +/- {statistics.pstdev(blues):5.1f}   "
          f"red {statistics.fmean(reds):6.1f} +/- {statistics.pstdev(reds):5.1f}   blue wins {bwins}/{a.seeds}")
    print(f"  per-seed blue: {[int(x) for x in blues]}")
    print(f"  per-seed red : {[int(x) for x in reds]}")


if __name__ == "__main__":
    main()
