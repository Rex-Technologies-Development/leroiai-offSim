"""Record an Override match (the real game simulator) as an animated GIF.

Runs a deterministic scripted 2v2 match headlessly and captures the Pygame renderer
frames. Optionally enables the TENURE contested mechanic (Channel 1, dwell-based
Toggle reversal) so the recording shows territory flips.

Example
-------
    python -m offsim.record --out match.gif --seconds 40 --seed 7
    python -m offsim.record --out contested.gif --contested --opponent toggle
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def record(out: str, seed: int = 7, chassis: str = "tank", opponent: str = "mixed",
           seconds: float = 40.0, target_frames: int = 200, fps: int = 20,
           size: int = 560, contested: bool = False) -> None:
    from PIL import Image
    from offsim.sim.env import ObjectiveController, OverrideStrategyEnv
    from offsim.sim.config import Action, DT, DECISION_INTERVAL
    from offsim.sim.renderer import PygameRenderer

    env = OverrideStrategyEnv(chassis=chassis, render_mode=None, opponent=opponent)
    env.reset(seed=seed)
    field = env.field
    if contested:
        field.contested_enabled = True
        field.contested = {**field.contested, "enabled": True}
    controller = ObjectiveController(field)
    renderer = PygameRenderer("rgb_array", size=size)

    ticks_per_decision = int(round(DECISION_INTERVAL / DT))
    total_ticks = int(round(seconds / DT))
    stride = max(1, total_ticks // target_frames)

    frames, tick = [], 0
    while tick < total_ticks and not field.done:
        blue = env.autoplay_actions()
        actions = {0: Action(int(blue[0])), 1: Action(int(blue[1])),
                   2: env.opponent.action(env, 2), 3: env.opponent.action(env, 3)}
        for _ in range(ticks_per_decision):
            if tick >= total_ticks or field.done:
                break
            commands = {rid: controller.command(rid, act) for rid, act in actions.items()}
            field.physics_tick(commands, DT)
            for rid, act in actions.items():
                controller.interact(rid, act)
            if tick % stride == 0:
                frames.append(Image.fromarray(renderer.draw(field)))
            tick += 1

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    renderer.close()
    b, r = field.score(field.robots[0].alliance), field.score(field.robots[2].alliance)
    print(f"wrote {out}  ({len(frames)} frames @ {fps}fps)  final blue={b} red={r}"
          + (f"  reversals={len(field.reversal_events)}" if contested else ""))


def main() -> None:
    p = argparse.ArgumentParser(description="Record an Override match as a GIF")
    p.add_argument("--out", default="override_match.gif")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chassis", default="tank", choices=["tank", "mecanum"])
    p.add_argument("--opponent", default="mixed", choices=["greedy", "toggle", "mixed", "descore"])
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--size", type=int, default=560)
    p.add_argument("--contested", action="store_true", help="enable dwell-based Toggle reversal (Channel 1)")
    args = p.parse_args()
    record(args.out, args.seed, args.chassis, args.opponent, args.seconds,
           args.frames, args.fps, args.size, args.contested)


if __name__ == "__main__":
    main()
