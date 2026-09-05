"""Deploy a canonical-trained TENURE policy on the REAL Override game and record a GIF.

This is the demonstration step: a policy trained on the abstract ``contested`` env is
dropped onto the actual ``OverrideField`` (goals, pins, cups, toggles) through
``OverrideGraphEnv`` (the graph observation adapter) and drives the blue alliance, with
a scripted opponent on red. The real Override renderer draws every physics tick, so you
see the trained policy *play the game* — not the abstract dots.

Runs entirely on CPU (policy loaded with ``map_location='cpu'``), so it does not touch
a GPU training run in progress.

    python -m offsim.deploy_render runs/tenure_demo.pt --out recordings/override_tenure.gif
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def deploy(checkpoint: str, out: str, seed: int = 7, opponent: str = "mixed",
           contested: bool = True, seconds: float | None = None, target_frames: int = 220,
           fps: int = 20, size: int = 560, deterministic: bool = True) -> None:
    import torch
    from PIL import Image

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

    frames, tick = [], 0
    while tick < total_ticks and not env.field.done:
        obs = env.observation()
        with torch.no_grad():
            action = policy.act(obs, deterministic=deterministic)["action"][0].tolist()
        env.last_actions = action[: env.max_robots]
        reds = {i: env.opponent.action(env, i) for i in (2, 3)}          # scripted red alliance
        for _ in range(ticks_per_decision):
            if tick >= total_ticks or env.field.done:
                break
            commands = {0: env._ally_command(0, action[0]), 1: env._ally_command(1, action[1])}
            commands.update({i: env.controller.command(i, a) for i, a in reds.items()})
            env.field.physics_tick(commands, DT)
            env._ally_interact(0, action[0])
            env._ally_interact(1, action[1])
            for i, a in reds.items():
                env.controller.interact(i, a)
            if tick % stride == 0:
                frames.append(Image.fromarray(renderer.draw(env.field)))
            tick += 1
        env._update_change_times()                                       # keep obs timers honest

    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    renderer.close()
    b, r = env.field.score(env.ally), env.field.score(env.foe)
    print(f"wrote {out}  ({len(frames)} frames @ {fps}fps)  policy={mode}  "
          f"blue={b} red={r}  reversals={len(env.field.reversal_events)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy a TENURE checkpoint on the real Override game")
    p.add_argument("checkpoint")
    p.add_argument("--out", default="recordings/override_tenure.gif")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--opponent", default="mixed", choices=["greedy", "toggle", "mixed", "descore"])
    p.add_argument("--no-contested", dest="contested", action="store_false",
                   help="disable the Channel-1 dwell reversal mechanic")
    p.add_argument("--seconds", type=float, default=None, help="default: full match")
    p.add_argument("--frames", dest="target_frames", type=int, default=220)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--size", type=int, default=560)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    deploy(args.checkpoint, args.out, args.seed, args.opponent, args.contested,
           args.seconds, args.target_frames, args.fps, args.size, args.deterministic)


if __name__ == "__main__":
    main()
