"""Record a TENURE policy playing the REAL Override game (pins/cups/goals/robots) as a GIF -- the actual
sim, not the abstract canonical circles. Drives ``OverrideField`` through ``OverrideGraphEnv`` (the
Section 3.7 base interface) with a policy, rendering the Pygame field each tick.

Needs a BASE-interface checkpoint (task_dim=7, NON-symmetric). The symmetric toggle/(b)/dynexp policies
(task_dim=12) do NOT fit this interface. Omit --checkpoint for an UNTRAINED pipeline check.

    python -m offsim.record_graph --checkpoint tenure/checkpoints/e9_vanilla.pt \
        --out recordings/override_sim/tenure_override.gif --seed 3
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def record(out: str, checkpoint: str | None, seed: int = 3, chassis: str = "tank",
           opponent: str = "mixed", seconds: float | None = None, target_frames: int = 200,
           fps: int = 20, size: int = 560) -> None:
    import torch
    from PIL import Image
    from offsim.sim.graph_env import OverrideGraphEnv
    from offsim.sim.config import DT, DECISION_INTERVAL, MATCH_DURATION
    from offsim.sim.renderer import PygameRenderer
    from tenure.policy import TenurePolicy

    env = OverrideGraphEnv(chassis=chassis, opponent=opponent, contested=True, seed=seed)
    td = env.observation()["task_feat"].shape[-1]
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu")
        if ck.get("symmetric") or ck.get("task_dim", 7) != td:
            raise SystemExit(f"checkpoint task_dim={ck.get('task_dim')} symmetric={ck.get('symmetric')} does "
                             f"NOT fit the Override base interface (task_dim={td}, non-symmetric).")
        pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                           task_dim=ck.get("task_dim", td), retention_head=ck.get("retention_head", "regression"),
                           symmetric=False)
        pol.load_state_dict(ck["state_dict"]); tag = "trained"
    else:
        pol = TenurePolicy(d_model=128, retention_mode="multiplicative", task_dim=td, symmetric=False)
        tag = "UNTRAINED-pipeline-check"
    pol.eval()
    renderer = PygameRenderer("rgb_array", size=size)

    seconds = seconds or MATCH_DURATION
    ticks_per_dec = int(round(DECISION_INTERVAL / DT))
    total_ticks = int(round(seconds / DT))
    stride = max(1, total_ticks // target_frames)
    frames, tick = [], 0
    obs = env.observation()
    while tick < total_ticks and not env.field.done:
        with torch.no_grad():
            action = [int(a) for a in pol.act(obs, deterministic=True)["action"][0].tolist()]
        env.last_actions = action[: env.max_robots]
        reds = {i: env.opponent.action(env, i) for i in (2, 3)}
        for _ in range(ticks_per_dec):
            if tick >= total_ticks or env.field.done:
                break
            commands = {0: env._ally_command(0, action[0]), 1: env._ally_command(1, action[1])}
            commands.update({i: env.controller.command(i, a) for i, a in reds.items()})
            env.field.physics_tick(commands, DT)
            env._ally_interact(0, action[0]); env._ally_interact(1, action[1])
            for i, a in reds.items():
                env.controller.interact(i, a)
            if tick % stride == 0:
                frames.append(Image.fromarray(renderer.draw(env.field)))
            tick += 1
        env._update_change_times()
        obs = env.observation()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    renderer.close()
    b, r = env.field.score(env.ally), env.field.score(env.foe)
    print(f"wrote {out}  ({len(frames)} frames, {tag})  final blue={b} red={r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Record a TENURE policy on the real Override game")
    p.add_argument("--checkpoint", default=None, help="BASE-interface TENURE .pt (task_dim=7); omit for pipeline check")
    p.add_argument("--out", default="recordings/override_sim/tenure_override.gif")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--chassis", default="tank")
    p.add_argument("--opponent", default="mixed")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--frames", type=int, default=200)
    args = p.parse_args()
    record(args.out, args.checkpoint, args.seed, args.chassis, args.opponent, args.seconds, args.frames)


if __name__ == "__main__":
    main()
