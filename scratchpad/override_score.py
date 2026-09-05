"""Quick no-render Override score sweep: mean blue(TENURE)/red(scripted) over N seeds.
Contextualizes whether canonical->Override transfer buys any competitiveness vs an untrained floor.

    python -m scratchpad.override_score --checkpoint scratchpad/e9_snap_conv.pt --seeds 8
    python -m scratchpad.override_score --seeds 8            # untrained floor
"""
from __future__ import annotations
import argparse, os, statistics
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def run(checkpoint, n_seeds, opponent):
    import torch
    from offsim.sim.graph_env import OverrideGraphEnv
    from tenure.policy import TenurePolicy
    blues, reds = [], []
    for seed in range(n_seeds):
        env = OverrideGraphEnv(opponent=opponent, contested=True, seed=seed)
        td = env.observation()["task_feat"].shape[-1]
        if checkpoint:
            ck = torch.load(checkpoint, map_location="cpu")
            pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                               task_dim=ck.get("task_dim", td), retention_head=ck.get("retention_head", "regression"),
                               symmetric=ck.get("symmetric", False))
            pol.load_state_dict(ck["state_dict"])
        else:
            pol = TenurePolicy(d_model=128, retention_mode="multiplicative", task_dim=td, symmetric=False)
        pol.eval()
        obs = env.observation()
        while not env.field.done:
            with torch.no_grad():
                a = pol.act(obs, deterministic=True)["action"][0].tolist()
            obs, _r, done, info = env.step(a)
            if done:
                break
        blues.append(info["blue_score"]); reds.append(info["red_score"])
    return blues, reds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--opponent", default="mixed")
    a = ap.parse_args()
    b, r = run(a.checkpoint, a.seeds, a.opponent)
    tag = a.checkpoint or "UNTRAINED"
    wins = sum(1 for x, y in zip(b, r) if x > y)
    print(f"{tag}  vs red={a.opponent}  ({a.seeds} matches)")
    print(f"  blue {statistics.fmean(b):6.1f} +/- {statistics.pstdev(b):5.1f}   "
          f"red {statistics.fmean(r):6.1f} +/- {statistics.pstdev(r):5.1f}   "
          f"blue wins {wins}/{a.seeds}")
    print(f"  per-seed blue: {[int(x) for x in b]}")
    print(f"  per-seed red : {[int(x) for x in r]}")


if __name__ == "__main__":
    main()
