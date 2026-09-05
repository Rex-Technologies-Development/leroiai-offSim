"""What does TENURE actually DO on Override? Tally its per-decision action types and the
blue-score accrual curve to explain the flat ~107 cap.

    python -m scratchpad.override_diag --checkpoint scratchpad/e9_snap_conv.pt --seed 3
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="scratchpad/e9_snap_conv.pt")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    import torch
    from collections import Counter
    from offsim.sim.graph_env import OverrideGraphEnv
    from tenure.policy import TenurePolicy
    env = OverrideGraphEnv(opponent="mixed", contested=True, seed=a.seed)
    td = env.observation()["task_feat"].shape[-1]
    ck = torch.load(a.checkpoint, map_location="cpu")
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", td), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"]); pol.eval()
    T = env.max_tasks
    def classify(rid, act):
        if act >= 2 * T: return "IDLE"
        site = act if act < T else act - T
        if site >= env.n_sites: return "OOB"
        kind = env._sites[site][0]
        verb = "ACQ" if act < T else "DEF"
        return f"{verb}-{'TOG' if kind=='toggle' else 'GOAL'}"
    hist = Counter(); curve = []
    obs = env.observation(); dec = 0
    while not env.field.done:
        with torch.no_grad():
            act = [int(x) for x in pol.act(obs, deterministic=True)["action"][0].tolist()]
        for rid in (0, 1):
            hist[classify(rid, act[rid])] += 1
        obs, _r, done, info = env.step(act)
        dec += 1
        if dec % 6 == 0 or done:
            curve.append((dec, info["blue_score"], info["red_score"]))
        if done: break
    # inventory + held state at end
    inv = env.field.match_loads[env.ally]
    held = sum(1 for r in (env.field.robots[0], env.field.robots[1]) if r.held_pin is not None or r.held_cup is not None)
    print(f"seed {a.seed}  decisions={dec}")
    print("action histogram (both ally robots):", dict(hist))
    print("blue-score curve (dec, blue, red):", curve)
    print(f"end match_loads[blue] pin={inv['pin']} cup={inv['cup']}   ally robots still holding: {held}/2")


if __name__ == "__main__":
    main()
