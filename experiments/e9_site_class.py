"""E9: predicted retention R_hat by Override site class (alliance / neutral / toggle).

Runs a TENURE policy on the REAL Override game via ``OverrideGraphEnv`` and reports the
head's mean R_hat per site class over one match — WITHOUT telling the head which site is
which. A trained TENURE should give R_hat ~ 1 on alliance goals (protected, alpha=0),
intermediate on neutral goals, and low on toggle-dependent value (alpha >= 1); a random
policy gives ~uniform 0.5. This is the highest-value figure in the Override half of the
paper, so we exercise the plumbing end-to-end BEFORE the real checkpoint lands.

CPU-only (policy loaded with ``map_location='cpu'``), safe alongside a GPU training run.

    python -m experiments.e9_site_class --random            # plumbing smoke (numbers meaningless)
    python -m experiments.e9_site_class runs/tenure_off.pt   # preview on a trained checkpoint
"""
from __future__ import annotations

import argparse

import torch

from offsim.sim.graph_env import OverrideGraphEnv, retention_by_site_class
from tenure.policy import TenurePolicy


def main() -> None:
    ap = argparse.ArgumentParser(description="E9: predicted retention by Override site class")
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--random", action="store_true", help="random-weights policy (plumbing smoke)")
    ap.add_argument("--opponent", default="mixed", choices=["greedy", "toggle", "mixed", "descore"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env = OverrideGraphEnv(opponent=args.opponent, contested=True, seed=args.seed)
    obs_td = int(env.observation()["task_feat"].shape[-1])   # dim the adapter actually emits

    if args.random or args.checkpoint is None:
        policy = TenurePolicy(d_model=128, retention_mode="multiplicative", task_dim=obs_td)
        mode, td = "random", obs_td
    else:
        ck = torch.load(args.checkpoint, map_location="cpu")
        td = int(ck.get("task_dim", 7))
        if td != obs_td:
            print(f"ADAPTER GAP: checkpoint task_dim={td} but OverrideGraphEnv emits task_feat "
                  f"dim={obs_td}. Override does not expose the protected flag yet — its alliance "
                  f"goals ARE the protected sites, so the fix is to have the adapter mark them "
                  f"(task_dim -> {obs_td + 1}). Use a task_dim={obs_td} checkpoint for now.")
            return
        policy = TenurePolicy(d_model=ck.get("d_model", 128),
                              retention_mode=ck.get("retention_mode", "multiplicative"), task_dim=td,
                              retention_head=ck.get("retention_head", "regression"))
        policy.load_state_dict(ck["state_dict"])
        mode = ck.get("retention_mode", "?")
    policy.eval()

    result = retention_by_site_class(policy, env)
    print(f"E9 retention by site class  (policy={mode}, task_dim={td}, opponent={args.opponent})")
    print(f"  {'site class':>16} | {'mean R_hat':>10}")
    print("  " + "-" * 31)
    for cls, v in result.items():
        print(f"  {cls:>16} | {v:10.3f}")
    print("\n  expect (trained): alliance_goal ~1 > neutral_goal > toggle-value ; random ~0.5")


if __name__ == "__main__":
    main()
