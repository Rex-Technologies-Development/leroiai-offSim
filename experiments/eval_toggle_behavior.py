"""Behavioural eval of a trained toggle policy: does it hold toggles AND cash them in on goals?

J_H alone hides HOW a policy scores. This reports, for the checkpoint vs greedy/defensive on the
checkpoint's own regime:
  * J_H (multiplied differential objective),
  * toggles held by blue vs stolen by red (defence),
  * goals blue holds UNDER a blue-owned toggle (the multiplier actually realised) vs total goals held
    (coverage). A good retention policy holds its toggles AND the goals they triple -- not one or the
    other (the first learned policy camped toggles and neglected goals; see M8 notes).

    python -m experiments.eval_toggle_behavior tenure/checkpoints/toggle_aware.pt
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
def _rollout(cfg, actfn, seed, batch, dev):
    core = CanonicalCore(cfg, batch_size=batch, device=dev)
    core.reset(seed)
    for _ in range(cfg.n_decisions):
        core.step(actfn(core))
    s = core.state
    T = cfg.n_tasks
    istog = s.task_is_toggle[:, :T]
    bc, rc = s.task_c[:, :T], s.task_c_red[:, :T]
    blue_tog = (bc & istog).sum(-1).float()
    red_tog = (rc & istog).sum(-1).float()
    goals_held = (bc & ~istog).sum(-1).float()
    gov_blue = bc.gather(1, s.task_toggle_idx[:, :T])                    # blue owns each task's toggle
    goals_cashed = (bc & ~istog & gov_blue).sum(-1).float()             # goals under a blue toggle (x M)
    jh = core.telemetry.summary(s)["J_H_diff"].mean().item()
    return (jh, blue_tog.mean().item(), red_tog.mean().item(),
            goals_held.mean().item(), goals_cashed.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioural eval of a toggle policy")
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", dest="batch", type=int, default=96)
    args = ap.parse_args()
    dev = "cpu"

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(overrides=dict(ck.get("overrides") or {}))
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode", "multiplicative"),
                       task_dim=ck.get("task_dim", 7), retention_head=ck.get("retention_head", "regression"),
                       symmetric=ck.get("symmetric", False))
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    n_tog = cfg.toggle_regions

    def tenure_act(core):
        return pol.act(build_observation(core.state, cfg), deterministic=True)["action"]

    rows = [("AWARE-TENURE", tenure_act)]
    for b in ("greedy", "defensive"):
        pol_b = REGISTRY[b]()
        rows.append((b, (lambda c, p=pol_b: p.act(c.state, cfg))))

    print(f"regime: n_tasks={cfg.n_tasks} toggles={n_tog} M={cfg.toggle_multiplier} "
          f"red={cfg.adversary_population} horizon={cfg.horizon_T}\n")
    print(f"{'policy':>14} | {'J_H':>7} | blue_tog red_tog | goals_held goals_cashed")
    print("-" * 68)
    for name, fn in rows:
        rs = [_rollout(cfg, fn, s, args.batch, dev) for s in range(args.seeds)]
        m = [statistics.fmean(r[i] for r in rs) for i in range(5)]
        print(f"{name:>14} | {m[0]:+7.3f} | {m[1]:7.2f}/{n_tog} {m[2]:4.2f}/{n_tog} | "
              f"{m[3]:9.2f}  {m[4]:9.2f}")
    print("\nRead: a retention policy should hold its toggles (high blue_tog, low red_tog) AND cash them "
          "in\n(goals_cashed close to goals_held) -- defending the multiplier is only worth it if you "
          "own the goals.")


if __name__ == "__main__":
    main()
