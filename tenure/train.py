"""Train the TENURE policy on the canonical contested environment.

Example
-------
    python -m tenure.train --updates 200 --alpha 1.0 --retention-mode multiplicative
    python -m tenure.train --updates 200 --retention-mode off      # ablation baseline
"""
from __future__ import annotations

import argparse

import torch

from contested.config import load_config
from contested.observation import observation_spec
from .policy import TenurePolicy
from .ppo import PPOConfig, PPOTrainer
from .progress import ProgressBar, tail_mean


def main() -> None:
    p = argparse.ArgumentParser(description="Train TENURE on the canonical env")
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-seed", dest="init_seed", type=int, default=None,
                   help="policy-init seed, DECOUPLED from --seed (which seeds the env). For path-(b) "
                        "restarts: keep --seed (env conditions preserved for the paired comparison) but "
                        "pass a new --init-seed to escape a collapsed optimisation basin. Defaults to --seed.")
    p.add_argument("--converge-patience", dest="converge_patience", type=int, default=0,
                   help="train-to-CONVERGENCE (per-seed criterion, not fixed budget): stop once tail-mean "
                        "J_H sets no new best (by --converge-eps) for this many updates after --converge-min. "
                        "0 = off (fixed budget). --updates is the hard cap.")
    p.add_argument("--converge-min", dest="converge_min", type=int, default=200,
                   help="minimum updates before the convergence criterion may fire")
    p.add_argument("--converge-eps", dest="converge_eps", type=float, default=0.005,
                   help="tail-mean J_H gain below this counts as no-improvement (plateau)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", dest="d_model", type=int, default=128)
    p.add_argument("--retention-mode", dest="retention_mode",
                   choices=["multiplicative", "feature", "off"], default="multiplicative")
    p.add_argument("--retention-head", dest="retention_head",
                   choices=["regression", "classification"], default="regression",
                   help="Table III ablation: retention head as scalar regressor (default) or a "
                        "categorical classifier over retention bins (R_hat = its expected value)")
    p.add_argument("--expose-protected", dest="expose_protected", action="store_true",
                   help="killer-control 4th arm: put the protected flag in the obs "
                        "(pair with --retention-mode off = weight-only WITH protection)")
    # M8 toggle/multiplier env (Override's real retention structure)
    p.add_argument("--symmetric", action="store_true", help="two-team symmetric env (Phase 2/3)")
    p.add_argument("--stack-cap", dest="stack_cap", type=int, default=None,
                   help="stacking game: max stack height (>1 enables build/dismantle by height)")
    p.add_argument("--stack-value-power", dest="stack_value_power", type=float, default=None,
                   help="held value of a height-h stack = w*h**power (1=linear, >1=convex/retention pays)")
    p.add_argument("--toggle-regions", dest="toggle_regions", type=int, default=None,
                   help="M8: number of toggle regions (first task of each region is its toggle)")
    p.add_argument("--toggle-multiplier", dest="toggle_multiplier", type=float, default=None,
                   help="M8: a region's goals score this x while you hold its toggle")
    p.add_argument("--expose-toggle", dest="expose_toggle", action="store_true",
                   help="M8: show toggle leverage in the obs + gate effective value (aware); "
                        "omit for the retention-BLIND baseline")
    p.add_argument("--dynamic-exposure", dest="dynamic_exposure", action="store_true",
                   help="toggle value = CURRENT held-cluster value (continuous spectrum) + varied cluster "
                        "sizes; the env where representation should resolve out of noise + head's fair test")
    p.add_argument("--no-revisit-r", dest="no_revisit_r", action="store_true",
                   help="marginal R (hold-until-first-reversal); fixes the churn double-count")
    p.add_argument("--ablate-adversary-nodes", dest="ablate_adversary_nodes", action="store_true",
                   help="Table III ablation: hide all adversary nodes/edges from the policy's obs")
    p.add_argument("--lam-r", dest="lam_r", type=float, default=0.5)
    # environment overrides
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--contest-mode", dest="contest_mode", default=None)
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=None)
    p.add_argument("--n-robots", dest="n_robots", type=int, default=None)
    p.add_argument("--n-adversaries", dest="n_adversaries", type=int, default=None)
    p.add_argument("--horizon", dest="horizon_T", type=float, default=None)
    p.add_argument("--tau-com", dest="tau_com", type=float, default=None)
    p.add_argument("--reacquire-cost", dest="reacquire_cost", type=float, default=None,
                   help="rho: a re-completion needs tau_com*rho service (retention pays as rho rises)")
    p.add_argument("--save", default=None, help="path to write the trained checkpoint (.pt)")
    p.add_argument("--load", default=None,
                   help="resume: load policy weights from a .pt checkpoint before training "
                        "(architecture must match --d-model/--retention-mode)")
    p.add_argument("--log-every", dest="log_every", type=int, default=5,
                   help="progress-bar cadence in updates (the bar is always on)")
    p.add_argument("--save-every", dest="save_every", type=int, default=0,
                   help="also write --save every N updates (crash insurance for long runs; 0=end only)")
    # protected_fraction: fraction of irreversible tasks (Override alliance-goal analogue).
    # RETENTION IS DECORATIVE AT 0.0 — train the retention comparison near Override's ~0.15.
    p.add_argument("--protected-fraction", dest="protected_fraction", type=float, default=None)
    # single adversary archetype -> adversary_population=[that] (e.g. camper for max retention signal)
    p.add_argument("--adversary", nargs="+", default=None,
                   help="one or more archetypes -> adversary_population (e.g. builder toggle_raider)")
    args = p.parse_args()
    torch.manual_seed(args.init_seed if args.init_seed is not None else args.seed)  # policy init (decoupled from env --seed for (b) restarts)

    overrides = {k: getattr(args, k) for k in
                 ("alpha", "beta", "contest_mode", "n_tasks", "n_robots", "n_adversaries",
                  "horizon_T", "tau_com", "protected_fraction", "reacquire_cost",
                  "toggle_regions", "toggle_multiplier", "stack_cap", "stack_value_power")
                 if getattr(args, k) is not None}
    if args.adversary is not None:
        overrides["adversary_population"] = list(args.adversary)
    if args.expose_protected:
        overrides["expose_protected"] = True
    if args.symmetric:
        overrides["symmetric"] = True
    if args.expose_toggle:
        overrides["expose_toggle"] = True
    if args.dynamic_exposure:
        overrides["dynamic_exposure"] = True
    if args.no_revisit_r:
        overrides["retention_no_revisit"] = True
    if args.ablate_adversary_nodes:
        overrides["ablate_adversary_nodes"] = True
    cfg = load_config(overrides=overrides or None)
    task_dim = observation_spec(cfg)["task_feat"][1]
    policy = TenurePolicy(d_model=args.d_model, retention_mode=args.retention_mode, task_dim=task_dim,
                          retention_head=args.retention_head, symmetric=cfg.symmetric)
    if args.load:
        ck = torch.load(args.load, map_location="cpu")
        policy.load_state_dict(ck["state_dict"])
        print(f"resumed weights from {args.load} (mode={ck.get('retention_mode')}); "
              f"optimizer restarts", flush=True)
    trainer = PPOTrainer(cfg, policy, batch_size=args.batch_size,
                         ppo=PPOConfig(lr=args.lr, lam_r=args.lam_r))
    print(f"device={trainer.device} params={sum(p.numel() for p in policy.parameters())} "
          f"mode={args.retention_mode} alpha={cfg.alpha} D={cfg.n_decisions}", flush=True)
    def _ckpt() -> None:
        torch.save({"state_dict": policy.state_dict(), "d_model": args.d_model,
                    "retention_mode": args.retention_mode, "task_dim": task_dim,
                    "retention_head": args.retention_head, "symmetric": cfg.symmetric,
                    "overrides": overrides}, args.save)

    pbar = ProgressBar(args.updates, every=args.log_every)
    jh_hist, mae_hist = [], []
    best = 0.0
    tail_hist, i = [], 0
    for i, m in enumerate(trainer.train(args.updates, seed=args.seed), start=1):
        jh_hist.append(m["rollout_reward"]); mae_hist.append(m["retention_mae"])
        best = max(best, m["rollout_reward"])
        tm = tail_mean(jh_hist); tail_hist.append(tm)
        pbar.line(i, {"J_H": tm, "MAE": tail_mean(mae_hist), "ent": m["entropy"]})
        if args.save and args.save_every and i % args.save_every == 0 and i < args.updates:
            _ckpt()
            print(f"  [checkpoint @ {i}/{args.updates}] -> {args.save}", flush=True)
        # train-to-convergence (per-seed): stop once the tail-mean J_H gain over the last `patience`
        # updates drops below eps (diminishing returns) -- so a good run stops at its own plateau and a
        # collapsed run stops early at a low value, rather than every run hitting the fixed cap.
        if args.converge_patience and i >= args.converge_min and len(tail_hist) > args.converge_patience:
            gain = tm - tail_hist[-args.converge_patience - 1]
            if gain < args.converge_eps:
                print(f"  [converged @ {i}/{args.updates}: tail J_H {tm:.4f}, "
                      f"{args.converge_patience}-update gain {gain:+.4f} < {args.converge_eps}]", flush=True)
                break
    print(f"done: best J_H(train)={best:.4f} (stopped @ {i} updates)", flush=True)
    if args.save:
        _ckpt()
        print(f"saved checkpoint -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
