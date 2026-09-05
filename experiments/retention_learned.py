"""Learned retention test — the HONEST killer control (needs training; uses CUDA).

Counterpart to ``experiments/retention_probe.py``. The probe measures the *ceiling*
by handing ``R`` as the protected-flag ORACLE — a field that is NOT in the
observation. This script instead trains real TENURE policies whose retention head
must ESTIMATE retention from geometry (``adv_feat`` + ``e_at`` threat/closing edges)
and observable dynamics (``eta/tau_rev``); the protected flag is never in the obs.

Comparison, all at Override's protected fraction:
- ``off``            — learned weight+geometry allocator (retention severed from the
                       decoder; ``eff = 1``). This is the plain-allocator control.
- ``feature``        — retention as an additive learned feature.
- ``multiplicative`` — TENURE's claim: predicted retention scales task value (Eq. 10).

Verdict logic (paired over seeds, same layouts):
- if ``multiplicative`` J_H > ``off`` J_H (CI excludes 0) AND ret_MAE < 0.15, the
  retention head earns its keep beyond a plain learned allocator;
- if they tie, the head is decoration at this regime — report that honestly.

This is DEFINITIVE where the scripted ceiling is only suggestive. Still preliminary
until >=5 seeds; treat single runs as promising, not proven.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import torch

from contested.config import load_config
from contested.observation import build_observation
from tenure.policy import TenurePolicy
from tenure.ppo import PPOConfig, PPOTrainer


def _ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


@torch.no_grad()
def eval_jh(trainer: PPOTrainer, cfg, seed: int) -> float:
    """Deterministic held-out held-value (J_H) of the trained policy on a fresh seed."""
    core = trainer.core
    core.reset(seed)
    total = torch.zeros(trainer.B, device=trainer.device, dtype=trainer.dtype)
    for _ in range(cfg.n_decisions):
        obs = build_observation(core.state, cfg)
        step = trainer.policy.act(obs, deterministic=True)
        reward, _done, _info = core.step(step["action"])
        total += reward.to(trainer.dtype)
    return total.mean().item()


def train_one(mode, cfg, updates, seed, batch, lr, lam_r, window, on_update=None):
    """Train one policy; return held-out J_H + final-window training ret_MAE."""
    policy = TenurePolicy(retention_mode=mode)
    tr = PPOTrainer(cfg, policy, batch_size=batch, ppo=PPOConfig(lr=lr, lam_r=lam_r))
    jh_train, mae = [], []
    for i, m in enumerate(tr.train(updates, seed=seed), start=1):
        jh_train.append(m["rollout_reward"])
        mae.append(m["retention_mae"])
        if on_update is not None:
            on_update(i, jh_train, mae)
    w = min(window, len(mae))
    maes = [x for x in mae[-w:] if not math.isnan(x)]
    return {
        "device": str(tr.device),
        "params": sum(p.numel() for p in policy.parameters()),
        "J_H_eval": eval_jh(tr, cfg, seed=100_000 + seed),      # deterministic, held-out layout
        "J_H_train": statistics.fmean(jh_train[-w:]),           # on-policy final window
        "ret_MAE": statistics.fmean(maes) if maes else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Learned retention killer control (CUDA)")
    p.add_argument("--modes", nargs="+", default=["off", "multiplicative"],
                   choices=["off", "feature", "multiplicative"])
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--batch-size", dest="batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam-r", dest="lam_r", type=float, default=0.5)
    # regime (defaults = the ceiling regime, calibrated to Override's protected fraction)
    p.add_argument("--protected-fraction", dest="pf", type=float, default=0.15)
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=16)
    p.add_argument("--n-robots", dest="n_robots", type=int, default=2)
    p.add_argument("--n-adversaries", dest="n_adversaries", type=int, default=6)
    p.add_argument("--tau-com", dest="tau_com", type=float, default=5.0)
    p.add_argument("--adversary", default="camper")
    p.add_argument("--log-every", dest="log_every", type=int, default=10,
                   help="print a progress bar line every N updates within each sub-run")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    overrides = dict(alpha=args.alpha, n_tasks=args.n_tasks, n_robots=args.n_robots,
                     n_adversaries=args.n_adversaries, tau_com=args.tau_com,
                     adversary_population=[args.adversary], horizon_T=args.horizon,
                     protected_fraction=args.pf)
    cfg = load_config(overrides=overrides)

    total_runs = args.seeds * len(args.modes)
    print("LEARNED retention test (head ESTIMATES R from geometry; protected flag NOT in obs).")
    print("regime:", overrides, "| updates:", args.updates, "seeds:", args.seeds, flush=True)
    print(f"{total_runs} sub-runs ({len(args.modes)} modes x {args.seeds} seeds) x {args.updates} "
          f"updates; progress every {args.log_every} updates:\n", flush=True)

    results = {m: [] for m in args.modes}
    bar_w = 22
    t0 = time.time()
    done_prior = {"n": 0}          # updates completed in already-finished sub-runs

    def make_cb(run_idx, mode, seed):
        run_start = time.time()

        def cb(i, jh_hist, mae_hist):
            if i % args.log_every and i != args.updates:
                return
            per = (time.time() - run_start) / i
            eta_run = per * (args.updates - i)
            eta_all = per * (total_runs * args.updates - (done_prior["n"] + i))
            filled = round(bar_w * i / args.updates)
            bar = "#" * filled + "-" * (bar_w - filled)
            jh = statistics.fmean(jh_hist[-10:])
            maes = [x for x in mae_hist[-10:] if not math.isnan(x)]
            mv = statistics.fmean(maes) if maes else float("nan")
            print(f"[run {run_idx}/{total_runs}] {mode:>14} s{seed} [{bar}] "
                  f"{i:3d}/{args.updates} {100 * i // args.updates:3d}%  J_H~{jh:.3f} MAE~{mv:.3f}  "
                  f"eta_run {_fmt(eta_run)} eta_all {_fmt(eta_all)}", flush=True)
        return cb

    run_idx = 0
    for s in range(args.seeds):
        for mode in args.modes:
            run_idx += 1
            r = train_one(mode, cfg, args.updates, s, args.batch, args.lr, args.lam_r,
                          args.window, on_update=make_cb(run_idx, mode, s))
            results[mode].append(r)
            done_prior["n"] += args.updates
            print(f">>> done {mode:>14} s{s} | J_H(eval)={r['J_H_eval']:.3f} "
                  f"J_H(train)={r['J_H_train']:.3f} ret_MAE={r['ret_MAE']:.3f}\n", flush=True)

    # ---- per-mode means -------------------------------------------------------
    print(f"\n{'mode':>14} | {'J_H(eval)':>16} {'ret_MAE':>8}   (mean +/- 95% CI over seeds)")
    print("-" * 58)
    for mode in args.modes:
        jh = [r["J_H_eval"] for r in results[mode]]
        mae = [r["ret_MAE"] for r in results[mode] if not math.isnan(r["ret_MAE"])]
        print(f"{mode:>14} | {statistics.fmean(jh):7.3f} +/- {_ci95(jh):.3f}   "
              f"{(statistics.fmean(mae) if mae else float('nan')):8.3f}")

    # ---- paired killer control: multiplicative - off --------------------------
    if "multiplicative" in results and "off" in results:
        gaps = [results["multiplicative"][s]["J_H_eval"] - results["off"][s]["J_H_eval"]
                for s in range(args.seeds)]
        g, ci = statistics.fmean(gaps), _ci95(gaps)
        sig = "MULT > OFF" if (g - ci) > 0 else "CI includes 0 -> head is decoration here"
        print(f"\nKILLER CONTROL  (multiplicative - off, paired over seeds):")
        print(f"  gap = {g:+.3f}  [{g - ci:+.3f}, {g + ci:+.3f}]   {sig}")
        print("  (compare to the scripted CEILING at this pf; learned should be <= ceiling)")

    print(f"\nelapsed {time.time() - t0:.0f}s on {results[args.modes[0]][0]['device']}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"regime": overrides, "args": vars(args), "results": results}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
