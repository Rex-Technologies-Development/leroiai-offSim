"""Run a single sweep cell — one ``(method, alpha, seed)`` — and persist it.

A run directory contains everything needed to reproduce and aggregate the cell:
``config.json`` (resolved), ``meta.json`` (git SHA, timestamp, rules-compliance),
``metrics.json`` (training series + final evaluation), and ``checkpoint.pt`` for
learned methods. Methods:

- ``tenure`` / ``tenure_off`` / ``tenure_feature`` — the three retention-mode variants
  (the causal ablation), trained then evaluated.
- ``greedy`` / ``defensive`` / ``cbba`` — evaluated directly.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from contested.config import CanonicalConfig, load_config

METHODS = ("tenure", "tenure_off", "tenure_feature", "greedy", "defensive", "cbba")
_TENURE_MODE = {"tenure": "multiplicative", "tenure_off": "off", "tenure_feature": "feature"}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _cfg_dict(cfg: CanonicalConfig) -> dict:
    d = asdict(cfg)
    d["tau_rev"] = None if not math.isfinite(cfg.tau_rev) else cfg.tau_rev  # inf is not valid JSON
    d["n_decisions"] = cfg.n_decisions
    return d


def run_cell(method: str, alpha: float, seed: int, out_dir: str | Path, *,
             updates: int = 100, batch_size: int = 256, eval_batch: int = 256,
             eval_seeds: int = 3, cfg_overrides: Optional[dict] = None,
             rules_compliant: bool = True, device=None) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(overrides={"alpha": alpha, "seed": seed, **(cfg_overrides or {})})

    series: list[dict] = []
    if method in _TENURE_MODE:
        from tenure.policy import TenurePolicy
        from tenure.ppo import PPOTrainer
        policy = TenurePolicy(retention_mode=_TENURE_MODE[method])
        trainer = PPOTrainer(cfg, policy, batch_size=batch_size, device=device)
        for i, m in enumerate(trainer.train(updates, seed=seed)):
            series.append({"update": i, **m})
        dev = trainer.device
        final = _evaluate_tenure(policy, cfg, eval_batch, eval_seeds, dev)
        torch.save(policy.state_dict(), out / "checkpoint.pt")
    else:
        from baselines import REGISTRY
        from baselines.evaluate import evaluate_policy
        final = evaluate_policy(REGISTRY[method]().act, cfg, batch_size=eval_batch,
                                n_seeds=eval_seeds, device=device)

    meta = {"method": method, "alpha": alpha, "seed": seed, "git_sha": git_sha(),
            "timestamp": time.time(), "rules_compliant": rules_compliant,
            "updates": updates, "batch_size": batch_size}
    (out / "config.json").write_text(json.dumps(_cfg_dict(cfg), indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "metrics.json").write_text(json.dumps({"final": final, "series": series}, indent=2))
    return {"method": method, "alpha": alpha, "seed": seed, **final}


def _evaluate_tenure(policy, cfg, eval_batch, eval_seeds, device) -> dict:
    from baselines.evaluate import evaluate_policy, tenure_policy_fn
    policy.eval()
    with torch.no_grad():
        return evaluate_policy(tenure_policy_fn(policy), cfg, batch_size=eval_batch,
                               n_seeds=eval_seeds, device=device)
