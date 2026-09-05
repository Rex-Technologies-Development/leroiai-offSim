"""View & compare TENURE runs in the sim: render each checkpoint + report its J_H.

Each checkpoint is rendered in ITS OWN training regime (saved inside the .pt by
``tenure.train``), so you can watch what each learned policy actually does and compare
their held-value side by side. Writes ``recordings/<checkpoint-stem>.{gif,png}``.

    python -m experiments.view_runs runs/tenure_demo.pt                 # one run
    python -m experiments.view_runs runs/*.pt --mode both              # every run, gif+filmstrip
    python -m experiments.view_runs runs/mult.pt runs/off.pt --eval-seeds 5
"""
from __future__ import annotations

import argparse
import os

import torch

from baselines.evaluate import evaluate_policy, tenure_policy_fn
from contested.config import load_config
from contested.core import default_device
from contested.demo import (make_tenure_policy, read_checkpoint_overrides,
                            run_filmstrip, run_gif)
from tenure.policy import TenurePolicy


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _load_model(ck_path: str, device):
    ck = torch.load(ck_path, map_location="cpu")
    m = TenurePolicy(d_model=ck.get("d_model", 128),
                     retention_mode=ck.get("retention_mode", "multiplicative"),
                     task_dim=ck.get("task_dim", 7),
                     retention_head=ck.get("retention_head", "regression"))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck.get("retention_mode", "?")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render + eval TENURE checkpoints for viewing")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--outdir", default="recordings")
    ap.add_argument("--mode", choices=["gif", "filmstrip", "both"], default="gif")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=5,
                    help="deterministic eval seeds for the J_H table (0 = skip eval)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = default_device()

    rows = []
    for ck_path in args.checkpoints:
        over = read_checkpoint_overrides(ck_path)
        cfg = load_config(overrides=over or None)
        policy_fn = make_tenure_policy(ck_path)
        stem = _stem(ck_path)
        outs = []
        if args.mode in ("gif", "both"):
            out = os.path.join(args.outdir, stem + ".gif")
            run_gif(cfg, args.seed, out, policy_fn); outs.append(out)
        if args.mode in ("filmstrip", "both"):
            out = os.path.join(args.outdir, stem + ".png")
            run_filmstrip(cfg, args.seed, out, 8, policy_fn); outs.append(out)
        rmode, jh, jhs = "?", None, None
        if args.eval_seeds > 0:
            model, rmode = _load_model(ck_path, device)
            res = evaluate_policy(tenure_policy_fn(model), cfg, n_seeds=args.eval_seeds, device=device)
            jh, jhs = res["J_H"], res["J_H_std"]
        rows.append((stem, rmode, jh, jhs, ", ".join(outs)))

    print(f"\n{'run':>22} {'mode':>14} {'J_H':>16}   files")
    print("-" * 84)
    for stem, rmode, jh, jhs, files in rows:
        jt = f"{jh:.3f} +/-{jhs:.3f}" if jh is not None else "(skipped)"
        print(f"{stem:>22} {rmode:>14} {jt:>16}   {files}")


if __name__ == "__main__":
    main()
