"""Cartesian sweeps over ``{method} × {alpha} × {seed}`` (plan Section 7)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .runner import METHODS, run_cell


def run_sweep(methods: Sequence[str], alphas: Sequence[float], seeds: Sequence[int],
              out_dir: str | Path, *, cfg_overrides: Optional[dict] = None,
              rules_compliant: bool = True, **cell_kwargs) -> list[dict]:
    """Run every ``(method, alpha, seed)`` cell into its own run directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for method in methods:
        for alpha in alphas:
            for seed in seeds:
                cell = out / f"{method}_a{alpha:g}_s{seed}"
                results.append(run_cell(method, alpha, seed, cell,
                                        cfg_overrides=cfg_overrides,
                                        rules_compliant=rules_compliant, **cell_kwargs))
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    from .experiments_map import EXPERIMENTS
    p = argparse.ArgumentParser(description="Run a TENURE experiment sweep")
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=None,
                   help="named experiment (E1/E4/E6/...); overrides --methods/--alphas")
    p.add_argument("--methods", nargs="+", default=["tenure", "greedy"], choices=METHODS)
    p.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    p.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=3)
    args = p.parse_args()

    if args.experiment:
        spec = EXPERIMENTS[args.experiment]
        methods = spec.get("methods", args.methods)
        alphas = spec.get("alphas", args.alphas)
        overrides = spec.get("cfg_overrides")
        out = args.out if args.out != "runs/sweep" else f"runs/{args.experiment}"
    else:
        methods, alphas, overrides, out = args.methods, args.alphas, None, args.out

    results = run_sweep(methods, alphas, args.seeds, out, cfg_overrides=overrides,
                        updates=args.updates, batch_size=args.batch_size, eval_seeds=args.eval_seeds)
    from .analysis import aggregate, print_table
    print_table(aggregate(out))
    print(f"\n{len(results)} cells -> {out}/  (aggregate with:  python -m experiments.analysis {out})")


if __name__ == "__main__":
    main()
