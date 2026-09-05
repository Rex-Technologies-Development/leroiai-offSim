"""Aggregate run directories into tables/figures (plan Section 7).

Groups cells by ``(method, alpha)`` and reports the mean and a 95% confidence
interval over seeds. ``python -m experiments.analysis <run_dir>`` prints the table
and, if matplotlib is available, writes ``J_H`` vs ``alpha`` per method.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

_METRICS = ("J_H", "retention_rate", "retention_mae", "defense_fraction", "n_reversals")


def load_runs(out_dir: str | Path) -> list[dict]:
    """Read every cell's final metrics from its run directory."""
    runs = []
    for meta_path in sorted(Path(out_dir).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        metrics = json.loads((meta_path.parent / "metrics.json").read_text())
        rec = {"method": meta["method"], "alpha": meta["alpha"], "seed": meta["seed"],
               "rules_compliant": meta.get("rules_compliant", True)}
        rec.update({k: v for k, v in metrics["final"].items() if not k.endswith("_std")})
        if metrics["series"]:                                   # TENURE retention MAE lives in the series
            rec["retention_mae"] = metrics["series"][-1].get("retention_mae", float("nan"))
        runs.append(rec)
    return runs


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate(out_dir: str | Path) -> list[dict]:
    """Mean + 95% CI over seeds, grouped by (method, alpha)."""
    runs = load_runs(out_dir)
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["method"], r["alpha"]), []).append(r)

    rows = []
    for (method, alpha), cells in sorted(groups.items()):
        row = {"method": method, "alpha": alpha, "n_seeds": len(cells),
               "rules_compliant": all(c["rules_compliant"] for c in cells)}
        for m in _METRICS:
            vals = [c[m] for c in cells if m in c and c[m] == c[m]]  # drop NaN
            if vals:
                row[m] = statistics.fmean(vals)
                row[m + "_ci"] = _ci95(vals)
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no runs found)")
        return
    hdr = f"{'method':16s} {'alpha':>5s} {'n':>2s}  " + "  ".join(f"{m:>16s}" for m in _METRICS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for m in _METRICS:
            if m in r:
                cells.append(f"{r[m]:8.3f}±{r[m + '_ci']:.3f}")
            else:
                cells.append(f"{'-':>16s}")
        flag = "" if r["rules_compliant"] else " †non-compliant"
        print(f"{r['method']:16s} {r['alpha']:5.2f} {r['n_seeds']:2d}  " + "  ".join(cells) + flag)


def save_figure(rows: list[dict], path: str | Path) -> bool:
    """J_H vs alpha per method with CI bands. Returns False if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        pts = sorted([r for r in rows if r["method"] == method and "J_H" in r], key=lambda r: r["alpha"])
        if not pts:
            continue
        xs = [p["alpha"] for p in pts]
        ys = [p["J_H"] for p in pts]
        es = [p.get("J_H_ci", 0.0) for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
    ax.set_xlabel("alpha (tau_com / tau_rev)")
    ax.set_ylabel("J_H (normalized held value)")
    ax.set_title("Held value vs contest speed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate a sweep's run directories")
    p.add_argument("run_dir")
    p.add_argument("--figure", default=None, help="optional path for the J_H-vs-alpha PNG")
    args = p.parse_args()
    rows = aggregate(args.run_dir)
    print_table(rows)
    if args.figure:
        ok = save_figure(rows, args.figure)
        print(f"\nfigure -> {args.figure}" if ok else "\n(matplotlib not available; skipped figure)")


if __name__ == "__main__":
    main()
