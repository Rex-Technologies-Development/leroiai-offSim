"""Shared training progress bar.

File-friendly by design: one appended line per interval, no carriage returns, so it
reads cleanly in a captured background-task log (where `\r` bars turn to garbage).
Used by ``tenure.train`` and ``experiments.retention_learned`` so **every training
run** shows the same bar + running metrics + ETA.
"""
from __future__ import annotations

import math
import statistics
import time


def bar_str(i: int, total: int, width: int = 22) -> str:
    filled = round(width * i / total) if total else width
    return "#" * filled + "-" * (width - filled)


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


def tail_mean(xs, k: int = 10) -> float:
    """Mean of the last ``k`` values, ignoring NaNs; NaN if none are finite."""
    vals = [x for x in xs[-k:] if not math.isnan(x)]
    return statistics.fmean(vals) if vals else float("nan")


class ProgressBar:
    """Periodic progress bar for a training loop.

    total            : steps (updates) in THIS run.
    every            : print only every ``every`` steps (and always on the last).
    prefix           : e.g. ``"[run 2/6] multiplicative s0 "`` for multi-run drivers.
    eta_total/eta_done : for multi-run ETA — total steps across ALL runs and steps
                       already finished before this run, so ``eta`` covers the whole job.
    """

    def __init__(self, total: int, every: int = 5, width: int = 22, prefix: str = "",
                 eta_total: int | None = None, eta_done: int = 0):
        self.total = total
        self.every = max(1, every)
        self.width = width
        self.prefix = prefix
        self.eta_total = eta_total or total
        self.eta_done = eta_done
        self.start = time.time()

    def eta(self, i: int) -> float:
        per = (time.time() - self.start) / max(1, i)
        return per * (self.eta_total - (self.eta_done + i))

    def line(self, i: int, metrics: dict) -> None:
        """Print a bar line for step ``i`` (1-based). ``metrics`` -> ``key~value`` fields."""
        if i % self.every and i != self.total:
            return
        ms = "  ".join(f"{k}~{v:.3f}" if isinstance(v, float) else f"{k}~{v}"
                       for k, v in metrics.items())
        pct = 100 * i // self.total if self.total else 100
        print(f"{self.prefix}[{bar_str(i, self.total)}] {i:4d}/{self.total} {pct:3d}%  "
              f"{ms}  eta {fmt_time(self.eta(i))}", flush=True)
