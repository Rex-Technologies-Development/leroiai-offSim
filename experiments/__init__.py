"""Component E: experiment runner (plan Section 7).

Config-driven sweeps over ``{method} × {alpha} × {seed}``. Each cell gets its own
run directory containing the resolved config, the git SHA, the metric series, the
final checkpoint, and a flag recording whether the cell is rules-compliant. A single
command aggregates the run directories into tables/figures.
"""
from __future__ import annotations

from .runner import METHODS, run_cell
from .sweep import run_sweep
from .analysis import aggregate, print_table
from .experiments_map import EXPERIMENTS

__all__ = ["METHODS", "run_cell", "run_sweep", "aggregate", "print_table", "EXPERIMENTS"]
