"""Component A: the canonical contested-task environment.

A native, batched-tensor implementation of the abstract dynamics from the TENURE
paper (Sections III and V). This is the *scientific* environment where headline
claims are established by sweeping alpha, beta, N, M, K. It is deliberately
independent of ``offsim/`` (the applied Override environment, Component B).

See ``tenure-sim-development-plan.md`` Section 3.
"""
from __future__ import annotations

from .config import CanonicalConfig, load_config

__all__ = ["CanonicalConfig", "load_config"]
