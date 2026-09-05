"""MARL baselines via BenchMARL (plan Section 6): MAPPO, QMIX, MASAC.

These are learned, cooperative multi-agent baselines run through PettingZoo using
``contested/adapters/pettingzoo_par.py`` (``CanonicalParallelEnv``). BenchMARL is an
optional dependency (``pip install benchmarl``); it is not required to import this
package or run the self-contained baselines.

Integration sketch
------------------
    from baselines.marl import make_parallel_env
    env = make_parallel_env(alpha=1.0)            # a PettingZoo ParallelEnv
    # register `env` as a BenchMARL task and launch MAPPO/QMIX/MASAC from its configs.

Fair adaptation (enforced by the runner, not here): each agent's observation already
carries adversary features; every MARL method is retrained from scratch in the
contested environment; sample budget and wall-clock are matched and logged.
"""
from __future__ import annotations

from typing import Optional

from contested.config import CanonicalConfig, load_config


def make_parallel_env(cfg: Optional[CanonicalConfig] = None, **overrides):
    """Return a fresh ``CanonicalParallelEnv`` for BenchMARL to wrap."""
    from contested.adapters.pettingzoo_par import CanonicalParallelEnv
    cfg = cfg or load_config(overrides=overrides or None)
    return CanonicalParallelEnv(cfg)


ALGORITHMS = ("mappo", "qmix", "masac")
