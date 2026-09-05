"""PettingZoo ParallelEnv adapter over the shared core (plan Section 3.12).

Exposes the ``n_robots`` allied robots as parallel agents so BenchMARL baselines
(MAPPO / QMIX / MASAC) can train against the same dynamics. Wraps a single
environment (``B = 1``); BenchMARL vectorizes externally.

Subclasses ``pettingzoo.utils.env.ParallelEnv`` when PettingZoo is installed, but
the class is duck-typed and fully usable (and testable) without it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

from ..config import CanonicalConfig
from ..core import CanonicalCore
from ..observation import build_observation
from .gym_vec import build_single_observation_space, _FLOAT_KEYS, _MASK_KEYS

try:  # optional dependency
    from pettingzoo.utils.env import ParallelEnv as _Base
except Exception:  # pragma: no cover - exercised only when pettingzoo is absent
    _Base = object


class CanonicalParallelEnv(_Base):
    """One canonical environment as a PettingZoo parallel multi-agent env."""

    metadata = {"name": "contested_canonical_v0", "is_parallelizable": True}

    def __init__(self, cfg: CanonicalConfig, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32, seed: int = 0):
        self.cfg = cfg
        self.core = CanonicalCore(cfg, batch_size=1, device=device, dtype=dtype)
        self._base_seed = seed
        self._episode = 0
        self.possible_agents = [f"robot_{i}" for i in range(cfg.n_robots)]
        self.agents: list[str] = list(self.possible_agents)

        obs = dict(build_single_observation_space(cfg).spaces)
        obs["self_index"] = spaces.Box(0, cfg.max_robots, shape=(1,), dtype=np.int64)
        self._obs_space = spaces.Dict(obs)
        self._act_space = spaces.Discrete(cfg.action_dim)

    # PettingZoo API ---------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._base_seed = seed
            self._episode = 0
        self.core.reset(self._base_seed + self._episode)
        self.agents = list(self.possible_agents)
        return self._observations(), {a: {} for a in self.agents}

    def step(self, actions: dict[str, int]):
        a = torch.full((1, self.cfg.max_robots), self.cfg.idle_action,
                       dtype=torch.int64, device=self.core.device)
        for i, agent in enumerate(self.possible_agents):
            if agent in actions:
                a[0, i] = int(actions[agent])
        reward, done, _ = self.core.step(a)
        r = float(reward.reshape(-1)[0])
        d = bool(done.reshape(-1)[0])

        obs = self._observations()
        rewards = {ag: r for ag in self.agents}                 # shared team reward
        terminations = {ag: False for ag in self.agents}        # fixed horizon => truncation
        truncations = {ag: d for ag in self.agents}
        infos = {ag: {} for ag in self.agents}
        if d:
            self._episode += 1
            self.agents = []                                    # PettingZoo: no agents once done
        return obs, rewards, terminations, truncations, infos

    def close(self):
        pass

    # helpers ----------------------------------------------------------------
    def _observations(self) -> dict[str, dict]:
        o = build_observation(self.core.state, self.cfg)
        shared: dict[str, np.ndarray] = {}
        for key in _FLOAT_KEYS:
            shared[key] = o[key][0].detach().cpu().numpy().astype(np.float32)
        for key in _MASK_KEYS:
            shared[key] = o[key][0].detach().cpu().numpy().astype(np.int8)
        return {
            agent: {**shared, "self_index": np.array([i], dtype=np.int64)}
            for i, agent in enumerate(self.agents)
        }
