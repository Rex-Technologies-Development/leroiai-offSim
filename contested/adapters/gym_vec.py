"""Gymnasium vector adapter over the batched canonical core (plan Section 3.12).

The core is already batched, so ``num_envs`` maps directly onto the core's ``B``.
This class is a *pure view*: it converts the core's dict observation to NumPy and
back, and forwards actions — it holds no dynamics of its own.

Episodes share a fixed horizon, so all sub-environments terminate together. The
horizon is a time limit, hence reported as ``truncated`` (not ``terminated``); on
that boundary the batch auto-resets and the terminal telemetry/observation are
returned in ``info`` (Gymnasium ``final_observation`` convention).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from ..config import CanonicalConfig
from ..core import CanonicalCore
from ..observation import build_observation, observation_spec

_FLOAT_KEYS = ("robot_feat", "task_feat", "adv_feat", "e_rt", "e_at", "e_tt", "scalar")
_MASK_KEYS = ("action_mask", "robot_valid", "task_valid", "adv_valid")


def build_single_observation_space(cfg: CanonicalConfig) -> spaces.Dict:
    spec = observation_spec(cfg)
    d: dict[str, spaces.Space] = {}
    for key in _FLOAT_KEYS:
        d[key] = spaces.Box(-np.inf, np.inf, shape=spec[key], dtype=np.float32)
    d["action_mask"] = spaces.MultiBinary(list(spec["action_mask"]))
    for key in ("robot_valid", "task_valid", "adv_valid"):
        d[key] = spaces.MultiBinary(spec[key][0])
    return spaces.Dict(d)


class CanonicalVectorEnv:
    """A Gymnasium-style vectorized environment backed by one CanonicalCore."""

    metadata: dict = {"autoreset_mode": "same-step"}

    def __init__(
        self,
        cfg: CanonicalConfig,
        num_envs: int = 8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.core = CanonicalCore(cfg, batch_size=num_envs, device=device, dtype=dtype)
        self.single_observation_space = build_single_observation_space(cfg)
        self.single_action_space = spaces.MultiDiscrete([cfg.action_dim] * cfg.max_robots)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._base_seed = seed
        self._episode = 0

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._base_seed = seed
            self._episode = 0
        self.core.reset(self._base_seed + self._episode)
        return self._obs_numpy(), {}

    def step(self, actions):
        a = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.core.device)
        reward, done, info = self.core.step(a)
        terminated = np.zeros(self.num_envs, dtype=bool)      # fixed horizon => time limit
        truncated = done.detach().cpu().numpy().astype(bool)
        obs = self._obs_numpy()

        out_info: dict = {}
        if bool(done.all()):
            out_info["final_observation"] = obs
            if "telemetry" in info:
                out_info["telemetry"] = {k: v.detach().cpu().numpy() for k, v in info["telemetry"].items()}
            self._episode += 1
            self.core.reset(self._base_seed + self._episode)
            obs = self._obs_numpy()

        return obs, reward.detach().cpu().numpy(), terminated, truncated, out_info

    def close(self):
        pass

    # --------------------------------------------------------------- helpers
    def _obs_numpy(self) -> dict[str, np.ndarray]:
        obs = build_observation(self.core.state, self.cfg)
        out: dict[str, np.ndarray] = {}
        for key in _FLOAT_KEYS:
            out[key] = obs[key].detach().cpu().numpy().astype(np.float32)
        for key in _MASK_KEYS:
            out[key] = obs[key].detach().cpu().numpy().astype(np.int8)
        return out

    def action_masks(self) -> np.ndarray:
        """Current ``(num_envs, R, A)`` action mask, for masked policies (SB3-contrib)."""
        from ..actions import action_masks
        return action_masks(self.core.state, self.cfg).detach().cpu().numpy()
