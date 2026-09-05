"""The five adversary archetypes (plan Section 3.10), batched and deterministic.

All are implemented over the padded batch and are selectable *per environment
instance* (a batch may mix archetypes). Scripted archetypes are deterministic
given the reset seed; the stochastic one (``feinter``) draws from per-environment
RNGs that are independent of batch size, so a given environment behaves identically
across ``B`` values.

| id | name             | target selection                                            |
| -- | ---------------- | ----------------------------------------------------------- |
| 0  | greedy_nearest   | nearest task with c=1                                       |
| 1  | value_targeting  | argmax w / (d + eps) over c=1                               |
| 2  | camper           | a spatial cluster chosen at reset; cycles within it         |
| 3  | feinter          | commits, then re-samples w.p. p when within d_switch        |
| 4  | learned_selfplay | PPO vs a frozen defender (stub -> greedy until a net loads) |
| 5  | builder          | acquire+hold: argmax w/(d+eps) over tasks blue does NOT own |

The first four *revert* blue's work (they target ``c=1`` = blue-owned tasks). ``builder``
is the Phase-2 opposite — it *acquires and holds* neutral/red-owned tasks, so a symmetric red
actually holds something worth taking back. It is a no-op in the single-team env (red never
completes there), so it only matters under ``cfg.symmetric``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .config import CanonicalConfig
from .core import CanonicalState, _gather_target

ARCHETYPE_IDS = {
    "greedy_nearest": 0, "value_targeting": 1, "camper": 2, "feinter": 3, "learned_selfplay": 4,
    "builder": 5,
}

CAMPER_RADIUS_FRAC = 0.28      # cluster radius, as a fraction of min(field)
FEINT_PROB = 0.30              # p: chance to re-sample when close to the committed target
FEINT_D_SWITCH_FRAC = 0.12     # d_switch, as a fraction of the field diagonal


class AdversaryController:
    """Selects ``adv_target`` (a task slot or -1) for every adversary each decision."""

    def __init__(self, cfg: CanonicalConfig, device, dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.camper_radius = CAMPER_RADIUS_FRAC * min(cfg.field_size)
        self.d_switch = FEINT_D_SWITCH_FRAC * math.hypot(*cfg.field_size)
        self._rngs: list[np.random.Generator] = []
        self.camper_center: Tensor
        self.committed: Tensor

    def reset(self, state: CanonicalState, adv_rngs: list[np.random.Generator]) -> None:
        B, K = state.adv_archetype.shape
        self._rngs = adv_rngs
        nt = max(1, self.cfg.n_tasks)
        tp = state.task_pos.detach().cpu().numpy()
        centers = np.zeros((B, K, 2), np.float64)
        for b in range(B):
            idx = adv_rngs[b].integers(0, nt, size=K)     # anchor each adversary at a real task
            centers[b] = tp[b, idx]
        self.camper_center = torch.tensor(centers, device=self.device, dtype=self.dtype)
        self.committed = torch.full((B, K), -1, device=self.device, dtype=torch.int64)

    def select_targets(self, state: CanonicalState) -> Tensor:
        cfg = self.cfg
        adv_pos, task_pos = state.adv_pos, state.task_pos
        B, K = state.adv_archetype.shape
        T = cfg.max_tasks
        neg = torch.finfo(self.dtype).min

        dist = torch.linalg.vector_norm(adv_pos.unsqueeze(2) - task_pos.unsqueeze(1), dim=-1)  # (B,K,T)
        cand = state.task_c & state.task_valid                                                 # (B,T)
        cand_k = cand.unsqueeze(1).expand_as(dist)
        cand_any = cand.any(-1, keepdim=True).expand(-1, K)                                     # (B,K)
        w = state.task_w.unsqueeze(1).expand_as(dist)

        greedy = self._pick(torch.where(cand_k, -dist, neg), cand_any)
        value = self._pick(torch.where(cand_k, w / (dist + 1e-3), neg), cand_any)
        camper = self._camper(state, dist, cand)
        feinter = self._feinter(state, greedy, value, cand_any)
        builder = self._builder(state, dist, w, neg)               # id 5 (acquire+hold, symmetric)
        raider = self._toggle_raider(state, dist, neg)             # id 6 (attacks blue's toggles)

        arch = state.adv_archetype
        target = greedy.clone()                                    # id 0 and 4 (learned stub)
        target = torch.where(arch == 1, value, target)
        target = torch.where(arch == 2, camper, target)
        target = torch.where(arch == 3, feinter, target)
        target = torch.where(arch == 5, builder, target)
        target = torch.where(arch == 6, raider, target)
        return torch.where(state.adv_valid, target, torch.full_like(target, -1))

    def _toggle_raider(self, state: CanonicalState, dist: Tensor, neg: float) -> Tensor:
        """M8 smart opponent: go for the MULTIPLIER. Each red robot targets the nearest BLUE-owned
        TOGGLE (flip it to collapse the cluster); if none, the nearest blue-owned task. Retention
        matters precisely because this opponent contests the leverage points."""
        blue = state.task_c & state.task_valid                          # (B,T)
        tog = (blue & state.task_is_toggle).unsqueeze(1)                # (B,1,T) blue toggles
        goal = (blue & ~state.task_is_toggle).unsqueeze(1)
        tgt_tog = torch.where(tog, -dist, torch.full_like(dist, neg)).argmax(-1)   # (B,K)
        tgt_goal = torch.where(goal, -dist, torch.full_like(dist, neg)).argmax(-1)
        has_tog = tog.any(-1)
        has_goal = goal.any(-1)
        return torch.where(has_tog, tgt_tog, torch.where(has_goal, tgt_goal, torch.full_like(tgt_tog, -1)))

    @staticmethod
    def _pick(score: Tensor, has_candidate: Tensor) -> Tensor:
        idx = score.argmax(dim=-1)
        return torch.where(has_candidate, idx, torch.full_like(idx, -1))

    def _camper(self, state: CanonicalState, dist: Tensor, cand: Tensor) -> Tensor:
        neg = torch.finfo(self.dtype).min
        d_center = torch.linalg.vector_norm(
            state.task_pos.unsqueeze(1) - self.camper_center.unsqueeze(2), dim=-1)   # (B,K,T)
        in_cluster = d_center < self.camper_radius
        in_valid = in_cluster & state.task_valid.unsqueeze(1)
        in_cand = in_cluster & cand.unsqueeze(1)
        pick_complete = torch.where(in_cand, -dist, neg).argmax(-1)
        pick_any = torch.where(in_valid, -dist, neg).argmax(-1)
        has_complete = in_cand.any(-1)
        has_any = in_valid.any(-1)
        camper = torch.where(has_complete, pick_complete, pick_any)
        return torch.where(has_any, camper, torch.full_like(camper, -1))

    def _feinter(self, state: CanonicalState, greedy: Tensor, value: Tensor, cand_any: Tensor) -> Tensor:
        T = self.cfg.max_tasks
        committed = self.committed
        safe = committed.clamp(0, T - 1)
        still_good = (committed >= 0) & torch.gather(state.task_c, 1, safe) & torch.gather(state.task_valid, 1, safe)
        recommitted = torch.where(still_good, committed, greedy)          # re-commit if target lost

        target_pos = _gather_target(state.task_pos, state.task_valid, recommitted, state.adv_pos)
        d_committed = torch.linalg.vector_norm(state.adv_pos - target_pos, dim=-1)
        within = d_committed < self.d_switch

        rand = self._per_env_rand(committed.shape)
        switch = within & (rand < FEINT_PROB) & cand_any                 # feint only if an alt exists
        feint = torch.where(switch, value, recommitted)
        self.committed = torch.where(state.adv_valid, feint, torch.full_like(feint, -1))
        return self.committed

    def _builder(self, state: CanonicalState, dist: Tensor, w: Tensor, neg: float) -> Tensor:
        """Acquire + hold: value-weighted nearest task blue does NOT own (neutral or red-owned).
        Under ``cfg.symmetric`` red completes a neutral task it dwells on and revisits its own
        held tasks to defend them; a no-op in the single-team env (red never completes)."""
        buildable = (~state.task_c) & state.task_valid            # not blue-owned & valid
        score = torch.where(buildable.unsqueeze(1), w / (dist + 1e-3), neg)
        has = buildable.any(-1, keepdim=True).expand(-1, dist.shape[1])
        return self._pick(score, has)

    def _per_env_rand(self, shape) -> Tensor:
        B, K = shape
        out = np.empty((B, K), np.float64)
        for b in range(B):
            out[b] = self._rngs[b].random(K)
        return torch.tensor(out, device=self.device, dtype=self.dtype)
