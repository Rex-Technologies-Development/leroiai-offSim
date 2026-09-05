"""Action masking (plan Section 3.5) and greedy conflict resolution (Section 3.6).

The flat action layout is fixed forever:

    [0,        T)      ACQUIRE task i
    [T,       2T)      DEFEND  task (i - T)
    2T                 IDLE

where ``T = max_tasks``. Total width ``A = 2T + 1``. Masking and matching operate
on the padded, batched :class:`~contested.core.CanonicalState`.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .config import CanonicalConfig
from .core import CanonicalState


def action_masks(state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
    """Return the ``(B, R, A)`` boolean action mask (Section 3.5).

    - ``ACQUIRE i`` valid iff ``task_valid[i]`` and ``not task_c[i]``
    - ``DEFEND i``  valid iff ``task_valid[i]`` and ``task_c[i]``
    - ``IDLE`` always valid
    - rows for invalid robots are all False except ``IDLE``
    """
    B = state.batch_size
    T, A, R = cfg.max_tasks, cfg.action_dim, cfg.max_robots
    mask = torch.zeros((B, R, A), dtype=torch.bool, device=state.device)

    rv = state.robot_valid.unsqueeze(-1)                     # (B, R, 1)
    # ACQUIRE a NEUTRAL task. Single-team: task_c_red is all-False, so ~task_c_red is all-True
    # and this equals the old ``task_valid & ~task_c`` -> the non-symmetric mask is unchanged.
    acquire = (state.task_valid & ~state.task_c & ~state.task_c_red).unsqueeze(1)  # (B, 1, T)
    defend = (state.task_valid & state.task_c).unsqueeze(1)    # DEFEND a blue-owned task
    mask[:, :, 0:T] = acquire & rv
    mask[:, :, T:2 * T] = defend & rv
    if cfg.symmetric:                                        # REVERSE a red-owned task (Phase 2)
        reverse = (state.task_valid & state.task_c_red).unsqueeze(1)
        mask[:, :, cfg.reverse_base:cfg.reverse_base + T] = reverse & rv
    mask[:, :, cfg.idle_action] = True                       # IDLE always valid
    return mask


def greedy_assign(
    scores: Tensor,
    available: Tensor,
    cfg: CanonicalConfig,
) -> Tensor:
    """Greedy conflict-free assignment (Section 3.6).

    ``R`` sequential global-argmax picks (cheap since ``R <= 8``): each pick
    consumes its robot, and — unless multi-assignment is allowed — its target
    column, so no two robots share an ``ACQUIRE`` (or ``DEFEND``) target. ``IDLE``
    is never consumed. Returns ``assignment`` ``(B, R)`` of action indices.

    ``scores`` is ``(B, R, A)``; ``available`` is the ``(B, R, A)`` action mask.
    """
    B, R, A = scores.shape
    T = cfg.max_tasks
    device = scores.device
    available = available.clone()
    assign = torch.full((B, R), cfg.idle_action, dtype=torch.int64, device=device)
    arangeB = torch.arange(B, device=device)
    neg_inf = torch.finfo(scores.dtype).min

    for _ in range(R):
        masked = torch.where(available, scores, torch.full_like(scores, neg_inf))
        best = masked.view(B, -1).argmax(dim=-1)             # (B,)
        robot = torch.div(best, A, rounding_mode="floor")
        action = best % A
        picked = available[arangeB, robot, action]           # (B,) still available?

        assign[arangeB, robot] = torch.where(picked, action, assign[arangeB, robot])
        available[arangeB, robot, :] = False                 # consume the robot row

        # consume the target column when multi-assignment is disallowed
        is_acquire = (action < T) & ~torch.tensor(cfg.allow_multi_acquire, device=device)
        is_defend = (action >= T) & (action < 2 * T) & ~torch.tensor(cfg.allow_multi_defend, device=device)
        # REVERSE: multi-assign by default (two robots commit to one target to out-number a
        # defended red task and take it -- consultant). Single-assign if allow_multi_reverse=False.
        is_reverse = ((action >= 2 * T) & (action < 3 * T) & cfg.symmetric
                      & ~torch.tensor(cfg.allow_multi_reverse, device=device))
        consume = (is_acquire | is_defend | is_reverse) & picked
        sel = consume.nonzero(as_tuple=True)[0]
        if sel.numel():
            available[sel, :, action[sel]] = False
    return assign
