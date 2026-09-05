"""Graph observation builder (plan Section 3.7).

Returns a *dict of tensors*, never a flat vector — flattening is exactly what
stops the existing Override env generalizing across sizes. Node features are
type-specific; edge features ``e_rt``/``e_at``/``e_tt`` precompute the paper's
travel-time / horizon / threat geometry (Eq. 11-12) so the decoder does not have
to rediscover it from raw distances.

All tensors are on the state's device; features are the state's float dtype.
Padded (invalid) nodes are zeroed, so a config's valid sub-blocks are identical
regardless of how much padding surrounds them (``test_padding_invariance``).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .actions import action_masks
from .config import CanonicalConfig
from .core import CanonicalState, _gather_target


def build_observation(state: CanonicalState, cfg: CanonicalConfig) -> dict[str, Tensor]:
    dev = state.device
    dt = state.robot_pos.dtype
    B, T = state.batch_size, cfg.max_tasks
    field = torch.tensor(cfg.field_size, device=dev, dtype=dt)          # (2,)
    diag = float(math.hypot(*cfg.field_size))
    w_max = float(cfg.weight_range[1])
    horizon = cfg.horizon_T
    eps = 1e-9

    rv = state.robot_valid.unsqueeze(-1).to(dt)                         # (B, R, 1)
    tv = state.task_valid.unsqueeze(-1).to(dt)                          # (B, T, 1)
    kv = state.adv_valid.unsqueeze(-1).to(dt)                           # (B, K, 1)

    # ---- robot features (B, R, 9) --------------------------------------------
    a = state.robot_action
    is_acquire = a < T
    is_defend = (a >= T) & (a < 2 * T)
    is_idle = ~(is_acquire | is_defend)
    one_hot = torch.stack([is_acquire, is_defend, is_idle], dim=-1).to(dt)
    has_assign = (is_acquire | is_defend).to(dt).unsqueeze(-1)

    task_idx = torch.where(is_acquire, a, torch.where(is_defend, a - T, torch.full_like(a, -1)))
    target = _gather_target(state.task_pos, state.task_valid, task_idx, state.robot_pos)
    dist_target = torch.linalg.vector_norm(state.robot_pos - target, dim=-1, keepdim=True) / diag

    robot_feat = torch.cat([
        state.robot_pos / field,
        state.robot_vel / cfg.v_max,
        has_assign,
        one_hot,
        dist_target,
    ], dim=-1) * rv

    # ---- task features (B, T, 7..12) -----------------------------------------
    task_cols = [
        state.task_pos / field,
        (state.task_w / w_max).unsqueeze(-1),
        state.task_c.to(dt).unsqueeze(-1),
        (state.sigma / cfg.tau_com).unsqueeze(-1),
        (state.eta / cfg.tau_rev).unsqueeze(-1),          # tau_rev may be inf -> 0.0
        (state.t_since_change / horizon).unsqueeze(-1),
    ]
    if cfg.symmetric:                                    # opponent ownership (needed to contest/reverse)
        task_cols.append(state.task_c_red.to(dt).unsqueeze(-1))
    if cfg.expose_protected:                              # killer-control 4th arm: hand it the flag
        task_cols.append(state.task_protected.to(dt).unsqueeze(-1))
    if cfg.stack_cap > 1:                                 # STACKING: current height (needed to build/defend tall)
        task_cols.append((state.task_height.to(dt) / cfg.stack_cap).unsqueeze(-1))
    # ``task_value`` (B,T) is the value channel the decoder gates by retention. A retention-BLIND
    # policy sees only base weight; an EFFECTIVE-value one sees the toggle premium and/or stack height
    # (a tall stack / a toggle control point is worth more than a fresh task).
    aware = cfg.expose_toggle and cfg.toggle_regions > 0
    use_eff = aware or cfg.stack_cap > 1
    task_value = _effective_value(state, cfg, w_max) if use_eff else (state.task_w / w_max)
    if aware:                                            # TOGGLE leverage features
        gov_blue = state.task_c.gather(1, state.task_toggle_idx)
        gov_red = state.task_c_red.gather(1, state.task_toggle_idx)
        task_cols += [state.task_is_toggle.to(dt).unsqueeze(-1),
                      gov_blue.to(dt).unsqueeze(-1), gov_red.to(dt).unsqueeze(-1),
                      task_value.unsqueeze(-1)]
    task_feat = torch.cat(task_cols, dim=-1) * tv
    task_value = task_value * state.task_valid.to(dt)

    # ---- adversary features (B, K, 6) ----------------------------------------
    adv_speed = torch.linalg.vector_norm(state.adv_vel, dim=-1, keepdim=True) / cfg.adv_v_max
    adv_feat = torch.cat([
        state.adv_pos / field,
        state.adv_vel / cfg.adv_v_max,
        adv_speed,
        state.adv_valid.to(dt).unsqueeze(-1),
    ], dim=-1) * kv

    # ---- robot->task edges e_rt (B, R, T, 2): travel-time, horizon factor -----
    diff_rt = state.robot_pos.unsqueeze(2) - state.task_pos.unsqueeze(1)    # (B, R, T, 2)
    dist_rt = torch.linalg.vector_norm(diff_rt, dim=-1)                     # (B, R, T)
    travel_rt = dist_rt / cfg.v_max
    t_now = state.t.view(B, 1, 1)
    horizon_factor = torch.clamp((horizon - (t_now + travel_rt)) / horizon, 0.0, 1.0)  # Eq. 11
    e_rt = torch.stack([travel_rt / horizon, horizon_factor], dim=-1)
    e_rt = e_rt * (rv.unsqueeze(2) * tv.unsqueeze(1))

    # ---- adversary->task edges e_at (B, K, T, 2): threat time, closing --------
    diff_at = state.adv_pos.unsqueeze(2) - state.task_pos.unsqueeze(1)      # (B, K, T, 2)
    dist_at = torch.linalg.vector_norm(diff_at, dim=-1)
    threat = torch.clamp((dist_at / cfg.adv_v_max + cfg.tau_rev) / horizon, 0.0, 1.0)  # inf -> 1
    to_task = -diff_at                                                      # task - adv
    closing = ((state.adv_vel.unsqueeze(2) * to_task).sum(-1) > 0).to(dt)
    e_at = torch.stack([threat, closing], dim=-1)
    e_at = e_at * (kv.unsqueeze(2) * tv.unsqueeze(1))

    # ---- task->task edges e_tt (B, T, T, 1): proximity kernel ------------------
    diff_tt = state.task_pos.unsqueeze(2) - state.task_pos.unsqueeze(1)     # (B, T, T, 2)
    dist_tt = torch.linalg.vector_norm(diff_tt, dim=-1)
    e_tt = torch.exp(-dist_tt / cfg.service_radius).unsqueeze(-1)
    e_tt = e_tt * (tv.unsqueeze(2) * tv.unsqueeze(1))

    # ---- global scalars (B, 4) ------------------------------------------------
    n_valid_task = state.task_valid.sum(-1).clamp_min(1).to(dt)
    frac_complete = (state.task_c & state.task_valid).sum(-1).to(dt) / n_valid_task
    total_w = (state.task_w * state.task_valid.to(dt)).sum(-1).clamp_min(eps)
    held_norm = state.held_integral / (horizon * total_w)
    scalar = torch.stack([
        state.t / horizon,
        frac_complete,
        held_norm,
        torch.full((B,), float(cfg.alpha), device=dev, dtype=dt),
    ], dim=-1)

    # Table III ablation: blind the policy to adversaries — zero their node features and the
    # e_at adversary->task edges and force adv_valid False, so the encoder/head see no
    # adversary geometry. Obs shapes are unchanged (nodes zeroed, not removed).
    if cfg.ablate_adversary_nodes:
        adv_feat = torch.zeros_like(adv_feat)
        e_at = torch.zeros_like(e_at)
        adv_valid_out = torch.zeros_like(state.adv_valid)
    else:
        adv_valid_out = state.adv_valid.clone()

    out = {
        "robot_feat": robot_feat,
        "task_feat": task_feat,
        "adv_feat": adv_feat,
        "e_rt": e_rt,
        "e_at": e_at,
        "e_tt": e_tt,
        "scalar": scalar,
        "action_mask": action_masks(state, cfg),
        "robot_valid": state.robot_valid.clone(),
        "task_valid": state.task_valid.clone(),
        "adv_valid": adv_valid_out,
    }
    if use_eff:                       # only when the effective value differs from base weight (toggles/stacking);
        out["task_value"] = task_value  # the decoder falls back to task_feat[...,2] when absent
    return out


def _effective_value(state: CanonicalState, cfg: CanonicalConfig, w_max: float) -> Tensor:
    """Toggle-effective task value, normalised by w_max. STABLE by design: a GOAL is always worth its
    base weight and a TOGGLE is worth its base PLUS the (M-1)x boost it unlocks over its whole cluster
    of goals (its strategic leverage). We deliberately do NOT swing goal value by current toggle
    ownership -- that ownership-driven x3<->x1 flip made the policy chase/flee and never commit
    (entropy stuck). The multiplier is realised in the REWARD (holding the toggle triples its goals),
    so the policy learns to GRAB and DEFEND the toggle rather than value-chase. The stable is_toggle /
    who-holds-the-toggle FEATURES (built separately) still tell it when its cluster is threatened."""
    w = state.task_w
    is_tog = state.task_is_toggle
    # STACKING: a task is worth building/holding proportional to its height (the convex h**power payoff
    # is carried by the REWARD; a linear-in-height value channel is enough to order build/defend and
    # stays bounded). Neutral (h=0) clamps to 1 so a fresh task is worth its base weight.
    if cfg.stack_cap > 1:
        base = w * state.task_height.to(w.dtype).clamp_min(1.0)
    else:
        base = w
    if cfg.dynamic_exposure and cfg.toggle_regions > 0:
        # DYNAMIC cluster exposure: a toggle is worth its base PLUS what it CURRENTLY unlocks --
        # (M-1) x the value of its cluster's goals held by blue right now. Empty cluster -> ~base
        # (nearly worthless); loaded cluster -> large. Continuous and state-dependent (recomputed every
        # step), so the same toggle's obs value rises as its cluster fills and collapses when lost --
        # the spectrum a fixed premium can't express and blind can't read off geometry. Goals keep base.
        m = cfg.toggle_multiplier - 1.0
        goal_held = (state.task_c & ~is_tog & state.task_valid).to(w.dtype) * w          # (B,T) blue-held goal value
        cluster_held = torch.zeros_like(w).scatter_add(1, state.task_toggle_idx, goal_held)  # sum per governing toggle
        eff = torch.where(is_tog, w + m * cluster_held, base)
    elif cfg.toggle_regions > 0:
        # STATIC premium (M8): value a toggle like a top-tier task (max base weight) so the policy grabs
        # and DEFENDS it, but not ownership-dependent (thrash) nor coupled-to-held-goals (idle). The full
        # M x payoff is in the REWARD; the is_toggle / who-holds-it features flag threatened regions.
        eff = torch.where(is_tog, torch.full_like(w, float(w_max)), base)
    else:
        eff = base
    return eff / w_max


def observation_spec(cfg: CanonicalConfig) -> dict[str, tuple]:
    """Per-environment (unbatched) shapes for each observation key. Useful for
    building Gym/PettingZoo spaces without instantiating the env."""
    R, T, K, A = cfg.max_robots, cfg.max_tasks, cfg.max_adversaries, cfg.action_dim
    task_dim = 7 + int(cfg.symmetric) + int(cfg.expose_protected) + int(cfg.stack_cap > 1)
    aware = cfg.expose_toggle and cfg.toggle_regions > 0
    if aware:
        task_dim += 4                                    # is_toggle, gov_blue, gov_red, eff_value
    spec = {
        "robot_feat": (R, 9),
        "task_feat": (T, task_dim),
        "adv_feat": (K, 6),
        "e_rt": (R, T, 2),
        "e_at": (K, T, 2),
        "e_tt": (T, T, 1),
        "scalar": (4,),
        "action_mask": (R, A),
        "robot_valid": (R,),
        "task_valid": (T,),
        "adv_valid": (K,),
    }
    if aware or cfg.stack_cap > 1:                        # effective-value channel present iff non-trivial
        spec["task_value"] = (T,)
    return spec
