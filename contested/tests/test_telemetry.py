"""Episode telemetry tests (plan Section 3.11)."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore

F64 = torch.float64


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4, horizon_T=8.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _run(cfg, actions_fn, batch=6, seed=0):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
    core.reset(seed=seed)
    info = {}
    for _ in range(cfg.n_decisions):
        _, done, info = core.step(actions_fn(cfg, batch))
        if bool(done.all()):
            break
    return core, info["telemetry"]


def _acquire_nearest(cfg, batch):
    # each robot acquires task (robot_id mod n_tasks) — drives completions
    idx = torch.arange(cfg.max_robots) % cfg.n_tasks
    return idx.unsqueeze(0).expand(batch, -1).contiguous()


def test_summary_keys_and_ranges():
    cfg = _cfg()
    core, tel = _run(cfg, _acquire_nearest)
    expected = {"J_H", "J_H_red", "J_H_diff", "standoff_fraction", "J_T", "retention_rate",
                "defense_fraction", "n_reversals", "n_recompletions", "assignment_churn",
                "time_to_first_completion", "per_task_held_fraction"}
    assert set(tel) == expected
    # single-team env: red holds nothing, so the differential reduces to blue-only and no standoffs
    assert torch.allclose(tel["J_H_diff"], tel["J_H"]) and (tel["J_H_red"] == 0).all()
    assert (tel["standoff_fraction"] == 0).all()
    B = 6
    for key in ("J_H", "J_T", "retention_rate", "defense_fraction", "time_to_first_completion"):
        assert tel[key].shape == (B,), f"{key} wrong shape"
    assert tel["per_task_held_fraction"].shape == (B, cfg.max_tasks)
    for key in ("J_H", "J_T", "retention_rate", "defense_fraction"):
        v = tel[key]
        assert (v >= -1e-9).all() and (v <= 1 + 1e-9).all(), f"{key} out of [0,1]: {v}"
    assert (tel["n_reversals"] >= 0).all()
    assert (tel["time_to_first_completion"] <= cfg.horizon_T + 1e-9).all()


def test_no_reversals_at_alpha_zero():
    cfg = _cfg(alpha=0.0, horizon_T=12.0)
    core, tel = _run(cfg, _acquire_nearest, seed=3)
    assert (tel["n_reversals"] == 0).all(), "alpha=0 must yield zero reversals"
    assert (tel["n_recompletions"] == 0).all()


def test_defense_fraction_reflects_actions():
    cfg = _cfg()
    all_defend = lambda cfg, batch: torch.full((batch, cfg.max_robots), cfg.max_tasks, dtype=torch.int64)
    all_acquire = lambda cfg, batch: torch.zeros((batch, cfg.max_robots), dtype=torch.int64)
    _, tel_d = _run(cfg, all_defend)
    _, tel_a = _run(cfg, all_acquire)
    assert torch.allclose(tel_d["defense_fraction"], torch.ones_like(tel_d["defense_fraction"]))
    assert torch.allclose(tel_a["defense_fraction"], torch.zeros_like(tel_a["defense_fraction"]))


def test_j_h_matches_normalized_held_integral():
    cfg = _cfg()
    core, tel = _run(cfg, _acquire_nearest, seed=1)
    s = core.state
    total_w = (s.task_w * s.task_valid.to(F64)).sum(-1)
    expected = s.held_integral / (cfg.horizon_T * total_w)
    assert torch.allclose(tel["J_H"], expected, atol=1e-9)
