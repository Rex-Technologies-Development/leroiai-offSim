"""Retention label tests (plan Sections 3.9, 3.13)."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore
from contested.labels import RetentionLabelRecorder, compute_retention_labels

F64 = torch.float64


def test_retention_label_handbuilt():
    """A hand-constructed c/held trajectory yields the known reverse-scan labels."""
    # B=1, D=3 decisions, T=1 task; horizon 3s, decision 1s. Task completes during d0.
    decision_c = torch.tensor([[[0], [1], [1]]], dtype=torch.bool)          # (1,3,1)
    decision_held = torch.tensor([[[0.5], [1.0], [1.0]]], dtype=F64)        # int c dt per decision
    decision_sigma = torch.zeros(1, 3, 1, dtype=F64)
    final_c = torch.tensor([[1]], dtype=torch.bool)
    final_sigma = torch.zeros(1, 1, dtype=F64)
    t_decision = torch.tensor([0.0, 1.0, 2.0], dtype=F64)
    task_valid = torch.tensor([[True]])

    out = compute_retention_labels(decision_c, decision_held, decision_sigma,
                                   final_c, final_sigma, t_decision, task_valid,
                                   horizon_T=3.0, dt=1.0)
    # remaining = [2.5, 2.0, 1.0]; denom = [3, 2, 1] -> R = [0.8333, 1.0, 1.0]
    expected_R = torch.tensor([[[2.5 / 3.0], [1.0], [1.0]]], dtype=F64)
    assert torch.allclose(out["R"], expected_R, atol=1e-9)
    # dense labels are the complete-task decisions; the event label is the acquisition decision
    assert out["dense_mask"].squeeze(-1).tolist() == [[False, True, True]]
    assert out["event_mask"].squeeze(-1).tolist() == [[True, False, False]]


def test_no_revisit_r_truncates_at_first_reversal():
    """no_revisit R stops at the first reversal; standard R counts the later re-completion."""
    # D=4, T=1, horizon 4s: complete d0 & d1, revert during d1, re-complete d3.
    decision_c = torch.tensor([[[1], [1], [0], [1]]], dtype=torch.bool)
    decision_held = torch.tensor([[[1.0], [0.5], [0.0], [1.0]]], dtype=F64)
    decision_sigma = torch.zeros(1, 4, 1, dtype=F64)
    kw = dict(final_c=torch.tensor([[1]], dtype=torch.bool), final_sigma=torch.zeros(1, 1, dtype=F64),
              t_decision=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=F64),
              task_valid=torch.tensor([[True]]), horizon_T=4.0, dt=1.0)
    std = compute_retention_labels(decision_c, decision_held, decision_sigma, **kw)
    nr = compute_retention_labels(decision_c, decision_held, decision_sigma, no_revisit=True, **kw)
    # standard: remaining/denom counts the d3 re-completion; no-revisit truncates at the d1 revert
    assert torch.allclose(std["R"].squeeze(-1)[0], torch.tensor([0.625, 0.5, 0.5, 1.0], dtype=F64))
    assert torch.allclose(nr["R"].squeeze(-1)[0], torch.tensor([0.375, 1 / 6, 0.5, 1.0], dtype=F64), atol=1e-6)
    assert (nr["R"] <= std["R"] + 1e-9).all()      # never larger than standard


def test_labels_bounded_and_high_when_held_to_end():
    """Real rollout at alpha=0 (no reversal): completed tasks retain ~fully."""
    cfg = CanonicalConfig(n_robots=4, n_tasks=6, n_adversaries=3,
                          max_robots=6, max_tasks=8, max_adversaries=4,
                          horizon_T=12.0, alpha=0.0)
    core = CanonicalCore(cfg, batch_size=4, device="cpu", dtype=F64)
    core.reset(seed=2)
    rec = RetentionLabelRecorder()
    idx = torch.arange(cfg.max_robots) % cfg.n_tasks
    acts = idx.unsqueeze(0).expand(4, -1).contiguous()
    for _ in range(cfg.n_decisions):
        rec.before_decision(core)
        core.step(acts)
        rec.after_decision(core)
    out = rec.finalize(core)

    assert out["R"].shape == (4, cfg.n_decisions, cfg.max_tasks)
    assert torch.isfinite(out["R"]).all()
    assert (out["R"] >= 0).all() and (out["R"] <= 1).all()
    # at least some labels exist, and dense (complete-task) labels are high at alpha=0
    dense = out["dense_mask"]
    assert dense.any(), "expected some completed-task labels"
    assert out["R"][dense].mean() > 0.9, "held-to-end tasks should retain nearly fully"
    # the two regimes are disjoint (a task is either complete or in its acquisition run)
    assert not (out["dense_mask"] & out["event_mask"]).any()
