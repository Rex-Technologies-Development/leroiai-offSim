"""Experiment runner + analysis tests (plan Section 7)."""
from __future__ import annotations

import json

from experiments.runner import run_cell
from experiments.sweep import run_sweep
from experiments.analysis import aggregate, load_runs

# tiny overrides so cells run in seconds, not GPU-hours
_FAST = dict(updates=2, batch_size=8, eval_batch=16, eval_seeds=2,
             cfg_overrides={"n_tasks": 6, "n_adversaries": 3, "horizon_T": 3.0}, device="cpu")


def test_run_cell_writes_run_directory(tmp_path):
    out = tmp_path / "greedy_a1_s0"
    res = run_cell("greedy", alpha=1.0, seed=0, out_dir=out, **_FAST)
    assert res["method"] == "greedy" and "J_H" in res
    for f in ("config.json", "meta.json", "metrics.json"):
        assert (out / f).exists(), f
    meta = json.loads((out / "meta.json").read_text())
    assert meta["git_sha"] and "rules_compliant" in meta
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["alpha"] == 1.0 and cfg["seed"] == 0


def test_tenure_cell_trains_and_checkpoints(tmp_path):
    out = tmp_path / "tenure_a1_s0"
    run_cell("tenure", alpha=1.0, seed=0, out_dir=out, **_FAST)
    assert (out / "checkpoint.pt").exists(), "learned method must save a checkpoint"
    metrics = json.loads((out / "metrics.json").read_text())
    assert len(metrics["series"]) == 2 and "retention_mae" in metrics["series"][-1]


def test_sweep_and_aggregate(tmp_path):
    out = tmp_path / "sweep"
    results = run_sweep(["greedy", "defensive"], [0.0, 1.0], [0, 1], out, **_FAST)
    assert len(results) == 2 * 2 * 2                          # method × alpha × seed
    assert (out / "summary.json").exists()
    assert len(load_runs(out)) == 8

    rows = aggregate(out)
    assert len(rows) == 4                                     # 2 methods × 2 alphas, aggregated over seeds
    for r in rows:
        assert r["n_seeds"] == 2 and 0.0 <= r["J_H"] <= 1.0 and "J_H_ci" in r
