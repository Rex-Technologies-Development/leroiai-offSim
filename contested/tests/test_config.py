"""Strict config loader tests (plan Section 3.1)."""
from __future__ import annotations

import math

import pytest

from contested.config import CanonicalConfig, load_config


def test_default_yaml_loads():
    cfg = load_config()
    assert cfg.n_robots == 4 and cfg.n_tasks == 12 and cfg.n_adversaries == 4
    assert cfg.ticks_per_decision == 10 and cfg.action_dim == 2 * cfg.max_tasks + 1


def test_alpha_knob_derives_tau_rev():
    assert load_config(overrides={"alpha": 1.0}).tau_rev == 2.0
    assert load_config(overrides={"alpha": 2.0}).tau_rev == 1.0
    assert math.isinf(load_config(overrides={"alpha": 0.0}).tau_rev)  # explicit inf branch


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown canonical config keys"):
        CanonicalConfig.from_dict({"n_robots": 4, "bogus": 1})


def test_decision_dt_must_be_multiple_of_dt():
    with pytest.raises(ValueError, match="multiple of dt"):
        CanonicalConfig(dt=0.05, decision_dt=0.53)


def test_counts_within_padding_bounds():
    with pytest.raises(ValueError, match="exceeds its padding bound"):
        CanonicalConfig(n_tasks=30, max_tasks=20)


@pytest.mark.parametrize("field,value", [
    ("contest_mode", "bogus"), ("weight_dist", "bogus"), ("layout", "bogus"),
])
def test_enum_fields_validated(field, value):
    with pytest.raises(ValueError):
        CanonicalConfig(**{field: value})
