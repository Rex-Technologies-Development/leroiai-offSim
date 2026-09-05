"""Headless smoke tests for the renderer and scripted demo policy."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore
from contested.actions import action_masks
from contested.render import CanonicalRenderer, ensure_headless


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4, horizon_T=5.0)
    base.update(kw)
    return CanonicalConfig(**base)


def test_render_headless_smoke():
    ensure_headless()
    cfg = _cfg()
    core = CanonicalCore(cfg, batch_size=2, device="cpu")
    core.reset(seed=0)
    core.state.task_c[:, 0] = True                       # exercise the completed-task path
    renderer = CanonicalRenderer(cfg, field_px=320, panel_px=200)
    surf = renderer.draw(core.state, b=0, info={"n_reversals": 3})
    assert surf.get_size() == renderer.size


def test_scripted_policy_is_legal():
    ensure_headless()
    from contested.demo import scripted_ally_actions
    cfg = _cfg()
    core = CanonicalCore(cfg, batch_size=4, device="cpu")
    core.reset(seed=1)
    core.state.task_c[:, :2] = True                      # some defendable tasks
    actions = scripted_ally_actions(core)
    assert actions.shape == (4, cfg.max_robots)
    assert (actions >= 0).all() and (actions < cfg.action_dim).all()
    mask = action_masks(core.state, cfg)
    legal = mask.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    assert legal.all(), "scripted policy chose a masked action"
