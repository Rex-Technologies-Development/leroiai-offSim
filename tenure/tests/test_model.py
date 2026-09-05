"""TENURE model + PPO tests (plan Section 5)."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore
from contested.observation import build_observation
from tenure.heads import Decoder
from tenure.policy import TenurePolicy
from tenure.ppo import PPOConfig, PPOTrainer

F32 = torch.float32


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4, horizon_T=4.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _obs(cfg, batch=4, seed=0):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F32)
    core.reset(seed=seed)
    g = torch.Generator().manual_seed(seed)
    core.state.task_c = (torch.rand(core.state.task_c.shape, generator=g) < 0.4) & core.state.task_valid
    return build_observation(core.state, cfg)


def _policy(**kw):
    return TenurePolicy(d_model=32, n_heads=4, n_layers=2, ff=64, value_hidden=32, **kw).eval()


def test_policy_shapes_and_ranges():
    cfg = _cfg()
    obs = _obs(cfg, batch=4)
    out = _policy()(obs)
    assert out["logits"].shape == (4, cfg.max_robots, cfg.action_dim)
    assert out["value"].shape == (4,)
    assert out["r_hat"].shape == (4, cfg.max_tasks)
    assert (out["r_hat"] >= 0).all() and (out["r_hat"] <= 1).all()
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["value"]).all()


def test_classification_retention_head():
    """Table III ablation: the classification head is a drop-in — R_hat is still (B,T) in
    [0,1] and warm-started high — while exposing bin logits and a differentiable CE loss."""
    cfg = _cfg()
    obs = _obs(cfg, batch=4)
    pol = _policy(retention_head="classification")
    out = pol(obs)
    # same interface as regression: a scalar R_hat per task in [0,1], warm-started high
    assert out["r_hat"].shape == (4, cfg.max_tasks)
    assert (out["r_hat"] >= 0).all() and (out["r_hat"] <= 1).all()
    assert out["r_hat"].mean() > 0.7, "warm-start should keep R_hat high at init (avoids the idle trap)"
    # classification exposes (B,T,n_bins) logits; regression exposes none
    assert out["r_logits"] is not None and out["r_logits"].shape[-1] == pol.retention.n_bins
    assert _policy(retention_head="regression")(obs)["r_logits"] is None
    # head-owned retention loss is cross-entropy: finite, non-negative, and differentiable
    R = torch.rand(4, cfg.max_tasks)
    mask = obs["task_valid"].clone(); mask[:, 0] = True
    loss = pol.retention.retention_loss(out["r_hat"], out["r_logits"], R, mask)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in pol.retention.parameters())


def test_act_returns_legal_actions():
    cfg = _cfg()
    obs = _obs(cfg, batch=8)
    step = _policy().act(obs)
    assert step["action"].shape == (8, cfg.max_robots)
    legal = obs["action_mask"].bool().gather(2, step["action"].unsqueeze(-1)).squeeze(-1)
    assert legal[obs["robot_valid"]].all(), "policy sampled a masked action for a valid robot"


def test_retention_mode_off_recovers_conventional():
    """off-mode logits == multiplicative logits when R_hat===1; off ignores R_hat."""
    torch.manual_seed(0)
    B, R, T, d = 3, 4, 6, 32
    dec = Decoder(d).eval()
    h_robot = torch.randn(B, R, d)
    h_task = torch.randn(B, T, d)
    e_rt = torch.randn(B, R, T, 2)
    w = torch.rand(B, T)
    ones = torch.ones(B, T)
    r_rand = torch.rand(B, T)

    off = dec(h_robot, h_task, r_rand, e_rt, w, mode="off")
    mult_ones = dec(h_robot, h_task, ones, e_rt, w, mode="multiplicative")
    assert torch.allclose(off, mult_ones, atol=1e-6), "R_hat===1 must recover the conventional score"

    off2 = dec(h_robot, h_task, ones, e_rt, w, mode="off")
    assert torch.allclose(off, off2, atol=1e-6), "off mode must ignore R_hat"

    mult_rand = dec(h_robot, h_task, r_rand, e_rt, w, mode="multiplicative")
    assert not torch.allclose(mult_rand, mult_ones, atol=1e-4), "multiplicative mode must depend on R_hat"


def test_permutation_equivariance_in_robots():
    cfg = _cfg()
    obs = _obs(cfg, batch=2)
    policy = _policy()
    perm = torch.randperm(cfg.max_robots)
    permuted = dict(obs)
    permuted["robot_feat"] = obs["robot_feat"][:, perm]
    permuted["e_rt"] = obs["e_rt"][:, perm]
    permuted["action_mask"] = obs["action_mask"][:, perm]
    permuted["robot_valid"] = obs["robot_valid"][:, perm]

    with torch.no_grad():
        base = policy(obs)
        other = policy(permuted)
    assert torch.allclose(other["logits"], base["logits"][:, perm], atol=1e-5), "logits not robot-equivariant"
    assert torch.allclose(other["value"], base["value"], atol=1e-5), "value not robot-invariant"
    assert torch.allclose(other["r_hat"], base["r_hat"], atol=1e-5), "r_hat not robot-invariant"


def test_ppo_smoke_runs_and_updates_params():
    cfg = _cfg(n_robots=2, n_tasks=4, n_adversaries=2,
               max_robots=3, max_tasks=5, max_adversaries=3, horizon_T=1.5)
    policy = TenurePolicy(d_model=16, n_heads=2, n_layers=1, ff=32, value_hidden=32)
    trainer = PPOTrainer(cfg, policy, batch_size=8, device="cpu",
                         ppo=PPOConfig(epochs=2, minibatches=2))
    before = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
    metrics = list(trainer.train(updates=2, seed=0))
    assert len(metrics) == 2
    for m in metrics:
        for key in ("policy_loss", "value_loss", "retention_loss", "entropy", "rollout_reward"):
            assert torch.isfinite(torch.tensor(m[key])), f"{key} not finite: {m[key]}"
        assert m["retention_loss"] >= 0
    after = torch.cat([p.detach().flatten() for p in policy.parameters()])
    assert not torch.allclose(before, after), "PPO update did not change parameters"


def test_retention_mode_off_policy_ignores_retention_head():
    """A policy in off mode gives identical logits regardless of the retention head."""
    cfg = _cfg()
    obs = _obs(cfg, batch=2)
    policy = _policy(retention_mode="off")
    with torch.no_grad():
        base = policy(obs)["logits"]
        for p in policy.retention.parameters():   # perturb the retention head
            p.add_(torch.randn_like(p))
        after = policy(obs)["logits"]
    assert torch.allclose(base, after, atol=1e-6), "off-mode logits must not depend on the retention head"
