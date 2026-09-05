"""Observation builder tests (plan Sections 3.7, 3.13)."""
from __future__ import annotations

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore
from contested.observation import build_observation, observation_spec

F64 = torch.float64


def _cfg(**kw) -> CanonicalConfig:
    base = dict(n_robots=4, n_tasks=6, n_adversaries=3,
                max_robots=6, max_tasks=8, max_adversaries=4, horizon_T=10.0)
    base.update(kw)
    return CanonicalConfig(**base)


def _obs_after(cfg, batch=8, seed=0, n_dec=2):
    core = CanonicalCore(cfg, batch_size=batch, device="cpu", dtype=F64)
    core.reset(seed=seed)
    # make it nontrivial: some completions, then a couple decisions
    core.state.task_c = (torch.rand(core.state.task_c.shape, generator=torch.Generator().manual_seed(seed)) < 0.3) & core.state.task_valid
    idle = torch.full((batch, cfg.max_robots), cfg.idle_action, dtype=torch.int64)
    for _ in range(n_dec):
        core.step(idle)
    return core, build_observation(core.state, cfg)


def test_expose_protected_adds_flag_as_last_task_feature():
    # default: 7 task features, no protected flag exposed
    cfg7 = _cfg(protected_fraction=0.4)
    _, obs7 = _obs_after(cfg7, batch=8)
    assert obs7["task_feat"].shape[-1] == 7
    assert observation_spec(cfg7)["task_feat"] == (cfg7.max_tasks, 7)

    # exposed: 8th feature is exactly the (valid-masked) protected flag
    cfg8 = _cfg(protected_fraction=0.4, expose_protected=True)
    core, obs8 = _obs_after(cfg8, batch=8)
    assert obs8["task_feat"].shape[-1] == 8
    assert observation_spec(cfg8)["task_feat"] == (cfg8.max_tasks, 8)
    expected = (core.state.task_protected & core.state.task_valid).to(F64)
    assert torch.equal(obs8["task_feat"][..., 7], expected)
    # the first 7 features are unchanged by exposing the flag
    assert torch.allclose(obs8["task_feat"][..., :7], obs7["task_feat"])


def test_ablate_adversary_nodes_blinds_policy_to_adversaries():
    """Table III ablation: adversary nodes/edges are zeroed and marked invalid, while
    every non-adversary feature is untouched (same env, obs shapes unchanged)."""
    _, obs_on = _obs_after(_cfg(ablate_adversary_nodes=True), batch=8, seed=1)
    _, obs_off = _obs_after(_cfg(ablate_adversary_nodes=False), batch=8, seed=1)
    # adversary information is gone
    assert torch.equal(obs_on["adv_feat"], torch.zeros_like(obs_on["adv_feat"]))
    assert torch.equal(obs_on["e_at"], torch.zeros_like(obs_on["e_at"]))
    assert not obs_on["adv_valid"].any()
    # shapes unchanged so a single set of weights still runs
    assert obs_on["adv_feat"].shape == obs_off["adv_feat"].shape
    # everything NOT about adversaries is bit-identical
    for key in ("robot_feat", "task_feat", "e_rt", "e_tt", "scalar", "task_valid", "robot_valid"):
        assert torch.equal(obs_on[key], obs_off[key]), f"{key} changed by the adversary ablation"
    # sanity: without the ablation adversaries ARE visible
    assert obs_off["adv_valid"].any() and obs_off["adv_feat"].abs().sum() > 0


def test_observation_shapes_and_finiteness():
    cfg = _cfg()
    core, obs = _obs_after(cfg, batch=8)
    spec = observation_spec(cfg)
    for key, shape in spec.items():
        assert obs[key].shape == (8, *shape), f"{key} shape {tuple(obs[key].shape)} != {(8, *shape)}"

    for key in ("robot_feat", "task_feat", "adv_feat", "e_rt", "e_at", "e_tt", "scalar"):
        assert torch.isfinite(obs[key]).all(), f"{key} has non-finite values"
    assert obs["action_mask"].dtype == torch.bool
    for key in ("robot_valid", "task_valid", "adv_valid"):
        assert obs[key].dtype == torch.bool

    # ranges
    rv = core.state.robot_valid
    assert (obs["robot_feat"][rv][:, :2] >= 0).all() and (obs["robot_feat"][rv][:, :2] <= 1).all()
    c_channel = obs["task_feat"][..., 3]
    assert torch.logical_or(c_channel == 0, c_channel == 1).all(), "c channel not binary"
    assert (obs["e_rt"][..., 1] >= 0).all() and (obs["e_rt"][..., 1] <= 1).all(), "horizon factor out of [0,1]"
    assert (obs["e_at"][..., 0] >= 0).all() and (obs["e_at"][..., 0] <= 1).all(), "threat out of [0,1]"
    s = obs["scalar"]
    assert (s[:, 0] >= 0).all() and (s[:, 0] <= 1).all()          # t/T
    assert (s[:, 1] >= 0).all() and (s[:, 1] <= 1).all()          # fraction complete
    assert torch.allclose(s[:, 3], torch.full_like(s[:, 3], cfg.alpha))  # alpha


def test_alpha_zero_features_finite():
    """tau_rev = inf must not produce nan/inf in eta or threat features."""
    cfg = _cfg(alpha=0.0)
    core, obs = _obs_after(cfg, batch=8)
    assert torch.isfinite(obs["task_feat"]).all()
    assert torch.isfinite(obs["e_at"]).all()
    assert torch.equal(obs["task_feat"][..., 5], torch.zeros_like(obs["task_feat"][..., 5])), "eta/inf should be 0"


def test_padding_invariance():
    """Padding to larger max_robots/max_adversaries leaves valid sub-blocks unchanged."""
    common = dict(n_robots=4, n_tasks=6, n_adversaries=3, max_tasks=8, horizon_T=10.0)
    cfg_a = CanonicalConfig(max_robots=6, max_adversaries=4, **common)
    cfg_b = CanonicalConfig(max_robots=10, max_adversaries=7, **common)
    nr, nt, nk = 4, 6, 3
    B, seed = 8, 42

    core_a = CanonicalCore(cfg_a, batch_size=B, device="cpu", dtype=F64); core_a.reset(seed=seed)
    core_b = CanonicalCore(cfg_b, batch_size=B, device="cpu", dtype=F64); core_b.reset(seed=seed)

    def valid_blocks(core, cfg):
        o = build_observation(core.state, cfg)
        return {
            "robot_feat": o["robot_feat"][:, :nr],
            "task_feat": o["task_feat"][:, :nt],
            "adv_feat": o["adv_feat"][:, :nk],
            "e_rt": o["e_rt"][:, :nr, :nt],
            "e_at": o["e_at"][:, :nk, :nt],
            "e_tt": o["e_tt"][:, :nt, :nt],
            "scalar": o["scalar"],
            "action_mask": o["action_mask"][:, :nr, :],
        }

    idle_a = torch.full((B, cfg_a.max_robots), cfg_a.idle_action, dtype=torch.int64)
    idle_b = torch.full((B, cfg_b.max_robots), cfg_b.idle_action, dtype=torch.int64)
    assert cfg_a.idle_action == cfg_b.idle_action  # max_tasks held fixed

    for _ in range(3):
        for key, va in valid_blocks(core_a, cfg_a).items():
            vb = valid_blocks(core_b, cfg_b)[key]
            if va.dtype == torch.bool:
                assert torch.equal(va, vb), f"{key} padding-variant (bool)"
            else:
                assert torch.allclose(va, vb, atol=1e-10, rtol=0), f"{key} padding-variant (max {torch.abs(va-vb).max():.2e})"
        core_a.step(idle_a)
        core_b.step(idle_b)


def test_observation_runs_on_default_device():
    """Build obs on whatever default_device() is (CUDA on the RTX box)."""
    cfg = _cfg()
    core = CanonicalCore(cfg, batch_size=4)   # default device + float32
    core.reset(seed=1)
    obs = build_observation(core.state, cfg)
    assert obs["robot_feat"].device == core.state.device
    assert torch.isfinite(obs["e_rt"]).all()


def test_dynamic_exposure_prices_toggle_by_current_cluster_load():
    """dynamic_exposure: a toggle's obs value rises with the value its cluster CURRENTLY holds and
    collapses when empty -- the continuous, state-dependent spectrum. The static default is unchanged.
    (Dynamics are flag-independent -> the 400-seed reference-match already covers parity.)"""
    cfg = _cfg(symmetric=True, toggle_regions=2, toggle_multiplier=3.0,
               expose_toggle=True, dynamic_exposure=True)
    core = CanonicalCore(cfg, batch_size=4, device="cpu", dtype=F64)
    core.reset(seed=0)
    s = core.state
    s.task_c = torch.zeros_like(s.task_c)                      # start clusters empty
    istog = s.task_is_toggle[0]
    tog = istog.nonzero().flatten()
    t0 = tog[0].item()
    v_empty = build_observation(s, cfg)["task_value"][0, t0].item()
    cluster = (s.task_toggle_idx[0] == t0) & ~istog & s.task_valid[0]
    s.task_c[0, cluster] = True                                # blue holds t0's whole cluster
    v_full = build_observation(s, cfg)["task_value"][0, t0].item()
    assert v_full > v_empty + 1e-6, f"dynamic toggle value must rise with cluster load ({v_empty}->{v_full})"

    # STATIC default path: fixed premium, unchanged by cluster load
    cfg_s = _cfg(symmetric=True, toggle_regions=2, toggle_multiplier=3.0, expose_toggle=True)
    core2 = CanonicalCore(cfg_s, batch_size=4, device="cpu", dtype=F64)
    core2.reset(seed=0)
    s2 = core2.state
    tg = s2.task_is_toggle[0].nonzero().flatten()[0].item()
    a = build_observation(s2, cfg_s)["task_value"][0, tg].item()
    s2.task_c[0, (s2.task_toggle_idx[0] == tg) & ~s2.task_is_toggle[0] & s2.task_valid[0]] = True
    b = build_observation(s2, cfg_s)["task_value"][0, tg].item()
    assert abs(a - b) < 1e-9, "static toggle value must not change with cluster load"
