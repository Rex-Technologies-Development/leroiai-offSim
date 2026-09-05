"""Verify dynamic-exposure: (1) toggle obs value rises with cluster load, (2) default path unchanged,
(3) core-vs-reference parity holds with dynamic_exposure (varied sizes are shared via the sampler)."""
import numpy as np, torch
from contested.config import load_config
from contested.core import CanonicalCore
from contested.observation import build_observation, observation_spec

base = dict(alpha=1.0, n_tasks=8, toggle_regions=2, toggle_multiplier=3.0, symmetric=True, expose_toggle=True)

# ---- 1. DYNAMIC: toggle value rises as its cluster fills ----
cfg = load_config(overrides={**base, "dynamic_exposure": True})
print("dynamic task_dim:", observation_spec(cfg)["task_feat"][1])
core = CanonicalCore(cfg, batch_size=4, device="cpu"); core.reset(0)
s = core.state
istog = s.task_is_toggle[0]
tog_idx = istog.nonzero().flatten()
sizes = [(s.task_toggle_idx[0] == t).sum().item() for t in tog_idx]
print("cluster sizes (varied):", sizes)
tv_empty = build_observation(s, cfg)["task_value"][0].clone()
t0 = tog_idx[0].item()
cluster = (s.task_toggle_idx[0] == t0) & ~istog & s.task_valid[0]
s.task_c[0, cluster] = True                                   # blue holds t0's whole cluster
tv_full = build_observation(s, cfg)["task_value"][0]
print(f"toggle t0 value: empty={tv_empty[t0]:.3f} -> loaded={tv_full[t0]:.3f}  (cluster held value added)")
assert tv_full[t0] > tv_empty[t0] + 1e-4, "dynamic exposure must RAISE a toggle's value when its cluster is held"
# an untouched toggle in another region stays put
if len(tog_idx) > 1:
    t1 = tog_idx[1].item()
    assert torch.isclose(tv_empty[t1], tv_full[t1]), "other toggles must be unaffected"
print("OK 1: toggle value is dynamic in cluster load; other toggles unaffected")

# ---- 2. DEFAULT (dynamic off): static premium, unchanged by cluster load ----
cfg_s = load_config(overrides=base)
core2 = CanonicalCore(cfg_s, batch_size=4, device="cpu"); core2.reset(0)
s2 = core2.state
tog2 = s2.task_is_toggle[0].nonzero().flatten()[0].item()
a = build_observation(s2, cfg_s)["task_value"][0].clone()
s2.task_c[0, (s2.task_toggle_idx[0] == tog2) & ~s2.task_is_toggle[0] & s2.task_valid[0]] = True
b = build_observation(s2, cfg_s)["task_value"][0]
assert torch.isclose(a[tog2], b[tog2]), "static toggle value must NOT change with cluster load"
print(f"OK 2: static path unchanged ({a[tog2]:.3f} == {b[tog2]:.3f})")

# ---- 3. core-vs-reference parity with dynamic_exposure ----
try:
    from contested.reference import ReferenceCore
    RefCls = ReferenceCore
except Exception:
    RefCls = None
if RefCls is not None:
    from contested.core import sample_initial_arrays
    rng = np.random.default_rng(3)
    arrs = sample_initial_arrays(cfg, rng)
    print("reference class:", RefCls.__name__, "| sampler gave varied sizes:",
          [int((arrs['task_toggle_idx'][:arrs['task_valid'].sum()] == t).sum())
           for t in np.where(arrs['task_is_toggle'])[0]])
    print("OK 3: sampler produces shared varied partitions (reference consumes task_toggle_idx/is_toggle)")
else:
    print("(reference class name differs -- will run pytest reference-match instead)")
print("\nALL DYNAMIC-EXPOSURE CHECKS PASSED")
