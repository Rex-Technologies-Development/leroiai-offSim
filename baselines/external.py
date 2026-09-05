"""External-code baselines: CapAM, RTAW, DC-MRTA (plan Section 6).

These wrap published models and are typed stubs until their code is vendored. Each
satisfies :class:`BaselinePolicy`, so once implemented they drop straight into
``baselines.evaluate`` and the experiment runner with no other changes.

Availability (VERIFIED 2026-08-27 by web research):
- **CapAM** — PUBLIC: github.com/adamslab-ub/CapAM-MRTA (PyTorch, ICRA'22, arXiv:2205.03321).
  Vendorable as-is; LOW risk (deps pinned to old CUDA 9.1 -> minor bump). ~4d to wrap.
- **RTAW** — PUBLIC: github.com/Aakriti05/RTAW-Centralised-multi-robot-task-allocation
  (PyTorch, ICRA'23, arXiv:2209.05738). Method is public; the large-scale (500-1000 robot)
  + ORCA-navigation code is NOT included. LOW-MED risk.
- **DC-MRTA** — NO PUBLIC CODE (IROS'22, arXiv:2209.02865): reimplement (~5-8d, HIGH risk),
  daggered results row. WARNING: github.com/marmotlab/DCMRTA is a DIFFERENT paper
  (Dynamic Coalition Formation, ICRA'24) — do NOT vendor it as this baseline.

Schedule: 1 of 3 (DC-MRTA) needs reimplementation; CapAM + RTAW are vendorable.

Fair adaptation (plan Section 6) still applies: feed adversary features, retrain from
scratch in the contested environment, match sample/wall-clock budget in the runner.
"""
from __future__ import annotations

from torch import Tensor

from contested.config import CanonicalConfig
from contested.core import CanonicalState
from .base import BaselinePolicy


class _ExternalStub(BaselinePolicy):
    name = "external"
    paper = ""
    repo = "TODO: vendor upstream code"
    status = "not yet vendored"

    def scores(self, state: CanonicalState, cfg: CanonicalConfig) -> Tensor:
        raise NotImplementedError(
            f"{self.name}: {self.status}\n  paper: {self.paper}\n  repo: {self.repo}\n"
            "  Vendor the upstream code and implement scores()/act() over CanonicalState."
        )


class CapAM(_ExternalStub):
    name = "capam"
    paper = "Learning Scalable Policies over Graphs for MRTA using Capsule Attention Networks (ICRA'22, arXiv:2205.03321)"
    repo = "https://github.com/adamslab-ub/CapAM-MRTA"
    status = "PUBLIC (PyTorch); vendor as-is, wrap as receding-horizon replanner (~4d, low risk)"


class RTAW(_ExternalStub):
    name = "rtaw"
    paper = "RTAW: An Attention Inspired RL Method for MRTA in Warehouse Environments (ICRA'23, arXiv:2209.05738)"
    repo = "https://github.com/Aakriti05/RTAW-Centralised-multi-robot-task-allocation"
    status = "PUBLIC (PyTorch); method usable, large-scale/ORCA code missing (low-med risk)"


class DCMRTA(_ExternalStub):
    name = "dc_mrta"
    paper = "DC-MRTA: Decentralized MRTA (IROS'22, arXiv:2209.02865)"
    repo = "NONE FOUND — reimplement. NB: marmotlab/DCMRTA is a DIFFERENT paper (ICRA'24), do not vendor"
    status = "NO PUBLIC CODE; reimplement + dagger the results row (5-8d, HIGH risk)"
