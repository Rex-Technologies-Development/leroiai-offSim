"""Component C: the TENURE model.

A heterogeneous graph encoder, a retention head, and a masked decoder in which
retention enters the allocation score *multiplicatively* (so ``R_hat === 1``
exactly recovers a conventional allocator). Trained with PPO plus a retention
regression auxiliary (paper Eq. 14).

See ``tenure-sim-development-plan.md`` Section 5.
"""
from __future__ import annotations

from .encoder import HeteroEncoder
from .heads import Decoder, RetentionHead
from .policy import TenurePolicy

__all__ = ["HeteroEncoder", "RetentionHead", "Decoder", "TenurePolicy"]
