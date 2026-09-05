"""Retention head and masked decoder (plan Section 5).

The **decoder is the contribution**: retention enters the ``ACQUIRE``/``DEFEND``
score as an explicit multiplicative factor on task value ``w_i`` (paper Eq. 10),
not merely as another MLP input. Two reasons the plan is emphatic about this:

1. It makes the ablation *causal* — with ``retention_mode="off"`` the factor is
   fixed to 1 and the score is exactly a conventional value/geometry allocator, so
   ``R_hat === 1`` is literally true in code.
2. It stops the encoder from silently relearning retention as geometry, which would
   make "no retention head" look deceptively fine.

``retention_mode`` selects ``multiplicative`` (Eq. 10), ``feature`` (R_hat as an
additive learned input), or ``off`` (no retention) so all three can be compared.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .encoder import masked_mean

RETENTION_MODES = ("multiplicative", "feature", "off")
RETENTION_HEADS = ("regression", "classification")


class RetentionHead(nn.Module):
    """Predicts ``R_hat`` per task from the task embedding + pooled adversary context.

    ``kind="regression"`` (default, TENURE's head): a scalar sigmoid ``R_hat`` trained
    with MSE. ``kind="classification"`` (paper Table III ablation): ``R_hat`` is the mean
    of a categorical over ``n_bins`` retention bins, trained with cross-entropy against the
    binned label. The decoder still consumes a single scalar ``R_hat`` (the distribution's
    expected value), so the classification head is a drop-in swap that only changes how
    retention is *represented and trained*, keeping the multiplicative gate comparable.
    ``forward`` returns ``(r_hat, r_logits)`` — ``r_logits`` is ``None`` for regression and
    the ``(B, T, n_bins)`` bin logits for classification (consumed by :meth:`retention_loss`).
    """

    def __init__(self, d_model: int, hidden: int = 128, kind: str = "regression", n_bins: int = 8):
        super().__init__()
        assert kind in RETENTION_HEADS, kind
        self.kind = kind
        self.n_bins = n_bins
        out_dim = 1 if kind == "regression" else n_bins
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        # Warm-start R_hat ~ 1: zero the last layer's weights and bias the output high so
        # R_hat saturates near 1 at init. With multiplicative gating this makes the policy
        # START as a conventional value/geometry allocator (paper: R_hat == 1 recovers it)
        # and LEARN to pull retention down, instead of a cold start where a small R_hat zeros
        # out every task value -> idle -> no completions -> no labels -> no learning signal.
        nn.init.zeros_(self.net[-1].weight)
        if kind == "regression":
            nn.init.constant_(self.net[-1].bias, 3.0)              # sigmoid(3) ~= 0.953
            self.bin_centers = None
        else:
            centers = (torch.arange(n_bins).float() + 0.5) / n_bins  # (n_bins,) in (0, 1)
            self.register_buffer("bin_centers", centers)
            bias = torch.zeros(n_bins); bias[-1] = 5.0            # softmax mass -> top bin -> E[R] ~ 0.9
            with torch.no_grad():
                self.net[-1].bias.copy_(bias)

    def forward(self, h_task: Tensor, h_adv: Tensor, adv_mask: Tensor):
        pooled = masked_mean(h_adv, adv_mask)                       # (B, D)
        T = h_task.shape[1]
        x = torch.cat([h_task, pooled.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        z = self.net(x)                                            # (B, T, 1) or (B, T, n_bins)
        if self.kind == "regression":
            return torch.sigmoid(z).squeeze(-1), None             # r_hat (B, T), no logits
        r_hat = (torch.softmax(z, dim=-1) * self.bin_centers).sum(-1)  # expected value (B, T)
        return r_hat, z                                           # r_hat + bin logits (for CE)

    def retention_loss(self, r_hat: Tensor, r_logits: Optional[Tensor],
                       target_R: Tensor, mask: Tensor) -> Tensor:
        """Masked retention loss: MSE (regression) or bin cross-entropy (classification).
        Caller guarantees ``mask.any()``."""
        if self.kind == "regression":
            return (r_hat[mask] - target_R[mask]).pow(2).mean()
        bins = (target_R * self.n_bins).clamp(min=0.0, max=self.n_bins - 1).long()  # (B, T) bin idx
        return F.cross_entropy(r_logits[mask], bins[mask])


class Decoder(nn.Module):
    """Scores every ``(robot, action)`` pair; retention multiplies task value."""

    def __init__(self, d_model: int, retention_mode: str = "multiplicative", symmetric: bool = False):
        super().__init__()
        assert retention_mode in RETENTION_MODES
        self.retention_mode = retention_mode
        self.symmetric = symmetric
        self.d = d_model
        self.q_acq = nn.Linear(d_model, d_model)
        self.k_acq = nn.Linear(d_model, d_model)
        self.q_def = nn.Linear(d_model, d_model)
        self.k_def = nn.Linear(d_model, d_model)
        self.edge_acq = nn.Linear(2, 1)                            # e_rt -> acquire compat bias
        self.edge_def = nn.Linear(2, 1)
        self.idle = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # used only in "feature" mode: R_hat as an additive learned input
        self.rfeat_acq = nn.Linear(1, 1)
        self.rfeat_def = nn.Linear(1, 1)
        if symmetric:                                             # PHASE 2/3: REVERSE a red-held task
            self.q_rev = nn.Linear(d_model, d_model)
            self.k_rev = nn.Linear(d_model, d_model)
            self.edge_rev = nn.Linear(2, 1)
            self.rfeat_rev = nn.Linear(1, 1)

    def _compat(self, q_lin, k_lin, edge_lin, h_robot, h_task, e_rt) -> Tensor:
        q = q_lin(h_robot)                                        # (B, R, D)
        k = k_lin(h_task)                                         # (B, T, D)
        compat = q @ k.transpose(-2, -1) / math.sqrt(self.d)     # (B, R, T)
        return compat + edge_lin(e_rt).squeeze(-1)               # + geometry bias from e_rt

    def forward(self, h_robot: Tensor, h_task: Tensor, r_hat: Tensor, e_rt: Tensor,
                w_norm: Tensor, mode: Optional[str] = None) -> Tensor:
        """Returns raw ``(B, R, A)`` logits (A = 2T+1). Masking is applied by the policy."""
        mode = mode or self.retention_mode
        compat_a = self._compat(self.q_acq, self.k_acq, self.edge_acq, h_robot, h_task, e_rt)
        compat_d = self._compat(self.q_def, self.k_def, self.edge_def, h_robot, h_task, e_rt)

        if mode == "multiplicative":
            eff = r_hat                                           # Eq. 10: value scaled by retention
        else:
            eff = torch.ones_like(r_hat)                          # "off" and "feature": no value scaling
        value = (w_norm * eff).unsqueeze(1)                      # (B, 1, T)
        acquire = compat_a * value
        defend = compat_d * value
        if mode == "feature":                                    # R_hat as an additive learned feature
            acquire = acquire + self.rfeat_acq(r_hat.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            defend = defend + self.rfeat_def(r_hat.unsqueeze(-1)).squeeze(-1).unsqueeze(1)

        idle = self.idle(h_robot)                                # (B, R, 1)
        if self.symmetric:                                       # REVERSE block: take a red-held task
            compat_r = self._compat(self.q_rev, self.k_rev, self.edge_rev, h_robot, h_task, e_rt)
            reverse = compat_r * value
            if mode == "feature":
                reverse = reverse + self.rfeat_rev(r_hat.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
            return torch.cat([acquire, defend, reverse, idle], dim=-1)   # (B, R, 3T+1)
        return torch.cat([acquire, defend, idle], dim=-1)        # (B, R, 2T+1)
