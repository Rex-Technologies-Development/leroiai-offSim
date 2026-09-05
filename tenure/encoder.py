"""Heterogeneous graph encoder (plan Section 5).

Type-specific input projections feed a stack of masked multi-head attention layers
over the combined, padded node set ``[robots; tasks; adversaries]``. The graph's
edge features (``e_rt``/``e_at``/``e_tt``) enter as a learned per-head **attention
bias**, so the geometry the environment precomputed (travel time, horizon factor,
threat, task proximity) shapes attention directly.

No positional encodings are used, so the encoder is permutation-equivariant within
each node type — a property the tests check end to end. Dense attention is used
deliberately (plan Section 1.3): at N, M, K <= 20 the graph is effectively complete
and dense masked attention beats a GNN library with no extra dependency.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``x`` (B, N, D) over valid rows given ``mask`` (B, N) bool -> (B, D)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class _EncoderLayer(nn.Module):
    """Pre-norm transformer block with masked attention + additive edge bias."""

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, bias: Tensor) -> Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, S, self.h, self.dh).transpose(1, 2)   # (B, H, S, dh)
        k = self.k(h).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.v(h).view(B, S, self.h, self.dh).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.dh) + bias  # (B, H, S, S)

        # mask invalid keys; always allow the diagonal so no row is fully -inf
        eye = torch.eye(S, dtype=torch.bool, device=x.device).view(1, 1, S, S)
        allow = valid.view(B, 1, 1, S) | eye
        logits = logits.masked_fill(~allow, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.o(out))
        x = x + self.ff(self.norm2(x))
        return x


class HeteroEncoder(nn.Module):
    """Encodes the dict observation into per-type node embeddings."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 ff: int = 256, dropout: float = 0.0,
                 robot_dim: int = 9, task_dim: int = 7, adv_dim: int = 6, scalar_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        self.robot_in = nn.Linear(robot_dim, d_model)
        self.task_in = nn.Linear(task_dim, d_model)
        self.adv_in = nn.Linear(adv_dim, d_model)
        self.scalar_in = nn.Linear(scalar_dim, d_model)
        self.type_emb = nn.Parameter(torch.zeros(3, d_model))
        self.edge_proj = nn.Linear(3, n_heads)                       # 3 combined edge channels -> per-head bias
        self.layers = nn.ModuleList([_EncoderLayer(d_model, n_heads, ff, dropout) for _ in range(n_layers)])

    def _edge_bias(self, obs: dict, R: int, T: int, K: int) -> Tensor:
        e_rt, e_at, e_tt = obs["e_rt"], obs["e_at"], obs["e_tt"]
        B = e_rt.shape[0]
        S = R + T + K
        E = torch.zeros(B, S, S, 3, device=e_rt.device, dtype=e_rt.dtype)
        E[:, 0:R, R:R + T, 0:2] = e_rt                               # robot -> task
        E[:, R:R + T, 0:R, 0:2] = e_rt.transpose(1, 2)               # task -> robot (symmetric)
        E[:, R + T:S, R:R + T, 0:2] = e_at                           # adv -> task
        E[:, R:R + T, R + T:S, 0:2] = e_at.transpose(1, 2)           # task -> adv
        E[:, R:R + T, R:R + T, 2:3] = e_tt                           # task <-> task
        return self.edge_proj(E).permute(0, 3, 1, 2).contiguous()    # (B, H, S, S)

    def forward(self, obs: dict) -> tuple[Tensor, Tensor, Tensor]:
        rv, tv, av = obs["robot_valid"], obs["task_valid"], obs["adv_valid"]
        R, T, K = rv.shape[1], tv.shape[1], av.shape[1]
        hr = self.robot_in(obs["robot_feat"]) + self.type_emb[0]
        ht = self.task_in(obs["task_feat"]) + self.type_emb[1]
        ha = self.adv_in(obs["adv_feat"]) + self.type_emb[2]
        g = self.scalar_in(obs["scalar"]).unsqueeze(1)               # (B, 1, D) global token added to all
        x = torch.cat([hr, ht, ha], dim=1) + g
        valid = torch.cat([rv, tv, av], dim=1)                       # (B, S) bool
        bias = self._edge_bias(obs, R, T, K)
        for layer in self.layers:
            x = layer(x, valid, bias)
        hr, ht, ha = torch.split(x, [R, T, K], dim=1)
        # zero padded nodes so downstream pooling/heads never see garbage
        return (hr * rv.unsqueeze(-1), ht * tv.unsqueeze(-1), ha * av.unsqueeze(-1))
