"""TENURE actor-critic policy (plan Section 5).

Ties the encoder, retention head, and multiplicative decoder into an actor that
emits per-robot action logits, plus a critic that reads a global pooled embedding.
The joint action factorises over robots (each robot draws from a masked categorical
over the flat action layout); log-probs and entropy sum over *valid* robots only.

Masked-action handling is done with an explicit ``log_softmax`` + zeroed entropy
terms rather than ``torch.distributions.Categorical`` so that ``-inf`` logits on
masked actions can never produce NaNs.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .encoder import HeteroEncoder, masked_mean
from .heads import Decoder, RetentionHead


class TenurePolicy(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, ff: int = 256,
                 dropout: float = 0.0, retention_mode: str = "multiplicative", value_hidden: int = 128,
                 task_dim: int = 7, retention_head: str = "regression", symmetric: bool = False):
        super().__init__()
        self.encoder = HeteroEncoder(d_model, n_heads, n_layers, ff, dropout, task_dim=task_dim)
        self.retention = RetentionHead(d_model, kind=retention_head)
        self.decoder = Decoder(d_model, retention_mode, symmetric=symmetric)
        self.value_head = nn.Sequential(
            nn.Linear(3 * d_model + 4, value_hidden), nn.GELU(), nn.Linear(value_hidden, 1))

    @property
    def retention_mode(self) -> str:
        return self.decoder.retention_mode

    def forward(self, obs: dict, mode: Optional[str] = None) -> dict:
        hr, ht, ha = self.encoder(obs)
        r_hat, r_logits = self.retention(ht, ha, obs["adv_valid"])
        # value channel the decoder gates by retention: the toggle-EFFECTIVE value when the obs
        # provides it (M8 aware policy), else base weight w/w_max at task_feat[...,2].
        w_norm = obs["task_value"] if "task_value" in obs else obs["task_feat"][..., 2]
        logits = self.decoder(hr, ht, r_hat, obs["e_rt"], w_norm, mode)
        pooled = torch.cat([
            masked_mean(hr, obs["robot_valid"]),
            masked_mean(ht, obs["task_valid"]),
            masked_mean(ha, obs["adv_valid"]),
        ], dim=-1)
        value = self.value_head(torch.cat([pooled, obs["scalar"]], dim=-1)).squeeze(-1)
        return {"logits": logits, "value": value, "r_hat": r_hat, "r_logits": r_logits}

    # ------------------------------------------------------------ distribution
    @staticmethod
    def _masked_log_probs(logits: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        ml = logits.masked_fill(~action_mask.bool(), float("-inf"))
        log_p = torch.log_softmax(ml, dim=-1)                     # (B, R, A); masked -> -inf
        return log_p, log_p.exp()                                 # probs: masked -> 0

    def act(self, obs: dict, deterministic: bool = False) -> dict:
        out = self.forward(obs)
        log_p, p = self._masked_log_probs(out["logits"], obs["action_mask"])
        if deterministic:
            action = p.argmax(dim=-1)
        else:
            action = torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])
        lp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)   # (B, R)
        rv = obs["robot_valid"].to(lp.dtype)
        return {"action": action, "log_prob": (lp * rv).sum(-1), "value": out["value"], "r_hat": out["r_hat"]}

    def evaluate(self, obs: dict, action: Tensor, mode: Optional[str] = None) -> dict:
        out = self.forward(obs, mode)
        log_p, p = self._masked_log_probs(out["logits"], obs["action_mask"])
        lp = log_p.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        rv = obs["robot_valid"].to(lp.dtype)
        # zero the -inf log-probs of masked actions BEFORE multiplying: p is already 0
        # there, so p * 0 = 0. (p * (-inf) = NaN would poison the backward pass even
        # though torch.where discards it in the forward pass.)
        log_p_safe = log_p.masked_fill(~obs["action_mask"].bool(), 0.0)
        entropy = -(p * log_p_safe).sum(-1)                       # (B, R)
        return {
            "log_prob": (lp * rv).sum(-1),
            "entropy": (entropy * rv).sum(-1),
            "value": out["value"],
            "r_hat": out["r_hat"],
            "r_logits": out["r_logits"],
            "logits": out["logits"],
        }
