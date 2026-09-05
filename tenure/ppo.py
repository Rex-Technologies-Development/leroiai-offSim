"""PPO trainer with the retention auxiliary (plan Section 5; paper Eq. 14).

Loss = clipped policy objective + ``lam_v`` * value loss + ``lam_r`` * retention
regression + ``lam_e`` * entropy bonus. The retention target comes from
``contested.labels`` and is realised under the *current* behaviour policy, so the
label buffer is rebuilt from the fresh rollout every update and never accumulated
across updates (plan Section 3.9).

Whole episodes are collected per update (labels need the full remainder of the
episode), driving the batched :class:`~contested.core.CanonicalCore` directly on
device for speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from contested.config import CanonicalConfig
from contested.core import CanonicalCore, default_device
from contested.labels import RetentionLabelRecorder
from contested.observation import build_observation
from .policy import TenurePolicy


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lam_v: float = 0.5          # value loss weight
    lam_r: float = 0.5          # retention regression weight (tune this first)
    lam_e: float = 0.01         # entropy bonus weight
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(self, cfg: CanonicalConfig, policy: Optional[TenurePolicy] = None,
                 batch_size: int = 256, device=None, dtype: torch.dtype = torch.float32,
                 ppo: Optional[PPOConfig] = None):
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self.pp = ppo or PPOConfig()
        self.B = batch_size
        self.core = CanonicalCore(cfg, batch_size=batch_size, device=self.device, dtype=dtype)
        self.policy = (policy or TenurePolicy()).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.pp.lr)

    # ------------------------------------------------------------- collection
    @torch.no_grad()
    def collect(self, seed: Optional[int] = None) -> dict:
        core = self.core
        core.reset(seed)
        rec = RetentionLabelRecorder()
        D = self.cfg.n_decisions
        obs_seq, act_seq, logp_seq, val_seq, rew_seq, rhat_seq = [], [], [], [], [], []
        for _ in range(D):
            obs = build_observation(core.state, self.cfg)
            rec.before_decision(core)
            step = self.policy.act(obs)
            reward, _done, _info = core.step(step["action"])
            rec.after_decision(core)
            obs_seq.append(obs)
            act_seq.append(step["action"])
            logp_seq.append(step["log_prob"])
            val_seq.append(step["value"])
            rew_seq.append(reward.to(self.dtype))
            rhat_seq.append(step["r_hat"])
        labels = rec.finalize(core)
        returns, adv = self._gae(rew_seq, val_seq)
        telemetry = core.telemetry.summary(core.state) if core.telemetry is not None else {}
        rollout = {
            "obs_seq": obs_seq, "act": torch.cat(act_seq, 0),
            "log_prob": torch.cat(logp_seq, 0), "value": torch.cat(val_seq, 0),
            "returns": returns.reshape(-1), "adv": adv.reshape(-1),
            "labels": labels, "telemetry": telemetry,
            "rollout_reward": torch.stack(rew_seq, 0).sum(0).mean().item(),
            "rollout_retention_mae": self._label_mae(torch.stack(rhat_seq, 1), labels),
        }
        return rollout

    def _gae(self, rewards: list, values: list) -> tuple[torch.Tensor, torch.Tensor]:
        D, B = len(rewards), rewards[0].shape[0]
        adv = torch.zeros(D, B, device=self.device, dtype=self.dtype)
        last, next_value = torch.zeros(B, device=self.device, dtype=self.dtype), torch.zeros(B, device=self.device, dtype=self.dtype)
        for t in reversed(range(D)):
            delta = rewards[t] + self.pp.gamma * next_value - values[t]
            last = delta + self.pp.gamma * self.pp.gae_lambda * last
            adv[t] = last
            next_value = values[t]
        returns = adv + torch.stack(values, 0)
        return returns, adv

    @staticmethod
    def _label_mae(rhat_bdt: torch.Tensor, labels: dict) -> float:
        mask = labels["dense_mask"] | labels["event_mask"]
        if not mask.any():
            return float("nan")
        return (rhat_bdt[mask] - labels["R"][mask]).abs().mean().item()

    # ------------------------------------------------------------------ update
    def update(self, roll: dict) -> dict:
        cfg = self.cfg
        D, B = cfg.n_decisions, self.B
        N = D * B
        keys = roll["obs_seq"][0].keys()
        flat_obs = {k: torch.cat([o[k] for o in roll["obs_seq"]], dim=0) for k in keys}
        act = roll["act"]
        old_logp = roll["log_prob"]
        returns = roll["returns"]
        adv = roll["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # labels are (B, D, T); flatten to (D*B, T) matching the d-outer/b-inner obs order
        R = roll["labels"]["R"].permute(1, 0, 2).reshape(N, -1)
        lab_mask = (roll["labels"]["dense_mask"] | roll["labels"]["event_mask"]).permute(1, 0, 2).reshape(N, -1)

        mb = max(1, N // self.pp.minibatches)
        stats = {"policy": 0.0, "value": 0.0, "retention": 0.0, "entropy": 0.0, "n": 0}
        for _ in range(self.pp.epochs):
            for idx in torch.randperm(N, device=self.device).split(mb):
                obs_mb = {k: v[idx] for k, v in flat_obs.items()}
                out = self.policy.evaluate(obs_mb, act[idx])
                ratio = torch.exp(out["log_prob"] - old_logp[idx])
                a = adv[idx]
                pol = -torch.min(ratio * a, torch.clamp(ratio, 1 - self.pp.clip, 1 + self.pp.clip) * a).mean()
                vloss = 0.5 * (out["value"] - returns[idx]).pow(2).mean()
                m = lab_mask[idx]
                rloss = (self.policy.retention.retention_loss(out["r_hat"], out["r_logits"], R[idx], m)
                         if m.any() else torch.zeros((), device=self.device))
                ent = out["entropy"].mean()
                loss = pol + self.pp.lam_v * vloss + self.pp.lam_r * rloss - self.pp.lam_e * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.pp.max_grad_norm)
                self.opt.step()
                stats["policy"] += pol.item(); stats["value"] += vloss.item()
                stats["retention"] += float(rloss.detach()); stats["entropy"] += ent.item(); stats["n"] += 1

        n = max(1, stats["n"])
        return {
            "policy_loss": stats["policy"] / n, "value_loss": stats["value"] / n,
            "retention_loss": stats["retention"] / n, "entropy": stats["entropy"] / n,
            "rollout_reward": roll["rollout_reward"], "retention_mae": roll["rollout_retention_mae"],
        }

    def train(self, updates: int, seed: int = 0) -> Iterator[dict]:
        for u in range(updates):
            yield self.update(self.collect(seed + u))
