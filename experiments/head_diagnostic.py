"""Is the retention head DISCRIMINATING among contested tasks? (consultant diagnostic)

A J_H tie between mult+flag and off+flag is ambiguous. This logs, alongside J_H, the
head's predicted R_hat behaviour on CONTESTED (unprotected) tasks only:

  - within-decision spread of R_hat across contested tasks (does it separate them AT ALL);
  - correlation of R_hat with the realized held fraction on contested tasks (is the
    separation CORRECT).

Readout:
  * real spread + positive correlation  -> the head estimates correctly; a J_H tie means
    the POLICY isn't converting it into better allocation (different, more fixable problem).
  * near-flat spread                     -> the head learned nothing; policy tuning won't help.

Also splits early vs late in the episode: camper structure is revealed at reset, so if the
head only discriminates late, the contested structure may not be actionable early enough —
a property of the environment, not the architecture.

    python -m experiments.head_diagnostic runs/tenure_multflag_a05.pt --seeds 8
"""
from __future__ import annotations

import argparse

import torch

from contested.config import load_config
from contested.core import CanonicalCore, default_device
from contested.labels import RetentionLabelRecorder
from contested.observation import build_observation
from tenure.policy import TenurePolicy


def _within_decision(rhat, R, sel):
    """Mean over decisions of (R_hat spread, corr(R_hat,R)) across selected tasks per decision."""
    cnt = sel.sum(-1)                                             # (B, D)
    ok = cnt >= 2
    c = cnt.clamp_min(1).unsqueeze(-1)
    mrh = (rhat * sel).sum(-1, keepdim=True) / c
    mR = (R * sel).sum(-1, keepdim=True) / c
    drh, dR = (rhat - mrh) * sel, (R - mR) * sel
    std_rh = ((drh ** 2).sum(-1) / cnt.clamp_min(1)).sqrt()      # (B, D)
    std_R = ((dR ** 2).sum(-1) / cnt.clamp_min(1)).sqrt()
    cov = (drh * dR).sum(-1) / cnt.clamp_min(1)
    good = ok & (std_rh > 1e-4) & (std_R > 1e-4)
    corr = (cov / (std_rh * std_R + 1e-9))[good]
    spread = std_rh[ok]
    return (spread.mean().item() if spread.numel() else float("nan"),
            corr.mean().item() if corr.numel() else float("nan"),
            int(ok.sum()))


@torch.no_grad()
def diagnose(checkpoint: str, seeds: int = 8, batch: int = 64):
    dev = default_device()
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = load_config(overrides=ck.get("overrides") or None)
    pol = TenurePolicy(d_model=ck.get("d_model", 128), retention_mode=ck.get("retention_mode"),
                       task_dim=ck.get("task_dim", 7),
                       retention_head=ck.get("retention_head", "regression"))
    pol.load_state_dict(ck["state_dict"]); pol.to(dev).eval()

    RH, RR, MASK, PROT = [], [], [], []
    for s in range(seeds):
        core = CanonicalCore(cfg, batch_size=batch, device=dev)
        rec = RetentionLabelRecorder(); core.reset(s)
        rhats = []
        for _ in range(cfg.n_decisions):
            obs = build_observation(core.state, cfg)
            rec.before_decision(core)
            rhats.append(pol.forward(obs)["r_hat"])
            core.step(pol.act(obs, deterministic=True)["action"])
            rec.after_decision(core)
        lab = rec.finalize(core)
        RH.append(torch.stack(rhats, 1)); RR.append(lab["R"])
        MASK.append(lab["dense_mask"] | lab["event_mask"])
        PROT.append(core.state.task_protected.unsqueeze(1).expand_as(lab["R"]))
    rhat, R = torch.cat(RH), torch.cat(RR)
    m, prot = torch.cat(MASK), torch.cat(PROT)
    D = cfg.n_decisions

    def pooled(sel, name):
        rh, r = rhat[sel], R[sel]
        if rh.numel() < 2:
            print(f"  {name:22s} n={rh.numel()} (too few)"); return
        dc, dr = rh - rh.mean(), r - r.mean()
        corr = (dc * dr).sum() / (dc.norm() * dr.norm() + 1e-9)
        print(f"  {name:22s} n={rh.numel():>7}  R_hat mean={rh.mean():.3f} std={rh.std():.3f}"
              f"  realized_R mean={r.mean():.3f}  corr={corr:.3f}")

    print(f"checkpoint: {checkpoint}  (mode={ck.get('retention_mode')}, "
          f"expose_protected={ck['overrides'].get('expose_protected')})")
    c_sel, p_sel = m & ~prot, m & prot
    print("POOLED over all labeled (d,i):")
    pooled(c_sel, "contested tasks"); pooled(p_sel, "protected tasks")

    sp, co, n = _within_decision(rhat, R, c_sel.float())
    print(f"WITHIN-DECISION, contested only:  R_hat spread={sp:.3f}  corr(R_hat,R)={co:.3f}  (decisions={n})")
    early = torch.zeros_like(c_sel); early[:, :D // 3] = c_sel[:, :D // 3]
    late = torch.zeros_like(c_sel); late[:, 2 * D // 3:] = c_sel[:, 2 * D // 3:]
    for lbl, sel in (("early (first 1/3)", early.float()), ("late (last 1/3)", late.float())):
        sp, co, n = _within_decision(rhat, R, sel)
        print(f"  {lbl:18s} spread={sp:.3f}  corr={co:.3f}  (decisions={n})")
    print("READ: real spread + positive corr = head estimates (policy problem);"
          " flat spread = head blind (dead end).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retention-head discrimination diagnostic")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--batch-size", dest="batch", type=int, default=64)
    args = ap.parse_args()
    for ck in args.checkpoints:
        diagnose(ck, seeds=args.seeds, batch=args.batch)
        print()


if __name__ == "__main__":
    main()
