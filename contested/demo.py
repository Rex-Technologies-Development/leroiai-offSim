"""Interactive / headless demo for the canonical contested environment.

Drives the core with a simple scripted allied policy (ACQUIRE high-value nearby
incomplete tasks; DEFEND held tasks under threat) so every mechanic is exercised
and visible. This is a *sanity visualiser*, not the TENURE model.

Examples
--------
    # live window (needs a display)
    python -m contested.demo --mode human --alpha 1.0 --seed 3

    # headless filmstrip PNG (works anywhere)
    python -m contested.demo --mode filmstrip --out filmstrip.png --frames 8

    # per-frame PNGs into a directory
    python -m contested.demo --mode frames --out frames_dir --frames 24
"""
from __future__ import annotations

import argparse
import math
import os

import torch

from .actions import action_masks, greedy_assign
from .config import load_config
from .core import CanonicalCore
from .render import CanonicalRenderer, ensure_headless, save_png


def scripted_ally_actions(core: CanonicalCore) -> torch.Tensor:
    """Greedy value/defense heuristic resolved through the real masking + matcher."""
    s = core.state
    cfg = core.cfg
    dev, dt = core.device, core.dtype
    T, A = cfg.max_tasks, cfg.action_dim
    B, R = s.robot_pos.shape[0], cfg.max_robots

    dist_rt = torch.linalg.vector_norm(s.robot_pos.unsqueeze(2) - s.task_pos.unsqueeze(1), dim=-1)  # (B,R,T)
    adv_dist = torch.linalg.vector_norm(s.adv_pos.unsqueeze(2) - s.task_pos.unsqueeze(1), dim=-1)   # (B,K,T)
    threat = ((adv_dist < 3 * cfg.service_radius).sum(1)).to(dt)          # (B,T)
    w = s.task_w.unsqueeze(1)                                             # (B,1,T)
    incomplete = (~s.task_c & s.task_valid).unsqueeze(1).to(dt)
    complete = (s.task_c & s.task_valid).unsqueeze(1).to(dt)

    acquire = incomplete * w / (dist_rt + 0.2)                           # (B,R,T)
    defend = complete * (0.5 + threat.unsqueeze(1)) * w / (dist_rt + 0.2)

    scores = torch.full((B, R, A), -1e30, device=dev, dtype=dt)
    scores[:, :, 0:T] = acquire
    scores[:, :, T:2 * T] = defend
    scores[:, :, cfg.idle_action] = 0.0                                  # act if anything is worthwhile
    return greedy_assign(scores, action_masks(s, cfg), cfg)


def make_policy(name: str):
    """Return a ``callable(core) -> actions``: the built-in ``scripted`` heuristic,
    or any M4 baseline by name (``greedy``/``defensive``/``cbba``)."""
    if name == "scripted":
        return scripted_ally_actions
    from baselines import REGISTRY
    baseline = REGISTRY[name]()
    return lambda core: baseline.act(core.state, core.cfg)


def make_tenure_policy(checkpoint: str, device: str = "cpu"):
    """Load a trained TENURE checkpoint (from ``tenure.train --save``) as a
    ``callable(core) -> actions`` for the visualiser. Deterministic (argmax)."""
    from tenure.policy import TenurePolicy
    from contested.observation import build_observation
    ckpt = torch.load(checkpoint, map_location=device)
    model = TenurePolicy(d_model=ckpt.get("d_model", 128),
                         retention_mode=ckpt.get("retention_mode", "multiplicative"),
                         task_dim=ckpt.get("task_dim", 7),
                         retention_head=ckpt.get("retention_head", "regression"),
                         symmetric=ckpt.get("symmetric", False)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded TENURE checkpoint {checkpoint} "
          f"(mode={ckpt.get('retention_mode')}, overrides={ckpt.get('overrides')})")

    @torch.no_grad()
    def fn(core):
        obs = build_observation(core.state, core.cfg)
        return model.act(obs, deterministic=True)["action"]
    return fn


def _decide(core: CanonicalCore, policy_fn) -> None:
    core.state.robot_action = policy_fn(core)
    if core.telemetry is not None:
        core.telemetry.on_decision(core.state, core.state.robot_action)
    core._refresh_adversary_targets()


def _hud_info(core: CanonicalCore, b: int) -> dict:
    n_rev = 0 if core.telemetry is None else int(core.telemetry.n_reversals[b].item())
    return {"n_reversals": n_rev}


def run_filmstrip(cfg, seed: int, out: str, n_frames: int, policy_fn, b: int = 0) -> None:
    ensure_headless()
    renderer = CanonicalRenderer(cfg, field_px=360, panel_px=210, margin=16)
    core = CanonicalCore(cfg, batch_size=max(1, b + 1), device="cpu", dtype=torch.float32)
    core.reset(seed)
    total = int(round(cfg.horizon_T / cfg.dt))
    tpd = cfg.ticks_per_decision
    grab = {int(round(x)) for x in torch.linspace(0, total - 1, n_frames).tolist()}

    frames = []
    for t in range(total):
        if t % tpd == 0:
            _decide(core, policy_fn)
        core.tick()
        if t in grab:
            frames.append(renderer.draw(core.state, b=b, info=_hud_info(core, b)).copy())

    import pygame
    fw, fh = renderer.size
    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    sheet = pygame.Surface((cols * fw, rows * fh))
    sheet.fill((10, 10, 14))
    for i, fr in enumerate(frames):
        sheet.blit(fr, ((i % cols) * fw, (i // cols) * fh))
    save_png(sheet, out)
    print(f"wrote {out}  ({cols}x{rows} frames, {fw}x{fh} each)")


def run_frames(cfg, seed: int, out_dir: str, n_frames: int, policy_fn, b: int = 0) -> None:
    ensure_headless()
    os.makedirs(out_dir, exist_ok=True)
    renderer = CanonicalRenderer(cfg)
    core = CanonicalCore(cfg, batch_size=max(1, b + 1), device="cpu", dtype=torch.float32)
    core.reset(seed)
    total = int(round(cfg.horizon_T / cfg.dt))
    tpd = cfg.ticks_per_decision
    grab = {int(round(x)) for x in torch.linspace(0, total - 1, n_frames).tolist()}
    saved = 0
    for t in range(total):
        if t % tpd == 0:
            _decide(core, policy_fn)
        core.tick()
        if t in grab:
            save_png(renderer.draw(core.state, b=b, info=_hud_info(core, b)), os.path.join(out_dir, f"frame_{saved:03d}.png"))
            saved += 1
    print(f"wrote {saved} frames to {out_dir}/")


def run_gif(cfg, seed: int, out: str, policy_fn, n_frames: int = 150, fps: int = 20, b: int = 0) -> None:
    """Record a rollout as an animated GIF (needs Pillow)."""
    from PIL import Image
    ensure_headless()
    renderer = CanonicalRenderer(cfg)
    core = CanonicalCore(cfg, batch_size=max(1, b + 1), device="cpu", dtype=torch.float32)
    core.reset(seed)
    total = int(round(cfg.horizon_T / cfg.dt))
    tpd = cfg.ticks_per_decision
    stride = max(1, total // n_frames)

    frames = []
    for t in range(total):
        if t % tpd == 0:
            _decide(core, policy_fn)
        core.tick()
        if t % stride == 0:
            surf = renderer.draw(core.state, b=b, info=_hud_info(core, b))
            frames.append(Image.frombytes("RGB", surf.get_size(), pygame_image_bytes(surf)))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"wrote {out}  ({len(frames)} frames @ {fps}fps)")


def pygame_image_bytes(surface):
    import pygame
    return pygame.image.tobytes(surface, "RGB")


def run_human(cfg, seed: int, policy_fn, b: int = 0, speed: float = 1.0) -> None:
    import pygame
    renderer = CanonicalRenderer(cfg)
    screen = pygame.display.set_mode(renderer.size)
    pygame.display.set_caption("Contested — canonical env")
    clock = pygame.time.Clock()
    core = CanonicalCore(cfg, batch_size=max(1, b + 1), device="cpu", dtype=torch.float32)
    core.reset(seed)
    total = int(round(cfg.horizon_T / cfg.dt))
    tpd = cfg.ticks_per_decision

    t, running, paused = 0, True, False
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    seed += 1; core.reset(seed); t = 0
        if not paused:
            if t % tpd == 0:
                _decide(core, policy_fn)
            core.tick()
            t += 1
            if t >= total:
                seed += 1; core.reset(seed); t = 0
        screen.blit(renderer.draw(core.state, b=b, info=_hud_info(core, b)), (0, 0))
        pygame.display.flip()
        clock.tick(max(1, int(1.0 / cfg.dt * speed)))
    pygame.quit()


def read_checkpoint_overrides(checkpoint: str) -> dict:
    """The env regime a TENURE checkpoint was trained under (saved by ``tenure.train``)."""
    ck = torch.load(checkpoint, map_location="cpu")
    return dict(ck.get("overrides") or {})


def build_cfg(args, defaults: dict | None = None):
    # start from the checkpoint's own regime (so a TENURE render matches training),
    # then let any explicit CLI flag win over it.
    overrides = dict(defaults or {})
    for key in ("alpha", "beta", "contest_mode", "layout", "n_tasks", "n_robots",
                "n_adversaries", "horizon_T", "tau_com", "protected_fraction", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "adversary", None) is not None:
        overrides["adversary_population"] = [args.adversary]
    return load_config(overrides=overrides or None)


def main() -> None:
    p = argparse.ArgumentParser(description="Canonical contested-env visual demo")
    p.add_argument("--mode", choices=["human", "filmstrip", "frames", "gif"], default="filmstrip")
    p.add_argument("--policy", default="scripted", help="scripted | greedy | defensive | cbba | tenure")
    p.add_argument("--checkpoint", default=None, help="TENURE .pt checkpoint (required for --policy tenure)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="filmstrip.png")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--contest-mode", dest="contest_mode", default=None)
    p.add_argument("--layout", default=None, help="uniform | clustered | polarized")
    p.add_argument("--n-tasks", dest="n_tasks", type=int, default=None)
    p.add_argument("--n-robots", dest="n_robots", type=int, default=None)
    p.add_argument("--n-adversaries", dest="n_adversaries", type=int, default=None)
    p.add_argument("--horizon", dest="horizon_T", type=float, default=None)
    p.add_argument("--tau-com", dest="tau_com", type=float, default=None)
    p.add_argument("--protected-fraction", dest="protected_fraction", type=float, default=None)
    p.add_argument("--adversary", default=None, help="single archetype, e.g. camper")
    args = p.parse_args()

    if args.policy == "tenure":
        if not args.checkpoint:
            p.error("--policy tenure requires --checkpoint PATH")
        cfg = build_cfg(args, read_checkpoint_overrides(args.checkpoint))  # regime from ckpt
        policy_fn = make_tenure_policy(args.checkpoint)
    else:
        cfg = build_cfg(args)
        policy_fn = make_policy(args.policy)
    if args.mode == "human":
        run_human(cfg, args.seed, policy_fn, speed=args.speed)
    elif args.mode == "frames":
        run_frames(cfg, args.seed, args.out, args.frames, policy_fn)
    elif args.mode == "gif":
        run_gif(cfg, args.seed, args.out, policy_fn, n_frames=max(60, args.frames), fps=args.fps)
    else:
        run_filmstrip(cfg, args.seed, args.out, args.frames, policy_fn)


if __name__ == "__main__":
    main()
