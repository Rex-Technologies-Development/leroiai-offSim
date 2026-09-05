"""Pygame renderer for the canonical contested-task environment.

The canonical env is abstract, so this exists to make every mechanic *visually*
legible for correctness checks:

- **Tasks** are circles; radius grows with weight ``w``. Grey/orange = incomplete,
  green = complete (value held). A faint ring marks the ``service_radius``. A
  yellow arc shows service progress ``sigma / tau_com`` (filling toward completion);
  a red arc shows reversal pressure ``eta / tau_rev`` on complete tasks.
- **Allies** are blue dots with a line to their assigned target, tinted by action
  (cyan = ACQUIRE, blue = DEFEND, dim = IDLE) and a short velocity whisker.
- **Adversaries** are red dots labelled by archetype, with a line to their target.
- **HUD** reports t/T, held fraction J_H, completion fraction, alpha, tau_com,
  tau_rev, contest_mode and reversal count.

Works headless (``SDL_VIDEODRIVER=dummy``) for filmstrip PNGs, or in a live window.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from .config import CanonicalConfig
from .core import CanonicalState

# palette (dark theme)
_BG = (18, 18, 24)
_FIELD = (26, 27, 36)
_BORDER = (70, 72, 90)
_GRID = (34, 36, 48)
_TASK_INCOMPLETE = (206, 148, 70)
_TASK_COMPLETE = (74, 202, 128)
_SERVICE_RING = (72, 74, 92)
_SIGMA = (240, 214, 84)
_ETA = (232, 84, 84)
_ALLY = (86, 152, 255)
_ACQUIRE = (86, 220, 220)
_DEFEND = (130, 168, 255)
_IDLE = (120, 122, 140)
_ADV = (232, 96, 96)
_TASK_RED = (214, 96, 110)                 # red-owned task (symmetric)
_TOGGLE = (247, 205, 74)                   # toggle (multiplier control point) — gold ring
_TEXT = (228, 228, 236)
_DIM = (150, 152, 168)

_ARCH_LABEL = {0: "G", 1: "V", 2: "C", 3: "F", 4: "L"}


class CanonicalRenderer:
    """Draws one environment (batch index ``b``) of a :class:`CanonicalState`."""

    def __init__(self, cfg: CanonicalConfig, field_px: int = 720, panel_px: int = 300, margin: int = 24):
        self.cfg = cfg
        self.field_px = field_px
        self.panel_px = panel_px
        self.margin = margin
        self.size = (field_px + panel_px, field_px)
        import pygame  # imported lazily so the package has no hard pygame dependency
        self.pygame = pygame
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        # scale fonts to frame size so filmstrip thumbnails stay legible and in-panel
        base = max(12, field_px // 40)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", base)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", max(10, base - 3))
        self.big = pygame.font.SysFont("consolas,menlo,monospace", base + 3, bold=True)
        self._line_h = base + 6

    # ------------------------------------------------------------ coordinates
    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        W, H = self.cfg.field_size
        inner = self.field_px - 2 * self.margin
        sx = self.margin + (x / W) * inner
        sy = self.margin + (1.0 - y / H) * inner   # flip so +y is up
        return int(sx), int(sy)

    def _px(self, meters: float) -> int:
        W = self.cfg.field_size[0]
        return int(meters / W * (self.field_px - 2 * self.margin))

    # ------------------------------------------------------------------ draw
    def draw(self, state: CanonicalState, b: int = 0, info: Optional[dict] = None):
        pg = self.pygame
        surf = pg.Surface(self.size)
        surf.fill(_BG)
        self._draw_field(surf)
        self._draw_task_edges(surf, state, b)
        self._draw_tasks(surf, state, b)
        self._draw_adversaries(surf, state, b)
        self._draw_allies(surf, state, b)
        self._draw_panel(surf, state, b, info)
        return surf

    def _draw_field(self, surf) -> None:
        pg = self.pygame
        rect = pg.Rect(self.margin, self.margin,
                       self.field_px - 2 * self.margin, self.field_px - 2 * self.margin)
        pg.draw.rect(surf, _FIELD, rect)
        for i in range(1, 6):
            f = i / 6.0
            x = self.margin + f * (self.field_px - 2 * self.margin)
            pg.draw.line(surf, _GRID, (x, self.margin), (x, self.field_px - self.margin))
            pg.draw.line(surf, _GRID, (self.margin, x), (self.field_px - self.margin, x))
        pg.draw.rect(surf, _BORDER, rect, 2)

    def _draw_tasks(self, surf, state: CanonicalState, b: int) -> None:
        pg = self.pygame
        cfg = self.cfg
        w_max = float(cfg.weight_range[1])
        service_r = self._px(cfg.service_radius)
        for i in range(cfg.max_tasks):
            if not bool(state.task_valid[b, i]):
                continue
            x, y = float(state.task_pos[b, i, 0]), float(state.task_pos[b, i, 1])
            cx, cy = self._to_screen(x, y)
            complete = bool(state.task_c[b, i])
            red = bool(getattr(state, "task_c_red")[b, i]) if hasattr(state, "task_c_red") else False
            is_tog = bool(state.task_is_toggle[b, i]) if hasattr(state, "task_is_toggle") else False
            w = float(state.task_w[b, i])
            r = 6 + int((w / w_max) * 12)
            if is_tog:
                r += 3                                         # toggles drawn a touch larger
            pg.draw.circle(surf, _SERVICE_RING, (cx, cy), service_r, 1)
            color = _TASK_COMPLETE if complete else (_TASK_RED if red else _TASK_INCOMPLETE)
            pg.draw.circle(surf, color, (cx, cy), r)
            pg.draw.circle(surf, (12, 12, 16), (cx, cy), r, 1)
            if is_tog:                                         # gold ring marks the multiplier control point
                pg.draw.circle(surf, _TOGGLE, (cx, cy), r + 4, 3)
            # progress arcs
            if not complete:
                frac = min(1.0, float(state.sigma[b, i]) / cfg.tau_com)
                self._arc(surf, cx, cy, r + 4, frac, _SIGMA)
            else:
                tau_rev = cfg.tau_rev
                if math.isfinite(tau_rev):
                    frac = min(1.0, float(state.eta[b, i]) / tau_rev)
                    self._arc(surf, cx, cy, r + 4, frac, _ETA)
            tag = f"T{i}" if is_tog else f"{i}:{w:.1f}"        # toggles labelled T<idx>
            lbl = self.small.render(tag, True, _TOGGLE if is_tog else _DIM)
            surf.blit(lbl, (cx + r + 3, cy - 8))

    def _arc(self, surf, cx: int, cy: int, radius: int, frac: float, color) -> None:
        if frac <= 0:
            return
        pg = self.pygame
        rect = pg.Rect(cx - radius, cy - radius, 2 * radius, 2 * radius)
        start = math.pi / 2
        pg.draw.arc(surf, color, rect, start - frac * 2 * math.pi, start, 3)

    def _draw_task_edges(self, surf, state: CanonicalState, b: int) -> None:
        """Faint links between nearby tasks (the e_tt proximity structure)."""
        pg = self.pygame
        cfg = self.cfg
        thresh = 2.5 * cfg.service_radius
        pts = state.task_pos[b]
        for i in range(cfg.max_tasks):
            if not bool(state.task_valid[b, i]):
                continue
            for j in range(i + 1, cfg.max_tasks):
                if not bool(state.task_valid[b, j]):
                    continue
                d = float(torch.linalg.vector_norm(pts[i] - pts[j]))
                if d < thresh:
                    a = self._to_screen(float(pts[i, 0]), float(pts[i, 1]))
                    c = self._to_screen(float(pts[j, 0]), float(pts[j, 1]))
                    pg.draw.line(surf, _GRID, a, c, 1)

    def _draw_allies(self, surf, state: CanonicalState, b: int) -> None:
        pg = self.pygame
        cfg = self.cfg
        T = cfg.max_tasks
        for r in range(cfg.max_robots):
            if not bool(state.robot_valid[b, r]):
                continue
            x, y = float(state.robot_pos[b, r, 0]), float(state.robot_pos[b, r, 1])
            cx, cy = self._to_screen(x, y)
            act = int(state.robot_action[b, r])
            if act < T:
                tint, tgt = _ACQUIRE, act
            elif act < 2 * T:
                tint, tgt = _DEFEND, act - T
            else:
                tint, tgt = _IDLE, -1
            if tgt >= 0 and bool(state.task_valid[b, tgt]):
                tx, ty = self._to_screen(float(state.task_pos[b, tgt, 0]), float(state.task_pos[b, tgt, 1]))
                pg.draw.line(surf, tint, (cx, cy), (tx, ty), 2)
            vx, vy = float(state.robot_vel[b, r, 0]), float(state.robot_vel[b, r, 1])
            pg.draw.line(surf, _DIM, (cx, cy), self._to_screen(x + vx * 0.15, y + vy * 0.15), 1)
            pg.draw.circle(surf, _ALLY, (cx, cy), 8)
            pg.draw.circle(surf, tint, (cx, cy), 8, 2)
            surf.blit(self.small.render(f"R{r}", True, _TEXT), (cx - 8, cy - 22))

    def _draw_adversaries(self, surf, state: CanonicalState, b: int) -> None:
        pg = self.pygame
        cfg = self.cfg
        for k in range(cfg.max_adversaries):
            if not bool(state.adv_valid[b, k]):
                continue
            x, y = float(state.adv_pos[b, k, 0]), float(state.adv_pos[b, k, 1])
            cx, cy = self._to_screen(x, y)
            tgt = int(state.adv_target[b, k])
            if 0 <= tgt < cfg.max_tasks and bool(state.task_valid[b, tgt]):
                tx, ty = self._to_screen(float(state.task_pos[b, tgt, 0]), float(state.task_pos[b, tgt, 1]))
                pg.draw.line(surf, _ADV, (cx, cy), (tx, ty), 1)
            pg.draw.circle(surf, _ADV, (cx, cy), 7)
            pg.draw.circle(surf, (12, 12, 16), (cx, cy), 7, 1)
            arch = _ARCH_LABEL.get(int(state.adv_archetype[b, k]), "?")
            surf.blit(self.small.render(arch, True, (20, 16, 16)), (cx - 4, cy - 7))

    def _draw_panel(self, surf, state: CanonicalState, b: int, info: Optional[dict]) -> None:
        pg = self.pygame
        cfg = self.cfg
        x0 = self.field_px + 16
        y = 22
        tv = state.task_valid[b]
        total_w = float((state.task_w[b] * tv).sum().clamp_min(1e-9))
        j_h = float(state.held_integral[b]) / (cfg.horizon_T * total_w)
        n_complete = int((state.task_c[b] & tv).sum())
        n_task = int(tv.sum())
        frac = n_complete / max(1, n_task)
        tau_rev = cfg.tau_rev

        lh = self._line_h
        surf.blit(self.big.render("CONTESTED env", True, _TEXT), (x0, y)); y += lh + 10
        rows = [
            (f"t       {float(state.t[b]):5.1f}/{cfg.horizon_T:.0f}s", _TEXT),
            (f"J_H     {j_h:5.3f}", _TASK_COMPLETE),
            (f"done    {n_complete}/{n_task} ({frac*100:.0f}%)", _TEXT),
            ("", _TEXT),
            (f"alpha   {cfg.alpha:.2f}", _DIM),
            (f"tau_com {cfg.tau_com:.1f}s", _DIM),
            (f"tau_rev {'inf' if not math.isfinite(tau_rev) else f'{tau_rev:.1f}s'}", _DIM),
            (f"beta    {cfg.beta:.2f}", _DIM),
            (f"contest {cfg.contest_mode}", _DIM),
        ]
        if info and "n_reversals" in info:
            rows.append((f"revert  {int(info['n_reversals'])}", _ETA))
        for text, color in rows:
            if text:
                surf.blit(self.font.render(text, True, color), (x0, y))
            y += lh

        y += 10
        surf.blit(self.small.render("LEGEND", True, _DIM), (x0, y)); y += lh
        legend = [
            (_TASK_INCOMPLETE, "task incomplete"),
            (_TASK_COMPLETE, "task held"),
            (_SIGMA, "sigma->done"),
            (_ETA, "eta->revert"),
            (_ALLY, "ally"),
            (_ADV, "adversary"),
        ]
        for color, text in legend:
            pg.draw.circle(surf, color, (x0 + 6, y + 7), 6)
            surf.blit(self.small.render(text, True, _DIM), (x0 + 20, y))
            y += lh


def save_png(surface, path: str) -> None:
    import pygame
    pygame.image.save(surface, path)


def ensure_headless() -> None:
    """Select the dummy SDL driver when no display is present."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
