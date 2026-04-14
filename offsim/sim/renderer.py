"""Pygame 2D top-down renderer — VEX Push Back 2025-2026.

Layout:
  Left  720px  — field (144"x144" at 5px/in)
  Right 320px  — info panel (objective, status, state editor)
  Bottom 60px  — HUD strip (spans full width under field only)

Field origin: bottom-left (y=0 is bottom of field, y increases upward).
Screen origin: top-left (pygame convention — y flipped via _to_screen).

Controls:
  Space     — pause / resume
  S         — step one RL decision while paused
  H         — toggle heatmap overlay
  R         — reset episode
  +/-       — speed up / slow down
  1/2       — highlight robot 0 / robot 1
  C         — change selected ball color (red ↔ blue)
  Delete    — delete selected ball
  Left-click on field   — select a ball
  Right-click on field  — add ball (current brush color shown in panel)
"""

from __future__ import annotations
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from sim.config import (
    Action, FIELD_W, FIELD_H, ROBOT_W,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP, OBJ_REMOVED,
    BALL_RED, BALL_BLUE,
    OUR_LONG_GOAL, OPP_LONG_GOAL, CENTER_MID_GOAL, CENTER_LOW_GOAL,
    LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX, LONG_GOAL_DEPTH,
    MATCH_DURATION, HEATMAP_W, HEATMAP_H,
)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCALE    = 5.0
FIELD_PX = int(FIELD_W * SCALE)   # 720
FIELD_PY = int(FIELD_H * SCALE)   # 720
HUD_H    = 60
PANEL_W  = 320
SCREEN_W = FIELD_PX + PANEL_W     # 1040
SCREEN_H = FIELD_PY + HUD_H       # 780

BTN_W = 144
BTN_H = 28

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BLACK      = (  0,   0,   0)
WHITE      = (255, 255, 255)
GRAY       = ( 50,  50,  50)
DARK_GRAY  = ( 22,  22,  22)
MID_GRAY   = ( 38,  38,  38)
LIGHT_GRAY = (120, 120, 120)
RED        = (220,  50,  50)
BLUE       = ( 50, 100, 220)
LIGHT_BLUE = (100, 170, 255)
GREEN      = ( 50, 200,  80)
YELLOW     = (230, 200,  40)
ORANGE     = (230, 140,  30)
LIGHT_RED  = (255, 130, 130)
PANEL_BG   = ( 18,  18,  28)
SECTION_BG = ( 28,  28,  42)
ACCENT     = ( 80, 140, 255)


def _to_screen(x: float, y: float) -> tuple[int, int]:
    """Field coords (origin bottom-left, inches) → screen pixels (top-left)."""
    return int(x * SCALE), int((FIELD_H - y) * SCALE)


def _from_screen(sx: int, sy: int) -> tuple[float, float]:
    """Screen pixels → field inches."""
    return sx / SCALE, FIELD_H - sy / SCALE


# ---------------------------------------------------------------------------
# Renderer class
# ---------------------------------------------------------------------------
class PygameRenderer:

    def __init__(self, env=None, render_every: int = 1):
        if pygame is None:
            raise ImportError("pygame required: pip install pygame-ce")
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("VEX Push Back 2025-26 — offsim")
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("consolas", 13)
        self.font_sm = pygame.font.SysFont("consolas", 11)
        self.font_lg = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_hd = pygame.font.SysFont("consolas", 17, bold=True)

        self.render_every    = render_every
        self.paused          = True        # start paused — press Space or click STEP
        self.show_heatmap    = False
        self.highlight_robot = -1
        self.sim_speed       = 1.0
        self.step_once       = False
        self.should_reset    = False

        # State editor
        self.selected_ball = -1           # index into field.objects, -1 = none
        self.brush_color   = BALL_BLUE    # color for right-click add

        # Pre-compute button rects (absolute screen coords)
        px = FIELD_PX + 10
        self.btn_step         = pygame.Rect(px,       8,  PANEL_W - 20, 36)
        self.btn_heatmap      = pygame.Rect(px,       0,  PANEL_W - 20, BTN_H)
        self.btn_change_color = pygame.Rect(px,       0,  BTN_W,        BTN_H)
        self.btn_delete       = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_add_red      = pygame.Rect(px,       0,  BTN_W,        BTN_H)
        self.btn_add_blue     = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_brush_toggle = pygame.Rect(px,       0,  PANEL_W - 20, BTN_H)
        # y-positions for editor buttons are set dynamically in _draw_panel

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_events(self, env=None) -> dict:
        signals = {
            "step_once":    False,
            "reset":        False,
            "change_color": False,
            "delete_ball":  False,
            "mouse_left":   None,
            "mouse_right":  None,
        }
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    signals["step_once"] = True
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_r:
                    signals["reset"] = True
                    self.selected_ball = -1
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.sim_speed = min(self.sim_speed * 2, 64.0)
                elif event.key == pygame.K_MINUS:
                    self.sim_speed = max(self.sim_speed / 2, 0.25)
                elif event.key == pygame.K_1:
                    self.highlight_robot = 0 if self.highlight_robot != 0 else -1
                elif event.key == pygame.K_2:
                    self.highlight_robot = 1 if self.highlight_robot != 1 else -1
                elif event.key == pygame.K_c:
                    signals["change_color"] = True
                elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                    signals["delete_ball"] = True

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:   # left click
                    signals["mouse_left"] = event.pos
                elif event.button == 3: # right click
                    signals["mouse_right"] = event.pos

        return signals

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------
    def draw(self, env):
        signals = self.handle_events(env)
        self.step_once    = signals["step_once"]
        self.should_reset = signals["reset"]

        # Apply editor actions
        if signals["change_color"] and self.selected_ball >= 0:
            env.field.change_ball_color(self.selected_ball)

        if signals["delete_ball"] and self.selected_ball >= 0:
            env.field.remove_ball(self.selected_ball)
            self.selected_ball = -1

        if signals["mouse_left"] is not None:
            self._handle_left_click(signals["mouse_left"], env)

        if signals["mouse_right"] is not None:
            mx, my = signals["mouse_right"]
            if mx < FIELD_PX and my < FIELD_PY:
                fx, fy = _from_screen(mx, my)
                env.field.add_ball(round(fx, 2), round(fy, 2), self.brush_color)

        # --- Draw field ---
        self.screen.fill(DARK_GRAY)
        pygame.draw.rect(self.screen, GRAY, (0, 0, FIELD_PX, FIELD_PY))
        self._draw_grid()
        pygame.draw.rect(self.screen, WHITE, (0, 0, FIELD_PX, FIELD_PY), 2)

        if self.show_heatmap:
            self._draw_heatmap(env)

        self._draw_goals(env)
        self._draw_balls(env)

        num_allies = getattr(env, "num_allies", 2)
        for idx in range(num_allies):
            robot = env.field.allies[idx]
            highlighted = (idx == self.highlight_robot)
            self._draw_robot(robot, BLUE if idx == 0 else LIGHT_BLUE,
                             f"R{idx}", highlighted, env)

        num_opponents = getattr(env, "num_opponents", 2)
        for idx in range(num_opponents):
            robot = env.field.opponents[idx]
            self._draw_robot(robot, RED if idx == 0 else LIGHT_RED,
                             f"O{idx}", False, env)

        # --- HUD strip ---
        self._draw_hud(env)

        # --- Right panel ---
        self._draw_panel(env)

        pygame.display.flip()
        self.clock.tick(int(60 * self.sim_speed))

    # ------------------------------------------------------------------
    # Field elements
    # ------------------------------------------------------------------
    def _draw_grid(self):
        for i in range(1, 12):
            x = int(i * 12 * SCALE)
            pygame.draw.line(self.screen, (58, 58, 58), (x, 0), (x, FIELD_PY))
            y = int(i * 12 * SCALE)
            pygame.draw.line(self.screen, (58, 58, 58), (0, y), (FIELD_PX, y))

    def _goal_counts(self, env) -> dict:
        """Count balls per goal by scored_in_goal tag."""
        counts: dict[str, int] = {}
        for obj in env.field.objects:
            if obj.status in (OBJ_SCORED_US, OBJ_SCORED_OPP) and obj.scored_in_goal:
                counts[obj.scored_in_goal] = counts.get(obj.scored_in_goal, 0) + 1
        return counts

    def _draw_goals(self, env=None):
        """Draw L-bracket long goals and center rectangular goals."""
        # ---- Blue long goal (right wall) ----
        y0 = int((FIELD_H - LONG_GOAL_Y_MAX) * SCALE)
        y1 = int((FIELD_H - LONG_GOAL_Y_MIN) * SCALE)
        h  = y1 - y0
        depth_px = int(LONG_GOAL_DEPTH * SCALE)
        arm_h    = 15  # pixels

        # Back bar (wall side)
        pygame.draw.rect(self.screen, BLUE,
                         (FIELD_PX - 22, y0, 22, h), 0)
        pygame.draw.rect(self.screen, LIGHT_BLUE,
                         (FIELD_PX - 22, y0, 22, h), 2)
        # Top arm
        pygame.draw.rect(self.screen, BLUE,
                         (FIELD_PX - 22 - depth_px, y0, depth_px, arm_h), 0)
        pygame.draw.rect(self.screen, LIGHT_BLUE,
                         (FIELD_PX - 22 - depth_px, y0, depth_px, arm_h), 2)
        # Bottom arm
        pygame.draw.rect(self.screen, BLUE,
                         (FIELD_PX - 22 - depth_px, y1 - arm_h, depth_px, arm_h), 0)
        pygame.draw.rect(self.screen, LIGHT_BLUE,
                         (FIELD_PX - 22 - depth_px, y1 - arm_h, depth_px, arm_h), 2)
        counts = self._goal_counts(env) if env else {}

        our_n = counts.get("our_long", 0)
        lbl = self.font_sm.render(f"OUR GOAL  [{our_n}]", True, LIGHT_BLUE)
        self.screen.blit(lbl, (FIELD_PX - 22 - depth_px - 4, y0 + h // 2 - 6))

        # ---- Red long goal (left wall) ----
        pygame.draw.rect(self.screen, RED,
                         (0, y0, 22, h), 0)
        pygame.draw.rect(self.screen, LIGHT_RED,
                         (0, y0, 22, h), 2)
        pygame.draw.rect(self.screen, RED,
                         (22, y0, depth_px, arm_h), 0)
        pygame.draw.rect(self.screen, LIGHT_RED,
                         (22, y0, depth_px, arm_h), 2)
        pygame.draw.rect(self.screen, RED,
                         (22, y1 - arm_h, depth_px, arm_h), 0)
        pygame.draw.rect(self.screen, LIGHT_RED,
                         (22, y1 - arm_h, depth_px, arm_h), 2)
        opp_n = counts.get("opp_long", 0)
        lbl = self.font_sm.render(f"OPP GOAL  [{opp_n}]", True, LIGHT_RED)
        self.screen.blit(lbl, (22 + depth_px + 2, y0 + h // 2 - 6))

        # ---- Center mid goal ----
        sx, sy = _to_screen(CENTER_MID_GOAL[0], CENTER_MID_GOAL[1])
        rect = pygame.Rect(sx - 30, sy - 25, 60, 50)
        pygame.draw.rect(self.screen, (100, 80, 20), rect)
        pygame.draw.rect(self.screen, ORANGE, rect, 2)
        mid_n = counts.get("center_mid", 0)
        lbl = self.font_sm.render(f"MID [{mid_n}]", True, ORANGE)
        self.screen.blit(lbl, (sx - 18, sy - 6))

        # ---- Center low goal ----
        sx, sy = _to_screen(CENTER_LOW_GOAL[0], CENTER_LOW_GOAL[1])
        rect = pygame.Rect(sx - 30, sy - 25, 60, 50)
        pygame.draw.rect(self.screen, (80, 60, 10), rect)
        pygame.draw.rect(self.screen, YELLOW, rect, 2)
        low_n = counts.get("center_low", 0)
        lbl = self.font_sm.render(f"LOW [{low_n}]", True, YELLOW)
        self.screen.blit(lbl, (sx - 18, sy - 6))

    def _draw_balls(self, env):
        """Draw all on-field and scored balls with correct colors.
        Fast-rolling balls get a speed trail behind them."""
        import math
        TRAIL_SPEED_MIN = 6.0   # in/s — faster than this gets a trail

        for i, obj in enumerate(env.field.objects):
            if obj.status == OBJ_REMOVED:
                continue

            # Ball colour based on alliance color and status
            if obj.status == OBJ_ON_FIELD:
                color   = RED      if obj.color == BALL_RED else BLUE
                outline = LIGHT_RED if obj.color == BALL_RED else LIGHT_BLUE
            elif obj.status == OBJ_HELD:
                color   = (200, 100, 100) if obj.color == BALL_RED else (100, 160, 240)
                outline = WHITE
            elif obj.status == OBJ_SCORED_US:
                color   = LIGHT_BLUE
                outline = WHITE
            elif obj.status == OBJ_SCORED_OPP:
                color   = LIGHT_RED
                outline = WHITE
            else:
                continue

            sx, sy = _to_screen(obj.x, obj.y)
            radius = 5

            # Rolling trail — draw a fading line behind moving balls
            if obj.status == OBJ_ON_FIELD:
                spd = math.sqrt(obj.vx ** 2 + obj.vy ** 2)
                if spd > TRAIL_SPEED_MIN:
                    trail_len = min(spd * 0.18, 20.0)   # pixels, capped
                    if spd > 0:
                        norm_vx = obj.vx / spd
                        norm_vy = obj.vy / spd
                    tx = sx - int(norm_vx * trail_len * SCALE / 10)
                    ty = sy + int(norm_vy * trail_len * SCALE / 10)
                    alpha = min(200, int(spd * 3))
                    fade_col = tuple(max(0, c - 80) for c in color)
                    pygame.draw.line(self.screen, fade_col, (tx, ty), (sx, sy), 2)

            # Highlight selected ball
            if i == self.selected_ball:
                pygame.draw.circle(self.screen, WHITE, (sx, sy), radius + 4, 2)

            pygame.draw.circle(self.screen, color,   (sx, sy), radius)
            pygame.draw.circle(self.screen, outline, (sx, sy), radius, 1)

    def _draw_robot(self, robot, color, label_prefix, highlighted, env):
        """Draw a 15x15in robot square with heading arrow and target line."""
        sx, sy = _to_screen(robot.x, robot.y)
        half   = int(ROBOT_W / 2 * SCALE)
        border = 3 if highlighted else 2

        rect = pygame.Rect(sx - half, sy - half, half * 2, half * 2)
        pygame.draw.rect(self.screen, color, rect, border)

        # Heading arrow
        arrow_len = half + 6
        ax = sx + int(arrow_len * np.cos(-robot.heading))
        ay = sy + int(arrow_len * np.sin(-robot.heading))
        pygame.draw.line(self.screen, WHITE, (sx, sy), (ax, ay), 2)

        # Target line
        if robot.target is not None and robot.moving:
            tx, ty = _to_screen(robot.target[0], robot.target[1])
            pygame.draw.line(self.screen, (90, 90, 90), (sx, sy), (tx, ty), 1)
            pygame.draw.circle(self.screen, (140, 140, 140), (tx, ty), 3)

        # Label (below robot)
        action_name = ""
        if hasattr(env, "current_actions"):
            for i, ally in enumerate(env.field.allies):
                if ally is robot:
                    action_name = Action(env.current_actions[i]).name[:12]
                    break
        lbl = f"{label_prefix}({robot.balls_held}) {action_name}"
        surf = self.font_sm.render(lbl, True, WHITE)
        self.screen.blit(surf, (sx - half, sy + half + 3))

    def _draw_heatmap(self, env):
        heatmap = env.field.get_heatmap()
        mx      = heatmap.max() if heatmap.max() > 0 else 1.0
        cell_w  = FIELD_PX / HEATMAP_W
        cell_h  = FIELD_PY / HEATMAP_H

        overlay = pygame.Surface((FIELD_PX, FIELD_PY), pygame.SRCALPHA)
        for gy in range(HEATMAP_H):
            for gx in range(HEATMAP_W):
                val = heatmap[gy, gx] / mx
                if val > 0.01:
                    screen_gy = HEATMAP_H - 1 - gy
                    rect = pygame.Rect(int(gx * cell_w), int(screen_gy * cell_h),
                                       int(cell_w) + 1, int(cell_h) + 1)
                    pygame.draw.rect(overlay,
                                     (int(val * 255), int((1 - val) * 80), 40, int(val * 100)),
                                     rect)
        self.screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # HUD strip (bottom of field)
    # ------------------------------------------------------------------
    def _draw_hud(self, env):
        y0 = FIELD_PY + 4
        t  = max(0, env.field.time_remaining)
        state = "PAUSED" if self.paused else "RUNNING"
        hm    = "ON"     if self.show_heatmap else "OFF"

        line1 = (f"Time: {t:6.2f}s  |  Us (Blue): {env.field.my_score:3d}  "
                 f"Opp (Red): {env.field.opponent_score:3d}  |  Speed: {self.sim_speed:.1f}x  [{state}]")
        self.screen.blit(self.font_lg.render(line1, True, WHITE), (10, y0))

        num_allies = getattr(env, "num_allies", 2)
        r0 = env.field.allies[0]
        robot_str = f"R0({r0.x:.1f},{r0.y:.1f}) held:{r0.balls_held}"
        if num_allies >= 2:
            r1 = env.field.allies[1]
            robot_str += f"  R1({r1.x:.1f},{r1.y:.1f}) held:{r1.balls_held}"
        line2 = (f"{robot_str}  Heatmap:{hm}  "
                 f"Brush:{'RED' if self.brush_color == BALL_RED else 'BLUE'}")
        self.screen.blit(self.font.render(line2, True, LIGHT_GRAY), (10, y0 + 20))

        line3 = "Space:pause  S:step  H:hmap  R:reset  +/-:speed  1/2:robot  C:color  Del:delete  RClick:add"
        self.screen.blit(self.font_sm.render(line3, True, (90, 90, 90)), (10, y0 + 38))

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------
    def _draw_panel(self, env):
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG,
                         (FIELD_PX, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(self.screen, ACCENT,
                         (FIELD_PX, 0), (FIELD_PX, SCREEN_H), 2)

        mouse = pygame.mouse.get_pos()
        y = self._draw_panel_step_btn(env, y_start=8, mouse=mouse)
        y = self._draw_panel_heatmap_btn(y + 4, mouse)
        y = self._draw_panel_timer(env, y + 4)
        y = self._draw_panel_section("OBJECTIVE", y + 6)
        y = self._draw_panel_objective(env, y)
        y = self._draw_panel_section("MATCH STATUS", y + 6)
        y = self._draw_panel_status(env, y)
        y = self._draw_panel_section("ROBOTS", y + 6)
        y = self._draw_panel_robots(env, y)
        y = self._draw_panel_section("STATE EDITOR", y + 6)
        y = self._draw_panel_editor(env, y, mouse)
        y = self._draw_panel_section("CONTROLS", y + 6)
        self._draw_panel_controls(y)

    def _draw_panel_section(self, title: str, y: int) -> int:
        """Draw a section header. Returns y after header."""
        pygame.draw.line(self.screen, ACCENT,
                         (FIELD_PX + 4, y), (FIELD_PX + PANEL_W - 4, y), 1)
        surf = self.font_lg.render(title, True, ACCENT)
        self.screen.blit(surf, (FIELD_PX + 8, y + 3))
        return y + 22

    def _draw_panel_step_btn(self, env, y_start: int, mouse) -> int:
        rect = self.btn_step
        rect.y = y_start
        hover = rect.collidepoint(mouse)
        color = (60, 120, 255) if hover else (40, 90, 200)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, LIGHT_BLUE, rect, 1, border_radius=4)

        label = "STEP (S)" if self.paused else "RUNNING — press Space to pause"
        surf = self.font_lg.render(label, True, WHITE)
        self.screen.blit(surf, (rect.x + rect.w // 2 - surf.get_width() // 2,
                                rect.y + rect.h // 2 - surf.get_height() // 2))

        # Detect click on step button
        if pygame.mouse.get_pressed()[0] and hover:
            self.step_once = True

        return y_start + rect.h

    def _draw_panel_heatmap_btn(self, y_start: int, mouse) -> int:
        px = FIELD_PX + 10
        self.btn_heatmap.update(px, y_start, PANEL_W - 20, BTN_H)
        on  = self.show_heatmap
        bg  = (20, 60, 40) if on else (35, 35, 50)
        fg  = GREEN        if on else LIGHT_GRAY
        label = "Heatmap: ON  (H)" if on else "Heatmap: OFF (H)"
        hover = self.btn_heatmap.collidepoint(mouse)
        if hover:
            bg = tuple(min(255, c + 20) for c in bg)
        pygame.draw.rect(self.screen, bg, self.btn_heatmap, border_radius=3)
        pygame.draw.rect(self.screen, fg, self.btn_heatmap, 1, border_radius=3)
        surf = self.font_sm.render(label, True, fg)
        self.screen.blit(surf, (self.btn_heatmap.x + self.btn_heatmap.w // 2 - surf.get_width() // 2,
                                self.btn_heatmap.y + (BTN_H - surf.get_height()) // 2))
        if pygame.mouse.get_pressed()[0] and hover:
            self.show_heatmap = not self.show_heatmap
        return y_start + BTN_H

    def _draw_panel_timer(self, env, y_start: int) -> int:
        """Large match timer + decision-progress bar."""
        from sim.config import TICKS_PER_DECISION
        t   = max(0.0, env.field.time_remaining)
        mins = int(t // 60)
        secs = t % 60
        px   = FIELD_PX + 10
        bw   = PANEL_W - 20   # bar width

        # Large MM:SS.ss display
        time_str = f"{mins}:{secs:05.2f}"
        # Choose colour by urgency
        if t > 60:
            tcol = WHITE
        elif t > 30:
            tcol = YELLOW
        else:
            tcol = RED

        # Background box
        box = pygame.Rect(px, y_start, bw, 38)
        pygame.draw.rect(self.screen, SECTION_BG, box, border_radius=4)
        pygame.draw.rect(self.screen, tcol, box, 1, border_radius=4)

        # Time text centred in box
        big = pygame.font.SysFont("consolas", 26, bold=True)
        tsurf = big.render(time_str, True, tcol)
        self.screen.blit(tsurf, (px + bw // 2 - tsurf.get_width() // 2,
                                 y_start + 19 - tsurf.get_height() // 2))

        y = y_start + 42

        # Decision progress bar
        executing = getattr(env, "executing", False)
        tick      = getattr(env, "decision_tick", 0)
        progress  = min(tick / TICKS_PER_DECISION, 1.0)

        bar_bg = pygame.Rect(px, y, bw, 14)
        pygame.draw.rect(self.screen, (35, 35, 50), bar_bg, border_radius=3)

        if executing and progress > 0:
            fill_w = int(bw * progress)
            bar_fill = pygame.Rect(px, y, fill_w, 14)
            pygame.draw.rect(self.screen, ACCENT, bar_fill, border_radius=3)

        pygame.draw.rect(self.screen, LIGHT_GRAY, bar_bg, 1, border_radius=3)

        # Status label
        if executing:
            status     = f"EXECUTING  {tick}/{TICKS_PER_DECISION}"
            status_col = GREEN
        elif self.paused:
            status     = "PAUSED — click STEP or press S"
            status_col = YELLOW
        else:
            status     = "RUNNING"
            status_col = LIGHT_BLUE

        ssurf = self.font_sm.render(status, True, status_col)
        self.screen.blit(ssurf, (px + bw // 2 - ssurf.get_width() // 2, y + 16))

        return y + 32

    def _draw_panel_objective(self, env, y_start: int) -> int:
        lines = self._compute_strategy(env)
        y = y_start
        for line in lines:
            col = YELLOW if line.startswith("PHASE") else (
                  LIGHT_RED if "BEHIND" in line or "TRAILING" in line else
                  GREEN if "LEADING" in line or "AHEAD" in line else
                  WHITE)
            surf = self.font.render(line, True, col)
            self.screen.blit(surf, (FIELD_PX + 10, y))
            y += 16
        return y

    def _draw_panel_status(self, env, y_start: int) -> int:
        y = y_start
        t = max(0, env.field.time_remaining)
        diff = env.field.my_score - env.field.opponent_score

        on_field  = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD)
        red_field = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD and o.color == BALL_RED)
        blu_field = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD and o.color == BALL_BLUE)

        lines = [
            (f"Time:  {t:6.2f}s", WHITE),
            (f"Us  (Blue):  {env.field.my_score:3d} pts", LIGHT_BLUE),
            (f"Opp (Red):  {env.field.opponent_score:3d} pts", LIGHT_RED),
            (f"Margin: {'+' if diff >= 0 else ''}{diff}", GREEN if diff > 0 else (RED if diff < 0 else WHITE)),
            (f"Balls on field: {on_field}  (R:{red_field} B:{blu_field})", LIGHT_GRAY),
        ]
        for text, col in lines:
            surf = self.font.render(text, True, col)
            self.screen.blit(surf, (FIELD_PX + 10, y))
            y += 17
        return y

    def _draw_panel_robots(self, env, y_start: int) -> int:
        y = y_start
        num_allies = getattr(env, "num_allies", 2)
        for i in range(num_allies):
            robot = env.field.allies[i]
            action_name = ""
            if hasattr(env, "current_actions"):
                action_name = Action(env.current_actions[i]).name
            color = LIGHT_BLUE if i == 0 else (180, 210, 255)
            lines = [
                f"R{i}: ({robot.x:.1f}, {robot.y:.1f})",
                f"     heading:{np.degrees(robot.heading):.1f}°  held:{robot.balls_held}",
                f"     action:{action_name}",
            ]
            for line in lines:
                self.screen.blit(self.font_sm.render(line, True, color),
                                 (FIELD_PX + 10, y))
                y += 14
            y += 3
        return y

    def _draw_panel_editor(self, env, y_start: int, mouse) -> int:
        y = y_start

        # Selected ball info
        if self.selected_ball >= 0 and self.selected_ball < len(env.field.objects):
            obj = env.field.objects[self.selected_ball]
            if obj.status != OBJ_REMOVED:
                col_name = "RED" if obj.color == BALL_RED else "BLUE"
                status_names = {0: "ON FIELD", 1: "HELD", 2: "SCORED US",
                                3: "SCORED OPP", 4: "REMOVED"}
                info = [
                    f"Ball #{self.selected_ball}",
                    f"Color:  {col_name}",
                    f"Pos:  ({obj.x:.1f}, {obj.y:.1f})",
                    f"Status: {status_names.get(obj.status, '?')}",
                ]
                for line in info:
                    self.screen.blit(self.font_sm.render(line, True, YELLOW),
                                     (FIELD_PX + 10, y))
                    y += 14
            else:
                self.screen.blit(self.font_sm.render("Ball was removed", True, LIGHT_GRAY),
                                 (FIELD_PX + 10, y))
                y += 14
                self.selected_ball = -1
        else:
            self.screen.blit(self.font_sm.render("No ball selected", True, LIGHT_GRAY),
                             (FIELD_PX + 10, y))
            self.screen.blit(self.font_sm.render("Click on field to select", True, LIGHT_GRAY),
                             (FIELD_PX + 10, y + 13))
            y += 28

        y += 4

        # Editor buttons — update y-positions
        px = FIELD_PX + 10
        self.btn_change_color.update(px, y, BTN_W, BTN_H)
        self.btn_delete.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        self._draw_btn(self.btn_change_color, "Change Color (C)",
                       active=(self.selected_ball >= 0), mouse=mouse)
        self._draw_btn(self.btn_delete, "Delete (Del)",
                       active=(self.selected_ball >= 0), danger=True, mouse=mouse)
        y += BTN_H + 6

        # Detect button clicks
        if pygame.mouse.get_pressed()[0]:
            if self.btn_change_color.collidepoint(mouse) and self.selected_ball >= 0:
                env.field.change_ball_color(self.selected_ball)
            if self.btn_delete.collidepoint(mouse) and self.selected_ball >= 0:
                env.field.remove_ball(self.selected_ball)
                self.selected_ball = -1

        self.btn_add_red.update(px, y, BTN_W, BTN_H)
        self.btn_add_blue.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        self._draw_btn(self.btn_add_red,  "Add Red  (RClick)",
                       active=True, color_override=RED, mouse=mouse)
        self._draw_btn(self.btn_add_blue, "Add Blue (RClick)",
                       active=True, color_override=BLUE, mouse=mouse)
        y += BTN_H + 6

        if pygame.mouse.get_pressed()[0]:
            if self.btn_add_red.collidepoint(mouse):
                self.brush_color = BALL_RED
            if self.btn_add_blue.collidepoint(mouse):
                self.brush_color = BALL_BLUE

        # Brush indicator
        brush_name = "RED" if self.brush_color == BALL_RED else "BLUE"
        brush_col  = LIGHT_RED if self.brush_color == BALL_RED else LIGHT_BLUE
        surf = self.font_sm.render(f"Right-click brush: {brush_name}", True, brush_col)
        self.screen.blit(surf, (FIELD_PX + 10, y))
        y += 16

        return y

    def _draw_panel_controls(self, y_start: int):
        y = y_start
        lines = [
            ("Space", "pause / resume"),
            ("S",     "step one decision"),
            ("H",     "toggle heatmap"),
            ("R",     "reset episode"),
            ("+/-",   "speed up / slow down"),
            ("1/2",   "highlight robot"),
            ("C",     "change ball color"),
            ("Del",   "delete selected ball"),
            ("LClick","select ball"),
            ("RClick","add ball (brush color)"),
        ]
        for key, desc in lines:
            if y > SCREEN_H - 14:
                break
            key_surf  = self.font_sm.render(f"{key:<8}", True, YELLOW)
            desc_surf = self.font_sm.render(desc, True, LIGHT_GRAY)
            self.screen.blit(key_surf,  (FIELD_PX + 10, y))
            self.screen.blit(desc_surf, (FIELD_PX + 70, y))
            y += 14

    def _draw_btn(self, rect, label: str, active: bool = True,
                  danger: bool = False, color_override=None, mouse=None):
        hover = rect.collidepoint(mouse) if mouse else False
        if not active:
            bg = (35, 35, 45)
            fg = (70, 70, 80)
        elif color_override:
            bg = tuple(max(0, c - 80) for c in color_override)
            fg = color_override
        elif danger:
            bg = (80, 20, 20) if not hover else (120, 30, 30)
            fg = LIGHT_RED
        else:
            bg = (35, 55, 95) if not hover else (50, 80, 140)
            fg = LIGHT_BLUE
        pygame.draw.rect(self.screen, bg, rect, border_radius=3)
        pygame.draw.rect(self.screen, fg, rect, 1, border_radius=3)
        surf = self.font_sm.render(label, True, fg if active else (60, 60, 70))
        self.screen.blit(surf, (rect.x + 5, rect.y + (BTN_H - surf.get_height()) // 2))

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------
    def _handle_left_click(self, pos, env):
        mx, my = pos

        # Click on step button
        if self.btn_step.collidepoint(pos) and self.paused:
            self.step_once = True
            return

        # Click on field → select nearest ball
        if mx < FIELD_PX and my < FIELD_PY:
            best_idx, best_dist = -1, 20.0  # 20px max select radius
            for i, obj in enumerate(env.field.objects):
                if obj.status in (OBJ_ON_FIELD,):
                    sx, sy = _to_screen(obj.x, obj.y)
                    d = np.sqrt((sx - mx) ** 2 + (sy - my) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_idx  = i
            self.selected_ball = best_idx

    # ------------------------------------------------------------------
    # Strategy computation
    # ------------------------------------------------------------------
    def _compute_strategy(self, env) -> list[str]:
        t    = env.field.time_remaining
        us   = env.field.my_score
        opp  = env.field.opponent_score
        diff = us - opp
        on_f = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD)
        held = sum(r.balls_held for r in env.field.allies)

        lines = []
        if t > 90:
            lines.append("PHASE: EARLY GAME")
            lines.append("Collect balls near our")
            lines.append("long goal (right side)")
        elif t > 60:
            lines.append("PHASE: MID GAME")
            lines.append("Score in long goal,")
            lines.append("contest center goals")
        elif t > 30:
            lines.append("PHASE: LATE GAME")
            if diff < -3:
                lines.append("BEHIND — rush center")
                lines.append("goals for fast points")
            elif diff > 3:
                lines.append("LEADING — defend our")
                lines.append("long goal, hold lead")
            else:
                lines.append("TIED — contest both")
                lines.append("center goals now")
        else:
            lines.append("PHASE: FINAL PUSH")
            if diff < 0:
                lines.append("TRAILING — score all")
                lines.append("held balls ASAP!")
            else:
                lines.append("AHEAD — guard goals")
                lines.append("and run out clock")

        lines.append("")
        lines.append(f"Field balls: {on_f}")
        if held > 0:
            lines.append(f"We hold {held} — SCORE!")
        return lines

    def close(self):
        if pygame is not None:
            pygame.quit()
