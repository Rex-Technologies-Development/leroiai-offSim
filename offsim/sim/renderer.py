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
import math
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from sim.config import (
    Action, FIELD_W, FIELD_H, ROBOT_W,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP, OBJ_REMOVED,
    BALL_RED, BALL_BLUE,
    LONG_GOAL_Y_MIN, LONG_GOAL_Y_MAX, LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH,
    CENTER_GOAL_ARM_LEN, CENTER_GOAL_ARM_W,
    MATCHLOAD_TUBES, MATCHLOAD_TUBE_RADIUS,
    PARK_ZONE_X_MIN, PARK_ZONE_X_MAX, PARK_ZONE_BOTTOM, PARK_ZONE_TOP,
    VISION_HALF_ANGLE, VISION_RANGE,
    MATCH_DURATION, HEATMAP_W, HEATMAP_H,
)

# Pre-computed goal body extents (for setup-mode ball scoring)
_R_GOAL_X_LO = FIELD_W - LONG_GOAL_WALL_GAP - LONG_GOAL_WIDTH
_R_GOAL_X_HI = FIELD_W - LONG_GOAL_WALL_GAP
_L_GOAL_X_LO = LONG_GOAL_WALL_GAP
_L_GOAL_X_HI = LONG_GOAL_WALL_GAP + LONG_GOAL_WIDTH

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCALE    = 6.0
FIELD_PX = int(FIELD_W * SCALE)   # 864
FIELD_PY = int(FIELD_H * SCALE)   # 864
HUD_H    = 60
PANEL_W  = 420
SCREEN_W = FIELD_PX + PANEL_W     # 1284
SCREEN_H = FIELD_PY + HUD_H       # 924

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


def _user_heading_deg(sim_heading: float) -> float:
    """Convert sim heading (radians, 0=right, CCW) to user degrees (0=N/up, CW, range 0-360)."""
    return (90.0 - math.degrees(sim_heading)) % 360.0


def _user_heading_disp(sim_heading: float) -> float:
    """Return user heading in -180..+180 range (0=N, CW)."""
    deg = _user_heading_deg(sim_heading)
    return deg if deg <= 180.0 else deg - 360.0


def _user_heading_label(sim_heading: float) -> str:
    """Return a short compass label e.g. '+0°(N)' or '-90°(W)'."""
    deg_360 = _user_heading_deg(sim_heading)
    dirs  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    label = dirs[int((deg_360 + 22.5) / 45) % 8]
    disp  = deg_360 if deg_360 <= 180.0 else deg_360 - 360.0
    return f"{disp:+.0f}°({label})"


def _sim_heading_from_user(user_deg: float) -> float:
    """Convert user degrees (0=N, CW, any range) to sim heading (radians, 0=right, CCW)."""
    return math.radians(90.0 - user_deg)


def _robot_corners(cx: int, cy: int, half: int, heading: float):
    """Return 4 screen-space corner points of a rotated square robot.

    heading: sim heading in radians (0=right, CCW).
    Screen has y-down, so we negate the sin component.
    """
    fwd_x =  math.cos(heading)
    fwd_y = -math.sin(heading)   # screen y is inverted
    rgt_x =  math.sin(heading)
    rgt_y =  math.cos(heading)
    h = half
    return [
        (cx + int(h * fwd_x + h * rgt_x), cy + int(h * fwd_y + h * rgt_y)),
        (cx + int(h * fwd_x - h * rgt_x), cy + int(h * fwd_y - h * rgt_y)),
        (cx + int(-h * fwd_x - h * rgt_x), cy + int(-h * fwd_y - h * rgt_y)),
        (cx + int(-h * fwd_x + h * rgt_x), cy + int(-h * fwd_y + h * rgt_y)),
    ]


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

        self.render_every       = render_every
        self.paused             = True        # start paused — press Space or click STEP
        self.show_heatmap       = False
        self.highlight_robot    = -1
        self.sim_speed          = 1.0
        self.step_once          = False
        self.should_reset       = False
        self.auto_collect       = False   # when True: force COLLECT action each step
        self.demo_score         = False   # when True: give robots balls, force SCORE actions
        self._click_this_frame  = None    # set once per frame from MOUSEBUTTONDOWN event

        # State editor
        self.selected_ball = -1           # index into field.objects, -1 = none
        self.brush_color   = BALL_BLUE    # color for right-click add

        # Setup mode
        self.setup_mode       = False
        # Tool options: "robot0", "robot1", "red_ball", "blue_ball"
        self.setup_tool       = "robot0"
        self.setup_confirmed  = False
        # Per-robot heading in user degrees (-180..+180, 0=N/up, CW). Both start facing north.
        self.setup_headings      = [0.0, 0.0]
        self.setup_slider_drag   = False      # currently dragging the heading slider
        self.setup_slider_rect   = pygame.Rect(0, 0, 1, 1)  # updated each frame

        # Pre-compute button rects (absolute screen coords)
        px = FIELD_PX + 10
        self.btn_step         = pygame.Rect(px,       8,  PANEL_W - 20, 36)
        self.btn_heatmap      = pygame.Rect(px,       0,  PANEL_W - 20, BTN_H)
        self.btn_change_color = pygame.Rect(px,       0,  BTN_W,        BTN_H)
        self.btn_delete       = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_add_red      = pygame.Rect(px,       0,  BTN_W,        BTN_H)
        self.btn_add_blue     = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_test_anim    = pygame.Rect(px,       0,  PANEL_W - 20, BTN_H)
        self.btn_brush_toggle = pygame.Rect(px,       0,  PANEL_W - 20, BTN_H)
        # Setup mode buttons (y-positions set dynamically)
        self.btn_setup_toggle  = pygame.Rect(px, 0, PANEL_W - 20, BTN_H)
        self.btn_setup_clear   = pygame.Rect(px, 0, PANEL_W - 20, BTN_H)
        self.btn_setup_robot0  = pygame.Rect(px, 0, BTN_W, BTN_H)
        self.btn_setup_robot1  = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_setup_red     = pygame.Rect(px, 0, BTN_W, BTN_H)
        self.btn_setup_blue    = pygame.Rect(px + BTN_W + 10, 0, BTN_W, BTN_H)
        self.btn_setup_confirm = pygame.Rect(px, 0, PANEL_W - 20, BTN_H)
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
            "toggle_setup": False,
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
        self.step_once         = signals["step_once"]
        self.should_reset      = signals["reset"]
        # Store single-click position for event-based button handling (avoids
        # toggle-every-frame bug when mouse button is held down).
        self._click_this_frame = signals["mouse_left"]

        if signals["toggle_setup"]:
            self.setup_mode = not self.setup_mode

        if self.setup_mode:
            # In setup mode: left-click places robot or adds ball
            if signals["mouse_left"] is not None:
                self._handle_setup_click(signals["mouse_left"], env)
            # Right-click in setup mode also adds balls (same color as active brush)
            if signals["mouse_right"] is not None:
                mx, my = signals["mouse_right"]
                if mx < FIELD_PX and my < FIELD_PY:
                    fx, fy = _from_screen(mx, my)
                    color = BALL_RED if self.setup_tool == "red_ball" else BALL_BLUE
                    self._setup_ball_placement(fx, fy, color, env)
        else:
            # Normal editor actions
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
        self._draw_route_overlay(env)

        if self.setup_mode:
            # Always draw both robots in setup mode regardless of num_allies
            for idx in range(2):
                robot = env.field.allies[idx]
                self._draw_robot(robot, BLUE if idx == 0 else LIGHT_BLUE,
                                 f"R{idx}", False, env, is_ally=True)
        else:
            num_allies = getattr(env, "num_allies", 1)
            for idx in range(num_allies):
                robot = env.field.allies[idx]
                highlighted = (idx == self.highlight_robot)
                self._draw_robot(robot, BLUE if idx == 0 else LIGHT_BLUE,
                                 f"R{idx}", highlighted, env, is_ally=True)

        num_opponents = getattr(env, "num_opponents", 0)
        for idx in range(num_opponents):
            robot = env.field.opponents[idx]
            self._draw_robot(robot, RED if idx == 0 else LIGHT_RED,
                             f"O{idx}", False, env, is_ally=False)

        # --- Score animations (balls flying to goal) — skip in setup mode ---
        if not self.setup_mode:
            self._draw_score_animations(env)

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
        """6×6 tile grid — VEX tiles are 24" each."""
        for i in range(1, 6):
            x = int(i * 24 * SCALE)
            pygame.draw.line(self.screen, (58, 58, 58), (x, 0), (x, FIELD_PY))
            y = int(i * 24 * SCALE)
            pygame.draw.line(self.screen, (58, 58, 58), (0, y), (FIELD_PX, y))

    def _goal_counts(self, env) -> dict:
        """Count balls per goal by scored_in_goal tag."""
        counts: dict[str, int] = {}
        for obj in env.field.objects:
            if obj.status in (OBJ_SCORED_US, OBJ_SCORED_OPP) and obj.scored_in_goal:
                counts[obj.scored_in_goal] = counts.get(obj.scored_in_goal, 0) + 1
        return counts

    def _draw_goals(self, env=None):
        """Draw long goals, center X-structure with diamond scoring zones, and matchload tubes."""
        counts = self._goal_counts(env) if env else {}

        # ---- Long goals (neutral — not owned by either alliance) ----
        # Each goal body: LONG_GOAL_WIDTH wide, spanning Y=LONG_GOAL_Y_MIN–MAX
        # Outer face is LONG_GOAL_WALL_GAP from the wall; inner face faces the field.
        y0 = int((FIELD_H - LONG_GOAL_Y_MAX) * SCALE)
        y1 = int((FIELD_H - LONG_GOAL_Y_MIN) * SCALE)
        h  = y1 - y0
        gap_px  = int(LONG_GOAL_WALL_GAP * SCALE)
        body_px = int(LONG_GOAL_WIDTH    * SCALE)
        GOAL_FILL = (110, 105, 90)
        GOAL_OUT  = (190, 185, 165)

        # Right goal: outer face at FIELD_PX - gap_px, body extends left
        rx_outer = FIELD_PX - gap_px
        rx_inner = rx_outer - body_px
        pygame.draw.rect(self.screen, GOAL_FILL, (rx_inner, y0, body_px, h))
        pygame.draw.rect(self.screen, GOAL_OUT,  (rx_inner, y0, body_px, h), 2)
        pygame.draw.line(self.screen, WHITE, (rx_inner, y0), (rx_inner, y1), 2)

        # Left goal: outer face at gap_px from left, body extends right
        lx_outer = gap_px
        lx_inner = lx_outer + body_px
        pygame.draw.rect(self.screen, GOAL_FILL, (lx_outer, y0, body_px, h))
        pygame.draw.rect(self.screen, GOAL_OUT,  (lx_outer, y0, body_px, h), 2)
        pygame.draw.line(self.screen, WHITE, (lx_inner, y0), (lx_inner, y1), 2)

        # Ball contents inside each long goal
        gs = env.field.goal_state if env and hasattr(env.field, "goal_state") else None
        our_balls = gs.our_long  if gs else []
        opp_balls = gs.opp_long  if gs else []
        self._draw_goal_ball_stack(our_balls, rx_inner, rx_outer, y0, y1)
        self._draw_goal_ball_stack(opp_balls, lx_outer, lx_inner, y0, y1)

        # Goal labels (outside each body, showing red/blue ball counts)
        def _rb_str(ball_list) -> str:
            r = sum(1 for _, c in ball_list if c == BALL_RED)
            b = len(ball_list) - r
            return f"[{r}r {b}b]"

        lbl = self.font_sm.render(f"R.GOAL {_rb_str(our_balls)}", True, GOAL_OUT)
        self.screen.blit(lbl, (rx_inner - lbl.get_width() - 4, y0 + h // 2 - 6))
        lbl = self.font_sm.render(f"L.GOAL {_rb_str(opp_balls)}", True, GOAL_OUT)
        self.screen.blit(lbl, (lx_inner + 4, y0 + h // 2 - 6))

        # Quadrant control indicators (small colored dots in field corners)
        if gs:
            ctrl = gs.compute_quadrant_control()
            _ctrl_positions = {
                "bottom_left":  _to_screen(12.0,  12.0),
                "bottom_right": _to_screen(132.0, 12.0),
                "top_left":     _to_screen(12.0,  132.0),
                "top_right":    _to_screen(132.0, 132.0),
            }
            for qname, (qsx, qsy) in _ctrl_positions.items():
                color = ctrl.get(qname)
                if color is None:
                    continue
                col = RED if color == BALL_RED else BLUE
                pygame.draw.circle(self.screen, col, (qsx, qsy), 8)
                pygame.draw.circle(self.screen, WHITE, (qsx, qsy), 8, 1)

        # ---- Park zones (raised platforms, top and bottom walls) ----
        park_x0, park_x1 = _to_screen(PARK_ZONE_X_MIN, 0)[0], _to_screen(PARK_ZONE_X_MAX, 0)[0]
        park_w  = park_x1 - park_x0
        # Bottom park zone (y=0 to PARK_ZONE_BOTTOM)
        bot_y0 = _to_screen(0, PARK_ZONE_BOTTOM)[1]
        bot_y1 = FIELD_PY
        pygame.draw.rect(self.screen, (50, 50, 50),
                         (park_x0, bot_y0, park_w, bot_y1 - bot_y0))
        pygame.draw.rect(self.screen, (100, 100, 100),
                         (park_x0, bot_y0, park_w, bot_y1 - bot_y0), 2)
        lbl = self.font_sm.render("PARK", True, (140, 140, 140))
        self.screen.blit(lbl, (park_x0 + park_w // 2 - lbl.get_width() // 2,
                                bot_y0 + (bot_y1 - bot_y0) // 2 - 6))
        # Top park zone (y=PARK_ZONE_TOP to 144)
        top_y0 = 0
        top_y1 = _to_screen(0, PARK_ZONE_TOP)[1]
        pygame.draw.rect(self.screen, (50, 50, 50),
                         (park_x0, top_y0, park_w, top_y1 - top_y0))
        pygame.draw.rect(self.screen, (100, 100, 100),
                         (park_x0, top_y0, park_w, top_y1 - top_y0), 2)
        lbl = self.font_sm.render("PARK", True, (140, 140, 140))
        self.screen.blit(lbl, (park_x0 + park_w // 2 - lbl.get_width() // 2,
                                top_y0 + (top_y1 - top_y0) // 2 - 6))

        # ---- Center X-structure — two clean full crossing bars ----
        # Both bars share the same fill so the overlap center is seamless.
        # MID = upper half of X (labeled above), LOW = lower half (labeled below).
        cx, cy  = 72.0, 72.0
        arm_len = CENTER_GOAL_ARM_LEN
        arm_hw  = CENTER_GOAL_ARM_W / 2.0
        x_fill    = (95, 72, 10)
        x_outline = (160, 118, 22)

        def _full_bar_pts(angle_rad):
            """Return the 4 screen-space corners of a full bar at angle_rad."""
            ca, sa = math.cos(angle_rad), math.sin(angle_rad)
            cp, sp = -sa, ca   # perpendicular
            return [
                _to_screen(cx + arm_len*ca + arm_hw*cp, cy + arm_len*sa + arm_hw*sp),
                _to_screen(cx + arm_len*ca - arm_hw*cp, cy + arm_len*sa - arm_hw*sp),
                _to_screen(cx - arm_len*ca - arm_hw*cp, cy - arm_len*sa - arm_hw*sp),
                _to_screen(cx - arm_len*ca + arm_hw*cp, cy - arm_len*sa + arm_hw*sp),
            ]

        bar1 = _full_bar_pts( math.pi / 4)   # NE–SW
        bar2 = _full_bar_pts(-math.pi / 4)   # NW–SE
        # Fill both bars (same color → seamless center)
        pygame.draw.polygon(self.screen, x_fill, bar1)
        pygame.draw.polygon(self.screen, x_fill, bar2)
        # Single-pixel outlines on top
        pygame.draw.polygon(self.screen, x_outline, bar1, 2)
        pygame.draw.polygon(self.screen, x_outline, bar2, 2)

        # Labels in the upper and lower open quadrants of the X
        mid_balls = gs.center_mid if gs else []
        low_balls = gs.center_low if gs else []
        mid_n = len(mid_balls)
        low_n = len(low_balls)
        off   = arm_len * 0.62
        msx, msy = _to_screen(cx, cy + off)
        lsx, lsy = _to_screen(cx, cy - off)

        def _goal_color_bar(ball_list) -> str:
            """Build a compact color string like 'BRRB' for the ball order."""
            return "".join("R" if c == BALL_RED else "B" for _, c in ball_list)

        mid_bar = _goal_color_bar(mid_balls)
        low_bar = _goal_color_bar(low_balls)
        lbl = self.font_sm.render(f"MID [{mid_n}] {mid_bar}", True, ORANGE)
        self.screen.blit(lbl, (msx - lbl.get_width() // 2, msy - 6))
        lbl = self.font_sm.render(f"LOW [{low_n}] {low_bar}", True, YELLOW)
        self.screen.blit(lbl, (lsx - lbl.get_width() // 2, lsy - 6))

        # ---- Matchload tubes (top-down: orange circles at tile-corner positions) ----
        TUBE_COL     = (200, 130,  30)
        TUBE_OUTLINE = (255, 180,  60)
        for tx, ty in MATCHLOAD_TUBES:
            sx, sy = _to_screen(tx, ty)
            r = max(4, int(MATCHLOAD_TUBE_RADIUS * SCALE))
            pygame.draw.circle(self.screen, TUBE_COL,     (sx, sy), r)
            pygame.draw.circle(self.screen, TUBE_OUTLINE, (sx, sy), r, 2)

    def _draw_goal_ball_stack(self, ball_list, x_lo_px: int, x_hi_px: int,
                              y_top_px: int, y_bot_px: int):
        """Draw stacked colored dots for balls in a long goal.

        ball_list: ordered [(ball_idx, color), ...] — index 0 = south/first outer.
        x_lo_px/x_hi_px: screen x bounds of the goal body.
        y_top_px: screen y for north end (small y value = top of screen).
        y_bot_px: screen y for south end (large y value = bottom of screen).
        """
        if not ball_list:
            return
        n   = len(ball_list)
        cx  = (x_lo_px + x_hi_px) // 2
        for i, (_, color) in enumerate(ball_list):
            # i=0 → south outer (near y_bot_px), i=n-1 → north outer (near y_top_px)
            t  = (i + 0.5) / n
            sy = int(y_bot_px - t * (y_bot_px - y_top_px))
            dot_col = RED      if color == BALL_RED else BLUE
            dot_out = LIGHT_RED if color == BALL_RED else LIGHT_BLUE
            r = 4
            pygame.draw.circle(self.screen, dot_col, (cx, sy), r)
            pygame.draw.circle(self.screen, dot_out, (cx, sy), r, 1)
            # White ring on the two outermost positions
            if i == 0 or i == n - 1:
                pygame.draw.circle(self.screen, WHITE, (cx, sy), r + 2, 1)

    def _draw_balls(self, env):
        """Draw all on-field and scored balls with correct colors.
        Fast-rolling balls get a speed trail behind them."""
        TRAIL_SPEED_MIN = 6.0   # in/s — faster than this gets a trail

        for i, obj in enumerate(env.field.objects):
            if obj.status == OBJ_REMOVED:
                continue

            # Ball colour based on alliance color and status
            if obj.status == OBJ_ON_FIELD:
                color   = RED      if obj.color == BALL_RED else BLUE
                outline = LIGHT_RED if obj.color == BALL_RED else LIGHT_BLUE
            elif obj.status == OBJ_HELD:
                continue   # held balls are inside the robot — not visible on field
            elif obj.status in (OBJ_SCORED_US, OBJ_SCORED_OPP):
                continue   # rendered by _draw_goal_ball_stack
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

    def _draw_route_overlay(self, env):
        """Draw the recommended ball-collection route for each active robot."""
        from sim.route_planner import compute_collection_route
        if self.setup_mode:
            return

        num_allies = getattr(env, "num_allies", 1)
        palette = [(80, 240, 120), (255, 200, 60)]   # green for R0, yellow for R1

        for idx in range(num_allies):
            robot  = env.field.allies[idx]
            route  = compute_collection_route(
                robot.position, env.field,
                already_held=robot.balls_held,
                max_volley=5,
                robot=robot,
            )
            if not route:
                continue

            col = palette[idx % len(palette)]
            # Build waypoint screen positions: robot → waypoint1 → waypoint2 → …
            waypoints = [_to_screen(robot.x, robot.y)]
            for ball_indices, wpos, score in route:
                waypoints.append(_to_screen(float(wpos[0]), float(wpos[1])))

            # Draw dashed path
            route_surf = pygame.Surface((FIELD_PX, FIELD_PY), pygame.SRCALPHA)
            dash_len = 8
            gap_len  = 5
            for i in range(len(waypoints) - 1):
                sx0, sy0 = waypoints[i]
                sx1, sy1 = waypoints[i + 1]
                dx, dy = sx1 - sx0, sy1 - sy0
                seg_len = math.sqrt(dx * dx + dy * dy)
                if seg_len < 1:
                    continue
                nx, ny = dx / seg_len, dy / seg_len
                t = 0.0
                drawing = True
                while t < seg_len:
                    t_end = min(t + (dash_len if drawing else gap_len), seg_len)
                    if drawing:
                        p0 = (int(sx0 + nx * t),     int(sy0 + ny * t))
                        p1 = (int(sx0 + nx * t_end),  int(sy0 + ny * t_end))
                        pygame.draw.line(route_surf, (*col, 180), p0, p1, 2)
                    t = t_end
                    drawing = not drawing

            self.screen.blit(route_surf, (0, 0))

            # Number each waypoint; larger circle for multi-ball clusters
            for order, (ball_indices, wpos, score) in enumerate(route, start=1):
                sx, sy = _to_screen(float(wpos[0]), float(wpos[1]))
                cluster_size = len(ball_indices)
                radius = 8 + (cluster_size - 1) * 3   # bigger ring for clusters
                pygame.draw.circle(self.screen, col, (sx, sy), radius, 2)
                label = str(order) if cluster_size == 1 else f"{order}×{cluster_size}"
                num_surf = self.font_sm.render(label, True, col)
                self.screen.blit(num_surf, (sx - num_surf.get_width() // 2,
                                            sy - num_surf.get_height() // 2))

    def _draw_vision_cone(self, robot, cone_color=(80, 160, 255)):
        """Draw a semi-transparent wide-angle vision cone in front of the robot."""
        NUM_SEGS = 20
        heading  = robot.heading
        rx, ry   = robot.x, robot.y

        # Build polygon: robot centre + arc of NUM_SEGS+1 points
        pts = [_to_screen(rx, ry)]
        for i in range(NUM_SEGS + 1):
            frac  = i / NUM_SEGS
            angle = heading - VISION_HALF_ANGLE + frac * 2 * VISION_HALF_ANGLE
            px = rx + VISION_RANGE * math.cos(angle)
            py = ry + VISION_RANGE * math.sin(angle)
            # Clamp to field bounds
            px = max(0.0, min(float(FIELD_W), px))
            py = max(0.0, min(float(FIELD_H), py))
            pts.append(_to_screen(px, py))

        if len(pts) >= 3:
            cone_surf = pygame.Surface((FIELD_PX, FIELD_PY), pygame.SRCALPHA)
            pygame.draw.polygon(cone_surf, (*cone_color, 22), pts)   # very faint fill
            # Arc outline and side edges on same alpha surface
            for i in range(1, len(pts) - 1):
                pygame.draw.line(cone_surf, (*cone_color, 70), pts[i], pts[i + 1], 1)
            pygame.draw.line(cone_surf, (*cone_color, 90), pts[0], pts[1],  1)
            pygame.draw.line(cone_surf, (*cone_color, 90), pts[0], pts[-1], 1)
            self.screen.blit(cone_surf, (0, 0))

    def _draw_robot(self, robot, color, label_prefix, highlighted, env, is_ally=True):
        """Draw a 15x15in robot square (rotated to heading) with arrow, vision cone, and target line."""
        sx, sy = _to_screen(robot.x, robot.y)
        half   = int(ROBOT_W / 2 * SCALE)
        border = 3 if highlighted else 2

        # Vision cone behind the robot so it draws on top
        if is_ally:
            self._draw_vision_cone(robot)

        # Faint footprint boundary — shows the physical 15" extents the path
        # planner respects (drawn before body so body renders on top)
        boundary_surf = pygame.Surface((FIELD_PX, FIELD_PY), pygame.SRCALPHA)
        bnd_corners = _robot_corners(sx, sy, half + 3, robot.heading)
        pygame.draw.polygon(boundary_surf, (*color, 35), bnd_corners)   # very faint fill
        pygame.draw.polygon(boundary_surf, (*color, 90), bnd_corners, 1)  # faint outline
        self.screen.blit(boundary_surf, (0, 0))

        # Rotated robot body (polygon that matches actual heading)
        corners = _robot_corners(sx, sy, half, robot.heading)
        pygame.draw.polygon(self.screen, color, corners, border)

        # ── Held balls: small colored dots inside robot body ──────────────────
        if robot.balls_held > 0 and hasattr(env, "field"):
            # Screen-space directional unit vectors
            rgt_x =  math.sin(robot.heading)
            rgt_y =  math.cos(robot.heading)   # y-down: cos gives correct right
            DOT_R   = 4
            spacing = (half * 2 - 6) / max(robot.balls_held, 1)
            for i, obj_idx in enumerate(robot.held_object_ids):
                obj_color = env.field.objects[obj_idx].color
                dot_col  = RED      if obj_color == BALL_RED else BLUE
                dot_ring = LIGHT_RED if obj_color == BALL_RED else LIGHT_BLUE
                # Position: evenly spread across robot width
                t    = -half + 3 + int(spacing * (i + 0.5))
                bx   = sx + int(t * rgt_x)
                by   = sy + int(t * rgt_y)
                pygame.draw.circle(self.screen, dot_col,  (bx, by), DOT_R)
                pygame.draw.circle(self.screen, dot_ring, (bx, by), DOT_R, 1)

        # ── Intake face indicator ───────────────────────────────────────────
        # Front-face screen endpoints  (corners[1] = front-left, corners[0] = front-right)
        fl = corners[1]   # front-left
        fr = corners[0]   # front-right
        if is_ally and getattr(robot, "intake_active", False):
            # Spinning intake — animated green conveyor dots
            t_ms  = pygame.time.get_ticks()
            phase = (t_ms % 400) / 400.0    # 0..1 per 400ms cycle
            pygame.draw.line(self.screen, GREEN, fl, fr, 2)
            for i in range(3):
                p   = (phase + i / 3.0) % 1.0
                dx  = int(fl[0] + p * (fr[0] - fl[0]))
                dy  = int(fl[1] + p * (fr[1] - fl[1]))
                bright = int(120 + 135 * abs(math.sin(p * math.pi)))
                pygame.draw.circle(self.screen, (0, bright, 0), (dx, dy), 3)
        else:
            # Stopped / wall — thick orange bar
            wall_col = ORANGE if is_ally else (180, 100, 30)
            pygame.draw.line(self.screen, wall_col, fl, fr, 4)

        # Heading arrow (from center toward forward face)
        arrow_len = half + 6
        ax = sx + int(arrow_len * math.cos(-robot.heading))
        ay = sy + int(arrow_len * math.sin(-robot.heading))
        pygame.draw.line(self.screen, WHITE, (sx, sy), (ax, ay), 2)

        # Target line
        if robot.target is not None and robot.moving:
            tx, ty = _to_screen(robot.target[0], robot.target[1])
            pygame.draw.line(self.screen, (90, 90, 90), (sx, sy), (tx, ty), 1)
            pygame.draw.circle(self.screen, (140, 140, 140), (tx, ty), 3)

        # Label (below robot)
        action_name = ""
        if hasattr(env, "current_actions") and is_ally:
            for i, ally in enumerate(env.field.allies):
                if ally is robot:
                    action_name = Action(env.current_actions[i]).name[:12]
                    break
        intake_sym = ">" if getattr(robot, "intake_active", False) else "|"
        lbl = f"{label_prefix}{intake_sym}({robot.balls_held}) {action_name}"
        surf = self.font_sm.render(lbl, True, WHITE)
        self.screen.blit(surf, (sx - half, sy + half + 3))

    def _start_demo_score(self, env):
        """Give each robot a full load of balls and activate DEMO SCORE mode.

        The main loop sees renderer.demo_score=True and forces SCORE_LONG_GOAL
        actions until all robots have finished scoring (balls_held == 0).
        Balls are taken from on-field objects nearest each robot.
        """
        from sim.config import MAX_CARRY, OBJ_HELD, OBJ_ON_FIELD

        num_allies = getattr(env, "num_allies", 1)
        for i in range(num_allies):
            robot = env.field.allies[i]
            if robot.balls_held >= MAX_CARRY:
                continue  # already loaded

            # Find nearest on-field balls and hand them to this robot
            available = [
                obj for obj in env.field.objects
                if obj.status == OBJ_ON_FIELD
            ]
            # Sort by distance to robot
            available.sort(key=lambda o: (o.x - robot.x) ** 2 + (o.y - robot.y) ** 2)
            for obj in available:
                if robot.balls_held >= MAX_CARRY:
                    break
                obj.status = OBJ_HELD
                robot.held_object_ids.append(obj.obj_id)
                robot.balls_held += 1

        self.demo_score = True
        self.auto_collect = False   # override auto-collect so main loop sees demo_score
        self.paused = False         # start running

    def _draw_score_animations(self, env):
        """Animate balls flying from robot position to goal on score events.

        Each animation dict: {x0, y0, x1, y1, color, start_ms, duration}
        start_ms is None until first draw, then set to pygame.time.get_ticks().
        Finished animations are removed from env.score_animations.
        """
        if not hasattr(env, "score_animations") or not env.score_animations:
            return

        now_ms = pygame.time.get_ticks()
        to_remove = []

        for i, anim in enumerate(env.score_animations):
            # Initialise start time on first render, honouring optional stagger delay
            if anim["start_ms"] is None:
                delay_ms = anim.get("_delay_ms", 0)
                anim["start_ms"] = now_ms + delay_ms

            if now_ms < anim["start_ms"]:
                continue   # stagger delay not yet elapsed

            elapsed = (now_ms - anim["start_ms"]) / 1000.0
            t = elapsed / anim["duration"]

            if t >= 1.0:
                to_remove.append(i)
                continue

            # Linear interpolation with a parabolic arc upward
            x = anim["x0"] + t * (anim["x1"] - anim["x0"])
            y = anim["y0"] + t * (anim["y1"] - anim["y0"])
            arc_h = 10.0               # peak height of arc in field inches
            y_arc = arc_h * 4.0 * t * (1.0 - t)   # parabola: 0 at ends, arc_h at t=0.5
            y += y_arc

            sx, sy = _to_screen(x, y)

            color   = RED      if anim["color"] == BALL_RED else BLUE
            outline = LIGHT_RED if anim["color"] == BALL_RED else LIGHT_BLUE

            # Size pulses slightly and fades toward end
            radius = max(3, int(5 + 2 * math.sin(t * math.pi)))
            alpha  = int(255 * (1.0 - 0.4 * t))

            # Draw with alpha by blending on a small surface
            dot_surf = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
            cx_ = radius + 4
            pygame.draw.circle(dot_surf, (*color,   alpha), (cx_, cx_), radius)
            pygame.draw.circle(dot_surf, (*outline, alpha), (cx_, cx_), radius, 1)
            pygame.draw.circle(dot_surf, (255, 255, 255, alpha // 2), (cx_, cx_), radius + 2, 1)
            self.screen.blit(dot_surf, (sx - cx_, sy - cx_))

        # Remove completed (reverse order to keep indices valid)
        for i in reversed(to_remove):
            env.score_animations.pop(i)

    def _draw_heatmap(self, env):
        # Use a high-res display heatmap (48×48 = 3" cells) independent of RL state
        from sim.heatmap import compute_heatmap as _hm
        DISP_W, DISP_H = 48, 48
        heatmap = _hm(
            env.field.get_obj_positions(),
            env.field.get_obj_statuses(),
            w=DISP_W, h=DISP_H, sigma=1.8,
        )
        mx     = heatmap.max() if heatmap.max() > 0 else 1.0
        cell_w = FIELD_PX / DISP_W
        cell_h = FIELD_PY / DISP_H

        overlay = pygame.Surface((FIELD_PX, FIELD_PY), pygame.SRCALPHA)
        for gy in range(DISP_H):
            for gx in range(DISP_W):
                val = heatmap[gy, gx] / mx
                if val > 0.02:
                    screen_gy = DISP_H - 1 - gy
                    rect = pygame.Rect(int(gx * cell_w), int(screen_gy * cell_h),
                                       int(cell_w) + 1, int(cell_h) + 1)
                    pygame.draw.rect(overlay,
                                     (int(val * 255), int((1 - val) * 80), 40, int(val * 110)),
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

        if self.setup_mode:
            self._draw_panel_setup(env, mouse)
        else:
            y = self._draw_panel_step_btn(env, y_start=8, mouse=mouse)
            y = self._draw_panel_timer(env, y + 6)
            y = self._draw_panel_section("SCOREBOARD", y + 8)
            y = self._draw_panel_scoreboard(env, y)
            y = self._draw_panel_section("ROBOTS", y + 8)
            y = self._draw_panel_robots(env, y)
            y = self._draw_panel_section("GOALS", y + 8)
            y = self._draw_panel_goals(env, y)
            y = self._draw_panel_section("TOOLS", y + 8)
            y = self._draw_panel_editor(env, y, mouse)
            y = self._draw_panel_section("KEYS", y + 8)
            self._draw_panel_controls(y)

    def _draw_panel_setup(self, env, mouse) -> None:
        """Full-panel UI for setup mode: place robots + balls, then Confirm."""
        px = FIELD_PX + 10
        bw = PANEL_W - 20
        y  = 8

        # Title bar
        title_surf = self.font_hd.render("SETUP MODE", True, YELLOW)
        self.screen.blit(title_surf,
                         (FIELD_PX + bw // 2 - title_surf.get_width() // 2 + 10, y))
        y += title_surf.get_height() + 6

        # Subtitle
        sub = self.font_sm.render("Click field to place selected tool", True, LIGHT_GRAY)
        self.screen.blit(sub, (px, y))
        y += 18

        # ── CLEAR ALL ──
        self.btn_setup_clear.update(px, y, bw, BTN_H)
        hover = self.btn_setup_clear.collidepoint(mouse)
        bg = (100, 40, 40) if hover else (70, 20, 20)
        pygame.draw.rect(self.screen, bg,       self.btn_setup_clear, border_radius=3)
        pygame.draw.rect(self.screen, LIGHT_RED, self.btn_setup_clear, 1, border_radius=3)
        lbl = self.font_sm.render("CLEAR ALL BALLS", True, LIGHT_RED)
        self.screen.blit(lbl, (self.btn_setup_clear.x + self.btn_setup_clear.w // 2 - lbl.get_width() // 2,
                                self.btn_setup_clear.y + (BTN_H - lbl.get_height()) // 2))
        if self._click_this_frame is not None and self.btn_setup_clear.collidepoint(self._click_this_frame):
            env.field.clear_all_balls()
        y += BTN_H + 8

        # ── Section label ──
        pygame.draw.line(self.screen, ACCENT, (px, y), (px + bw, y), 1)
        self.screen.blit(self.font_lg.render("PLACE TOOL", True, ACCENT), (px, y + 3))
        y += 22

        # ── Robot 0 / Robot 1 buttons ──
        self.btn_setup_robot0.update(px, y, BTN_W, BTN_H)
        self.btn_setup_robot1.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        for btn, label, tool, col in [
            (self.btn_setup_robot0, "Robot 0",  "robot0", LIGHT_BLUE),
            (self.btn_setup_robot1, "Robot 1",  "robot1", (180, 210, 255)),
        ]:
            active = (self.setup_tool == tool)
            bg  = (20, 50, 120) if active else (28, 28, 42)
            bdr = col if active else LIGHT_GRAY
            hover = btn.collidepoint(mouse)
            if hover:
                bg = tuple(min(255, c + 15) for c in bg)
            pygame.draw.rect(self.screen, bg,  btn, border_radius=3)
            pygame.draw.rect(self.screen, bdr, btn, 2 if active else 1, border_radius=3)
            s = self.font_sm.render(label, True, col)
            self.screen.blit(s, (btn.x + btn.w // 2 - s.get_width() // 2,
                                 btn.y + (BTN_H - s.get_height()) // 2))
            if pygame.mouse.get_pressed()[0] and hover:
                self.setup_tool = tool
        y += BTN_H + 6

        # ── Red ball / Blue ball buttons ──
        self.btn_setup_red.update(px, y, BTN_W, BTN_H)
        self.btn_setup_blue.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        for btn, label, tool, col in [
            (self.btn_setup_red,  "Red Ball",  "red_ball",  LIGHT_RED),
            (self.btn_setup_blue, "Blue Ball", "blue_ball", LIGHT_BLUE),
        ]:
            active = (self.setup_tool == tool)
            bg  = (80, 20, 20) if (active and col == LIGHT_RED) else (20, 40, 100) if active else (28, 28, 42)
            bdr = col if active else LIGHT_GRAY
            hover = btn.collidepoint(mouse)
            if hover:
                bg = tuple(min(255, c + 15) for c in bg)
            pygame.draw.rect(self.screen, bg,  btn, border_radius=3)
            pygame.draw.rect(self.screen, bdr, btn, 2 if active else 1, border_radius=3)
            s = self.font_sm.render(label, True, col)
            self.screen.blit(s, (btn.x + btn.w // 2 - s.get_width() // 2,
                                 btn.y + (BTN_H - s.get_height()) // 2))
            if pygame.mouse.get_pressed()[0] and hover:
                self.setup_tool = tool
        y += BTN_H + 8

        # ── Heading slider (shown only when a robot tool is active) ──
        if self.setup_tool in ("robot0", "robot1"):
            robot_idx = 0 if self.setup_tool == "robot0" else 1
            hdeg      = self.setup_headings[robot_idx]   # -180..+180

            pygame.draw.line(self.screen, ACCENT, (px, y), (px + bw, y), 1)
            y += 4

            # Slider track
            track = pygame.Rect(px, y + 12, bw, 10)
            self.setup_slider_rect = pygame.Rect(track)   # save for drag detection
            pygame.draw.rect(self.screen, (50, 50, 70), track, border_radius=4)

            # Tick marks at -180, -90, 0, +90, +180
            for tick_deg in (-180, -90, 0, 90, 180):
                tx = track.x + int((tick_deg + 180) / 360.0 * track.w)
                tcol = YELLOW if tick_deg == 0 else (80, 80, 100)
                th   = 14 if tick_deg == 0 else 8
                pygame.draw.line(self.screen, tcol,
                                 (tx, track.centery - th // 2),
                                 (tx, track.centery + th // 2), 1)
                if tick_deg in (-180, -90, 0, 90, 180):
                    lbl_s = self.font_sm.render(str(tick_deg), True, tcol)
                    self.screen.blit(lbl_s, (tx - lbl_s.get_width() // 2, track.y - 14))

            # Filled bar from center to handle
            t_val  = (hdeg + 180.0) / 360.0
            cx_    = track.x + track.w // 2
            hx_    = track.x + int(t_val * track.w)
            bar_r  = pygame.Rect(min(cx_, hx_), track.y, abs(hx_ - cx_) + 1, track.h)
            pygame.draw.rect(self.screen, ACCENT, bar_r)
            pygame.draw.rect(self.screen, LIGHT_GRAY, track, 1, border_radius=4)

            # Drag handle
            pygame.draw.circle(self.screen, LIGHT_BLUE, (hx_, track.centery), 9)
            pygame.draw.circle(self.screen, WHITE,      (hx_, track.centery), 9, 2)

            # Handle mouse drag
            mouse_pressed = pygame.mouse.get_pressed()[0]
            expanded_track = track.inflate(0, 24)
            if mouse_pressed and (expanded_track.collidepoint(mouse) or self.setup_slider_drag):
                self.setup_slider_drag = True
                raw_t = max(0.0, min(1.0, (mouse[0] - track.x) / max(track.w, 1)))
                raw_deg = raw_t * 360.0 - 180.0
                # Snap to 1° increments
                raw_deg = round(raw_deg)
                raw_deg = max(-180.0, min(180.0, raw_deg))
                self.setup_headings[robot_idx] = raw_deg
                env.field.set_robot_heading(robot_idx, _sim_heading_from_user(raw_deg))
            elif not mouse_pressed:
                self.setup_slider_drag = False

            y += 36   # track height + label row

            # Value readout
            dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            deg360 = hdeg % 360
            compass = dirs[int((deg360 + 22.5) / 45) % 8]
            val_s = self.font_sm.render(f"Heading: {hdeg:+.0f}°  ({compass})", True, LIGHT_BLUE)
            self.screen.blit(val_s, (px + bw // 2 - val_s.get_width() // 2, y))
            y += 16

        # ── Current tool indicator ──
        tool_label = {
            "robot0":   "Active: Place Robot 0",
            "robot1":   "Active: Place Robot 1",
            "red_ball": "Active: Place Red Ball",
            "blue_ball":"Active: Place Blue Ball",
        }.get(self.setup_tool, "")
        tool_col = (LIGHT_BLUE if "robot" in self.setup_tool
                    else LIGHT_RED if "red" in self.setup_tool else LIGHT_BLUE)
        self.screen.blit(self.font_sm.render(tool_label, True, tool_col), (px, y))
        y += 18

        # ── Ball count ──
        n_balls = sum(1 for o in env.field.objects if o.status not in (4,))
        n_red   = sum(1 for o in env.field.objects if o.status not in (4,) and o.color == BALL_RED)
        n_blue  = sum(1 for o in env.field.objects if o.status not in (4,) and o.color == BALL_BLUE)
        self.screen.blit(self.font_sm.render(f"Balls placed: {n_balls}  (R:{n_red} B:{n_blue})",
                                              True, LIGHT_GRAY), (px, y))
        y += 18

        # Robot positions
        for i, robot in enumerate(env.field.allies):
            col = LIGHT_BLUE if i == 0 else (180, 210, 255)
            self.screen.blit(
                self.font_sm.render(f"Robot {i}: ({robot.x:.0f}, {robot.y:.0f})", True, col),
                (px, y))
            y += 15
        y += 4

        # ── CONFIRM ──
        pygame.draw.line(self.screen, ACCENT, (px, y), (px + bw, y), 1)
        y += 4
        self.btn_setup_confirm.update(px, y, bw, 36)
        hover = self.btn_setup_confirm.collidepoint(mouse)
        bg = (30, 120, 50) if hover else (20, 90, 35)
        pygame.draw.rect(self.screen, bg,   self.btn_setup_confirm, border_radius=4)
        pygame.draw.rect(self.screen, GREEN, self.btn_setup_confirm, 2, border_radius=4)
        clbl = self.font_lg.render("CONFIRM & START", True, GREEN)
        self.screen.blit(clbl, (self.btn_setup_confirm.x + self.btn_setup_confirm.w // 2 - clbl.get_width() // 2,
                                self.btn_setup_confirm.y + self.btn_setup_confirm.h // 2 - clbl.get_height() // 2))
        if self._click_this_frame is not None and self.btn_setup_confirm.collidepoint(self._click_this_frame):
            self.setup_mode      = False
            self.setup_confirmed = True
            env.setup_reset()             # partial reset: keep positions, clear scores/physics

        y += 40

        # ── Exit setup (without confirming) ──
        self.btn_setup_toggle.update(px, y, bw, BTN_H)
        hover = self.btn_setup_toggle.collidepoint(mouse)
        bg = (50, 50, 70) if hover else (35, 35, 50)
        pygame.draw.rect(self.screen, bg,         self.btn_setup_toggle, border_radius=3)
        pygame.draw.rect(self.screen, LIGHT_GRAY, self.btn_setup_toggle, 1, border_radius=3)
        xlbl = self.font_sm.render("Cancel (exit setup)", True, LIGHT_GRAY)
        self.screen.blit(xlbl, (self.btn_setup_toggle.x + self.btn_setup_toggle.w // 2 - xlbl.get_width() // 2,
                                self.btn_setup_toggle.y + (BTN_H - xlbl.get_height()) // 2))
        if self._click_this_frame is not None and self.btn_setup_toggle.collidepoint(self._click_this_frame):
            self.setup_mode = False

    def _draw_panel_section(self, title: str, y: int) -> int:
        """Draw a section header. Returns y after header."""
        pygame.draw.line(self.screen, ACCENT,
                         (FIELD_PX + 4, y), (FIELD_PX + PANEL_W - 4, y), 1)
        surf = self.font_lg.render(title, True, ACCENT)
        self.screen.blit(surf, (FIELD_PX + 8, y + 3))
        return y + 22

    def _draw_panel_step_btn(self, env, y_start: int, mouse) -> int:
        px  = FIELD_PX + 10
        bw  = PANEL_W - 20
        half = (bw - 6) // 2

        # ── STEP button (left half) ──
        step_rect = pygame.Rect(px, y_start, half, 36)
        hover_step = step_rect.collidepoint(mouse)
        sc = (60, 120, 255) if hover_step else (40, 90, 200)
        pygame.draw.rect(self.screen, sc, step_rect, border_radius=4)
        pygame.draw.rect(self.screen, LIGHT_BLUE, step_rect, 1, border_radius=4)
        slbl = self.font_lg.render("STEP (S)", True, WHITE)
        self.screen.blit(slbl, (step_rect.centerx - slbl.get_width() // 2,
                                step_rect.centery - slbl.get_height() // 2))
        if self._click_this_frame is not None and step_rect.collidepoint(self._click_this_frame):
            self.step_once = True
            self.paused    = True

        # ── RUN / PAUSE button (right half) ──
        run_rect = pygame.Rect(px + half + 6, y_start, half, 36)
        hover_run = run_rect.collidepoint(mouse)
        if self.paused:
            rc = (30, 130, 60) if hover_run else (20, 100, 40)
            rlabel = "▶ RUN"
            rout   = GREEN
        else:
            rc = (140, 80, 20) if hover_run else (110, 60, 10)
            rlabel = "⏸ PAUSE"
            rout   = ORANGE
        pygame.draw.rect(self.screen, rc,   run_rect, border_radius=4)
        pygame.draw.rect(self.screen, rout, run_rect, 1, border_radius=4)
        rlbl = self.font_lg.render(rlabel, True, WHITE)
        self.screen.blit(rlbl, (run_rect.centerx - rlbl.get_width() // 2,
                                run_rect.centery - rlbl.get_height() // 2))
        if self._click_this_frame is not None and run_rect.collidepoint(self._click_this_frame):
            self.paused = not self.paused

        y = y_start + 36 + 4

        # ── AUTO COLLECT toggle ──
        ac_rect = pygame.Rect(px, y, bw, BTN_H)
        hover_ac = ac_rect.collidepoint(mouse)
        if self.auto_collect:
            ac_bg  = (20, 90, 50) if hover_ac else (10, 70, 35)
            ac_fg  = GREEN
            ac_lbl = "AUTO PLAY: ON  (greedy collect+score)"
        else:
            ac_bg  = (45, 45, 60) if hover_ac else (35, 35, 50)
            ac_fg  = LIGHT_GRAY
            ac_lbl = "AUTO PLAY: OFF (random actions)"
        pygame.draw.rect(self.screen, ac_bg, ac_rect, border_radius=3)
        pygame.draw.rect(self.screen, ac_fg, ac_rect, 1, border_radius=3)
        als = self.font_sm.render(ac_lbl, True, ac_fg)
        self.screen.blit(als, (ac_rect.x + ac_rect.w // 2 - als.get_width() // 2,
                               ac_rect.y + (BTN_H - als.get_height()) // 2))
        if self._click_this_frame is not None and ac_rect.collidepoint(self._click_this_frame):
            self.auto_collect = not self.auto_collect

        # Keep btn_step rect updated so keyboard shortcut still works
        self.btn_step.update(px, y_start, bw, 36)

        return y + BTN_H

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
        if self._click_this_frame is not None and self.btn_heatmap.collidepoint(self._click_this_frame):
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

    def _draw_panel_scoreboard(self, env, y_start: int) -> int:
        """Compact scoreboard: us vs opp with margin and ball counts."""
        y  = y_start
        px = FIELD_PX + 10
        bw = PANEL_W - 20

        us  = env.field.my_score
        opp = env.field.opponent_score
        diff = us - opp

        # Score boxes side by side
        half = (bw - 10) // 2
        # Us box
        us_rect = pygame.Rect(px, y, half, 32)
        pygame.draw.rect(self.screen, (15, 30, 70), us_rect, border_radius=4)
        pygame.draw.rect(self.screen, LIGHT_BLUE, us_rect, 1, border_radius=4)
        us_lbl = self.font_sm.render("US (Blue)", True, LIGHT_BLUE)
        us_val = self.font_lg.render(str(us), True, WHITE)
        self.screen.blit(us_lbl, (us_rect.x + 6, us_rect.y + 2))
        self.screen.blit(us_val, (us_rect.right - us_val.get_width() - 8,
                                  us_rect.y + 8))
        # Opp box
        opp_rect = pygame.Rect(px + half + 10, y, half, 32)
        pygame.draw.rect(self.screen, (60, 15, 15), opp_rect, border_radius=4)
        pygame.draw.rect(self.screen, LIGHT_RED, opp_rect, 1, border_radius=4)
        opp_lbl = self.font_sm.render("OPP (Red)", True, LIGHT_RED)
        opp_val = self.font_lg.render(str(opp), True, WHITE)
        self.screen.blit(opp_lbl, (opp_rect.x + 6, opp_rect.y + 2))
        self.screen.blit(opp_val, (opp_rect.right - opp_val.get_width() - 8,
                                   opp_rect.y + 8))
        y += 36

        # Margin + balls on field
        if diff > 0:
            d_col, d_pfx = GREEN, "+"
        elif diff < 0:
            d_col, d_pfx = LIGHT_RED, ""
        else:
            d_col, d_pfx = LIGHT_GRAY, ""

        on_field  = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD)
        red_field = sum(1 for o in env.field.objects if o.status == OBJ_ON_FIELD and o.color == BALL_RED)
        blu_field = on_field - red_field
        info = f"Margin: {d_pfx}{diff}   |   Field: {on_field} ({red_field}r {blu_field}b)"
        self.screen.blit(self.font_sm.render(info, True, d_col), (px, y))
        y += 16

        return y

    def _draw_panel_robots(self, env, y_start: int) -> int:
        """Per-robot status card: position, heading, held, action, context."""
        from sim.route_planner import compute_collection_route
        from sim.config import Action as Act, FIELD_W

        y  = y_start
        px = FIELD_PX + 10
        bw = PANEL_W - 20

        num_allies = getattr(env, "num_allies", 2)
        for i in range(num_allies):
            robot = env.field.allies[i]
            r_col = LIGHT_BLUE if i == 0 else (180, 210, 255)

            act_id = int(env.current_actions[i]) if hasattr(env, "current_actions") else 0
            try:
                act = Act(act_id)
                act_name = act.name
            except ValueError:
                act_name = str(act_id)
                act = None

            # Robot card background
            card = pygame.Rect(px, y, bw, 52)
            pygame.draw.rect(self.screen, SECTION_BG, card, border_radius=4)
            pygame.draw.rect(self.screen, r_col, card, 1, border_radius=4)

            # Header line: R0  (3 balls)  SCORE_LONG_GOAL
            held = robot.balls_held
            held_str = f"{held} ball{'s' if held != 1 else ''}" if held > 0 else "empty"
            hdr = f"R{i}  ({held_str})"
            self.screen.blit(self.font.render(hdr, True, r_col), (px + 6, y + 3))
            act_surf = self.font_sm.render(act_name, True, YELLOW)
            self.screen.blit(act_surf, (card.right - act_surf.get_width() - 8, y + 4))

            # Position + heading line
            pos_line = f"({robot.x:.0f}, {robot.y:.0f})  {_user_heading_label(robot.heading)}"
            self.screen.blit(self.font_sm.render(pos_line, True, LIGHT_GRAY), (px + 6, y + 19))

            # Context line (what robot is doing)
            ctx = ""
            col2 = LIGHT_GRAY
            if act == Act.SCORE_LONG_GOAL:
                from sim.config import OUR_LONG_GOAL, OPP_LONG_GOAL
                if robot.x >= FIELD_W / 2:
                    goal_pos, goal_name = OUR_LONG_GOAL, "R.Goal"
                else:
                    goal_pos, goal_name = OPP_LONG_GOAL, "L.Goal"
                dist = float(np.linalg.norm(robot.position - goal_pos))
                if robot.score_timer > 0 and held > 0:
                    fill = int(robot.score_timer / (1.5 / max(held, 1)) * 5)
                    bar = "\u2593" * fill + "\u2591" * (5 - fill)
                    ctx = f"-> {goal_name} {dist:.0f}\"  [{bar}]"
                    col2 = YELLOW
                else:
                    ctx = f"-> {goal_name} {dist:.0f}\"  (navigating)"
            elif act == Act.COLLECT_NEAREST_BALL:
                route = compute_collection_route(
                    robot.position, env.field,
                    already_held=held, max_volley=1, robot=robot,
                )
                if route:
                    wp = route[0][1]
                    d = float(np.linalg.norm(robot.position - wp))
                    n_b = len(route[0][0])
                    ctx = f"-> {d:.0f}\" away  ({n_b} ball cluster)"
                    col2 = GREEN
                else:
                    ctx = "no accessible balls"
                    col2 = LIGHT_RED
            elif act == Act.IDLE:
                ctx = "idle"
            else:
                ctx = act_name.lower()
            self.screen.blit(self.font_sm.render(ctx, True, col2), (px + 6, y + 34))

            y += 56

        return y

    def _draw_panel_goals(self, env, y_start: int) -> int:
        """Show ordered ball contents per goal with color dots."""
        y  = y_start
        px = FIELD_PX + 10

        gs = getattr(env.field, "goal_state", None)
        if gs is None:
            return y

        all_goals = [
            ("our_long",   "Right Long Goal"),
            ("opp_long",   "Left Long Goal"),
            ("center_mid", "Center MID"),
            ("center_low", "Center LOW"),
        ]

        for gname, label in all_goals:
            lst = gs._list(gname)
            n_r = sum(1 for _, c in lst if c == BALL_RED)
            n_b = len(lst) - n_r
            count_str = f"{n_r}r {n_b}b" if lst else "empty"
            header = self.font_sm.render(f"{label}  ({count_str})", True, LIGHT_GRAY)
            self.screen.blit(header, (px, y))
            y += 14
            if lst:
                sx = px + 4
                for i, (_, col) in enumerate(lst):
                    dot_col = LIGHT_RED if col == BALL_RED else LIGHT_BLUE
                    is_outer = (i == 0 or i == len(lst) - 1)
                    r = 5 if is_outer else 4
                    pygame.draw.circle(self.screen, dot_col, (sx + r, y + 5), r)
                    if is_outer:
                        pygame.draw.circle(self.screen, WHITE, (sx + r, y + 5), r, 2)
                    sx += r * 2 + 3
                y += 14
            y += 2

        # Quadrant control — compact inline
        ctrl = gs.compute_quadrant_control()
        parts = []
        for qlabel, qkey in [("TL", "top_left"), ("TR", "top_right"),
                             ("BL", "bottom_left"), ("BR", "bottom_right")]:
            c = ctrl.get(qkey)
            if c == BALL_RED:
                parts.append((f"{qlabel}:R", LIGHT_RED))
            elif c == BALL_BLUE:
                parts.append((f"{qlabel}:B", LIGHT_BLUE))
            else:
                parts.append((f"{qlabel}:-", (70, 70, 80)))

        ctrl_lbl = self.font_sm.render("Control:", True, LIGHT_GRAY)
        self.screen.blit(ctrl_lbl, (px, y))
        sx = px + ctrl_lbl.get_width() + 6
        for text, col in parts:
            s = self.font_sm.render(text, True, col)
            self.screen.blit(s, (sx, y))
            sx += s.get_width() + 8
        y += 16

        return y

    def _draw_panel_editor(self, env, y_start: int, mouse) -> int:
        y  = y_start
        px = FIELD_PX + 10
        bw = PANEL_W - 20

        # ── Row 1: Setup mode + Heatmap toggle ──
        half = (bw - 6) // 2
        self.btn_setup_toggle.update(px, y, half, BTN_H)
        hover = self.btn_setup_toggle.collidepoint(mouse)
        bg = (30, 70, 30) if hover else (20, 50, 20)
        pygame.draw.rect(self.screen, bg,    self.btn_setup_toggle, border_radius=3)
        pygame.draw.rect(self.screen, GREEN, self.btn_setup_toggle, 1, border_radius=3)
        slbl = self.font_sm.render("SETUP MODE", True, GREEN)
        self.screen.blit(slbl, (self.btn_setup_toggle.centerx - slbl.get_width() // 2,
                                self.btn_setup_toggle.y + (BTN_H - slbl.get_height()) // 2))
        if self._click_this_frame is not None and self.btn_setup_toggle.collidepoint(self._click_this_frame):
            self.setup_mode = True
            self.paused     = True

        self.btn_heatmap.update(px + half + 6, y, half, BTN_H)
        on = self.show_heatmap
        hm_bg = (20, 60, 40) if on else (35, 35, 50)
        hm_fg = GREEN if on else LIGHT_GRAY
        hover_h = self.btn_heatmap.collidepoint(mouse)
        if hover_h:
            hm_bg = tuple(min(255, c + 20) for c in hm_bg)
        pygame.draw.rect(self.screen, hm_bg, self.btn_heatmap, border_radius=3)
        pygame.draw.rect(self.screen, hm_fg, self.btn_heatmap, 1, border_radius=3)
        hlbl = self.font_sm.render("Heatmap: ON" if on else "Heatmap: OFF", True, hm_fg)
        self.screen.blit(hlbl, (self.btn_heatmap.centerx - hlbl.get_width() // 2,
                                self.btn_heatmap.y + (BTN_H - hlbl.get_height()) // 2))
        if self._click_this_frame is not None and self.btn_heatmap.collidepoint(self._click_this_frame):
            self.show_heatmap = not self.show_heatmap
        y += BTN_H + 6

        # ── Demo scoring button ──
        self.btn_test_anim.update(px, y, bw, BTN_H)
        hover = self.btn_test_anim.collidepoint(mouse)
        demo_active = getattr(self, "demo_score", False)
        bg = (20, 80, 50) if demo_active else ((20, 55, 80) if hover else (12, 35, 55))
        pygame.draw.rect(self.screen, bg,     self.btn_test_anim, border_radius=3)
        pygame.draw.rect(self.screen, ACCENT, self.btn_test_anim, 1, border_radius=3)
        albl_text = "DEMO SCORING [ACTIVE]" if demo_active else "DEMO: GIVE BALLS & SCORE"
        albl_col  = GREEN if demo_active else ACCENT
        albl = self.font_sm.render(albl_text, True, albl_col)
        self.screen.blit(albl, (self.btn_test_anim.centerx - albl.get_width() // 2,
                                self.btn_test_anim.y + (BTN_H - albl.get_height()) // 2))
        if self._click_this_frame is not None and self.btn_test_anim.collidepoint(self._click_this_frame):
            self._start_demo_score(env)
        y += BTN_H + 6

        # ── Selected ball info ──
        if self.selected_ball >= 0 and self.selected_ball < len(env.field.objects):
            obj = env.field.objects[self.selected_ball]
            if obj.status != OBJ_REMOVED:
                col_name = "RED" if obj.color == BALL_RED else "BLUE"
                status_names = {0: "FIELD", 1: "HELD", 2: "SCORED US",
                                3: "SCORED OPP", 4: "REMOVED"}
                info = f"Ball #{self.selected_ball}  {col_name}  ({obj.x:.0f},{obj.y:.0f})  {status_names.get(obj.status, '?')}"
                self.screen.blit(self.font_sm.render(info, True, YELLOW), (px, y))
                y += 14
            else:
                self.selected_ball = -1
        else:
            self.screen.blit(self.font_sm.render("Click field to select a ball", True, (80, 80, 90)), (px, y))
            y += 14
        y += 2

        # ── Ball edit buttons ──
        self.btn_change_color.update(px, y, BTN_W, BTN_H)
        self.btn_delete.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        self._draw_btn(self.btn_change_color, "Change Color (C)",
                       active=(self.selected_ball >= 0), mouse=mouse)
        self._draw_btn(self.btn_delete, "Delete (Del)",
                       active=(self.selected_ball >= 0), danger=True, mouse=mouse)
        if self._click_this_frame is not None:
            if self.btn_change_color.collidepoint(self._click_this_frame) and self.selected_ball >= 0:
                env.field.change_ball_color(self.selected_ball)
            if self.btn_delete.collidepoint(self._click_this_frame) and self.selected_ball >= 0:
                env.field.remove_ball(self.selected_ball)
                self.selected_ball = -1
        y += BTN_H + 4

        # ── Brush color selector ──
        self.btn_add_red.update(px, y, BTN_W, BTN_H)
        self.btn_add_blue.update(px + BTN_W + 10, y, BTN_W, BTN_H)
        for btn, label, bcolor, col in [
            (self.btn_add_red,  "Brush: RED",  BALL_RED,  RED),
            (self.btn_add_blue, "Brush: BLUE", BALL_BLUE, BLUE),
        ]:
            active = (self.brush_color == bcolor)
            bg = tuple(max(0, c - 60) for c in col) if active else (35, 35, 45)
            fg = col if active else (80, 80, 90)
            hover_b = btn.collidepoint(mouse)
            if hover_b:
                bg = tuple(min(255, c + 20) for c in bg)
            pygame.draw.rect(self.screen, bg, btn, border_radius=3)
            pygame.draw.rect(self.screen, fg, btn, 2 if active else 1, border_radius=3)
            s = self.font_sm.render(label, True, fg)
            self.screen.blit(s, (btn.centerx - s.get_width() // 2,
                                 btn.y + (BTN_H - s.get_height()) // 2))
        if self._click_this_frame is not None:
            if self.btn_add_red.collidepoint(self._click_this_frame):
                self.brush_color = BALL_RED
            elif self.btn_add_blue.collidepoint(self._click_this_frame):
                self.brush_color = BALL_BLUE
        y += BTN_H + 2

        brush_col = LIGHT_RED if self.brush_color == BALL_RED else LIGHT_BLUE
        self.screen.blit(self.font_sm.render("RClick on field to add ball", True, brush_col), (px, y))
        y += 14

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
    # Click handlers
    # ------------------------------------------------------------------
    def _setup_ball_placement(self, fx: float, fy: float, color: int, env):
        """Place a ball at (fx, fy) in setup mode.

        - Snap to matchload tube if within snap distance.
        - Auto-score if placed inside/near a long-goal body (expanded hit zone).
        - Place near center X arms → score in appropriate center goal.
        - Otherwise place as on-field ball at clicked position.
        """
        from sim.config import OBJ_SCORED_US

        # Snap to matchload tube?
        snap_dist = MATCHLOAD_TUBE_RADIUS * 3.0
        for tube in MATCHLOAD_TUBES:
            if math.hypot(fx - tube[0], fy - tube[1]) < snap_dist:
                fx, fy = float(tube[0]), float(tube[1])
                break

        # Expanded hit zones for goal placement (easier clicking)
        _GOAL_PAD_X = 8.0   # extra inches on each side for click detection
        _GOAL_PAD_Y = 4.0

        # Auto-score if near right long goal body
        if (_R_GOAL_X_LO - _GOAL_PAD_X <= fx <= _R_GOAL_X_HI + _GOAL_PAD_X
                and LONG_GOAL_Y_MIN - _GOAL_PAD_Y <= fy <= LONG_GOAL_Y_MAX + _GOAL_PAD_Y):
            # Clamp Y to goal bounds, X to goal center
            gy = float(np.clip(fy, LONG_GOAL_Y_MIN + 2.0, LONG_GOAL_Y_MAX - 2.0))
            gx = (_R_GOAL_X_LO + _R_GOAL_X_HI) / 2.0
            obj_id = len(env.field.objects)
            env.field.add_ball(round(gx, 2), round(gy, 2), color)
            obj = env.field.objects[-1]
            obj.status = OBJ_SCORED_US
            obj.scored_in_goal = "our_long"
            env.field.goal_state.score_ball("our_long", obj_id, color)
            return

        # Auto-score if near left long goal body
        if (_L_GOAL_X_LO - _GOAL_PAD_X <= fx <= _L_GOAL_X_HI + _GOAL_PAD_X
                and LONG_GOAL_Y_MIN - _GOAL_PAD_Y <= fy <= LONG_GOAL_Y_MAX + _GOAL_PAD_Y):
            gy = float(np.clip(fy, LONG_GOAL_Y_MIN + 2.0, LONG_GOAL_Y_MAX - 2.0))
            gx = (_L_GOAL_X_LO + _L_GOAL_X_HI) / 2.0
            obj_id = len(env.field.objects)
            env.field.add_ball(round(gx, 2), round(gy, 2), color)
            obj = env.field.objects[-1]
            obj.status = OBJ_SCORED_US
            obj.scored_in_goal = "opp_long"
            env.field.goal_state.score_ball("opp_long", obj_id, color)
            return

        # Check near center X arms (expanded hit zone for easier clicking)
        dx, dy = fx - 72.0, fy - 72.0
        arm_len = CENTER_GOAL_ARM_LEN
        arm_pad = 10.0   # generous perpendicular hit zone
        for angle in (math.pi / 4, -math.pi / 4):
            ca, sa = math.cos(angle), math.sin(angle)
            along = dx * ca + dy * sa
            perp  = dx * (-sa) + dy * ca
            if abs(along) <= arm_len and abs(perp) <= arm_pad:
                gname = "center_mid" if fy >= 72.0 else "center_low"
                # Snap position onto the arm centerline
                snap_x = 72.0 + along * ca
                snap_y = 72.0 + along * sa
                obj_id = len(env.field.objects)
                env.field.add_ball(round(snap_x, 2), round(snap_y, 2), color)
                obj = env.field.objects[-1]
                obj.status = OBJ_SCORED_US
                obj.scored_in_goal = gname
                env.field.goal_state.score_ball(gname, obj_id, color)
                return

        # Default: place on field at clicked position
        env.field.add_ball(round(fx, 2), round(fy, 2), color)

    def _handle_setup_click(self, pos, env):
        """Handle left-click while in setup mode."""
        mx, my = pos
        if mx >= FIELD_PX or my >= FIELD_PY:
            return
        fx, fy = _from_screen(mx, my)
        if self.setup_tool == "robot0":
            heading = _sim_heading_from_user(self.setup_headings[0])
            env.field.set_robot_start(0, fx, fy, heading)
        elif self.setup_tool == "robot1":
            heading = _sim_heading_from_user(self.setup_headings[1])
            env.field.set_robot_start(1, fx, fy, heading)
        elif self.setup_tool == "red_ball":
            self._setup_ball_placement(fx, fy, BALL_RED, env)
        elif self.setup_tool == "blue_ball":
            self._setup_ball_placement(fx, fy, BALL_BLUE, env)

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
