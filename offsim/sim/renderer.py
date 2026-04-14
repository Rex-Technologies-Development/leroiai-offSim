"""Pygame 2D top-down field renderer.

Shows all 4 robots (15x15in squares), 24 game objects, goals,
and optional heatmap overlay. Animates tank-drive movement in
real time at the sim tick rate.

Controls:
    Space  — pause / unpause
    S      — step one decision while paused
    H      — toggle heatmap overlay
    R      — reset episode
    +/-    — speed up / slow down sim
    1/2    — highlight robot 0 / robot 1 details
"""

from __future__ import annotations
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

from sim.config import (
    Action, FIELD_W, FIELD_H, ROBOT_W,
    OBJ_ON_FIELD, OBJ_HELD, OBJ_SCORED_US, OBJ_SCORED_OPP,
    OUR_LONG_GOAL, OUR_MID_GOAL, OPP_LONG_GOAL, OPP_MID_GOAL,
    MATCH_DURATION, HEATMAP_W, HEATMAP_H,
)

# Scale: inches -> pixels  (144in * 5px/in = 720px)
SCALE = 5.0
SCREEN_W = int(FIELD_W * SCALE)
SCREEN_H = int(FIELD_H * SCALE)
HUD_HEIGHT = 70

# Colours
BLACK      = (0, 0, 0)
WHITE      = (255, 255, 255)
GRAY       = (50, 50, 50)
DARK_GRAY  = (30, 30, 30)
RED        = (220, 50, 50)
BLUE       = (50, 100, 220)
LIGHT_BLUE = (100, 170, 255)
GREEN      = (50, 200, 80)
YELLOW     = (230, 200, 40)
ORANGE     = (230, 140, 30)
LIGHT_RED  = (255, 130, 130)


def _to_screen(x: float, y: float) -> tuple[int, int]:
    """Field coords (origin bottom-left, inches) -> screen pixels (origin top-left)."""
    return int(x * SCALE), int((FIELD_H - y) * SCALE)


class PygameRenderer:
    def __init__(self, env=None, render_every: int = 1):
        if pygame is None:
            raise ImportError("pygame required. pip install pygame")
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H + HUD_HEIGHT))
        pygame.display.set_caption("VEX AI Sim — 144\" x 144\" field")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 13)
        self.big_font = pygame.font.SysFont("consolas", 16)

        self.render_every = render_every
        self.paused = False
        self.show_heatmap = False
        self.highlight_robot = -1
        self.sim_speed = 1.0
        self.step_once = False
        self.should_reset = False

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_events(self) -> dict:
        signals = {"step_once": False, "reset": False}
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
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.sim_speed = min(self.sim_speed * 2, 64.0)
                elif event.key == pygame.K_MINUS:
                    self.sim_speed = max(self.sim_speed / 2, 0.25)
                elif event.key == pygame.K_1:
                    self.highlight_robot = 0 if self.highlight_robot != 0 else -1
                elif event.key == pygame.K_2:
                    self.highlight_robot = 1 if self.highlight_robot != 1 else -1
        return signals

    # ------------------------------------------------------------------
    # Main draw
    # ------------------------------------------------------------------
    def draw(self, env):
        signals = self.handle_events()
        self.step_once = signals["step_once"]
        self.should_reset = signals["reset"]

        self.screen.fill(DARK_GRAY)

        # Field background
        pygame.draw.rect(self.screen, GRAY, (0, 0, SCREEN_W, SCREEN_H))

        # Grid (every 12 inches = 1 foot)
        for i in range(1, 12):
            x = int(i * 12 * SCALE)
            pygame.draw.line(self.screen, (60, 60, 60), (x, 0), (x, SCREEN_H))
            y = int(i * 12 * SCALE)
            pygame.draw.line(self.screen, (60, 60, 60), (0, y), (SCREEN_W, y))

        # Field border
        pygame.draw.rect(self.screen, WHITE, (0, 0, SCREEN_W, SCREEN_H), 2)

        # Heatmap overlay
        if self.show_heatmap:
            self._draw_heatmap(env)

        # Goals
        goal_px = int(10 * SCALE)
        for goal_pos, color, label in [
            (OUR_LONG_GOAL, BLUE, "OL"),
            (OUR_MID_GOAL, BLUE, "OM"),
            (OPP_LONG_GOAL, RED, "XL"),
            (OPP_MID_GOAL, RED, "XM"),
        ]:
            sx, sy = _to_screen(goal_pos[0], goal_pos[1])
            rect = pygame.Rect(sx - goal_px // 2, sy - goal_px // 2, goal_px, goal_px)
            pygame.draw.rect(self.screen, color, rect, 3)
            lbl = self.font.render(label, True, color)
            self.screen.blit(lbl, (sx - 7, sy - 7))

        # Game objects (rings)
        for obj in env.field.objects:
            color = {
                OBJ_ON_FIELD: YELLOW,
                OBJ_HELD: GREEN,
                OBJ_SCORED_US: BLUE,
                OBJ_SCORED_OPP: RED,
            }.get(obj.status, WHITE)
            sx, sy = _to_screen(obj.x, obj.y)
            pygame.draw.circle(self.screen, color, (sx, sy), 4)
            if obj.status in (OBJ_SCORED_US, OBJ_SCORED_OPP):
                pygame.draw.circle(self.screen, WHITE, (sx, sy), 4, 1)

        # Allied robots (15x15in squares)
        for idx, robot in enumerate(env.field.allies):
            self._draw_robot(robot, BLUE if idx == 0 else LIGHT_BLUE,
                             f"R{idx}", idx == self.highlight_robot, env)

        # Opponent robots
        for idx, robot in enumerate(env.field.opponents):
            self._draw_robot(robot, RED if idx == 0 else LIGHT_RED,
                             f"O{idx}", False, env)

        # HUD
        self._draw_hud(env)

        pygame.display.flip()
        self.clock.tick(int(60 * self.sim_speed))

    def _draw_robot(self, robot, color, label_prefix, highlighted, env):
        """Draw a robot as a 15x15in square with heading indicator."""
        sx, sy = _to_screen(robot.x, robot.y)
        half = int(ROBOT_W / 2 * SCALE)
        border = 3 if highlighted else 2

        # Rotated square — approximate with rect for now
        rect = pygame.Rect(sx - half, sy - half, half * 2, half * 2)
        pygame.draw.rect(self.screen, color, rect, border)

        # Heading arrow
        arrow_len = half + 5
        ax = sx + int(arrow_len * np.cos(-robot.heading))
        ay = sy + int(arrow_len * np.sin(-robot.heading))
        pygame.draw.line(self.screen, WHITE, (sx, sy), (ax, ay), 2)

        # Target line (if moving)
        if robot.target is not None and robot.moving:
            tx, ty = _to_screen(robot.target[0], robot.target[1])
            pygame.draw.line(self.screen, (100, 100, 100), (sx, sy), (tx, ty), 1)
            pygame.draw.circle(self.screen, (150, 150, 150), (tx, ty), 3)

        # Label
        action_name = ""
        if hasattr(env, 'current_actions'):
            for i, ally in enumerate(env.field.allies):
                if ally is robot:
                    action_name = Action(env.current_actions[i]).name[:10]
                    break
        lbl = f"{label_prefix}({robot.balls_held}) {action_name}"
        surf = self.font.render(lbl, True, WHITE)
        self.screen.blit(surf, (sx - half, sy + half + 3))

    def _draw_heatmap(self, env):
        heatmap = env.field.get_heatmap()
        max_val = heatmap.max() if heatmap.max() > 0 else 1.0
        cell_w = SCREEN_W / HEATMAP_W
        cell_h = SCREEN_H / HEATMAP_H

        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        for gy in range(HEATMAP_H):
            for gx in range(HEATMAP_W):
                val = heatmap[gy, gx] / max_val
                if val > 0.01:
                    alpha = int(val * 100)
                    r = int(val * 255)
                    g = int((1 - val) * 80)
                    screen_gy = HEATMAP_H - 1 - gy
                    rect = pygame.Rect(
                        int(gx * cell_w), int(screen_gy * cell_h),
                        int(cell_w) + 1, int(cell_h) + 1,
                    )
                    pygame.draw.rect(overlay, (r, g, 40, alpha), rect)
        self.screen.blit(overlay, (0, 0))

    def _draw_hud(self, env):
        y0 = SCREEN_H + 4
        t = max(0, env.field.time_remaining)

        line1 = (f"Time: {t:6.2f}s  |  Us: {env.field.my_score:3d}  "
                 f"Opp: {env.field.opponent_score:3d}  |  Speed: {self.sim_speed:.1f}x")
        self.screen.blit(self.big_font.render(line1, True, WHITE), (10, y0))

        r0 = env.field.allies[0]
        r1 = env.field.allies[1]
        state = "PAUSED" if self.paused else "RUNNING"
        hm = "ON" if self.show_heatmap else "OFF"
        line2 = (f"R0({r0.x:.1f},{r0.y:.1f}) held:{r0.balls_held}  "
                 f"R1({r1.x:.1f},{r1.y:.1f}) held:{r1.balls_held}  |  "
                 f"[{state}] Heatmap:{hm}")
        self.screen.blit(self.font.render(line2, True, (180, 180, 180)), (10, y0 + 20))

        line3 = "Space:pause  S:step  H:heatmap  R:reset  +/-:speed  1/2:robot detail"
        self.screen.blit(self.font.render(line3, True, (120, 120, 120)), (10, y0 + 38))

        if self.highlight_robot >= 0:
            r = env.field.allies[self.highlight_robot]
            line4 = (f"Robot {self.highlight_robot} | "
                     f"pos:({r.x:.2f}, {r.y:.2f}) heading:{np.degrees(r.heading):.1f}deg "
                     f"success:{r.success_ratio():.0%}")
            self.screen.blit(self.font.render(line4, True, LIGHT_BLUE), (10, y0 + 54))

    def close(self):
        if pygame is not None:
            pygame.quit()
