"""Pygame renderer for the Override 2D prototype.

Geometry follows the official Override Appendix A / VEX GPS coordinates. Object
clusters and vertical stack appearance remain deliberate top-down 2D proxies.
"""
from __future__ import annotations
import math
from .config import Alliance, FIELD_WIDTH, LOAD_ZONE_DEPTH, LOAD_ZONE_SPAN
from .field import AUTONOMOUS_TAPE_SEGMENTS, MIDFIELD_DIAMOND

RED = (218, 48, 57)
BLUE = (42, 115, 224)
YELLOW = (244, 205, 55)
WHITE = (235, 238, 238)
DARK = (25, 30, 34)
NEUTRAL = (48, 53, 57)
TILE = (72, 76, 77)
BG = (16, 20, 23)

# How many rendered frames a pickup pulse lasts. In human mode draw() is called
# once per 0.05 s physics tick, so 12 frames is roughly a 0.6 s flash.
PICKUP_PULSE_FRAMES = 12

class PygameRenderer:
    GOAL_LEGEND_LABELS = (
        "Base: Goal type / protection",
        "Halo: yellow-Pin owner",
        "Pips: visible Pin halves",
        "Center #: stack entries",
        "T: tall Midfield Goal",
    )

    def __init__(self, render_mode: str = "human", size: int = 760):
        import pygame
        self.pg = pygame
        self.render_mode = render_mode
        self.size = size
        self.margin = 30
        self.panel = 300
        pygame.init(); pygame.font.init()
        flags = 0 if render_mode == "human" else pygame.HIDDEN
        self.screen = pygame.display.set_mode((size+self.panel, size), flags)
        pygame.display.set_caption("Override 2D Prototype — VEX U Layout")
        self.font = pygame.font.SysFont("Arial", 17)
        self.small = pygame.font.SysFont("Arial", 13)
        self.tiny = pygame.font.SysFont("Arial", 11)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.step_once = False
        self.should_reset = False
        self.running = True
        self.speed = 1.0
        # pickup-animation state (renderer is otherwise stateless per tick)
        self._prev_pin: dict[int, int | None] = {}
        self._prev_cup: dict[int, int | None] = {}
        self._pickup_pulse: dict[int, int] = {}
        self._field_id: int | None = None

    @property
    def scale(self) -> float:
        return (self.size-2*self.margin)/FIELD_WIDTH

    def _xy(self, x, y):
        return int(self.margin+x*self.scale), int(self.size-self.margin-y*self.scale)

    @staticmethod
    def _element_color(value):
        raw = value.value if isinstance(value, Alliance) else value
        return BLUE if raw == "blue" else RED if raw == "red" else YELLOW

    @staticmethod
    def _octagon(center, radius):
        cx, cy = center
        return [(int(cx+radius*math.cos(math.pi/8+i*math.pi/4)),
                 int(cy+radius*math.sin(math.pi/8+i*math.pi/4))) for i in range(8)]

    def handle_events(self):
        p = self.pg
        for event in p.event.get():
            if event.type == p.QUIT: self.running = False
            elif event.type == p.KEYDOWN:
                if event.key == p.K_SPACE: self.paused = not self.paused
                elif event.key == p.K_s: self.step_once = True
                elif event.key == p.K_r: self.should_reset = True
                elif event.key in (p.K_PLUS, p.K_EQUALS, p.K_KP_PLUS): self.speed = min(16.0, self.speed*2.0)
                elif event.key in (p.K_MINUS, p.K_KP_MINUS): self.speed = max(0.25, self.speed/2.0)
        return self.running

    def _draw_dashed_line(self, color, start, end, width, dash=8, gap=5):
        p = self.pg
        x0, y0 = start; x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        drawn = 0.0
        while drawn < length:
            a = drawn
            b = min(length, drawn + dash)
            p.draw.line(self.screen, color,
                        (int(x0 + ux * a), int(y0 + uy * a)),
                        (int(x0 + ux * b), int(y0 + uy * b)), width)
            drawn += dash + gap

    def _draw_load_zone(self, alliance, left, top):
        """L-tape for a rectangular Load Zone (depth × span), not a corner square.

        Depth lands on the half-tile midpoint from the alliance wall; span runs
        farther along the N/S wall so the inner L corner sits mid-tile-edge.
        """
        p = self.pg
        color = RED if alliance is Alliance.RED else BLUE
        width = max(2, int(round(1.5 * self.scale)))
        depth, span = LOAD_ZONE_DEPTH, LOAD_ZONE_SPAN
        inner_x = depth if left else 144.0 - depth
        inner_y = 144.0 - span if top else span
        # Vertical leg: from N/S wall to the inner corner (span long).
        vertical = (self._xy(inner_x, 144.0 if top else 0.0), self._xy(inner_x, inner_y))
        # Horizontal leg: from alliance wall to the inner corner (depth short).
        horizontal = (self._xy(0.0 if left else 144.0, inner_y), self._xy(inner_x, inner_y))
        # Official figure: red Load Zones are solid L-tape; blue are dashed.
        if alliance is Alliance.RED:
            p.draw.line(self.screen, color, vertical[0], vertical[1], width)
            p.draw.line(self.screen, color, horizontal[0], horizontal[1], width)
        else:
            self._draw_dashed_line(color, vertical[0], vertical[1], width)
            self._draw_dashed_line(color, horizontal[0], horizontal[1], width)

    def _draw_paired_tape(self, start, end, stroke, gap_inches=1.15):
        """Draw two parallel white strokes straddling a diagonal centerline."""
        p = self.pg
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        nx, ny = -dy / length, dx / length
        half = gap_inches / 2.0
        for sign in (-1.0, 1.0):
            a = (start[0] + sign * nx * half, start[1] + sign * ny * half)
            b = (end[0] + sign * nx * half, end[1] + sign * ny * half)
            p.draw.line(self.screen, WHITE, self._xy(*a), self._xy(*b), stroke)

    def _draw_field_markings(self):
        p = self.pg
        # Official 6x6 foam-tile grid.
        for value in range(24, 144, 24):
            p.draw.line(self.screen, TILE, self._xy(value,0), self._xy(value,144), 1)
            p.draw.line(self.screen, TILE, self._xy(0,value), self._xy(144,value), 1)
        self._draw_load_zone(Alliance.RED, True, False)
        self._draw_load_zone(Alliance.RED, True, True)
        self._draw_load_zone(Alliance.BLUE, False, False)
        self._draw_load_zone(Alliance.BLUE, False, True)
        # Diagonal X: paired strokes from each corner to diamond edge midpoints.
        stroke = max(2, int(round(1.1 * self.scale)))
        for start, end in AUTONOMOUS_TAPE_SEGMENTS:
            self._draw_paired_tape(start, end, stroke)
        diamond_width = max(3, int(round(2.0 * self.scale)))
        points = [self._xy(*point) for point in MIDFIELD_DIAMOND]
        p.draw.polygon(self.screen, WHITE, points, diamond_width)

    def _draw_pin(self, pin):
        p = self.pg
        cx, cy = self._xy(pin.x, pin.y)
        # Top-down 4-blade star: two colored halves on perpendicular axes.
        angle = (pin.object_id % 8) * math.pi / 4
        length = max(6, int(round(2.2 * self.scale)))
        width = max(3, int(round(1.1 * self.scale)))
        colors = [self._element_color(h) for h in pin.halves]
        for axis, color in ((0, colors[0]), (1, colors[1])):
            theta = angle + axis * math.pi / 2
            ux, uy = math.cos(theta), -math.sin(theta)
            for sign in (-1, 1):
                end = (int(cx + sign * ux * length), int(cy + sign * uy * length))
                p.draw.line(self.screen, color, (cx, cy), end, width)
                p.draw.circle(self.screen, color, end, max(2, width // 2))
        p.draw.circle(self.screen, (28, 28, 28), (cx, cy), 2)

    def _draw_cup(self, cup):
        p = self.pg
        cx, cy = self._xy(cup.x, cup.y)
        radius = max(5, int(round(1.6 * self.scale)))
        # Dark rounded Cup body with light ring — matches Appendix A top-down look.
        p.draw.circle(self.screen, (28, 32, 34), (cx, cy), radius)
        p.draw.circle(self.screen, (150, 158, 160), (cx, cy), radius, 2)
        p.draw.circle(self.screen, (55, 60, 62), (cx, cy), max(2, radius // 3))

    def _draw_goal_badge(self, center, radius, base, owner, visible, count, tall=False, label=None, rim=None):
        p = self.pg
        if owner is not None:
            p.draw.polygon(self.screen, self._element_color(owner), self._octagon(center, radius + 4), 3)
        p.draw.polygon(self.screen, base, self._octagon(center, radius))
        p.draw.polygon(self.screen, rim or (12, 15, 17), self._octagon(center, radius), 2)
        p.draw.circle(self.screen, (9, 12, 14), center, max(4, radius // 2))
        if tall:
            p.draw.polygon(self.screen, (190, 194, 194), self._octagon(center, radius - 3), 2)
            self.screen.blit(self.tiny.render("T", True, WHITE), (center[0] - 3, center[1] - radius - 12))
        count_img = self.small.render(str(count), True, WHITE)
        self.screen.blit(count_img, count_img.get_rect(center=center))
        for index, half in enumerate(visible[:2]):
            p.draw.circle(self.screen, self._element_color(half),
                          (center[0] - 6 + index * 12, center[1] - radius - 5), 5)
            p.draw.circle(self.screen, (20, 20, 20),
                          (center[0] - 6 + index * 12, center[1] - radius - 5), 5, 1)
        if label:
            tag = self.tiny.render(label, True, WHITE)
            self.screen.blit(tag, (center[0] + radius + 3, center[1] - radius))

    def _draw_goal(self, field, goal):
        """Field Goals: rounded squares matching Appendix A; status lives in the panel."""
        p = self.pg
        cx, cy = self._xy(goal.x, goal.y)
        tall = goal.kind == "neutral_tall"
        half = max(10, int(round((5.4 if tall else 4.6) * self.scale)))
        if goal.protected_by is Alliance.RED:
            fill, border = (168, 42, 48), (220, 70, 70)
        elif goal.protected_by is Alliance.BLUE:
            fill, border = (36, 88, 180), (70, 140, 230)
        else:
            fill, border = (22, 24, 26), (8, 10, 12)
        rect = p.Rect(cx - half, cy - half, half * 2, half * 2)
        p.draw.rect(self.screen, fill, rect, border_radius=max(4, half // 3))
        p.draw.rect(self.screen, border, rect, max(2, half // 8), border_radius=max(4, half // 3))
        if tall:
            inset = p.Rect(cx - half + 4, cy - half + 4, half * 2 - 8, half * 2 - 8)
            p.draw.rect(self.screen, (160, 164, 166), inset, 2, border_radius=max(3, half // 4))
        self._draw_goal_contents(field, goal, cx, cy, half)
        # Subtle ownership halo when a yellow Pin is owned (gameplay cue, not diagram ink).
        owner = field.goal_status_owner(goal)
        if owner is not None:
            p.draw.rect(self.screen, self._element_color(owner), rect.inflate(8, 8), 2,
                        border_radius=max(5, half // 3 + 2))

    def _draw_goal_contents(self, field, goal, cx, cy, half):
        """Show what's scored in a Goal on the field: a filled core that grows with
        stack height, the visible Pin-half pips, and the entry count."""
        p = self.pg
        count = len(goal.stack)
        if count == 0:
            return
        visible = goal.visible_pin_halves(field.pins)
        frac = min(1.0, count / 8.0)                      # goal capacity is 8
        inner = max(3, int(half * (0.35 + 0.5 * frac)))
        core = p.Rect(cx - inner, cy - inner, 2 * inner, 2 * inner)
        br = max(2, inner // 2)
        if len(visible) >= 2:                             # two-tone core shows BOTH pin halves
            p.draw.rect(self.screen, self._element_color(visible[0]), core, border_radius=br)
            clip = self.screen.get_clip()
            self.screen.set_clip(p.Rect(cx, cy - inner, inner, 2 * inner))   # right half only
            p.draw.rect(self.screen, self._element_color(visible[1]), core, border_radius=br)
            self.screen.set_clip(clip)
        else:
            p.draw.rect(self.screen, (150, 155, 158), core, border_radius=br)
        p.draw.rect(self.screen, (15, 17, 19), core, 1, border_radius=br)
        # visible Pin-half pips just above the Goal
        for i, half_color in enumerate(visible[:2]):
            px, py = cx - 5 + i * 10, cy - half - 6
            p.draw.circle(self.screen, self._element_color(half_color), (px, py), 4)
            p.draw.circle(self.screen, (18, 18, 18), (px, py), 4, 1)
        # entry count centered over the core
        num = self.tiny.render(str(count), True, WHITE)
        self.screen.blit(num, num.get_rect(center=(cx, cy)))

    def _draw_toggle(self, toggle):
        p = self.pg
        center = self._xy(toggle.x,toggle.y)
        length = int(round(25.8*self.scale)); thick = 9
        face = self._element_color(toggle.owner)
        horizontal = toggle.compass in ("N","S")
        rect = p.Rect(center[0]-length//2,center[1]-thick//2,length,thick) if horizontal else p.Rect(center[0]-thick//2,center[1]-length//2,thick,length)
        p.draw.rect(self.screen,(25,28,30),rect.inflate(6,6),border_radius=3)
        p.draw.rect(self.screen,face,rect,border_radius=3)
        p.draw.circle(self.screen,YELLOW,center,7)
        self.screen.blit(self.tiny.render(toggle.compass,True,(20,20,20)),(center[0]-4,center[1]-6))

    def _draw_loader(self, loader):
        p = self.pg
        cx, cy = self._xy(loader.x,loader.y)
        color = RED if loader.alliance is Alliance.RED else BLUE
        p.draw.rect(self.screen,(20,24,27),(cx-10,cy-17,20,34),border_radius=3)
        p.draw.rect(self.screen,color,(cx-8,cy-15,16,30),3,border_radius=3)
        p.draw.circle(self.screen,(170,215,224),(cx,cy),5,2)
        # Arrow points from the wall chute toward the playing field.
        direction = 1 if loader.alliance is Alliance.RED else -1
        p.draw.polygon(self.screen,color,[(cx+direction*15,cy),(cx+direction*8,cy-5),(cx+direction*8,cy+5)])

    @staticmethod
    def _lerp(a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _update_pickup_pulses(self, field):
        """Detect None->held transitions per robot and arm a pulse; reset on a new match."""
        if id(field) != self._field_id:                      # new field (reset): seed, don't flash
            self._field_id = id(field)
            self._pickup_pulse.clear()
            self._prev_pin = {r.robot_id: r.held_pin for r in field.robots}
            self._prev_cup = {r.robot_id: r.held_cup for r in field.robots}
            return
        for r in field.robots:
            rid = r.robot_id
            if self._prev_pin.get(rid) is None and r.held_pin is not None:
                self._pickup_pulse[rid] = PICKUP_PULSE_FRAMES
            if self._prev_cup.get(rid) is None and r.held_cup is not None:
                self._pickup_pulse[rid] = PICKUP_PULSE_FRAMES
            self._prev_pin[rid] = r.held_pin
            self._prev_cup[rid] = r.held_cup

    def _decay_pickup_pulses(self):
        for rid in list(self._pickup_pulse):
            self._pickup_pulse[rid] -= 1
            if self._pickup_pulse[rid] <= 0:
                del self._pickup_pulse[rid]

    def _carry_anchor(self, robot, forward=5.0, perp=0.0):
        fx, fy = math.cos(robot.heading), math.sin(robot.heading)
        return self._xy(robot.x + forward * fx - perp * fy, robot.y + forward * fy + perp * fx)

    def _draw_mini_pin(self, center, halves):
        p = self.pg
        cx, cy = center
        colors = [self._element_color(h) for h in halves]
        for axis, color in ((0, colors[0]), (1, colors[1])):
            ux, uy = math.cos(axis * math.pi / 2), -math.sin(axis * math.pi / 2)
            p.draw.line(self.screen, color, (int(cx - ux * 6), int(cy - uy * 6)), (int(cx + ux * 6), int(cy + uy * 6)), 3)
        p.draw.circle(self.screen, (18, 18, 18), (int(cx), int(cy)), 2)

    def _draw_mini_cup(self, center):
        p = self.pg
        cx, cy = int(center[0]), int(center[1])
        p.draw.circle(self.screen, (30, 34, 36), (cx, cy), 5)
        p.draw.circle(self.screen, (160, 168, 170), (cx, cy), 5, 2)

    def _draw_carried(self, field, robot):
        """Draw the Pin/Cup the robot is currently holding, in its 'gripper' (heading)."""
        if robot.held_pin is not None and robot.held_pin in field.pins:
            self._draw_mini_pin(self._carry_anchor(robot, forward=6, perp=3.5), field.pins[robot.held_pin].halves)
        if robot.held_cup is not None:
            self._draw_mini_cup(self._carry_anchor(robot, forward=6, perp=-3.5))

    def _robot_rect_points(self, robot):
        """Screen corners of the robot's rotated rectangular footprint (length along heading)."""
        hl, hw = robot.length / 2.0, robot.width / 2.0
        c, s = math.cos(robot.heading), math.sin(robot.heading)
        return [self._xy(robot.x + lx * c - ly * s, robot.y + lx * s + ly * c)
                for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]

    def _draw_robot(self, field, robot):
        p = self.pg
        pos = self._xy(robot.x, robot.y)
        color = BLUE if robot.alliance is Alliance.BLUE else RED
        rpx = max(10, int(max(robot.width, robot.length) / 2 * self.scale))
        pulse = self._pickup_pulse.get(robot.robot_id, 0)
        if pulse > 0:                                        # expanding, fading pickup ring
            frac = pulse / PICKUP_PULSE_FRAMES
            p.draw.circle(self.screen, self._lerp(YELLOW, BG, 1 - frac), pos, rpx + 3 + int((1 - frac) * 16), 3)
        points = self._robot_rect_points(robot)
        p.draw.polygon(self.screen, color, points)
        p.draw.polygon(self.screen, (17, 20, 22), points, 2)
        # heading indicator: from center to the middle of the front edge
        front = self._xy(robot.x + (robot.length / 2) * math.cos(robot.heading),
                         robot.y + (robot.length / 2) * math.sin(robot.heading))
        p.draw.line(self.screen, WHITE, pos, front, 3)
        self._draw_carried(field, robot)
        label = self.tiny.render(f"R{robot.robot_id} P{int(robot.held_pin is not None)} C{int(robot.held_cup is not None)}", True, WHITE)
        self.screen.blit(label, (pos[0] - 24, pos[1] + rpx + 4))

    def _draw_compass(self, x, y):
        p = self.pg
        p.draw.circle(self.screen,(93,101,105),(x,y),28,1)
        p.draw.line(self.screen,WHITE,(x,y+22),(x,y-22),2)
        p.draw.polygon(self.screen,WHITE,[(x,y-29),(x-5,y-18),(x+5,y-18)])
        p.draw.line(self.screen,(130,135,138),(x-22,y),(x+22,y),1)
        for text,dx,dy in (("N",-5,-45),("E",34,-7),("S",-4,31),("W",-44,-7)):
            self.screen.blit(self.small.render(text,True,WHITE),(x+dx,y+dy))

    def _draw_panel(self, field):
        x = self.size+18
        lines = ["OVERRIDE — VEX U", field.phase.value.upper(),
                 f"Time   {field.time_remaining:05.1f}",
                 f"Blue   {field.score(Alliance.BLUE)}", f"Red    {field.score(Alliance.RED)}",
                 f"Bonus  {field.opening_bonus.value if field.opening_bonus else '-'}",
                 f"AWP    B:{int(field.awp[Alliance.BLUE])} R:{int(field.awp[Alliance.RED])}",
                 f"Speed  {self.speed:g}x"]
        for i,line in enumerate(lines):
            self.screen.blit((self.font if i<7 else self.small).render(line,True,WHITE),(x,18+i*25))
        self._draw_compass(x+225,82)
        self.screen.blit(self.font.render("GOAL STATUS KEY",True,WHITE),(x,230))
        self._draw_goal_badge((x+42,285),20,NEUTRAL,Alliance.BLUE,("blue","yellow"),3,True)
        for i,label in enumerate(self.GOAL_LEGEND_LABELS):
            self.screen.blit(self.tiny.render(label,True,(205,210,212)),(x+78,258+i*18))
        self.screen.blit(self.small.render("LIVE GOALS  (ID ORDER)",True,WHITE),(x,365))
        # Compact live badges make every Goal's current symbolic status readable.
        for index, goal in enumerate(field.goals):
            col, row = index%3, index//3
            center = (x+35+col*86, 410+row*58)
            base = (128,38,43) if goal.protected_by is Alliance.RED else (30,67,126) if goal.protected_by is Alliance.BLUE else NEUTRAL
            self._draw_goal_badge(center,14,base,field.goal_status_owner(goal),goal.visible_pin_halves(field.pins),len(goal.stack),goal.kind=="neutral_tall",f"G{goal.goal_id}")
        controls = ["Space  pause / resume", "S      one strategy step", "R      reset match", "+ / -  playback speed"]
        for i,line in enumerate(controls):
            self.screen.blit(self.small.render(line,True,(190,196,198)),(x,610+i*23))
        self.screen.blit(self.tiny.render("Layout: official GPS / Appendix A",True,(135,145,150)),(x,720))
        self.screen.blit(self.tiny.render("Stacks: symbolic 2D proxy",True,(135,145,150)),(x,737))

    def draw(self, field):
        p = self.pg
        self.handle_events()
        self._update_pickup_pulses(field)
        self.screen.fill(BG)
        low, high = self._xy(0,0), self._xy(144,144)
        rect = p.Rect(low[0],high[1],high[0]-low[0],low[1]-high[1])
        p.draw.rect(self.screen,(57,62,63),rect)
        self._draw_field_markings()
        for cup in field.objects_on_field("cup"): self._draw_cup(cup)
        for pin in field.objects_on_field("pin"): self._draw_pin(pin)
        for goal in field.goals: self._draw_goal(field,goal)
        for loader in field.loaders: self._draw_loader(loader)
        for toggle in field.toggles: self._draw_toggle(toggle)
        for robot in field.robots: self._draw_robot(field, robot)
        self._decay_pickup_pulses()
        p.draw.rect(self.screen,(206,211,212),rect,4)
        self._draw_panel(field)
        p.display.flip()
        if self.render_mode == "human":
            self.clock.tick(30 if self.paused else max(1,int(round(20*self.speed))))
        if self.render_mode == "rgb_array":
            return p.surfarray.array3d(self.screen).swapaxes(0,1)
        return None

    def close(self):
        self.pg.quit()
