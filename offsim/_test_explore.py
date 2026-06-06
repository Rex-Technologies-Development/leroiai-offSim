import numpy as np, math
from sim.env import VexAIEnv, Action
from sim.config import BALL_BLUE, BALL_RED, OBJ_ON_FIELD, OBJ_REMOVED, FIELD_H

env = VexAIEnv(num_allies=1, num_opponents=0)
env.reset(seed=3)

# Clear the field, then place blue balls ONLY in the top half (y>90), and put the
# robot in the bottom. Simulates "collected all bottom blocks".
for o in env.field.objects:
    o.status = OBJ_REMOVED
top_positions = [(40,120),(60,128),(95,118),(110,125),(72,132)]
placed = 0
for o in env.field.objects:
    if placed < len(top_positions):
        o.status = OBJ_ON_FIELD
        o.color = BALL_BLUE
        o.x, o.y = top_positions[placed]
        placed += 1
r = env.field.allies[0]
r.x, r.y = 49.0, 36.0   # bottom-left scan area
r.heading = 0.0
r.balls_held = 0
r.held_object_ids = []

print(f"Blue balls all in TOP half. Robot starts bottom at ({r.x},{r.y}).")
print(f"{'dec':>3} {'pos':>14} {'barren':>6} {'held':>4}  note")
prev_y = r.y
for d in range(16):
    env.step(np.array([int(Action.COLLECT_NEAREST_BALL), int(Action.IDLE)]))
    r = env.field.allies[0]
    note = ""
    if r.y > prev_y + 2: note = "moved NORTH"
    elif r.y < prev_y - 2: note = "moved south"
    prev_y = r.y
    print(f"{d:>3} ({r.x:6.1f},{r.y:6.1f}) {r.explore_barren:>6} {r.balls_held:>4}  {note}")
    if r.balls_held > 0:
        print("   >>> reached top & started collecting!")
        break
print(f"\nfinal pos y={r.y:.1f} (start 36) — reached top half? {r.y > 72}")
