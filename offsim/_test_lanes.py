import numpy as np
from sim.route_planner import _los_blocked, _NAV_MARGIN
from sim.config import LONG_GOAL_WALL_GAP, LONG_GOAL_WIDTH, FIELD_W

print("LEFT goal inner face x =", LONG_GOAL_WALL_GAP+LONG_GOAL_WIDTH, " RIGHT goal inner x =", FIELD_W-LONG_GOAL_WALL_GAP-LONG_GOAL_WIDTH)
print("_NAV_MARGIN =", _NAV_MARGIN)
print("\nVertical traversal test (y=40 -> y=104), which x is clear end-to-end:")
for x in [30,32,34,35,36,38,40, 104,106,108,109,110,112,114]:
    blocked = _los_blocked(x,40, x,104, margin=_NAV_MARGIN)
    print(f"  x={x:3d}: {'BLOCKED' if blocked else 'clear'}")

print("\nCan a lane point bridge bottom<->top? test (49,40)->(L,72)->(49,104):")
for L in [30,33,35,37]:
    a = _los_blocked(49,40, L,72, margin=_NAV_MARGIN)
    b = _los_blocked(L,72, 49,104, margin=_NAV_MARGIN)
    print(f"  L=({L},72): (49,40)->L {'blk' if a else 'ok'}, L->(49,104) {'blk' if b else 'ok'}")
for R in [109,111,113]:
    a = _los_blocked(95,40, R,72, margin=_NAV_MARGIN)
    b = _los_blocked(R,72, 95,104, margin=_NAV_MARGIN)
    print(f"  R=({R},72): (95,40)->R {'blk' if a else 'ok'}, R->(95,104) {'blk' if b else 'ok'}")
