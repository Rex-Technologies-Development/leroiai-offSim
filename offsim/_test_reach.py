import numpy as np
from sim.env import _build_nav_waypoints

pts = {
    "SE(95,36)": (95,36), "SW(49,36)": (49,36),
    "BELOW_X(72,41)": (72,41), "ABOVE_X(72,103)": (72,103),
    "NE_scan(95,108)": (95,108), "NW_scan(49,108)": (49,108),
    "RIGHT_HIGH(95,104)": (95,104), "LEFT_HIGH(49,104)": (49,104),
}
starts = {
    "bottom-left (30,40)": (30,40),
    "bottom-right (100,40)": (100,40),
    "SW scan (49,36)": (49,36),
    "SE scan (95,36)": (95,36),
}
for sname, s in starts.items():
    print(f"\nfrom {sname}:")
    for pname, p in pts.items():
        wq = _build_nav_waypoints(np.array(s, float), np.array(p, float))
        reach = "REACH" if wq else "block"
        n = len(wq) if wq else 0
        print(f"   {pname:20s} {reach}  ({n} legs)")
