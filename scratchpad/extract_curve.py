"""How many of the 15 extension runs are STUCK (plateaued far below their arm's median)?
Determines whether a 'restart clearly-bimodal stuck runs' protocol (path b) is viable (few stuck) or
whether stuck runs are common (need many seeds, path a)."""
import re, statistics
runs, cur = {}, None
for line in open("scratchpad/hd2_ext_w0.log", encoding="utf-8", errors="ignore"):
    m = re.search(r'EXTEND (\S+)', line)
    if m:
        cur = m.group(1); runs.setdefault(cur, [])
    jm = re.search(r'(\d+)/150.*J_H~\s*([-\d.]+)', line)
    if jm and cur:
        runs[cur].append(float(jm.group(2)))
end = {k: v[-1] for k, v in runs.items() if v}
print("final train-J_H per run (220->370 extension):\n")
for arm in ("off_aware", "off_blind", "mult_aware"):
    vals = {s: end[f"{arm}_s{s}"] for s in range(5) if f"{arm}_s{s}" in end}
    if not vals:
        continue
    med = statistics.median(vals.values())
    row = "  ".join(f"s{s}:{v:+.3f}{'  <STUCK' if v < 0.5*med else ''}" for s, v in vals.items())
    print(f"  {arm:11s} median={med:.3f} | {row}")
allv = list(end.values())
stuck = [k for k, v in end.items() if v < 0.5 * statistics.median(allv)]
print(f"\n{len(end)} runs logged; STUCK (< 50% of overall median {statistics.median(allv):.3f}): "
      f"{stuck if stuck else 'none'}")
