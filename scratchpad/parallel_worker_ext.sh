#!/usr/bin/env bash
# Extension worker: --load each 220-update hd2 checkpoint and train +150 more (-> 370 total, matching the
# banked recipe that gave clean separation). Saves as hd2370_* so the 220s are preserved for a 220-vs-370
# comparison. Lock-queue so any number of workers run collision-free. Diagnoses whether convergence
# sharpens the head/isolation gaps out of the seed noise (they were within-noise at 220).
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CKDIR=tenure/checkpoints
LOCKDIR=scratchpad/hd2370_locks
WID="${1:-0}"
COMMON="--updates 150 --symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 --horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --log-every 20 --adversary toggle_raider greedy_nearest"
names=(off_aware off_blind mult_aware); modes=(off off multiplicative); extras=("--expose-toggle" "" "--expose-toggle")

progress=1
while [ "$progress" -eq 1 ]; do
  progress=0
  for s in 0 1 2 3 4; do
    for j in 0 1 2; do
      name=${names[$j]}
      src="$CKDIR/hd2_${name}_s${s}.pt"; out="$CKDIR/hd2370_${name}_s${s}.pt"
      [ -f "$out" ] && continue                                  # done
      [ -f "$src" ] || continue                                  # 220 checkpoint not ready yet
      lock="$LOCKDIR/${name}_s${s}"
      if mkdir "$lock" 2>/dev/null; then
        progress=1
        echo "[x$WID $(date +%H:%M:%S)] EXTEND ${name}_s${s} (220->370)"
        if $PY -m tenure.train $COMMON --seed "$s" --retention-mode "${modes[$j]}" ${extras[$j]} \
              --load "$src" --save "$out"; then
          echo "[x$WID $(date +%H:%M:%S)] DONE ${name}_s${s}"
        else
          echo "[x$WID $(date +%H:%M:%S)] FAIL ${name}_s${s}"; rmdir "$lock" 2>/dev/null
        fi
      fi
    done
  done
done
echo "[x$WID $(date +%H:%M:%S)] EXT WORKER $WID EXIT (queue drained)"
