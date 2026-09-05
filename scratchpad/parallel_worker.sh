#!/usr/bin/env bash
# One parallel training worker. Pulls jobs from a shared queue via atomic mkdir-locks so ANY number of
# workers can run concurrently with zero collision. A job is "done" iff its checkpoint exists (locks only
# gate concurrent claiming). Skips done jobs; releases its lock on failure so another worker can retry.
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2      # cap per-process CPU threads (16 cores / many workers)
CKDIR=tenure/checkpoints
LOCKDIR=scratchpad/hd2_locks
WID="${1:-0}"
COMMON="--updates 220 --symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 --horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --log-every 20 --adversary toggle_raider greedy_nearest"
names=(off_aware off_blind mult_aware); modes=(off off multiplicative); extras=("--expose-toggle" "" "--expose-toggle")

progress=1
while [ "$progress" -eq 1 ]; do
  progress=0
  for s in 0 1 2 3 4; do
    for j in 0 1 2; do
      name=${names[$j]}; out="$CKDIR/hd2_${name}_s${s}.pt"
      [ -f "$out" ] && continue                                  # done -> skip
      lock="$LOCKDIR/${name}_s${s}"
      if mkdir "$lock" 2>/dev/null; then                         # atomic claim
        progress=1
        echo "[w$WID $(date +%H:%M:%S)] START ${name}_s${s}"
        if $PY -m tenure.train $COMMON --seed "$s" --retention-mode "${modes[$j]}" ${extras[$j]} --save "$out"; then
          echo "[w$WID $(date +%H:%M:%S)] DONE ${name}_s${s}"
        else
          echo "[w$WID $(date +%H:%M:%S)] FAIL ${name}_s${s}"; rmdir "$lock" 2>/dev/null
        fi
      fi
    done
  done
done
echo "[w$WID $(date +%H:%M:%S)] WORKER $WID EXIT (queue drained)"
