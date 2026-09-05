#!/usr/bin/env bash
# DYNAMIC-EXPOSURE decisive test (pre-registered): 10 seeds x 3 arms, FRESH to the unbiased fixed-420
# budget, with --dynamic-exposure (toggle value = current cluster load; varied cluster sizes). Same three
# arms; off_aware now scores by EXPOSURE (task_value=exposure -> decoder w_norm). Collapses (<50% arm
# median) restarted post-hoc per convergence-protocol.md. Fresh runs ~4.8GB -> 2 workers safe. Prefix hd2dyn_.
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CKDIR=tenure/checkpoints
LOCKDIR=scratchpad/hd2dyn_locks
WID="${1:-0}"
CONV="--updates 420 --dynamic-exposure"
COMMON="--symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 --horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --log-every 40 --adversary toggle_raider greedy_nearest"
names=(off_aware off_blind mult_aware); modes=(off off multiplicative); extras=("--expose-toggle" "" "--expose-toggle")

progress=1
while [ "$progress" -eq 1 ]; do
  progress=0
  for s in 0 1 2 3 4 5 6 7 8 9; do
    for j in 0 1 2; do
      name=${names[$j]}; out="$CKDIR/hd2dyn_${name}_s${s}.pt"
      [ -f "$out" ] && continue
      lock="$LOCKDIR/${name}_s${s}"
      if mkdir "$lock" 2>/dev/null; then
        progress=1
        echo "[d$WID $(date +%H:%M:%S)] DYNEXP ${name}_s${s}"
        if $PY -m tenure.train $CONV $COMMON --seed "$s" --retention-mode "${modes[$j]}" ${extras[$j]} --save "$out"; then
          echo "[d$WID $(date +%H:%M:%S)] DONE ${name}_s${s}"
        else
          echo "[d$WID $(date +%H:%M:%S)] FAIL ${name}_s${s}"; rmdir "$lock" 2>/dev/null
        fi
      fi
    done
  done
done
echo "[d$WID $(date +%H:%M:%S)] DYNEXP WORKER $WID EXIT (queue drained)"
