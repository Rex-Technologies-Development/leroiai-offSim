#!/usr/bin/env bash
# Path-(b) FIRST PASS: train each arm x seed FRESH to a J_H-plateau convergence criterion (not a fixed
# budget), so we measure the decomposition on converged policies. Collapsed cells (< 50% of arm median)
# are restarted separately with a new --init-seed (see convergence-protocol.md). Fresh runs are ~4.8GB
# -> 2 workers ~12GB (safe). Lock-queue; prefix hd2conv_.
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CKDIR=tenure/checkpoints
LOCKDIR=scratchpad/hd2conv_locks
WID="${1:-0}"
# Fixed budget PAST the empirical plateau (good runs plateau ~380, collapses far earlier), NO early-stop
# -- an early-stop criterion differentially under-trains slow-converging arms (blind), biasing the
# isolation gap. Convergence is demonstrated post-hoc per-seed from the J_H trajectories in these logs.
CONV="--updates 420"
COMMON="--symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 --horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --log-every 20 --adversary toggle_raider greedy_nearest"
names=(off_aware off_blind mult_aware); modes=(off off multiplicative); extras=("--expose-toggle" "" "--expose-toggle")

progress=1
while [ "$progress" -eq 1 ]; do
  progress=0
  for s in 0 1 2 3 4; do
    for j in 0 1 2; do
      name=${names[$j]}; out="$CKDIR/hd2conv_${name}_s${s}.pt"
      [ -f "$out" ] && continue
      lock="$LOCKDIR/${name}_s${s}"
      if mkdir "$lock" 2>/dev/null; then
        progress=1
        echo "[c$WID $(date +%H:%M:%S)] CONVERGE ${name}_s${s}"
        if $PY -m tenure.train $CONV $COMMON --seed "$s" --retention-mode "${modes[$j]}" ${extras[$j]} --save "$out"; then
          echo "[c$WID $(date +%H:%M:%S)] DONE ${name}_s${s}"
        else
          echo "[c$WID $(date +%H:%M:%S)] FAIL ${name}_s${s}"; rmdir "$lock" 2>/dev/null
        fi
      fi
    done
  done
done
echo "[c$WID $(date +%H:%M:%S)] CONV WORKER $WID EXIT (queue drained)"
