#!/usr/bin/env bash
# Uniform single-stage hardening-v2 batch: 3 arms x 5 seeds, 220 updates, no optimizer-restart confound.
#   off_aware  = retention head OFF, toggle leverage VISIBLE  (M9 winner + isolation "aware" arm)
#   off_blind  = retention head OFF, leverage HIDDEN          (same game+reward, isolation "blind" arm)
#   mult_aware = retention head ON,  toggle leverage VISIBLE  (head-decorative row)
# Skips any checkpoint that already exists, so a rate-limit interruption just resumes.
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
CKDIR=tenure/checkpoints
COMMON="--updates 220 --symmetric --alpha 1.0 --n-tasks 8 --n-robots 4 --n-adversaries 4 --horizon 45 --toggle-regions 2 --toggle-multiplier 3.0 --log-every 20 --adversary toggle_raider greedy_nearest"

run () {  # $1=name  $2..=extra args
  local name="$1"; shift
  local out="$CKDIR/hd2_${name}.pt"
  if [ -f "$out" ]; then echo "[$(date +%H:%M:%S)] SKIP $name (exists)"; return 0; fi
  echo "[$(date +%H:%M:%S)] START $name"
  $PY -m tenure.train $COMMON "$@" --save "$out" || { echo "[$(date +%H:%M:%S)] FAIL $name"; return 1; }
  echo "[$(date +%H:%M:%S)] DONE $name"
}

for s in 0 1 2 3 4; do
  run "off_aware_s${s}"  --seed "$s" --retention-mode off            --expose-toggle
  run "off_blind_s${s}"  --seed "$s" --retention-mode off
  run "mult_aware_s${s}" --seed "$s" --retention-mode multiplicative --expose-toggle
done
echo "[$(date +%H:%M:%S)] BATCH COMPLETE"
