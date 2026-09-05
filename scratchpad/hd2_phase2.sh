#!/usr/bin/env bash
# Phase 2 (queued): after the hd2 batch finishes, train the E9 vanilla checkpoint on the BASE interface
# (task_dim=7, non-symmetric, retention head active) so retention_by_site_class can run on real Override.
set -u
cd "c:/Users/xiele/Documents/Rex_Technologies/leroiai-offSim" || exit 1
PY=.venv313/Scripts/python.exe
until [ "$(ls tenure/checkpoints/hd2_*.pt 2>/dev/null | wc -l)" -ge 15 ] || grep -q "BATCH COMPLETE" scratchpad/hd2_train.log 2>/dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] phase2: hd2 complete -> training e9_vanilla"
if [ ! -f tenure/checkpoints/e9_vanilla.pt ]; then
  $PY -m tenure.train --updates 220 --alpha 1.0 --protected-fraction 0.15 \
      --retention-mode multiplicative --log-every 20 --save tenure/checkpoints/e9_vanilla.pt
  echo "[$(date +%H:%M:%S)] phase2: e9_vanilla done"
else
  echo "[$(date +%H:%M:%S)] phase2: e9_vanilla already exists, skip"
fi
echo "[$(date +%H:%M:%S)] PHASE2 COMPLETE"
