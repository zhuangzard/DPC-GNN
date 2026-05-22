#!/bin/bash
set -uo pipefail
cd /root/dpcgnn_forensic/src
OUT=$(ls -td /root/results/ablations_* | head -1)
echo "OUT=$OUT"
for seed in 42 123 456 789 2026; do
  LOG="$OUT/mgn/bone_seed${seed}.log"
  echo "[$(date +%H:%M:%S)] re-run BONE MGN seed=$seed -> $LOG"
  (python3 solid_tissue_train_mgn.py \
    --tissue bone --E 10000000 --nu 0.30 --rho 1900 \
    --epochs 2000 --seed "$seed" \
    --output-root "$OUT/mgn" > "$LOG" 2>&1) &
done
wait
echo "BONE MGN DONE: $(date)"
