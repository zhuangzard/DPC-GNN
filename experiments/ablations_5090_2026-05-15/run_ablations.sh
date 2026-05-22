#!/bin/bash
# ============================================================
# Master driver for the AMP + EBC ablation sweep.
# RTX 5090 (Singapore-A), 2026-05-15.
# Wall time: ~30 minutes @ MAX_PARALLEL=4.
# ============================================================
set -euo pipefail
cd /root/dpcgnn_forensic/src

OUT=/root/results/ablations_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT/mgn" "$OUT/barrier"
MAX_PARALLEL=${MAX_PARALLEL:-4}

SEEDS=(42 123 456 789 2026)

declare -A E_PA=(
  [brain]=1000 [kidney]=10000 [myocardium]=30000
  [vessel]=400000 [cartilage]=500000 [bone]=10000000
)
declare -A NU=(
  [brain]=0.49 [kidney]=0.45 [myocardium]=0.40
  [vessel]=0.49 [cartilage]=0.30 [bone]=0.30
)
declare -A RHO=(
  [brain]=1040 [kidney]=1050 [myocardium]=1060
  [vessel]=1050 [cartilage]=1100 [bone]=1900
)
declare -A EPOCHS=(
  [brain]=3000 [kidney]=3000 [myocardium]=3000
  [vessel]=2500 [cartilage]=3000 [bone]=2000
)

job_count=0
wait_if_full() {
  if (( job_count % MAX_PARALLEL == 0 )); then wait; fi
}

echo "=== MGN baseline: 6 tissues x 5 seeds = 30 runs ===" | tee "$OUT/run.log"
for tissue in brain kidney myocardium vessel cartilage bone; do
  for seed in "${SEEDS[@]}"; do
    LOG="$OUT/mgn/${tissue}_seed${seed}.log"
    echo "[$(date +%H:%M:%S)] MGN $tissue seed=$seed" | tee -a "$OUT/run.log"
    (python3 solid_tissue_train_mgn.py \
      --tissue "$tissue" \
      --E "${E_PA[$tissue]}" --nu "${NU[$tissue]}" --rho "${RHO[$tissue]}" \
      --epochs "${EPOCHS[$tissue]}" --seed "$seed" \
      --output-root "$OUT/mgn" > "$LOG" 2>&1) &
    job_count=$((job_count+1)); wait_if_full
  done
done
wait
echo "=== MGN baseline DONE: $(date) ===" | tee -a "$OUT/run.log"

echo "=== Barrier ablation: 3 tissues x 5 seeds x 2 settings = 30 runs ===" | tee -a "$OUT/run.log"
for tissue in cartilage bone vessel; do
  for seed in "${SEEDS[@]}"; do
    for coef in 0 100; do
      LOG="$OUT/barrier/${tissue}_coef${coef}_seed${seed}.log"
      echo "[$(date +%H:%M:%S)] BAR $tissue seed=$seed coef=$coef" | tee -a "$OUT/run.log"
      (python3 solid_tissue_train_no_barrier.py \
        --tissue "$tissue" \
        --E "${E_PA[$tissue]}" --nu "${NU[$tissue]}" --rho "${RHO[$tissue]}" \
        --epochs "${EPOCHS[$tissue]}" --seed "$seed" \
        --barrier-coef "$coef" \
        --output-root "$OUT/barrier" > "$LOG" 2>&1) &
      job_count=$((job_count+1)); wait_if_full
    done
  done
done
wait
echo "=== Barrier ablation DONE: $(date) ===" | tee -a "$OUT/run.log"
echo "Outputs in: $OUT" | tee -a "$OUT/run.log"
