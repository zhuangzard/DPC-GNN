#!/bin/bash
# ============================================================
# MGN-Baseline Run — AMP Ablation (R1 Minor 4)
#
# Six tissues × 5 seeds = 30 jobs.  Same physics-loss training and
# canonical hyperparameters as DPC-GNN's run_20seed.sh; only the
# message-passing primitive (AntisymMP → DirectedMP) differs.
# Output: /root/results/mgn_baseline/{tissue}_seed{N}.log per run, plus
# /root/results/mgn_baseline/{tissue}_mgn/{best_model.pt,results.json,history.json}
#
# Expected wall time per run on RTX 5090: ~70 s (cartilage/bone) to
# ~120 s (vessel high-res).  Five seeds × six tissues ~ 30 min wall
# serial; ~15 min with MAX_PARALLEL=2.
#
# To run:   bash run_mgn_baseline.sh
# ============================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/workspace/DPC-GNN}"
cd "$REPO_ROOT"

OUTDIR=/root/results/mgn_baseline
TRAIN_SCRIPT="$REPO_ROOT/multi_tissue/src/solid_tissue_train_mgn.py"
MAX_PARALLEL=${MAX_PARALLEL:-2}

# 5 representative seeds from the canonical 20-seed list (cheaper than full
# 20-seed sweep; AMP-vs-MGN gap is dramatic enough to be detectable with 5).
SEEDS=(42 123 456 789 2026)

# Tissues to ablate.  Soft + near-incompressible to cover the full range.
TISSUES=(brain kidney myocardium vessel cartilage bone)

declare -A EPOCHS=(
  [brain]=3000      [kidney]=3000   [myocardium]=3000
  [vessel]=2500     [cartilage]=3000 [bone]=2000
)

mkdir -p "$OUTDIR"
echo "MGN-baseline ablation start: $(date)" > "$OUTDIR/run.log"

# Override TISSUES dict for bone (the canonical trainer's dict uses 1e6 Pa;
# the published bone runs use 1e7 Pa = 10 MPa).  Pass E/nu/rho explicitly
# from the bash side so the script doesn't need a hardcoded bone entry.
declare -A E_PA=(
  [brain]=1000       [kidney]=10000  [myocardium]=30000
  [vessel]=400000    [cartilage]=500000  [bone]=10000000
)
declare -A NU=(
  [brain]=0.49 [kidney]=0.45 [myocardium]=0.40
  [vessel]=0.49 [cartilage]=0.30 [bone]=0.30
)
declare -A RHO=(
  [brain]=1040 [kidney]=1050 [myocardium]=1060
  [vessel]=1050 [cartilage]=1100 [bone]=1900
)

job_count=0
for tissue in "${TISSUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    LOG="$OUTDIR/${tissue}_seed${seed}.log"
    echo "[$(date +%H:%M:%S)] Starting $tissue seed=$seed → $LOG" | tee -a "$OUTDIR/run.log"

    (
      python3 "$TRAIN_SCRIPT" \
        --tissue "$tissue" \
        --E "${E_PA[$tissue]}" --nu "${NU[$tissue]}" --rho "${RHO[$tissue]}" \
        --epochs "${EPOCHS[$tissue]}" \
        --seed "$seed" \
        --output-root "$OUTDIR" \
        > "$LOG" 2>&1
    ) &

    job_count=$((job_count + 1))
    if (( job_count % MAX_PARALLEL == 0 )); then
      wait
    fi
  done
done

wait

echo "MGN-baseline ablation done: $(date)" | tee -a "$OUTDIR/run.log"
echo "Output: $OUTDIR"
ls -la "$OUTDIR" | tail -20
