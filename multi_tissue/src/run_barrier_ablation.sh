#!/bin/bash
# ============================================================
# Barrier Ablation Run — EBC On/Off  (F10 D-14 / F13)
#
# Trains DPC-GNN with the linear ReLU barrier coefficient k_E set to 0
# (ablation, "no barrier") and compares against the canonical k_E = 100
# baseline.  Cartilage + bone + vessel are the inversion-prone tissues
# (largest deformation under compressive loading), so this is where the
# barrier matters most.
#
# Per-epoch n_inverted is recorded in each history.json, and the
# script also surfaces the cumulative max_n_inverted_during_training
# and epoch_first_inversion in results.json.  The abstract's "hundreds
# of inverted elements per simulation step" claim becomes quantifiable.
#
# Three tissues × five seeds × two barrier settings = 30 jobs.
# Expected wall time on RTX 5090: ~25 min serial, ~13 min with MAX_PARALLEL=2.
#
# To run:   bash run_barrier_ablation.sh
# ============================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/workspace/DPC-GNN}"
cd "$REPO_ROOT"

OUTDIR=/root/results/barrier_ablation
TRAIN_SCRIPT="$REPO_ROOT/multi_tissue/src/solid_tissue_train_no_barrier.py"
MAX_PARALLEL=${MAX_PARALLEL:-2}

SEEDS=(42 123 456 789 2026)
TISSUES=(cartilage bone vessel)

declare -A EPOCHS=(
  [cartilage]=3000 [bone]=2000 [vessel]=2500
)
declare -A E_PA=(
  [cartilage]=500000 [bone]=10000000 [vessel]=400000
)
declare -A NU=(
  [cartilage]=0.30 [bone]=0.30 [vessel]=0.49
)
declare -A RHO=(
  [cartilage]=1100 [bone]=1900 [vessel]=1050
)

mkdir -p "$OUTDIR"
echo "Barrier ablation start: $(date)" > "$OUTDIR/run.log"

run_one () {
  local tissue=$1 seed=$2 coef=$3
  local tag
  if [[ "$coef" == "0" || "$coef" == "0.0" ]]; then tag="no_barrier"; else tag="barrier_${coef}"; fi
  local LOG="$OUTDIR/${tissue}_${tag}_seed${seed}.log"
  echo "[$(date +%H:%M:%S)] $tissue seed=$seed barrier=$coef → $LOG" | tee -a "$OUTDIR/run.log"
  python3 "$TRAIN_SCRIPT" \
    --tissue "$tissue" \
    --E "${E_PA[$tissue]}" --nu "${NU[$tissue]}" --rho "${RHO[$tissue]}" \
    --epochs "${EPOCHS[$tissue]}" \
    --seed "$seed" \
    --barrier-coef "$coef" \
    --output-root "$OUTDIR" \
    > "$LOG" 2>&1
}

job_count=0
for tissue in "${TISSUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for coef in 0 100; do
      run_one "$tissue" "$seed" "$coef" &
      job_count=$((job_count + 1))
      if (( job_count % MAX_PARALLEL == 0 )); then
        wait
      fi
    done
  done
done
wait

echo "Barrier ablation done: $(date)" | tee -a "$OUTDIR/run.log"
echo "Output: $OUTDIR"
ls -la "$OUTDIR" | tail -20

# Summary parser — count first-inversion epochs across all runs.
echo ""
echo "Summary: epoch of first inversion (across all no_barrier runs)"
echo "============================================================="
python3 - <<'PY'
import json, os, glob
OUTDIR = "/root/results/barrier_ablation"
results = []
for f in sorted(glob.glob(f"{OUTDIR}/*_no_barrier_seed*.log")):
    # Find matching results.json
    base = os.path.basename(f).replace(".log","").replace("_seed", "/seed")
    # Walk through results dirs
    pass

# Better: read all results.json files
rows = []
for d in sorted(glob.glob(f"{OUTDIR}/*_no_barrier")):
    rj = os.path.join(d, "results.json")
    if not os.path.exists(rj): continue
    r = json.load(open(rj))
    rows.append((r["tissue"], r.get("n_inverted_max_during_training", 0),
                 r.get("epoch_first_inversion"), r.get("n_inverted_final", 0)))
print(f"{'Tissue':>12} | {'Max n_inv':>10} | {'First Ep':>9} | {'Final n_inv':>11}")
print("-" * 60)
for t,m,fe,fin in rows:
    fe_str = str(fe) if fe is not None else "—"
    print(f"{t:>12} | {m:>10} | {fe_str:>9} | {fin:>11}")
PY
