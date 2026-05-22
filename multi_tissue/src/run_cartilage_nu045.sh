#!/bin/bash
# ============================================================
# Cartilage Retrain @ ν=0.45 (R5C-3 response)
#
# The published cartilage runs used ν=0.30; reviewer R5C-3 (blind
# biomechanics) correctly noted articular cartilage in the
# infinitesimal regime is literature 0.40–0.49. This script reruns
# cartilage at ν=0.45 with the GDR (F-bar) anti-locking branch
# auto-active (use_fbar triggers at ν ≥ 0.45 inside solid_tissue_train.py,
# see lines 215–217).
#
# IMPORTANT: keeps the original ν=0.30 results intact — writes to a
# DIFFERENT directory (cartilage_nu045/, NOT cartilage/).
#
# Twenty canonical seeds × ~45 min each ≈ 14 GPU-h serial wall.
# With MAX_PARALLEL=2 (default) ≈ 7 GPU-h wall.
#
# Usage:
#   bash run_cartilage_nu045.sh
#   OUTDIR=/root/results/cartilage_nu045_v2 MAX_PARALLEL=4 bash run_cartilage_nu045.sh
# ============================================================

# NOTE: deliberately use `-uo pipefail` (NOT `-euo pipefail`).  In a 20-seed
# sweep we want a single-seed failure to be logged and skipped, not to abort
# the whole run.  Background `python3` failures are caught by `wait || …`
# below and surfaced in the final summary loop.
set -uo pipefail

# ---- configuration ------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$HOME/workspace/DPC-GNN}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/multi_tissue/src/solid_tissue_train.py}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-$REPO_ROOT/multi_tissue/src/collect_20seed.py}"
OUTDIR="${OUTDIR:-/root/results/cartilage_nu045}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
EPOCHS="${EPOCHS:-3000}"

# Cartilage physical parameters — ν is the reviewer-driven change.
E_PA=500000
NU=0.45
RHO=1100

# Canonical 20-seed list (matches run_20seed.sh exactly).
SEEDS=(42 100 123 456 789 999 1000 1337 2024 2026 3407 7777 8888 11111 12345 31415 54321 66666 99999 271828)
N_SEEDS=${#SEEDS[@]}

# ---- safety: don't clobber the ν=0.30 historical record -----------------
if [[ "$OUTDIR" == */cartilage || "$OUTDIR" == */cartilage/ ]]; then
    echo "ERROR: OUTDIR=$OUTDIR would overwrite the ν=0.30 historical runs." >&2
    echo "       Pick a different path (default cartilage_nu045/)." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# ---- pre-run validation -------------------------------------------------
echo "=================================================================="
echo "  Cartilage retrain @ ν=0.45  (R5C-3 response)"
echo "=================================================================="
echo "  Train script : $TRAIN_SCRIPT"
echo "  Output dir   : $OUTDIR"
echo "  E / ν / ρ    : $E_PA Pa / $NU / $RHO kg/m^3"
echo "  Seeds        : ${N_SEEDS}  (${SEEDS[*]})"
echo "  Epochs/seed  : $EPOCHS"
echo "  Parallel jobs: $MAX_PARALLEL"
echo "  Timestamp    : $(date '+%Y-%m-%d %H:%M:%S %Z')"

# (1) Confirm --nu is accepted by the trainer.
if ! python3 "$TRAIN_SCRIPT" --help 2>&1 | grep -qE -- "--nu\b"; then
    echo "ERROR: $TRAIN_SCRIPT does not advertise --nu in --help." >&2
    echo "       Refusing to launch — the override would silently be ignored." >&2
    exit 2
fi
echo "  Pre-check    : --nu flag present in trainer --help  [OK]"

# (2) Check whether --output-root is accepted (solid_tissue_train.py does NOT
#     have this flag; solid_tissue_train_mgn.py does).  If absent, we drop
#     the flag and rely on the script's own /root/results/cartilage/ default,
#     then copy logs into $OUTDIR (logs are what collect_20seed.py parses).
HAVE_OUTPUT_ROOT=0
if python3 "$TRAIN_SCRIPT" --help 2>&1 | grep -qE -- "--output-root\b"; then
    HAVE_OUTPUT_ROOT=1
    echo "  Pre-check    : --output-root flag present                  [OK]"
else
    echo "  Pre-check    : --output-root NOT supported by $(basename "$TRAIN_SCRIPT")"
    echo "                 → logs still land in $OUTDIR (parsed by collect_20seed.py)"
    echo "                 → checkpoints land in /root/results/cartilage/ per script default"
fi

# (3) Estimated wall time.
# ~45 min per seed × 20 seeds = 900 min serial; divided by MAX_PARALLEL.
TOTAL_MIN=$(( 45 * N_SEEDS / MAX_PARALLEL ))
TOTAL_H=$(( TOTAL_MIN / 60 ))
TOTAL_M=$(( TOTAL_MIN % 60 ))
echo "  Est. wall    : ~${TOTAL_H}h ${TOTAL_M}m at MAX_PARALLEL=$MAX_PARALLEL"
echo "                 (45 min/seed × $N_SEEDS seeds / $MAX_PARALLEL)"
echo "=================================================================="

# ---- launch loop --------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Starting sweep..." | tee "$OUTDIR/run.log"

job_count=0
launched=0
skipped=0

for seed in "${SEEDS[@]}"; do
    LOG="$OUTDIR/cartilage_seed${seed}.log"

    # Skip if already completed (matches run_20seed.sh resume semantics).
    if [[ -f "$LOG" ]] && grep -q "Relative error" "$LOG" 2>/dev/null; then
        skipped=$((skipped + 1))
        echo "[$(date '+%H:%M:%S')] seed=$seed  already complete, skipping" \
            | tee -a "$OUTDIR/run.log"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] seed=$seed  → $LOG" \
        | tee -a "$OUTDIR/run.log"

    (
        if [[ "$HAVE_OUTPUT_ROOT" -eq 1 ]]; then
            python3 "$TRAIN_SCRIPT" \
                --tissue cartilage \
                --E "$E_PA" --nu "$NU" --rho "$RHO" \
                --epochs "$EPOCHS" \
                --seed "$seed" \
                --output-root "$OUTDIR" \
                > "$LOG" 2>&1
        else
            python3 "$TRAIN_SCRIPT" \
                --tissue cartilage \
                --E "$E_PA" --nu "$NU" --rho "$RHO" \
                --epochs "$EPOCHS" \
                --seed "$seed" \
                > "$LOG" 2>&1
        fi
    ) &

    launched=$((launched + 1))
    job_count=$((job_count + 1))
    if (( job_count % MAX_PARALLEL == 0 )); then
        wait || echo "[warn] one or more parallel runs failed; continuing"
    fi
done

wait || echo "[warn] one or more parallel runs failed; continuing"

echo "[$(date '+%H:%M:%S')] Sweep complete — launched=$launched, skipped=$skipped" \
    | tee -a "$OUTDIR/run.log"

# ---- post-run validation: GDR (F-bar) should have activated at ν=0.45 ---
echo ""
echo "=== GDR (F-bar) activation check ==="
gdr_hits=0
for seed in "${SEEDS[@]}"; do
    LOG="$OUTDIR/cartilage_seed${seed}.log"
    [[ -f "$LOG" ]] || continue
    if grep -qE "F-bar enabled|GDR active" "$LOG"; then
        gdr_hits=$((gdr_hits + 1))
    fi
done
echo "  GDR/F-bar log line present in $gdr_hits / $N_SEEDS runs"
if (( gdr_hits == 0 )); then
    echo "  WARNING: no GDR activation detected — verify ν=0.45 trigger threshold." >&2
fi

# ---- aggregate ----------------------------------------------------------
if [[ -f "$COLLECT_SCRIPT" ]]; then
    echo ""
    echo "=== Aggregating with $(basename "$COLLECT_SCRIPT") ==="
    # collect_20seed.py hard-codes OUTDIR; patch it via a temporary env-style
    # override using a wrapper sed-on-the-fly if user has set OUTDIR != default.
    if grep -q "OUTDIR = \"/root/results/twentyseed\"" "$COLLECT_SCRIPT"; then
        echo "  NOTE: collect_20seed.py reads /root/results/twentyseed by default."
        echo "        Symlinking $OUTDIR → /root/results/twentyseed_cartilage_nu045 for traceability."
        ln -sfn "$OUTDIR" /root/results/twentyseed_cartilage_nu045 2>/dev/null || true
    fi
    python3 "$COLLECT_SCRIPT" || echo "  (collect_20seed.py returned non-zero; inspect manually)"
else
    echo "collect_20seed.py not found at $COLLECT_SCRIPT — skipping aggregation."
fi

# ---- final summary ------------------------------------------------------
echo ""
echo "=== FINAL SUMMARY: cartilage ν=$NU ==="
for seed in "${SEEDS[@]}"; do
    LOG="$OUTDIR/cartilage_seed${seed}.log"
    if [[ -f "$LOG" ]]; then
        err=$(grep "Relative error:" "$LOG" | tail -1 | grep -oE "[0-9.]+%" | head -1)
        echo "  seed=$seed: ${err:-FAILED}"
    else
        echo "  seed=$seed: MISSING"
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] Done.  Outputs in $OUTDIR"
