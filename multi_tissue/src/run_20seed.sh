#!/bin/bash
set -u

SEEDS=(42 123 456 789 2026 1337 999 31415 271828 1000 2024 3407 7777 8888 12345 54321 99999 11111 66666 100)
TISSUES=(brain kidney myocardium cartilage vessel bone)
OUTDIR=/root/results/twentyseed
MAX_PARALLEL=2

mkdir -p $OUTDIR

get_epochs() {
    local tissue=$1
    case $tissue in
        vessel) echo 2500 ;;
        *)      echo 3000 ;;
    esac
}

running_count() {
    ps aux | grep "solid_tissue_train_hires" | grep -v grep | wc -l
}

total=0
done_skip=0
for tissue in "${TISSUES[@]}"; do
    epochs=$(get_epochs $tissue)
    for seed in "${SEEDS[@]}"; do
        logfile=$OUTDIR/${tissue}_seed${seed}.log
        
        # Skip if already completed (check for "Relative error" in log)
        if [ -f "$logfile" ] && grep -q "Relative error" "$logfile" 2>/dev/null; then
            done_skip=$((done_skip + 1))
            continue
        fi
        
        # Wait for slot
        while [ $(running_count) -ge $MAX_PARALLEL ]; do
            sleep 15
        done
        
        total=$((total + 1))
        echo "[$(date +%H:%M:%S)] Starting $tissue seed=$seed epochs=$epochs (job #$total, skipped $done_skip)"
        
        nohup python3 /root/solid_tissue_train_hires.py \
            --tissue $tissue --seed $seed --epochs $epochs \
            > $logfile 2>&1 &
        
        sleep 3
    done
done

echo "[$(date +%H:%M:%S)] All $total jobs submitted (skipped $done_skip already done). Waiting..."
wait
echo "[$(date +%H:%M:%S)] ALL COMPLETE. Total launched: $total, skipped: $done_skip"

# Quick summary
echo ""
echo "=== QUICK SUMMARY ==="
for tissue in "${TISSUES[@]}"; do
    echo "--- $tissue ---"
    for seed in "${SEEDS[@]}"; do
        logfile=$OUTDIR/${tissue}_seed${seed}.log
        if [ -f "$logfile" ]; then
            err=$(grep "Relative error:" "$logfile" | tail -1 | grep -oP "[\d.]+(?=%)")
            echo "  seed=$seed: ${err:-FAILED}%"
        else
            echo "  seed=$seed: MISSING"
        fi
    done
done
