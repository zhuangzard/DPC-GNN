#!/bin/bash
# ============================================================
# Bone 29-seed Complete Experiment (F-bar unified)
# Seeds 200-219 (20 new seeds to complement existing 101-109)
# 
# Config derived from bone_seed101-109 logs:
#   - Epochs: 2000
#   - LR: 1e-3 (warmup 50ep, cosine decay to 1e-5)
#   - Mesh: 22×7×7 (1472 nodes, 5390 tets)
#   - F-bar: FORCED ON (ν=0.30, global mean anti-locking)
#   - E=10,000,000 Pa (10 MPa), ν=0.30, ρ=1900
#   - Hidden dim: 64, GNN layers: 6, Params: 161,731
# 
# Expected per-seed time: ~70s on RTX 5090
# Total estimated time: ~25 min serial, ~13 min parallel-2
# 
# GPU requirement: ~2GB VRAM (very lightweight)
# Output: /root/results/bone_29seed_complete/seed_NNN.log
# ============================================================

set -euo pipefail
cd ~/workspace/DPC-GNN

OUTDIR=/root/results/bone_29seed_complete
TRAIN_SCRIPT=/root/solid_tissue_train_hires_fbar.py
MAX_PARALLEL=2

mkdir -p "$OUTDIR"

# ─── Step 1: Create force-fbar version of training script ───────────────────
# The standard script only enables F-bar when nu >= 0.45.
# Bone has nu=0.30, so we need to force F-bar on.
cp multi_tissue/src/solid_tissue_train_hires.py "$TRAIN_SCRIPT"

# Patch: change "use_fbar = (nu >= 0.45)" to always True for bone
# Also update the print message to show it's forced
python3 - << 'PATCH'
import re

with open('/root/solid_tissue_train_hires_fbar.py', 'r') as f:
    code = f.read()

# Replace the fbar condition to force it on (mirrors the version used for seeds 101-109)
old = '    # Use F-bar for near-incompressible materials (nu >= 0.45) to avoid volumetric locking\n    use_fbar = (nu >= 0.45)\n    if use_fbar:\n        print(f"  ⚡ F-bar enabled (ν={nu:.2f} ≥ 0.45, anti-locking)")'
new = '    # Force F-bar enabled for all solid tissues (global mean anti-locking)\n    use_fbar = True\n    if use_fbar:\n        print(f"  ⚡ F-bar enabled (ν={nu:.2f}, global mean anti-locking)")'

if old in code:
    code = code.replace(old, new)
    with open('/root/solid_tissue_train_hires_fbar.py', 'w') as f:
        f.write(code)
    print("Patch applied successfully")
else:
    print("WARNING: Expected pattern not found! Check script version.")
    # Fallback: use sed-based patch
    import subprocess
    subprocess.run(['sed', '-i', 
        's/use_fbar = (nu >= 0.45)/use_fbar = True  # forced for bone nu=0.30/',
        '/root/solid_tissue_train_hires_fbar.py'])
    print("Fallback sed patch applied")
PATCH

echo "[$(date +%H:%M:%S)] Training script prepared: $TRAIN_SCRIPT"

# ─── Step 2: Run 20 new seeds ───────────────────────────────────────────────
SEEDS=(200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219)

running_count() {
    ps aux | grep "solid_tissue_train_hires_fbar" | grep -v grep | wc -l
}

echo "[$(date +%H:%M:%S)] Starting 20 bone F-bar seeds (200-219)"
echo "[$(date +%H:%M:%S)] Output directory: $OUTDIR"
echo "[$(date +%H:%M:%S)] Max parallel: $MAX_PARALLEL"

total=0
done_skip=0

for seed in "${SEEDS[@]}"; do
    logfile="$OUTDIR/seed_${seed}.log"
    
    # Skip if already completed
    if [ -f "$logfile" ] && grep -q "Relative error:" "$logfile" 2>/dev/null; then
        done_skip=$((done_skip + 1))
        echo "[$(date +%H:%M:%S)] Skipping seed=$seed (already done)"
        continue
    fi
    
    # Wait for slot
    while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done
    
    total=$((total + 1))
    echo "[$(date +%H:%M:%S)] Starting seed=$seed (job #$total, skipped=$done_skip)"
    
    nohup python3 "$TRAIN_SCRIPT" \
        --tissue bone \
        --epochs 2000 \
        --lr 1e-3 \
        --seed "$seed" \
        > "$logfile" 2>&1 &
    
    sleep 2
done

echo "[$(date +%H:%M:%S)] All $total jobs submitted (skipped $done_skip). Waiting..."
wait
echo "[$(date +%H:%M:%S)] ALL SEEDS COMPLETE"

# ─── Step 3: Collect results ────────────────────────────────────────────────
echo ""
echo "=== RESULTS SUMMARY (Seeds 200-219) ==="
python3 - << 'COLLECT'
import os, re, json, statistics

seeds = list(range(200, 220))
outdir = '/root/results/bone_29seed_complete'
errors = []
seed_errors = {}

for seed in seeds:
    logfile = os.path.join(outdir, f'seed_{seed}.log')
    if not os.path.exists(logfile):
        print(f'  seed {seed}: MISSING')
        continue
    with open(logfile) as f:
        content = f.read()
    m = re.findall(r'Relative error:\s*([\d.]+)%', content)
    if m:
        err = float(m[-1])
        errors.append(err)
        seed_errors[seed] = err
        print(f'  seed {seed}: {err:.2f}%')
    else:
        print(f'  seed {seed}: FAILED (no result found)')

if errors:
    print(f'\nSummary (seeds 200-219):')
    print(f'  Mean:   {statistics.mean(errors):.2f}%')
    print(f'  Median: {statistics.median(errors):.2f}%')
    print(f'  Std:    {statistics.stdev(errors):.2f}%')
    print(f'  Min:    {min(errors):.2f}%')
    print(f'  Max:    {max(errors):.2f}%')
    print(f'  <5%:    {sum(1 for e in errors if e < 5)}/{len(errors)}')
    print(f'  <10%:   {sum(1 for e in errors if e < 10)}/{len(errors)}')
    
    # Combine with existing 9 seeds (101-109)
    existing = {
        101: 18.30, 102: 6.77, 103: 3.45, 104: 0.42, 105: 5.19,
        106: 0.94, 107: 0.19, 108: 2.64, 109: 0.95
    }
    all_errors = list(existing.values()) + errors
    all_seeds = list(existing.keys()) + seeds
    
    print(f'\nCombined 29-seed (101-109 + 200-219):')
    print(f'  Mean:   {statistics.mean(all_errors):.2f}%')
    print(f'  Median: {statistics.median(all_errors):.2f}%')
    print(f'  Std:    {statistics.stdev(all_errors):.2f}%')
    print(f'  Min:    {min(all_errors):.2f}%')
    print(f'  Max:    {max(all_errors):.2f}%')
    print(f'  <5%:    {sum(1 for e in all_errors if e < 5)}/{len(all_errors)}')
    print(f'  <10%:   {sum(1 for e in all_errors if e < 10)}/{len(all_errors)}')
    
    # Save JSON
    result = {
        'tissue': 'bone',
        'experiment': '29seed_unified_fbar',
        'fbar': True,
        'seeds_new': seeds,
        'errors_new': [seed_errors.get(s) for s in seeds],
        'seeds_existing': list(existing.keys()),
        'errors_existing': list(existing.values()),
        'all_seeds': all_seeds,
        'all_errors': all_errors,
        'mean_pct': round(statistics.mean(all_errors), 2),
        'median_pct': round(statistics.median(all_errors), 2),
        'std_pct': round(statistics.stdev(all_errors), 2),
        'min_pct': round(min(all_errors), 2),
        'max_pct': round(max(all_errors), 2),
        'success_5pct': sum(1 for e in all_errors if e < 5),
        'success_10pct': sum(1 for e in all_errors if e < 10),
        'n_total': len(all_errors),
        'note': 'Unified F-bar run: seeds 101-109 (original fbar) + 200-219 (new fbar). All seeds used identical config: epochs=2000, lr=1e-3, fbar=True, E=10MPa, nu=0.30'
    }
    
    outfile = '/root/results/bone_29seed_complete/bone_29seed_unified_results.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResults saved to: {outfile}')
COLLECT

echo ""
echo "[$(date +%H:%M:%S)] DONE. Copy results back:"
echo "  scp -r root@<gpuhub_ip>:/root/results/bone_29seed_complete/ ~/workspace/DPC-GNN/paper/unified-multitissue/data/"

# ─── Quick test (single seed, ~70s) ──────────────────────────────────────────
# To verify setup before full run:
# python3 /root/solid_tissue_train_hires_fbar.py --tissue bone --epochs 2000 --lr 1e-3 --seed 999
# Expected: ~70s, Relative error < 10%, ⚡ F-bar enabled (ν=0.30, global mean anti-locking)
