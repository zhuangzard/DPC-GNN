#!/bin/bash
# ============================================================
# DPC-GNN Bone 29-Seed Clean Experiment
# 创建时间: 2026-03-14
# 目的: 统一F-bar配置，全新seed列表200-228（29个连续seed）
# 注意: 完全独立于旧的bone_29seed_analysis.json
#
# 配置来源（derived from bone_seed101-109 logs）:
#   - 训练脚本: solid_tissue_train_hires.py + force_fbar patch
#   - Epochs: 2000
#   - LR: 1e-3 (warmup 50ep, cosine decay to 1e-5)
#   - Mesh: 22x7x7 (1472 nodes, 5390 tets, 20904 edges)
#   - F-bar: FORCED ON (ν=0.30, global mean anti-locking)
#   - E=10,000,000 Pa (10 MPa), ν=0.30, ρ=1900 kg/m³
#   - Beam: 10cm × 2cm × 2cm
#   - Expected tip displacement: 0.6990 mm (Euler-Bernoulli)
#   - Hidden dim: 64, GNN layers: 6, Params: ~161,731
#
# 预计运行时间: ~70-80s/seed on RTX 5090
#   Total: ~35 min serial | ~18 min parallel-2
# GPU需求: ~2 GB VRAM
# 输出: /root/results/bone_29seed_clean/seed_NNN.log
# ============================================================

set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR=/root/results/bone_29seed_clean
TRAIN_SCRIPT=/root/solid_tissue_train_hires_fbar.py
MAX_PARALLEL=2

mkdir -p "$OUTDIR"

echo "============================================================"
echo " DPC-GNN Bone 29-Seed Clean Experiment"
echo " Seeds: 200-228 (29 fresh seeds, all F-bar ON)"
echo " Output: $OUTDIR"
echo " Start: $(date)"
echo "============================================================"

# ─── Step 1: Prepare F-bar patched training script ──────────────────────────
# The standard script only enables F-bar when nu >= 0.45.
# Bone has nu=0.30, so we force F-bar on via patch.
# This matches the exact behavior in seeds 101-109 (verified from logs).

echo "[$(date +%H:%M:%S)] Preparing F-bar patched training script..."

cp "$REPO_DIR/multi_tissue/src/solid_tissue_train_hires.py" "$TRAIN_SCRIPT"

python3 - << 'PATCH'
import sys

with open('/root/solid_tissue_train_hires_fbar.py', 'r') as f:
    code = f.read()

# Patch 1: Force F-bar ON for all tissues (matches bone seeds 101-109 behavior)
old = (
    '    # Use F-bar for near-incompressible materials (nu >= 0.45) to avoid volumetric locking\n'
    '    use_fbar = (nu >= 0.45)\n'
    '    if use_fbar:\n'
    '        print(f"  ⚡ F-bar enabled (ν={nu:.2f} ≥ 0.45, anti-locking)")'
)
new = (
    '    # Force F-bar ON for all solid tissues (global mean anti-locking)\n'
    '    # Matches bone_seed101-109 config: nu=0.30 requires forced F-bar (GDR criterion)\n'
    '    use_fbar = True\n'
    '    if use_fbar:\n'
    '        print(f"  ⚡ F-bar enabled (ν={nu:.2f}, global mean anti-locking)")'
)

if old in code:
    code = code.replace(old, new)
    with open('/root/solid_tissue_train_hires_fbar.py', 'w') as f:
        f.write(code)
    print("✅ Patch applied: use_fbar forced True")
else:
    print("⚠️  Primary patch pattern not found. Trying fallback...")
    # Fallback: simpler sed-style patch
    import re
    code = re.sub(
        r'use_fbar = \(nu >= 0\.45\)',
        'use_fbar = True  # forced: GDR for bone nu=0.30',
        code
    )
    code = re.sub(
        r'print\(f"  ⚡ F-bar enabled \(ν=\{nu:\.2f\} ≥ 0\.45, anti-locking\)"\)',
        'print(f"  ⚡ F-bar enabled (ν={nu:.2f}, global mean anti-locking)")',
        code
    )
    with open('/root/solid_tissue_train_hires_fbar.py', 'w') as f:
        f.write(code)
    print("✅ Fallback patch applied")

# Verify patch
with open('/root/solid_tissue_train_hires_fbar.py', 'r') as f:
    patched = f.read()
if 'use_fbar = True' in patched and 'nu >= 0.45' not in patched:
    print("✅ Verification passed: fbar forced, nu>=0.45 removed")
else:
    print("❌ Verification FAILED — check script manually")
    sys.exit(1)
PATCH

echo "[$(date +%H:%M:%S)] Training script ready: $TRAIN_SCRIPT"

# ─── Step 2: Run 29 clean seeds (200-228) ───────────────────────────────────
SEEDS=(200 201 202 203 204 205 206 207 208 209 \
       210 211 212 213 214 215 216 217 218 219 \
       220 221 222 223 224 225 226 227 228)

running_count() {
    ps aux | grep "solid_tissue_train_hires_fbar" | grep -v grep | wc -l
}

echo "[$(date +%H:%M:%S)] Starting 29 bone seeds (200-228) with F-bar"
echo "[$(date +%H:%M:%S)] Config: tissue=bone E=10MPa nu=0.30 epochs=2000 lr=1e-3 fbar=True"
echo "[$(date +%H:%M:%S)] Max parallel: $MAX_PARALLEL"
echo ""

total=0
done_skip=0

for seed in "${SEEDS[@]}"; do
    logfile="$OUTDIR/seed_${seed}.log"

    # Skip if already completed (idempotent restarts)
    if [ -f "$logfile" ] && grep -q "Relative error:" "$logfile" 2>/dev/null; then
        done_skip=$((done_skip + 1))
        echo "[$(date +%H:%M:%S)] SKIP seed=$seed (already done)"
        continue
    fi

    # Wait for parallel slot
    while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
        sleep 10
    done

    total=$((total + 1))
    echo "[$(date +%H:%M:%S)] START seed=$seed (job #$total, skipped=$done_skip)"

    nohup python3 "$TRAIN_SCRIPT" \
        --tissue bone \
        --epochs 2000 \
        --lr 1e-3 \
        --seed "$seed" \
        > "$logfile" 2>&1 &

    sleep 2
done

echo "[$(date +%H:%M:%S)] All $total jobs submitted (skipped $done_skip). Waiting for completion..."
wait
echo "[$(date +%H:%M:%S)] ALL SEEDS COMPLETE"

# ─── Step 3: Collect and report results ─────────────────────────────────────
echo ""
echo "========================================================"
echo " RESULTS SUMMARY — Bone 29-Seed Clean (seeds 200-228)"
echo "========================================================"

python3 - << 'COLLECT'
import os, re, json, statistics

SEEDS = list(range(200, 229))  # 200-228 inclusive (29 seeds)
OUTDIR = '/root/results/bone_29seed_clean'

errors = []
seed_errors = {}
failed = []

for seed in SEEDS:
    logfile = os.path.join(OUTDIR, f'seed_{seed}.log')
    if not os.path.exists(logfile):
        print(f'  seed {seed}: MISSING')
        failed.append(seed)
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
        print(f'  seed {seed}: FAILED (no result)')
        failed.append(seed)

print()
if errors:
    n = len(errors)
    mean_e = statistics.mean(errors)
    med_e  = statistics.median(errors)
    std_e  = statistics.stdev(errors) if n > 1 else 0.0
    min_e  = min(errors)
    max_e  = max(errors)

    print(f'Completed: {n}/29 seeds')
    print(f'Mean:      {mean_e:.2f}%')
    print(f'Median:    {med_e:.2f}%')
    print(f'Std:       {std_e:.2f}%')
    print(f'Min/Max:   {min_e:.2f}% / {max_e:.2f}%')
    print(f'<5%:       {sum(1 for e in errors if e < 5)}/{n}')
    print(f'<10%:      {sum(1 for e in errors if e < 10)}/{n}')
    print(f'<15%:      {sum(1 for e in errors if e < 15)}/{n}')
    if failed:
        print(f'Failed:    {failed}')

    result = {
        'tissue': 'bone',
        'experiment': 'bone_29seed_clean',
        'description': 'Fresh 29-seed run, seeds 200-228, all unified F-bar ON. '
                       'Completely independent from bone_29seed_analysis.json.',
        'config': {
            'seeds': SEEDS,
            'fbar': True,
            'fbar_note': 'Forced ON (GDR criterion: bone nu=0.30 <= 0.35)',
            'E_Pa': 10000000.0,
            'nu': 0.30,
            'rho': 1900,
            'beam': '10cm x 2cm x 2cm',
            'mesh': '22x7x7',
            'nodes': 1472,
            'tets': 5390,
            'epochs': 2000,
            'lr': 1e-3,
            'hidden_dim': 64,
            'n_layers': 6,
            'params': 161731,
            'expected_tip_mm': 0.699,
            'training_script': 'solid_tissue_train_hires.py (fbar-forced patch)',
        },
        'results': {
            'seeds': SEEDS,
            'errors': [seed_errors.get(s) for s in SEEDS],
            'n_completed': n,
            'n_total': 29,
            'mean_pct': round(mean_e, 2),
            'median_pct': round(med_e, 2),
            'std_pct': round(std_e, 2),
            'min_pct': round(min_e, 2),
            'max_pct': round(max_e, 2),
            'success_5pct': sum(1 for e in errors if e < 5),
            'success_10pct': sum(1 for e in errors if e < 10),
            'success_15pct': sum(1 for e in errors if e < 15),
            'failed_seeds': failed,
        },
        'isolation': {
            'uses_old_bone_29seed_analysis': False,
            'reuses_old_seeds': False,
            'old_seeds_excluded': [42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                   123, 456, 789, 999, 1000, 1337, 2024, 2026, 3407,
                                   7777, 8888, 11111, 12345, 31415, 54321, 66666, 99999, 271828],
        }
    }

    outfile = os.path.join(OUTDIR, 'bone_29seed_clean_results.json')
    with open(outfile, 'w') as f:
        import json as _json
        _json.dump(result, f, indent=2)
    print(f'\nResults saved to: {outfile}')
else:
    print('ERROR: No seeds completed!')
    exit(1)
COLLECT

echo ""
echo "[$(date +%H:%M:%S)] =============================================="
echo "[$(date +%H:%M:%S)]  DONE. Copy results to paper data directory:"
echo "[$(date +%H:%M:%S)]  scp root@<gpuhub_ip>:/root/results/bone_29seed_clean/bone_29seed_clean_results.json \\"
echo "[$(date +%H:%M:%S)]      ~/workspace/DPC-GNN/paper/unified-multitissue/data/"
echo "[$(date +%H:%M:%S)] =============================================="
