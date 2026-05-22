#!/bin/bash
# Run all vessel improved experiments: 5 seeds × 2 methods
# Sequential to avoid GPU contention

set -e

SEEDS="42 123 456 789 2026"

echo "=========================================="
echo "  Vessel Improved Training - All Seeds"
echo "=========================================="

# Method A: Mixed Sampling
for seed in $SEEDS; do
    echo ""
    echo ">>> Method A, seed=$seed"
    python3 /root/vessel_improved_A.py --seed $seed --epochs 2000 2>&1 | tee /root/results/vessel_improved_A/seed_${seed}/log.txt
done

# Method B: Continuous Ramp
for seed in $SEEDS; do
    echo ""
    echo ">>> Method B, seed=$seed"
    python3 /root/vessel_improved_B.py --seed $seed --epochs 2000 2>&1 | tee /root/results/vessel_improved_B/seed_${seed}/log.txt
done

echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=========================================="

# Collect summary
python3 -c "
import json, os
methods = {'vessel_improved_A': 'Mixed Sampling', 'vessel_improved_B': 'Continuous Ramp'}
seeds = [42, 123, 456, 789, 2026]
summary = {}
for method_dir, method_name in methods.items():
    errs = []
    for seed in seeds:
        path = f'/root/results/{method_dir}/seed_{seed}/results.json'
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            errs.append(r['relative_error_pct'])
            print(f'{method_name} seed={seed}: error={r[\"relative_error_pct\"]:.2f}% tip={r[\"tip_disp_mm\"]:.4f}mm')
    if errs:
        import statistics
        mean_err = statistics.mean(errs)
        success = sum(1 for e in errs if e < 15.0)
        print(f'  → {method_name}: mean={mean_err:.2f}% min={min(errs):.2f}% max={max(errs):.2f}% success={success}/5')
        summary[method_dir] = {'mean': mean_err, 'min': min(errs), 'max': max(errs), 
                                'success': success, 'errors': errs}
with open('/root/results/vessel_improved_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Summary saved to /root/results/vessel_improved_summary.json')
"
