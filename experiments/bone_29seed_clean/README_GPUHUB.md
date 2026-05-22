# Bone 29-Seed Clean Experiment — GPUHub Deployment Guide

## Overview
- **Purpose**: Re-run bone 29-seed experiment with unified F-bar, fresh seed list 200-228
- **Isolation**: Completely independent from `bone_29seed_analysis.json` (old mixed seeds)
- **F-bar**: Forced ON for all seeds (GDR criterion: bone ν=0.30 ≤ 0.35)

## Configuration (from verified bone_seed101-109 logs)
| Parameter | Value |
|-----------|-------|
| Tissue    | bone  |
| E         | 10,000,000 Pa (10 MPa) |
| ν         | 0.30  |
| ρ         | 1900 kg/m³ |
| Epochs    | 2000  |
| LR        | 1e-3 (warmup 50ep, cosine decay to 1e-5) |
| Mesh      | 22×7×7 (1472 nodes, 5390 tets) |
| F-bar     | **FORCED ON** (global mean anti-locking) |
| Seeds     | 200–228 (29 consecutive, fully fresh) |
| Expected tip δ | 0.699 mm (Euler-Bernoulli) |
| Time/seed | ~70–80s on RTX 5090 |
| Total time | ~35 min serial, ~18 min with MAX_PARALLEL=2 |
| GPU VRAM  | ~2 GB |

## Deployment Steps

### 1. Upload files
```bash
scp solid_tissue_train_hires.py run_bone_29seed.sh requirements.txt root@<gpuhub_ip>:/root/
```

### 2. Install dependencies
```bash
pip install torch numpy  # usually pre-installed on GPUHub
```

### 3. Run experiment
```bash
cd /root
bash run_bone_29seed.sh 2>&1 | tee /root/results/bone_29seed_clean/run.log
```

### 4. Monitor progress
```bash
# Check running seeds
ps aux | grep solid_tissue_train_hires_fbar | grep -v grep

# Check specific seed
tail -20 /root/results/bone_29seed_clean/seed_200.log

# Quick error summary
grep "Relative error:" /root/results/bone_29seed_clean/seed_*.log
```

### 5. Retrieve results
```bash
scp root@<gpuhub_ip>:/root/results/bone_29seed_clean/bone_29seed_clean_results.json \
    ~/workspace/DPC-GNN/paper/unified-multitissue/data/
```

## What the script does
1. **Patches** `solid_tissue_train_hires.py` to force `use_fbar=True` (overrides nu>=0.45 threshold)
2. **Runs** 29 seeds (200-228) with `--tissue bone --epochs 2000 --lr 1e-3 --seed N`
3. **Collects** all results, computes mean/std/success rates, saves JSON
4. **Output**: `/root/results/bone_29seed_clean/bone_29seed_clean_results.json`

## Key differences from old bone_29seed_analysis.json
| | Old (mixed) | New (clean) |
|--|--|--|
| Seeds | 42, 100-109, 123, 456, 789, ... | 200-228 |
| F-bar | Mixed (some seeds no fbar) | All unified F-bar ON |
| Isolation | Mixed with 20-seed original run | Completely fresh |
| Output file | bone_29seed_analysis.json | bone_29seed_clean_results.json |

## Troubleshooting
- If patch fails: manually edit `solid_tissue_train_hires_fbar.py`, change `use_fbar = (nu >= 0.45)` to `use_fbar = True`
- If CUDA OOM: reduce `MAX_PARALLEL=1` in script (only ~2GB VRAM needed per run)
- If seeds fail (no convergence): check `J_min` in log; re-run with `--epochs 3000`
