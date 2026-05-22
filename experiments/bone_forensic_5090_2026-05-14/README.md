# Bone Forensic Retrain — 2026-05-14 (RTX 5090 GPUHub)

**Purpose**: Forensically reproduce the paper's bone 20-seed F-bar fix results
(`paper/unified-multitissue/data/final_results.json` →
`solid_tissues_main.bone_fbar.all_errors`) and the no-F-bar baseline
(`solid_tissues_main.bone`).

**Why this exists**: Prior to this retrain, R4 peer reviewer and F1/F2 forensic
auditors flagged that the paper's bone numbers were not reproducible from any
per-seed log on disk — the original 5090 instance was destroyed, the per-seed
logs were never archived to git, and the aggregate JSON did not specify which
metric (final-epoch error vs best-checkpoint error) the paper used. This
retrain settles all those questions in a single 20-minute run.

## Setup

- **Hardware**: NVIDIA RTX 5090 (GPUHub container `11b84db1d2-7fb33e18`,
  port 37581), Python 3.12.3, PyTorch 2.8.0+cu128
- **Trainer**: `multi_tissue/src/solid_tissue_train_hires.py` (copied verbatim
  from the v22-paper-time git history)
- **Config**: bone (E = 10 MPa, ν = 0.30, ρ = 1900), beam 10 cm × 2 cm × 2 cm,
  mesh 22×7×7, params 161,731, 3000 epochs, LR 1e-3, hidden_dim 64, n_layers 6
- **20 canonical seeds**: 42, 123, 456, 789, 2026, 1337, 999, 31415, 271828,
  1000, 2024, 3407, 7777, 8888, 12345, 54321, 99999, 11111, 66666, 100
  (same as `multi_tissue/src/run_20seed.sh`)

## Two experiment groups

### Group A — baseline (no F-bar)

Used the unmodified `solid_tissue_train_hires.py` where the line

```python
use_fbar = (nu >= 0.45)
```

evaluates to `False` for bone (ν = 0.30). Logs in `logs/group_A_baseline/`.

### Group B — F-bar fix (forensic override)

Patched a copy of `solid_tissue_train_hires.py` with a single sed substitution

```
use_fbar = (nu >= 0.45)   →   use_fbar = True
```

forcing the F-bar branch on for bone. Logs in `logs/group_B_fbar/`.

This is what `EXPERIMENT_REPORT.md:101-105` describes as the "Bone+F-bar"
experiment (CSV row B: `source: GPUHub:/root/results/fbar_all/`). The forensic
retrain reproduces that experiment from scratch.

## Two metrics per log

Each `bone_seed*.log` ends with **both**:

- **Best error**: smallest tip-displacement error vs Euler-Bernoulli analytical
  solution achieved at *any epoch* during training. Reported as `Best error: X% (epoch Y)`.
- **Relative error**: final-epoch tip-displacement error. Reported as `Relative error: Z%`.

**Paper convention**: the published numbers (0.87% mean, 0.07% median,
19/20 < 5%, sorted array ending in 6.50%) match the **Best error** metric in
this retrain. This forensically confirms R2's critical finding that paper
§3.9 / Conclusion's "trained exclusively via physics energy minimisation, no
supervision at any stage" claim must be softened to "no FEM supervision;
best-checkpoint selection uses the Euler-Bernoulli analytical tip displacement
as a reference."

## Results (aggregated to `bone_forensic_aggregated_v2.json`)

| Group | Metric | n | Mean | Std | Median | Min | Max | Success < 5% |
|---|---|---|---|---|---|---|---|---|
| A baseline | Best error  | 20 | **7.362%** | 0.212 | 7.345 | 7.06 | 7.71 | 0/20 |
| A baseline | Final error | 20 | 7.362% | 0.212 | 7.345 | 7.06 | 7.71 | 0/20 |
| B F-bar    | **Best error** | 20 | **0.801%** | 1.508 | **0.215** | 0.01 | 5.94 | **19/20** |
| B F-bar    | Final error | 20 | 10.35% | 13.96 | 3.63 | 0.27 | 50.10 | 12/20 |

**Comparison to paper claims**:

- Group A baseline 7.36% ± 0.21% **matches paper 7.36% ± 0.22% to within 0.01%**.
- Group B F-bar Best error 0.80% ± 1.51% (median 0.21%, 19/20 < 5%, range 0.01–5.94%)
  **matches paper 0.87% ± 1.70% (median 0.07%, 19/20 < 5%, range 0.01–6.50%)**
  within ordinary stochastic variation across two independent training runs.
- **Improvement factor**: forensic mean 7.362 → 0.801 = **9.19×**, paper claims 8.5× ✓
- **Seed 7777 is the worst seed in Group B** (5.94% Best error in forensic; 6.50%
  in paper); Grubbs outlier identity confirmed.
- **The 19-of-20 success rate is exact** (paper 19/20, forensic 19/20).

## Significance

1. **Paper bone numbers are real** — both Group A and Group B are independently
   reproducible to within stochastic seed variation.
2. **R2:[C2] critical finding confirmed** — paper's "no supervision at any stage"
   is misleading: best-checkpoint selection uses Euler-Bernoulli analytical
   reference. Paper text must be softened (see Stage 2 W3 patch list).
3. **The bone Final-relative-error story is grim**: without best-ckpt
   selection, bone bone tip-displacement diverges in 8/20 seeds (Final error
   range 13–50%, mean 10.35%). This explains why best-ckpt is needed and
   makes the Methods section's transparency requirement non-negotiable.
4. **Per-seed mapping recovered** — the placeholder ordering used in
   `figures/revision/regenerate_fig11_bone_20seed_fbar.py` (Stage 2 W2-A,
   commit `d1c0c13`) can now be replaced with the empirical seed-to-error
   mapping in `bone_forensic_aggregated_v2.json:per_seed_group_B`. The
   Grubbs outlier in the forensic run is still seed 7777 (5.94%), confirming
   the paper's outlier identification.

## File map

```
logs/
  group_A_baseline/         # 20 bone_seedXXXX.log files (no F-bar)
  group_B_fbar/             # 20 bone_seedXXXX.log files (F-bar forced)
aggregated/                 # original on-5090 aggregation (raw, has f-string bug)
  bone_forensic_2026-05-14.json
bone_forensic_aggregated_v2.json   # CORRECTED aggregation (this is the
                                   # authoritative file; cite this in N1/N2)
aggregate_v2.py             # generator for the above
README.md                   # this file
```

## Reproduction

```bash
# On any RTX 5090 with PyTorch 2.8+/CUDA 12.8:
SEEDS="42 123 456 789 2026 1337 999 31415 271828 1000 2024 3407 7777 8888 12345 54321 99999 11111 66666 100"

# Group A baseline (no F-bar):
for s in $SEEDS; do
  python3 solid_tissue_train_hires.py --tissue bone --seed $s --epochs 3000 > bone_seed${s}.log 2>&1
done

# Group B F-bar fix (forced):
sed -i 's|use_fbar = (nu >= 0.45)|use_fbar = True|' solid_tissue_train_hires.py
# ... same loop, different output dir
```

8-way parallel on a single RTX 5090: full 40-job sweep completes in ~20 min.

## Provenance

- Runner script: `GPUHub:/root/dpcgnn_forensic/run_bone_forensic.sh`
- Container: `gpuhub-container-11b84db1d2-7fb33e18`
- Forensic dir: `GPUHub:/root/dpcgnn_forensic/` (preserved at this commit)
- Local copy: `experiments/bone_forensic_5090_2026-05-14/`
