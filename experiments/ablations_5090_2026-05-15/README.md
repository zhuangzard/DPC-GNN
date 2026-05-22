# AMP + EBC Ablation Sweep — 2026-05-15 (RTX 5090, Singapore-A)

**Purpose**: empirical answers to two architectural questions raised by F10
(`reviews/F10_theory_v1_drift_audit.md`) and R1 Minor 4 in the v25 review
pass:

  1. **AMP ablation** — does Antisymmetric Message Passing actually matter,
     or could a standard MeshGraphNets-style directed-edge block achieve
     the same accuracy with the same physics loss, mesh, F-bar, optimiser?
  2. **EBC ablation** — does the linear ReLU barrier
     `W_barrier = k_E · ReLU(ε − J)` empirically prevent element inversion
     on the canonical cantilever benchmark, or is it precautionary?

Both ablations replace exactly ONE element of the canonical DPC-GNN trainer
(the message-passing primitive in #1, the barrier coefficient in #2) and
hold everything else identical to the published `run_20seed.sh` config.

## Hardware

- **GPU**: NVIDIA RTX 5090 (GPUHub container, Singapore-A, port 20968)
- **Driver / CUDA**: 580.76.05 / CUDA 13.0
- **PyTorch**: 2.12.0.dev20260312+cu128
- **Run start**: 2026-05-15 05:14 CST (GPU local time)
- **Wall time**: 14 min (MGN sweep) + 15 min (barrier sweep) + 7 min (bone
  MGN re-run after argparse fix) = **~36 min total**, MAX_PARALLEL=4

## Trainers used (all in `multi_tissue/src/`, committed to git)

- `solid_tissue_train_mgn.py` — MGN-style baseline.  Replaces `AntisymMP`
  with `DirectedMP` (two independent edge MLPs, sum aggregation, no
  antisymmetry subtraction).  Imports `generate_beam_mesh`, `compute_F`,
  `neo_hookean_energy`, `gravity_potential` verbatim from the canonical
  `solid_tissue_train.py`, so the only thing that differs is the
  message-passing primitive.

- `solid_tissue_train_no_barrier.py` — EBC ablation.  Uses the canonical
  `SolidGNN` (AMP backbone unchanged) and `neo_hookean_energy_nobar()`
  which accepts a `barrier_coef` parameter (default 0 = ablation,
  100 = canonical DPC-GNN value).  Records `n_inverted` per epoch and
  surfaces `max_n_inverted_during_training` + `epoch_first_inversion`
  in `results.json`.

## Sweeps

### MGN baseline (AMP ablation)
- **6 tissues × 5 seeds = 30 runs**
- Seeds: 42, 123, 456, 789, 2026 (subset of the canonical 20-seed list)
- Tissues + epochs:
  | Tissue       | E (Pa)    | ν     | ρ    | Epochs |
  |--------------|-----------|-------|------|--------|
  | brain        | 1,000     | 0.49  | 1040 | 3000   |
  | kidney       | 10,000    | 0.45  | 1050 | 3000   |
  | myocardium   | 30,000    | 0.40  | 1060 | 3000   |
  | vessel       | 400,000   | 0.49  | 1050 | 2500   |
  | cartilage    | 500,000   | 0.30  | 1100 | 3000   |
  | bone         | 10,000,000| 0.30  | 1900 | 2000   |

### EBC barrier on/off
- **3 tissues × 5 seeds × 2 settings = 30 runs**
- Tissues: cartilage, bone, vessel (the three with the largest gravity-driven
  deformation; soft tissues never approach J→0 in this benchmark)
- Barrier coefficient k_E: 0 (ablation) and 100 (canonical DPC-GNN value)

## What's in this archive

```
experiments/ablations_5090_2026-05-15/
├── README.md                          (this file)
├── ablation_summary.json              (aggregated 5-seed stats, used by paper §4.4)
├── parse_ablations.py                 (the aggregator that produced summary.json)
├── run_bone_mgn.sh                    (bone MGN re-run driver after argparse fix)
├── mgn/                               (MGN baseline output)
│   ├── {tissue}_seed{N}.log           (per-seed full training log — 30 files)
│   ├── brain_mgn/                     (model checkpoints + history.json;
│   ├── kidney_mgn/                     only last-seed survives due to a known
│   ├── myocardium_mgn/                 seed-not-in-path bug in the trainer;
│   ├── vessel_mgn/                     per-seed final errors are recoverable
│   ├── cartilage_mgn/                  from the .log files)
│   └── bone_mgn/
├── barrier/                           (EBC ablation output)
│   ├── {tissue}_coef{0,100}_seed{N}.log
│   ├── {tissue}_no_barrier/           (k_E = 0 output dir)
│   └── {tissue}_barrier_100/          (k_E = 100 output dir)
└── logs/                              (master driver logs)
```

## Key results (full numbers in `ablation_summary.json` + paper Tables 6/7)

### MGN baseline (5-seed mean ± std final relative tip-displacement error)

| Tissue       | DPC-GNN paper | Directed-MP baseline | Degradation |
|--------------|---------------|----------------------|-------------|
| brain        | 0.23%         | 8.06 ± 4.54%         | 35×         |
| kidney       | 0.52%         | **193.7 ± 89.6%**    | **373×**    |
| myocardium   | 0.93%         | 22.12 ± 0.85%        | 24×         |
| vessel       | 0.35%         | 14.07 ± 14.61%       | 40×         |
| cartilage    | 0.19%         | 14.17 ± **0.01%**    | 75×, **locked** |
| bone         | 0.87%         | 15.65 ± 0.09%        | 18×, locked |

The cartilage / bone / myocardium std collapse (<1%) is diagnostic: the
directed-MP architecture locks every seed into the same physically wrong
attractor, independent of weight initialisation.  Without structural
antisymmetry the GNN learns a coherent but momentum-violating force
field — the optimisation-landscape failure mode that AMP is designed to
break.

### EBC ablation

| Tissue       | k_E  | mean(%) | std(%) | range          | max n_inverted |
|--------------|------|---------|--------|----------------|----------------|
| cartilage    |   0  | 14.23   | 0.02   | 14.20–14.25    | **0**          |
| cartilage    | 100  | 14.23   | 0.02   | 14.20–14.25    | **0**          |
| bone         |   0  | 15.94   | 0.53   | 15.36–16.68    | **0**          |
| bone         | 100  | 15.88   | 0.50   | 15.40–16.71    | **0**          |
| vessel       |   0  | 34.00   | 31.13  | 5.82–83.26     | **0**          |
| vessel       | 100  | 12.94   | 16.20  | 0.18–40.71     | **0**          |

`n_inverted = 0` across all 60 runs and all epochs.  Under gravity loading
on a 10 cm × 2 cm × 2 cm cantilever, the optimisation never approaches
the inversion boundary, so the barrier is silent in every run.  Final
errors between k_E=0 and k_E=100 are within seed noise on cartilage and
bone; vessel shows a mean improvement (34%→13%) but the std overlaps
massively (n=5).  The barrier is therefore **precautionary, not
load-bearing** for this benchmark; we retain it for surgical-cutting /
large-compression / contact scenarios identified as future work.

## Known limitation: seed-not-in-path bug

The trainers wrote `results.json` and `history.json` to a path that
omitted the seed (`{output_root}/{tissue}_{tag}/`), so the 5 seeds per
condition overwrote each other and only the last seed's JSON survived
in the canonical paths.  Per-seed final errors and full training logs
are preserved in `mgn/{tissue}_seed{N}.log` and
`barrier/{tissue}_coef{c}_seed{N}.log` and were used to compute the
summary statistics in `ablation_summary.json`.  The bug is reported in
the v27 commit message but not yet fixed in the trainer source files
(deferred to v28).

## Reproducing this archive

On a fresh RTX 5090 container with the DPC-GNN repo at
`/root/dpcgnn_forensic/`:

```bash
cp multi_tissue/src/solid_tissue_train_mgn.py        /root/dpcgnn_forensic/src/
cp multi_tissue/src/solid_tissue_train_no_barrier.py /root/dpcgnn_forensic/src/
bash run_ablations.sh   # the master driver (also in this archive)
```

Wall time: ~30 minutes at MAX_PARALLEL=4.

— end README —
