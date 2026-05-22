# DPC-GNN

A unified physics-constrained Graph Neural Network that simulates biological tissues
spanning four orders of magnitude in stiffness, from brain parenchyma (~1 kPa) to
trabecular bone (~10 MPa), plus a companion SPH-GNN for non-Newtonian blood flow.

The network encodes continuum-mechanical invariants as architectural and constitutive
properties rather than additive loss terms:

- **Antisymmetric Message Passing (AMP)** enforces Newton's third law as an
  algebraic identity at every forward pass: a shared edge network is evaluated
  forward and reverse and combined by subtraction, so `m_ij = -m_ji` by construction.
- **Modified Neo-Hookean energy** with a compact-support barrier at `J → 0+`
  prevents Jacobian inversion.
- **Global Dilatation Regularization (GDR)** replaces element-local F-bar with a
  single mesh-level mean-dilatation projection, removing volumetric locking at
  near-incompressibility.
- **Stiffness-aware decoder rescaling** uses the analytic Euler–Bernoulli tip
  displacement to keep training signals at unit scale across four stiffness decades.
- **Velocity Verlet** integration preserves the symplectic structure of conservative
  dynamics.

Training is **unsupervised**: the model minimises Neo-Hookean strain energy plus
gravitational potential directly, without paired FEM displacement labels.

## Repository layout

```
multi_tissue/src/         FEM-GNN training (solid tissues) + SPH-GNN (blood)
                          + reference FEM solvers (scipy.sparse)
phase_d/                  small reusable modules (material_features.py,
                          material_scaling.py) with their own tests
experiments/              cached experimental results (JSON + log) for the
                          published 20-seed sweeps; backs the headline numbers
scripts/                  utility scripts
```

## Quick start

```bash
# Single tissue, single seed
python3 multi_tissue/src/solid_tissue_train.py --tissue bone --epochs 1000 --seed 42

# High-resolution variant used in the published 20-seed sweeps
python3 multi_tissue/src/solid_tissue_train_hires.py --tissue brain --seed 42 --epochs 3000

# Full 6-tissue × 20-seed sweep (uses /root paths; edit before running locally)
bash multi_tissue/src/run_20seed.sh

# Steady-state Poiseuille blood flow
python3 multi_tissue/src/poiseuille_v2.py --epochs 500 --dp 0.001 --delta-p 0.5

# Pulsatile Womersley (smoke test)
python3 multi_tissue/src/womersley_test.py --epochs 50 --quick
```

Tissue identifiers recognised by the trainer:
`brain`, `kidney`, `myocardium`, `cartilage`, `vessel`, `bone`.

## Reproducing the published numbers

Twenty seeds are used for every solid-tissue table entry:
`42, 123, 456, 789, 2026, 1337, 999, 31415, 271828, 1000,
2024, 3407, 7777, 8888, 12345, 54321, 99999, 11111, 66666, 100`.

Aggregated experimental results (per-tissue, per-seed) live under
`experiments/` as JSON.

## Architecture invariant

Any new message-passing layer in this codebase must preserve
`m_ij = -m_ji`. Breaking that identity invalidates the physics claim of the
underlying method.

## Runtime

Training was developed for an NVIDIA RTX 5090 (PyTorch 2.12 / CUDA 12.8).
The runner scripts hardcode `/root/...` paths used on the training box; edit
those before running on a different machine.

## License

Apache License, Version 2.0 — see [`LICENSE`](LICENSE).
