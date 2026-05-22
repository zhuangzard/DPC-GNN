# DPC-GNN Multi-Tissue Source Code

**Last updated**: 2026-03-13  
**GPU**: NVIDIA GeForce RTX 5090 (33.7 GB VRAM)  
**PyTorch**: 2.12.0.dev20260312+cu128

## File Inventory

### Solid Mechanics (FEM-GNN)

| File | Lines | Description |
|------|-------|-------------|
| `solid_tissue_train.py` | 387 | **Main training pipeline** for 5 solid tissues (brain, kidney, myocardium, cartilage, vessel). Neo-Hookean hyperelastic physics with gravity-loaded cantilever beam. AntisymmetricMP GNN (161,731 params). Supports F-bar anti-locking for near-incompressible tissues (ν≥0.45). |

### Fluid Mechanics (SPH-GNN)

| File | Lines | Description |
|------|-------|-------------|
| `sph_domain.py` | 535 | SPH particle domain generator. Creates cylindrical tube geometry (portal vein: D=7mm, L=80mm) with fluid/wall/inlet/outlet particle classification. Builds neighbor graph with Wendland C2 cutoff. |
| `sph_kernels.py` | 333 | Wendland C2 SPH kernel function and gradient. 3D normalized (∫W dV = 0.999998). Supports variable smoothing length h. |
| `sph_gnn_model.py` | 546 | SPH-GNN model extending DPC-GNN's AntisymmetricMP architecture. 9D input (position + velocity + density + pressure + particle_type), 11D edge features (relative pos/vel + kernel values). 304,707 trainable parameters. |
| `sph_physics_loss.py` | 645 | SPH physics loss functions: mass conservation, momentum conservation (Navier-Stokes), divergence-free constraint. Supports Carreau-Yasuda non-Newtonian blood model. |
| `sph_integrator.py` | 606 | Symplectic Euler time integrator for SPH. Density/pressure update, velocity/position integration with boundary enforcement (no-slip walls, inlet/outlet pressure BC). |
| `poiseuille_test.py` | 505 | Poiseuille steady-state flow validation (v1, dp=2mm, 962 particles). Compares GNN velocity profile against analytical parabolic solution. |
| `poiseuille_v2.py` | 371 | Improved Poiseuille validation (v2, dp=1mm, 5383 particles). Publication-quality resolution achieving v_max error 0.1%. |
| `womersley_test.py` | 586 | Womersley pulsatile flow validation. Single-step supervised training with cardiac phase cycling. Validates against Bessel-function analytical solution at Wo=5.40. |

## Architecture Summary

Both solid and fluid models share the same GNN backbone:

```
Input → MLP_encoder → [AntisymmetricMP × 5] → MLP_decoder → Output
                        hidden_dim = 96
```

| Property | Solid (FEM) | Fluid (SPH) |
|----------|-------------|-------------|
| Params | 161,731 | 304,707 |
| Input dim | Material-dependent | 9D |
| Edge features | Relative position + deformation | Relative pos/vel + SPH kernel |
| Output | Displacement/Force | Acceleration |
| Physics | Neo-Hookean hyperelastic | Navier-Stokes (SPH form) |
| Training | Unsupervised (energy minimization) | Supervised (analytical solutions) |

## Total: 9 Python files, 4,514 lines of code
