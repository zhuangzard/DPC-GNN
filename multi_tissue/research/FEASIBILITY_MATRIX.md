# Feasibility Matrix: Blood Flow Simulation Approaches
## DPC-GNN Extension — Multi-Expert Assessment

**Date:** 2026-03-11  
**Evaluators:** 6-Expert Council (CFD, GNN-Fluids, Hemodynamics, DPC-GNN Arch, FSI, Paper Strategy)

---

## Evaluation Dimensions

| Dimension | Description | Score Scale |
|-----------|-------------|-------------|
| **D1: DPC-GNN Compat** | How well approach fits existing AntisymmetricMP architecture | 1-5 |
| **D2: Physics Fidelity** | Accuracy of physical representation for blood flow | 1-5 |
| **D3: Impl Complexity** | Implementation effort (5=trivial, 1=months of work) | 1-5 |
| **D4: Clinical Value** | Relevance to surgical planning / MedIA reviewers | 1-5 |
| **D5: MedIA Novelty** | Novelty score for MedIA revision claim | 1-5 |

**Weighted Total** = 0.25*D1 + 0.20*D2 + 0.20*D3 + 0.20*D4 + 0.15*D5

---

## Approach 1: SPH-GNN (Smoothed Particle Hydrodynamics + GNN)

### Description
Replace tetrahedral mesh nodes with SPH fluid particles. Use AntisymmetricMP to compute inter-particle SPH forces. Physics loss = SPH conservation law residuals.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **5/5** | SPH inter-particle forces are F_ij = -F_ji by kernel symmetry. EXACT match to AntisymmetricMP. Graph edges = SPH neighbor pairs. Node features naturally encode [x, v, rho, p, mu]. |
| D2: Physics Fidelity | **4/5** | Weakly compressible SPH well-validated for hemodynamics. Carreau-Yasuda viscosity included. Main limitation: SPH has ~O(h) order accuracy (lower than FEM). |
| D3: Impl Complexity | **3/5** | Need: SPH kernel functions, radius graph, EOS pressure, dynamic graph rebuild. ~3-4 weeks new code. Significant but tractable. |
| D4: Clinical Value | **4/5** | WSS computable from SPH velocity gradient. TAWSS, OSI, RRT all derivable. Pulsatile flow captures cardiac waveform. |
| D5: MedIA Novelty | **5/5** | First physics-driven (zero training data) GNN-SPH for blood flow. GNS/MeshGraphNets are data-driven. PINN methods lack antisymmetric structure. Strongly novel. |

**Weighted Score: 0.25*5 + 0.20*4 + 0.20*3 + 0.20*4 + 0.15*5 = 1.25+0.80+0.60+0.80+0.75 = 4.20/5**

### Key Equations
```
SPH pressure force on particle i:
F_i^p = -Σj mi*mj * (pi/rhoi^2 + pj/rhoj^2) * gradW(ri-rj, h)

SPH viscous force:
F_i^v = Σj mi*mj * (muij * (vi-vj) / (rhoi*rhoj)) * 2*rij*gradW/|rij|^2

Physics loss:
L = ||Drhoi/Dt + rhoi*div(vi)||^2 + ||Dvi/Dt - F_i^p/mi - F_i^v/mi - g||^2
```

### Risks
- SPH tensile instability at low density regions (vessel inlet/outlet)
- Dynamic graph rebuild at each step increases computational cost
- Boundary treatment at curved vessel walls requires careful SPH kernel truncation

---

## Approach 2: Eulerian GNN (Fixed Grid + GNN)

### Description
Use a fixed Cartesian grid as GNN graph. Fluid flows through fixed nodes (Eulerian viewpoint). Physics loss = finite-difference Navier-Stokes residual.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **1/5** | Fundamentally incompatible. Fixed nodes cannot track material deformation. AntisymmetricMP enforces Newton's 3rd law for MATERIAL forces — irrelevant for Eulerian node fluxes. GNN edge structure would be static grid, not physics-based. |
| D2: Physics Fidelity | **4/5** | Eulerian CFD is standard high-fidelity approach. FVM schemes (OpenFOAM) are Eulerian. Excellent for incompressible flow. |
| D3: Impl Complexity | **2/5** | Requires complete reimplementation of CFD discretization. Pressure solver (Poisson), convection scheme (upwinding), time integration. This is essentially writing a new CFD solver. |
| D4: Clinical Value | **4/5** | Eulerian CFD standard in clinical hemodynamics literature. WSS computation straightforward from wall gradients. |
| D5: MedIA Novelty | **1/5** | "GNN on fixed grid for CFD" already exists (MeshGraphNets, GNS). No novelty over existing approaches. Does not leverage DPC-GNN's unique physics-driven strength. |

**Weighted Score: 0.25*1 + 0.20*4 + 0.20*2 + 0.20*4 + 0.15*1 = 0.25+0.80+0.40+0.80+0.15 = 2.40/5**

### Why Rejected
Incompatible with DPC-GNN architecture, no novelty, requires full CFD reimplementation.

---

## Approach 3: PINN (Physics-Informed Neural Network, Continuous)

### Description
Replace GNN with continuous MLP. Input: (x, y, z, t). Output: (vx, vy, vz, p). Physics loss = Navier-Stokes PDEs evaluated at collocation points.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **1/5** | Not a GNN at all — abandons the DPC-GNN architecture entirely. No graph structure, no message passing, no antisymmetry. Would require complete rewrite. |
| D2: Physics Fidelity | **4/5** | PINN can exactly enforce NS PDEs (Kissas 2020 validated for blood flow). Handles complex geometries via collocation. Main issue: spectral bias — misses high-frequency flow features. |
| D3: Impl Complexity | **3/5** | PINN itself simple (MLP + NS residual). But: geometry parameterization, collocation point sampling, and convergence issues make practical application difficult. |
| D4: Clinical Value | **3/5** | PINN validated for carotid, aorta (Kissas 2020, Arzani 2021). WSS computation requires gradient of network output — adds complexity. |
| D5: MedIA Novelty | **2/5** | PINN for blood flow well-established (2020-2024 literature). No novelty unless combined with DPC-GNN's unique physics-driven approach. |

**Weighted Score: 0.25*1 + 0.20*4 + 0.20*3 + 0.20*3 + 0.15*2 = 0.25+0.80+0.60+0.60+0.30 = 2.55/5**

### Why Rejected
Abandons DPC-GNN architecture. PINN for blood flow not novel enough for MedIA.

---

## Approach 4: ALE-GNN (Arbitrary Lagrangian-Eulerian + GNN)

### Description
Extend DPC-GNN to ALE framework. Solid nodes (vessel wall) move with material. Fluid nodes move with mesh but not material. Coupling at fluid-solid interface.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **3/5** | ALE is the natural FSI extension. Solid part is exactly DPC-GNN. Fluid part requires additional mesh velocity term (v_fluid - v_mesh) in convection. Not trivially compatible with current AntisymmetricMP (mesh velocity breaks antisymmetry). |
| D2: Physics Fidelity | **5/5** | ALE is the gold standard for FSI (SimVascular uses ALE). Exactly handles moving boundaries. Best physical representation of vessel wall-blood interaction. |
| D3: Impl Complexity | **1/5** | Most complex approach. Requires: ALE mesh update equations, fluid-solid coupling at moving interface, mesh quality control, potential remeshing. Multi-months implementation. |
| D4: Clinical Value | **5/5** | Captures compliance effects: vessel wall distension during systole changes blood flow patterns. Most physiologically accurate. Essential for compliance-related hemodynamics. |
| D5: MedIA Novelty | **4/5** | GNN-ALE-FSI is novel. Existing ALE methods are not GNN-based. However: complexity may preclude implementation for MedIA revision timeline. |

**Weighted Score: 0.25*3 + 0.20*5 + 0.20*1 + 0.20*5 + 0.15*4 = 0.75+1.00+0.20+1.00+0.60 = 3.55/5**

### Why Not for Revision
Too complex for MedIA revision timeline. Best suited for extended journal version.

---

## Approach 5: LBM-GNN (Lattice Boltzmann + GNN)

### Description
Use Lattice Boltzmann Method with GNN. LBM evolves probability distribution functions f(x,v,t) on a regular lattice. GNN predicts LBM collision operator.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **2/5** | LBM requires regular D3Q19 lattice — incompatible with DPC-GNN's unstructured graph. The 19 velocity directions per node have no natural analogue in message passing. Antisymmetry of DPC-GNN not relevant for LBM collision operator. |
| D2: Physics Fidelity | **4/5** | LBM well-validated for blood flow (mesoscopic, captures RBC effects better than continuum). Naturally handles incompressibility. Efficient parallelization. |
| D3: Impl Complexity | **2/5** | LBM itself straightforward, but adapting GNN to regular grid removes all advantages of graph-based approach. Essentially implementing GPU-accelerated LBM, not GNN. |
| D4: Clinical Value | **4/5** | LBM competitive with FVM-CFD for hemodynamics. Can model RBC-scale phenomena at higher fidelity than continuum models. |
| D5: MedIA Novelty | **2/5** | LBM-GNN exists in literature for some problems. No clear novelty given DPC-GNN's specific contributions (antisymmetric physics-driven GNN). |

**Weighted Score: 0.25*2 + 0.20*4 + 0.20*2 + 0.20*4 + 0.15*2 = 0.50+0.80+0.40+0.80+0.30 = 2.80/5**

### Why Rejected
Incompatible with DPC-GNN graph structure. Sacrifices main architectural innovation.

---

## Approach 6: Unified SPH-FEM GNN with FSI (Full System)

### Description
Combine SPH-GNN (Approach 1) for blood with existing DPC-GNN for vessel wall. Two subgraphs connected by FSI interface edges. Single training objective combining solid + fluid + coupling physics losses.

### Evaluation

| Dimension | Score | Justification |
|-----------|-------|--------------|
| D1: DPC-GNN Compat | **5/5** | Perfect extension of DPC-GNN. Solid subgraph = existing code. Fluid subgraph = SPH-GNN (Approach 1). Interface edges extend AntisymmetricMP to cross-physics coupling. Same encoder-process-decode structure throughout. |
| D2: Physics Fidelity | **5/5** | Captures both structural mechanics (Neo-Hookean vessel wall) and hemodynamics (SPH blood flow) with proper FSI coupling. Compliance effects included. Best overall physical representation. |
| D3: Impl Complexity | **2/5** | Most complex: requires SPH-GNN (Approach 1) PLUS FSI coupling module PLUS multi-physics loss balancing. Estimated 8-12 weeks for production-quality implementation. |
| D4: Clinical Value | **5/5** | Highest clinical value: full FSI enables prediction of compliance-related hemodynamics, vessel wall stress under pulsatile loading, interaction effects at bifurcations. |
| D5: MedIA Novelty | **5/5** | Highest novelty: first unified physics-driven GNN for simultaneous structural + hemodynamic simulation. No existing work combines antisymmetric physics-driven GNN with SPH-FSI. |

**Weighted Score: 0.25*5 + 0.20*5 + 0.20*2 + 0.20*5 + 0.15*5 = 1.25+1.00+0.40+1.00+0.75 = 4.40/5**

### Why Not for Revision (Despite Highest Score)
Implementation complexity exceeds MedIA revision timeline. **This is the target for the extended paper / next journal submission.**

---

## Summary Feasibility Matrix

| Approach | D1: Compat | D2: Physics | D3: Impl | D4: Clinical | D5: Novelty | **Weighted** | Timeline | Verdict |
|----------|-----------|-------------|----------|--------------|-------------|-------------|----------|---------|
| 1. SPH-GNN | 5 | 4 | 3 | 4 | 5 | **4.20** | 4-5 wks | **REVISION** |
| 2. Eulerian GNN | 1 | 4 | 2 | 4 | 1 | **2.40** | 2+ mo | Rejected |
| 3. PINN | 1 | 4 | 3 | 3 | 2 | **2.55** | 2-3 mo | Rejected |
| 4. ALE-GNN | 3 | 5 | 1 | 5 | 4 | **3.55** | 3+ mo | Extended paper |
| 5. LBM-GNN | 2 | 4 | 2 | 4 | 2 | **2.80** | 2+ mo | Rejected |
| 6. SPH-FEM FSI | 5 | 5 | 2 | 5 | 5 | **4.40** | 8-12 wks | Next journal |

---

## Decision Rationale

**For MedIA revision (4-5 week deadline):** → **Approach 1: SPH-GNN**

The highest feasibility/novelty combination given the timeline. SPH architecture perfectly matches DPC-GNN's AntisymmetricMP (both require F_ij = -F_ji). Implementation is challenging but tractable.

**For extended journal / next submission:** → **Approach 6: SPH-FEM FSI**

The architecturally most complete and clinically most valuable extension. Builds directly on Approach 1 by adding the FSI coupling layer.

---

## Implementation Priority Map

```
Week 1-2:  SPH kernel + radius graph + Poiseuille validation
           [Approach 1 foundation]

Week 3-4:  SPH physics loss + Womersley pulsatile demo
           [Approach 1 complete]

Week 5-6:  Portal vein application + OpenFOAM comparison
           [Approach 1 clinical validation]

Month 2-3: FSI coupling + vessel wall integration
           [Approach 6 prototype]

Month 4+:  Full FSI clinical validation + paper writing
           [Approach 6 complete]
```
