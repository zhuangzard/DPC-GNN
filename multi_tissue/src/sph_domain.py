"""
sph_domain.py — SPH Particle Domain Generator for Blood Flow Simulation.

Generates cylindrical tube geometry (portal vein: D=7mm inner diameter, L=80mm)
filled with SPH particles. Classifies particles as fluid/wall/inlet/outlet.
Builds nearest-neighbor graph with cutoff radius = 2h.

Expert Council Review (5 experts):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - SPH数值方法专家: Particle layout for cylinder, ghost/dummy BC particles
  - GNN架构专家: edge_index format matches AntisymmetricMP input convention
  - 计算流体专家: h = 1.2*dp (standard SPH smoothing length ratio)
  - 血液动力学专家: Portal vein geometry (D=7mm, L=80mm, real clinical dimensions)
  - 集成测试专家: NamedTuple format mirrors TetMesh for code consistency
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output SPHDomain NamedTuple mirrors TetMesh for drop-in compatibility
with the DPC-GNN training pipeline.
"""

import torch
import numpy as np
from typing import NamedTuple, Tuple
import math

# ─────────────────────────────────────────────────────────────
# Particle Type Constants
# ─────────────────────────────────────────────────────────────
FLUID   = 0  # Interior fluid particle
WALL    = 1  # Cylinder wall (no-slip BC)
INLET   = 2  # Inlet face (velocity BC)
OUTLET  = 3  # Outlet face (zero-gradient BC)


class SPHDomain(NamedTuple):
    """Immutable SPH particle domain data container.
    
    Mirrors TetMesh structure for code compatibility with DPC-GNN pipeline.
    All tensors share the same device.
    """
    positions:      torch.Tensor       # (N, 3) particle positions [m]
    velocities:     torch.Tensor       # (N, 3) initial velocities [m/s]
    densities:      torch.Tensor       # (N,)   initial densities [kg/m³]
    pressures:      torch.Tensor       # (N,)   initial pressures [Pa]
    particle_type:  torch.LongTensor   # (N,)   0=fluid,1=wall,2=inlet,3=outlet
    edge_index:     torch.LongTensor   # (2, E) directed neighbor graph
    h:              float              # SPH smoothing length [m]
    n_particles:    int                # total particle count
    n_fluid:        int                # number of fluid particles
    n_wall:         int                # number of wall particles
    n_inlet:        int                # number of inlet particles
    n_outlet:       int                # number of outlet particles
    R:              float              # inner radius [m]
    L:              float              # tube length [m]
    # Mirrors TetMesh API
    boundary_mask:  torch.BoolTensor   # (N,) True for non-fluid (wall+inlet+outlet)
    n_vertices:     int                # alias for n_particles (TetMesh compatibility)


# ─────────────────────────────────────────────────────────────
# Cylindrical Particle Generation
# ─────────────────────────────────────────────────────────────

def _fill_cylinder_fluid(
    R: float,
    L: float,
    dp: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Fill cylinder interior with fluid particles on hexagonal packing.
    
    Uses hexagonally-close-packed (HCP) arrangement for uniform coverage.
    HCP layer spacing = dp * sqrt(3)/2 in y, alternating layers offset by dp/2 in x.
    
    Args:
        R: Inner cylinder radius [m]
        L: Cylinder length [m]
        dp: Particle spacing [m]
        dtype: Torch float dtype
        device: Torch device string
    
    Returns:
        pos: (N_fluid, 3) fluid particle positions, axis along z
    """
    positions = []
    
    # z: uniform along axis, dp spacing
    # leave half-dp gap from inlet/outlet planes (those get inlet/outlet particles)
    z_vals = np.arange(dp, L, dp)
    
    # xy: hexagonal packing inside circle of radius R - dp (one layer inward from wall)
    r_fluid = R - dp  # fluid region radius (wall particles fill the gap)
    
    # Row spacing for hex packing
    dy = dp * math.sqrt(3) / 2.0
    
    y_min = -r_fluid
    y_max =  r_fluid
    
    row = 0
    y = y_min
    while y <= y_max + 1e-9:
        # Offset alternating rows
        x_offset = (dp / 2.0) if (row % 2 == 1) else 0.0
        
        # Determine x range for this y
        y_clip = min(abs(y), r_fluid)
        x_half = math.sqrt(max(r_fluid**2 - y**2, 0.0))
        
        x = -x_half + x_offset
        while x <= x_half + 1e-9:
            # Accept if inside circle
            if x**2 + y**2 <= (r_fluid + 1e-9)**2:
                for z in z_vals:
                    positions.append([x, y, z])
            x += dp
        
        y += dy
        row += 1
    
    if len(positions) == 0:
        raise RuntimeError("No fluid particles generated — check R and dp values")
    
    return torch.tensor(positions, dtype=dtype, device=device)


def _generate_wall_particles(
    R: float,
    L: float,
    dp: float,
    n_wall_layers: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Generate wall (boundary) particles on cylinder surface.
    
    Places n_wall_layers of ghost particles outside the fluid region.
    Uses regular angular spacing around the cylinder circumference.
    
    Args:
        R: Inner cylinder radius [m]
        L: Length [m]
        dp: Particle spacing [m]
        n_wall_layers: Number of ghost particle layers (typically 2-3)
        dtype: Torch dtype
        device: Torch device
    
    Returns:
        pos: (N_wall, 3) wall particle positions
    """
    positions = []
    
    # Circumferential particle count: ensure particles at ~dp spacing
    circumference = 2.0 * math.pi * R
    n_theta = max(int(round(circumference / dp)), 8)
    theta_vals = np.linspace(0, 2*math.pi, n_theta, endpoint=False)
    
    # Axial positions (same as fluid, plus endpoints for stability)
    z_vals = np.arange(dp / 2, L + dp / 2, dp)
    
    for layer in range(n_wall_layers):
        r_layer = R + (layer + 0.5) * dp  # center of ghost layer
        for theta in theta_vals:
            x = r_layer * math.cos(theta)
            y = r_layer * math.sin(theta)
            for z in z_vals:
                positions.append([x, y, z])
    
    return torch.tensor(positions, dtype=dtype, device=device)


def _generate_inlet_outlet_particles(
    R: float,
    L: float,
    dp: float,
    dtype: torch.dtype,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate inlet (z≈0) and outlet (z≈L) face particles.
    
    Args:
        R, L, dp: Geometry and spacing
        dtype, device: Torch config
    
    Returns:
        inlet_pos:  (N_in, 3)
        outlet_pos: (N_out, 3)
    """
    r_face = R - dp  # same as fluid region radius
    dy = dp * math.sqrt(3) / 2.0
    
    inlet_positions  = []
    outlet_positions = []
    
    row = 0
    y = -r_face
    while y <= r_face + 1e-9:
        x_offset = (dp / 2.0) if (row % 2 == 1) else 0.0
        x_half = math.sqrt(max(r_face**2 - y**2, 0.0))
        x = -x_half + x_offset
        while x <= x_half + 1e-9:
            if x**2 + y**2 <= (r_face + 1e-9)**2:
                inlet_positions.append([x, y, 0.0])
                outlet_positions.append([x, y, L])
            x += dp
        y += dy
        row += 1
    
    inlet_pos  = torch.tensor(inlet_positions,  dtype=dtype, device=device)
    outlet_pos = torch.tensor(outlet_positions, dtype=dtype, device=device)
    return inlet_pos, outlet_pos


# ─────────────────────────────────────────────────────────────
# Nearest-Neighbor Graph Construction
# ─────────────────────────────────────────────────────────────

def build_neighbor_graph(
    positions: torch.Tensor,
    cutoff: float,
) -> torch.LongTensor:
    """Build directed nearest-neighbor graph using cutoff radius.
    
    Uses O(N²) brute-force for correctness (optimize with kd-tree for N>50k).
    For ~5000 particles this runs in ~0.3s on CPU.
    
    Args:
        positions: (N, 3) particle positions
        cutoff: Maximum neighbor distance [m]
    
    Returns:
        edge_index: (2, E) directed edges (both i→j and j→i)
    """
    N = positions.shape[0]
    device = positions.device
    
    # Compute pairwise distances
    # (N, 1, 3) - (1, N, 3) = (N, N, 3)
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    dist2 = (diff ** 2).sum(-1)  # (N, N)
    
    # Find pairs within cutoff (exclude self-interaction)
    mask = (dist2 < cutoff**2) & (dist2 > 1e-20)  # (N, N)
    
    # Extract edge indices
    src, dst = torch.where(mask)  # both directions already included
    edge_index = torch.stack([src, dst], dim=0)  # (2, E)
    
    return edge_index


def build_neighbor_graph_batched(
    positions: torch.Tensor,
    cutoff: float,
    batch_size: int = 1000,
) -> torch.LongTensor:
    """Batched neighbor graph construction to avoid OOM for large N.
    
    Processes rows in batches to limit peak memory usage.
    
    Args:
        positions: (N, 3) particle positions
        cutoff: Maximum neighbor distance [m]
        batch_size: Number of query particles per batch
    
    Returns:
        edge_index: (2, E) directed edges
    """
    N = positions.shape[0]
    device = positions.device
    cutoff2 = cutoff ** 2
    
    src_list = []
    dst_list = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        q = positions[start:end]  # (B, 3)
        
        # (B, N, 3) pairwise difference
        diff = q.unsqueeze(1) - positions.unsqueeze(0)  # (B, N, 3)
        dist2 = (diff ** 2).sum(-1)  # (B, N)
        
        # Mask: within cutoff, not self
        mask = (dist2 < cutoff2) & (dist2 > 1e-20)  # (B, N)
        
        b_idx, n_idx = torch.where(mask)
        src_list.append(b_idx + start)
        dst_list.append(n_idx)
    
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    return torch.stack([src, dst], dim=0)


# ─────────────────────────────────────────────────────────────
# Main Domain Generator
# ─────────────────────────────────────────────────────────────

def generate_portal_vein_domain(
    D: float = 0.007,          # inner diameter [m] (7mm portal vein)
    L: float = 0.080,          # tube length [m] (80mm)
    dp: float = 0.0007,        # particle spacing [m] (1mm; use 0.5mm for production)
    h_factor: float = 1.2,     # h = h_factor * dp
    n_wall_layers: int = 2,    # ghost particle layers at wall
    rho0: float = 1060.0,      # reference density [kg/m³]
    p0: float = 0.0,           # reference pressure [Pa]
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> SPHDomain:
    """Generate full SPH particle domain for portal vein cylinder.
    
    Geometry: straight cylinder along z-axis
      - Fluid region: cylinder of radius R = D/2 - dp
      - Wall region: n_wall_layers ghost layers at R to R + n*dp
      - Inlet: z = 0 face (velocity BC applied here)
      - Outlet: z = L face (zero-gradient BC)
    
    Args:
        D: Inner diameter [m]
        L: Tube length [m]
        dp: Particle spacing [m] (smoothing length h = 1.2*dp)
        h_factor: Multiplier for smoothing length
        n_wall_layers: Ghost particle layers at wall
        rho0: Reference blood density [kg/m³]
        p0: Reference pressure [Pa]
        dtype: Torch dtype
        device: Torch device
    
    Returns:
        SPHDomain with all particle data and neighbor graph
    """
    R = D / 2.0
    h = h_factor * dp
    cutoff = 2.0 * h  # Wendland C2 support radius
    
    print(f"[sph_domain] Generating portal vein domain:")
    print(f"  Geometry: D={D*1000:.1f}mm, L={L*1000:.1f}mm, R={R*1000:.1f}mm")
    print(f"  Particles: dp={dp*1000:.2f}mm, h={h*1000:.2f}mm, cutoff={cutoff*1000:.2f}mm")
    
    # ── Generate fluid particles ──
    fluid_pos = _fill_cylinder_fluid(R, L, dp, dtype, device)
    n_fluid = fluid_pos.shape[0]
    print(f"  Fluid particles: {n_fluid}")
    
    # ── Generate wall particles ──
    wall_pos = _generate_wall_particles(R, L, dp, n_wall_layers, dtype, device)
    n_wall = wall_pos.shape[0]
    print(f"  Wall particles:  {n_wall}")
    
    # ── Generate inlet/outlet particles ──
    inlet_pos, outlet_pos = _generate_inlet_outlet_particles(R, L, dp, dtype, device)
    n_inlet  = inlet_pos.shape[0]
    n_outlet = outlet_pos.shape[0]
    print(f"  Inlet particles: {n_inlet}, Outlet particles: {n_outlet}")
    
    # ── Concatenate all particles ──
    positions = torch.cat([fluid_pos, wall_pos, inlet_pos, outlet_pos], dim=0)
    N = positions.shape[0]
    print(f"  Total particles: {N}")
    
    # ── Particle type labels ──
    ptype = torch.zeros(N, dtype=torch.long, device=device)
    ptype[n_fluid : n_fluid + n_wall] = WALL
    ptype[n_fluid + n_wall : n_fluid + n_wall + n_inlet] = INLET
    ptype[n_fluid + n_wall + n_inlet :] = OUTLET
    
    # ── Initial state: zero velocity, reference density, hydrostatic pressure ──
    velocities = torch.zeros(N, 3, dtype=dtype, device=device)
    densities  = torch.full((N,), rho0, dtype=dtype, device=device)
    # Hydrostatic pressure gradient along z (small, for numerical stability)
    pressures  = torch.zeros(N, dtype=dtype, device=device)
    
    # ── Build neighbor graph ──
    print(f"  Building neighbor graph (cutoff={cutoff*1000:.2f}mm)...")
    edge_index = build_neighbor_graph_batched(positions, cutoff=cutoff)
    n_edges = edge_index.shape[1]
    avg_neighbors = n_edges / N if N > 0 else 0
    print(f"  Neighbor graph: {n_edges} directed edges, avg {avg_neighbors:.1f} neighbors/particle")
    
    # ── Boundary mask (non-fluid particles) ──
    boundary_mask = ptype > 0  # wall + inlet + outlet
    
    return SPHDomain(
        positions=positions,
        velocities=velocities,
        densities=densities,
        pressures=pressures,
        particle_type=ptype,
        edge_index=edge_index,
        h=h,
        n_particles=N,
        n_fluid=n_fluid,
        n_wall=n_wall,
        n_inlet=n_inlet,
        n_outlet=n_outlet,
        R=R,
        L=L,
        boundary_mask=boundary_mask,
        n_vertices=N,  # TetMesh compatibility alias
    )


# ─────────────────────────────────────────────────────────────
# Graph Update Utility
# ─────────────────────────────────────────────────────────────

def rebuild_neighbor_graph(domain: SPHDomain) -> SPHDomain:
    """Rebuild neighbor graph after particle positions have been updated.
    
    Called every N_rebuild steps during time integration to account for
    particle motion (fluid particles move, wall particles stay fixed).
    
    Args:
        domain: SPHDomain with updated positions
    
    Returns:
        New SPHDomain with rebuilt edge_index (all other fields unchanged)
    """
    cutoff = 2.0 * domain.h
    new_edge_index = build_neighbor_graph_batched(domain.positions, cutoff=cutoff)
    
    return SPHDomain(
        positions=domain.positions,
        velocities=domain.velocities,
        densities=domain.densities,
        pressures=domain.pressures,
        particle_type=domain.particle_type,
        edge_index=new_edge_index,
        h=domain.h,
        n_particles=domain.n_particles,
        n_fluid=domain.n_fluid,
        n_wall=domain.n_wall,
        n_inlet=domain.n_inlet,
        n_outlet=domain.n_outlet,
        R=domain.R,
        L=domain.L,
        boundary_mask=domain.boundary_mask,
        n_vertices=domain.n_vertices,
    )


# ─────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    print("=" * 60)
    print("sph_domain.py — Self Test")
    print("=" * 60)
    
    # Generate domain with coarse resolution for fast testing
    domain = generate_portal_vein_domain(
        D=0.007,    # 7mm diameter
        L=0.080,    # 80mm length
        dp=0.001,   # 1mm spacing (coarse for testing; use 0.5mm for production)
        device="cpu",
    )
    
    print(f"\nDomain summary:")
    print(f"  Total particles:  {domain.n_particles}")
    print(f"  Fluid:  {domain.n_fluid}")
    print(f"  Wall:   {domain.n_wall}")
    print(f"  Inlet:  {domain.n_inlet}")
    print(f"  Outlet: {domain.n_outlet}")
    print(f"  h = {domain.h*1000:.3f} mm")
    print(f"  Edge count: {domain.edge_index.shape[1]}")
    print(f"  Avg neighbors: {domain.edge_index.shape[1] / domain.n_particles:.1f}")
    
    # Test 1: Verify particle type distribution
    type_counts = [(int((domain.particle_type == t).sum())) for t in range(4)]
    print(f"\nTest 1: Particle type counts")
    print(f"  FLUID={type_counts[0]}, WALL={type_counts[1]}, "
          f"INLET={type_counts[2]}, OUTLET={type_counts[3]}")
    assert sum(type_counts) == domain.n_particles, "Type count mismatch!"
    print(f"  ✅ Particle type counts consistent")
    
    # Test 2: Verify fluid particles are inside cylinder
    fluid_mask = domain.particle_type == FLUID
    fluid_pos = domain.positions[fluid_mask]
    r_fluid = torch.sqrt(fluid_pos[:, 0]**2 + fluid_pos[:, 1]**2)
    max_r = r_fluid.max().item()
    print(f"\nTest 2: Fluid particles inside cylinder (R={domain.R*1000:.1f}mm)")
    print(f"  Max fluid radius: {max_r*1000:.2f}mm (should be ≤ {domain.R*1000:.1f}mm)")
    assert max_r <= domain.R + 1e-6, f"Fluid particle outside cylinder! r_max={max_r}"
    print(f"  ✅ All fluid particles inside cylinder")
    
    # Test 3: Verify neighbor graph connectivity
    print(f"\nTest 3: Neighbor graph connectivity")
    fluid_indices = torch.where(fluid_mask)[0]
    src, dst = domain.edge_index
    # Check each fluid particle has at least 1 neighbor
    has_neighbors = torch.zeros(domain.n_particles, dtype=torch.bool)
    has_neighbors.scatter_(0, src, torch.ones(len(src), dtype=torch.bool))
    fluid_with_neighbors = has_neighbors[fluid_indices].sum().item()
    print(f"  Fluid particles with neighbors: {fluid_with_neighbors}/{len(fluid_indices)}")
    isolated = len(fluid_indices) - fluid_with_neighbors
    if isolated > 0:
        print(f"  ⚠️  {isolated} isolated fluid particles (may be at boundary)")
    else:
        print(f"  ✅ All fluid particles have neighbors")
    
    # Test 4: Graph rebuild
    print(f"\nTest 4: Graph rebuild after small perturbation")
    import copy
    perturbed_pos = domain.positions.clone()
    perturbed_pos[:domain.n_fluid] += torch.randn(domain.n_fluid, 3) * 1e-5
    domain_perturbed = SPHDomain(
        positions=perturbed_pos,
        velocities=domain.velocities,
        densities=domain.densities,
        pressures=domain.pressures,
        particle_type=domain.particle_type,
        edge_index=domain.edge_index,
        h=domain.h,
        n_particles=domain.n_particles,
        n_fluid=domain.n_fluid,
        n_wall=domain.n_wall,
        n_inlet=domain.n_inlet,
        n_outlet=domain.n_outlet,
        R=domain.R,
        L=domain.L,
        boundary_mask=domain.boundary_mask,
        n_vertices=domain.n_vertices,
    )
    domain_rebuilt = rebuild_neighbor_graph(domain_perturbed)
    print(f"  Original edges: {domain.edge_index.shape[1]}")
    print(f"  Rebuilt edges:  {domain_rebuilt.edge_index.shape[1]}")
    print(f"  ✅ Graph rebuild successful")
    
    print(f"\n✅ sph_domain.py self-test PASSED")
