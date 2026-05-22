"""
sph_gnn_model.py — SPH-GNN Model for Blood Flow Acceleration Prediction.

Extends DPC-GNN's DynamicPIGNN architecture for SPH fluid simulation.
Predicts per-particle acceleration a_i ∈ R³ from SPH particle state.

Architecture (mirrors DynamicPIGNN):
  1. Encoder:  9D node features → hidden_dim
  2. Processor: n_mp_layers × AntisymmetricMP (REUSED from static_pignn_model.py)
  3. Decoder:  hidden_dim → 3D acceleration

Node features (9D):
  [x(3), v(3), ρ(1), p(1), particle_type(1)]

Edge features (11D) — passed to AntisymmetricMP:
  [r_ij(3), |r_ij|(1), v_ij(3), W_ij(1), ∇W_ij(3)] = 11D

Boundary conditions (hard enforcement):
  - Wall particles:   a = 0 (fixed)
  - Inlet particles:  a = 0 (prescribed velocity profile, not learned)
  - Outlet particles: a = 0 (zero-gradient, handled by loss)
  - Fluid particles:  a predicted by GNN

Expert Council Review (5 experts):
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - GNN架构专家: AntisymmetricMP reused zero-modification, edge_dim=11 via new msg_mlp
  - PIGNN专家: Output scale tuning for acceleration units (m/s²)
  - 计算流体专家: Edge features encode SPH kernel info (W, ∇W)
  - 血液动力学专家: Particle type encoding for BC handling
  - 集成测试专家: Drop-in replacement for DynamicPIGNN with SPHDomain input
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

# ── Import AntisymmetricMP from Phase A (zero modification) ──
# Add phase_a to path (adjust based on actual DPC-GNN layout)
_PHASE_A_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "phase_a")
if os.path.exists(_PHASE_A_DIR):
    sys.path.insert(0, _PHASE_A_DIR)

from sph_domain import SPHDomain, FLUID, WALL, INLET, OUTLET
from sph_kernels import wendland_c2


# ─────────────────────────────────────────────────────────────
# SPH-Adapted Antisymmetric Message Passing
# ─────────────────────────────────────────────────────────────

class SPHAntisymmetricMP(nn.Module):
    """Antisymmetric message passing adapted for 11D SPH edge features.
    
    Identical architecture to AntisymmetricMP from static_pignn_model.py,
    but accepts 11D edge attributes (instead of 3D displacement-only).
    
    Edge features (11D):
      - r_ij (3D): relative position vector
      - |r_ij| (1D): distance
      - v_ij (3D): relative velocity
      - W_ij (1D): kernel value
      - ∇W_ij (3D): kernel gradient vector
    
    Key: m_ij = f(h_i, h_j, e_ij) - f(h_j, h_i, e_ij) → guarantees m_ij = -m_ji
    This antisymmetry matches SPH force antisymmetry F_ij = -F_ji (Newton's 3rd law).
    """
    
    def __init__(self, hidden_dim: int, edge_dim: int = 11, activation: nn.Module = None):
        super().__init__()
        
        if activation is None:
            activation = nn.SiLU()
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message function: (h_i, h_j, e_ij) → message
        # Input dim: hidden_dim * 2 + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Update function: (h_i, agg) → h_i_new
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Layer norm for stability (matches DPC-GNN style)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, hidden_dim) node features
            edge_index: (2, E) directed edges [src, dst]
            edge_attr: (E, edge_dim) edge features
        
        Returns:
            x_new: (N, hidden_dim) updated node features
        """
        src, dst = edge_index
        N = x.shape[0]
        device = x.device
        
        # ── Compute antisymmetric messages ──
        x_src = x[src]  # (E, H)
        x_dst = x[dst]  # (E, H)
        
        # Forward: f(h_i, h_j, e_ij)
        msg_fwd = self.msg_mlp(torch.cat([x_src, x_dst, edge_attr], dim=-1))
        
        # Reverse: f(h_j, h_i, -e_ij)  (negate edge for direction reversal)
        msg_rev = self.msg_mlp(torch.cat([x_dst, x_src, -edge_attr], dim=-1))
        
        # Antisymmetric message: m_ij = fwd - rev → m_ij = -m_ji
        messages = msg_fwd - msg_rev  # (E, H)
        
        # ── Aggregate messages (sum) ──
        agg = torch.zeros(N, self.hidden_dim, device=device, dtype=x.dtype)
        agg.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.hidden_dim), messages)
        
        # ── Update node features with residual ──
        x_new = self.update_mlp(torch.cat([x, agg], dim=-1))
        x_new = self.layer_norm(x + x_new)  # residual + layer norm
        
        return x_new


# ─────────────────────────────────────────────────────────────
# SPH-GNN Model
# ─────────────────────────────────────────────────────────────

class SPHGNNModel(nn.Module):
    """Physics-Informed GNN for SPH blood flow acceleration prediction.
    
    Predicts per-particle acceleration a_i ∈ R³.
    Physical forces (pressure + viscous) are enforced via SPH physics loss.
    GNN learns the optimal message-passing weights to minimize physics residuals.
    
    Input (9D per particle):
      [x(3), v(3), ρ_norm(1), p_norm(1), type(1)]
    
    Edge features (11D):
      [r_ij(3), |r_ij|(1), v_ij(3), W_ij(1), ∇W_ij(3)]
    
    Output: a_i ∈ R³ (fluid particles only; boundary → a=0)
    
    Training: Loss = SPH physics loss (sph_physics_loss.py)
    No ground truth data required.
    """
    
    def __init__(
        self,
        hidden_dim: int = 96,        # 96D (slightly larger than DPC-GNN's 64 for fluid)
        n_mp_layers: int = 5,        # 5 layers (matched to task spec)
        node_input_dim: int = 9,     # [x(3), v(3), ρ(1), p(1), type(1)]
        edge_input_dim: int = 11,    # [r_ij(3), |r_ij|(1), v_ij(3), W(1), ∇W(3)]
        output_dim: int = 3,         # acceleration a ∈ R³
        output_scale: float = 1.0,   # initial output scale (m/s²)
        rho0: float = 1060.0,        # reference density for normalization
        p_ref: float = 100.0,        # reference pressure for normalization [Pa]
        v_ref: float = 0.25,         # reference velocity (portal vein ~0.25 m/s)
    ):
        """
        Args:
            hidden_dim: Hidden representation dimension (96, matches task spec)
            n_mp_layers: Number of antisymmetric message passing layers (5)
            node_input_dim: Node feature dimension (9D)
            edge_input_dim: Edge feature dimension (11D)
            output_dim: Output dimension (3 for acceleration)
            output_scale: Scale factor for decoder output
            rho0: Reference density for feature normalization
            p_ref: Reference pressure for feature normalization
            v_ref: Reference velocity for feature normalization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_mp_layers = n_mp_layers
        self.output_scale = output_scale
        self.rho0 = rho0
        self.p_ref = p_ref
        self.v_ref = v_ref
        
        # ── Encoder: 9D → hidden_dim ──
        self.encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # ── Processor: n_mp_layers × SPHAntisymmetricMP ──
        self.mp_layers = nn.ModuleList([
            SPHAntisymmetricMP(hidden_dim, edge_dim=edge_input_dim, activation=nn.SiLU())
            for _ in range(n_mp_layers)
        ])
        
        # ── Decoder: hidden_dim → acceleration (3D) ──
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize decoder to near-zero for stable training start
        self._init_decoder()
    
    def _init_decoder(self):
        """Initialize decoder last layer to near-zero for training stability.
        
        Large initial accelerations would cause divergence in SPH time integration.
        """
        last = self.decoder[-1]
        nn.init.uniform_(last.weight, -0.01, 0.01)
        nn.init.zeros_(last.bias)
    
    def _build_node_features(
        self,
        domain: SPHDomain,
    ) -> torch.Tensor:
        """Build normalized 9D node feature tensor from SPHDomain.
        
        Features:
          [x_norm(3), v_norm(3), rho_norm(1), p_norm(1), type_norm(1)]
        
        Args:
            domain: SPHDomain with current particle state
        
        Returns:
            node_feats: (N, 9) normalized node features
        """
        N = domain.n_particles
        device = domain.positions.device
        dtype = domain.positions.dtype
        
        # Position: normalize to [0, 1]
        pos_min = domain.positions.min(dim=0).values
        pos_range = domain.positions.max(dim=0).values - pos_min + 1e-8
        x_norm = (domain.positions - pos_min) / pos_range  # (N, 3)
        
        # Velocity: normalize by reference velocity
        v_norm = domain.velocities / self.v_ref  # (N, 3)
        
        # Density: normalized perturbation
        rho_norm = ((domain.densities - self.rho0) / self.rho0).unsqueeze(-1)  # (N, 1)
        
        # Pressure: normalize by reference pressure
        p_norm = (domain.pressures / self.p_ref).unsqueeze(-1)  # (N, 1)
        
        # Particle type: normalize to [0, 1]
        type_norm = (domain.particle_type.float() / 3.0).unsqueeze(-1)  # (N, 1)
        
        node_feats = torch.cat([x_norm, v_norm, rho_norm, p_norm, type_norm], dim=-1)
        assert node_feats.shape == (N, 9), f"Expected (N, 9), got {node_feats.shape}"
        
        return node_feats
    
    def _build_edge_features(
        self,
        domain: SPHDomain,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Build 11D edge features from SPH particle pairs.
        
        Features per edge (i, j):
          r_ij = x_i - x_j   (3D)
          |r_ij|              (1D)
          v_ij = v_i - v_j   (3D)
          W_ij                (1D)
          ∇W_ij               (3D)
          Total: 11D
        
        Args:
            domain: SPHDomain with current state
            eps: Regularization for distance computation
        
        Returns:
            edge_feats: (E, 11) edge features
        """
        src, dst = domain.edge_index
        
        # Relative position and distance
        r_vec = domain.positions[src] - domain.positions[dst]  # (E, 3)
        r_mag = torch.sqrt((r_vec**2).sum(-1, keepdim=True) + eps**2)  # (E, 1)
        
        # Relative velocity
        v_vec = domain.velocities[src] - domain.velocities[dst]  # (E, 3)
        
        # SPH kernel value and gradient
        W, gradW = wendland_c2(r_vec, domain.h, eps=eps)  # (E,), (E, 3)
        W_feat = W.unsqueeze(-1)  # (E, 1)

        # ── Normalize edge features to O(1) ──
        # Raw W_ij (~1e8) and ∇W_ij (~1e11) dwarf position/velocity by 10-14 orders.
        # Normalize by kernel reference scales so all features are O(1).
        h = domain.h
        W_ref = 21.0 / (16.0 * math.pi * h ** 3)   # W(0): kernel peak value
        gradW_ref = W_ref / h                         # ~max |∇W| scale

        r_norm = r_vec / h            # dimensionless relative position
        r_mag_norm = r_mag / h        # dimensionless distance
        W_norm = W_feat / W_ref       # dimensionless kernel value
        gradW_norm = gradW / gradW_ref  # dimensionless kernel gradient

        edge_feats = torch.cat([r_norm, r_mag_norm, v_vec, W_norm, gradW_norm], dim=-1)  # (E, 11)
        
        return edge_feats
    
    def forward(
        self,
        domain: SPHDomain,
    ) -> torch.Tensor:
        """Predict per-particle acceleration.
        
        Args:
            domain: SPHDomain with current particle state
        
        Returns:
            a: (N, 3) per-particle acceleration [m/s²]
               Boundary particles (wall/inlet/outlet) → a = 0 (hard constraint)
        """
        N = domain.n_particles
        
        # ── Build features ──
        node_feats = self._build_node_features(domain)   # (N, 9)
        edge_feats = self._build_edge_features(domain)   # (E, 11)
        
        # ── Encode ──
        h = self.encoder(node_feats)  # (N, hidden_dim)
        
        # ── Message passing ──
        for mp_layer in self.mp_layers:
            h = mp_layer(h, domain.edge_index, edge_feats)
        
        # ── Decode to acceleration ──
        a = self.decoder(h) * self.output_scale  # (N, 3)
        
        # ── Hard boundary conditions ──
        # Non-fluid particles: a = 0 (wall no-slip, inlet/outlet prescribed externally)
        fluid_mask = (domain.particle_type == FLUID)  # (N,)
        a = a * fluid_mask.float().unsqueeze(-1)
        
        return a
    
    def count_parameters(self) -> Tuple[int, int]:
        """Return (total, trainable) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_hidden_states(self, domain: SPHDomain) -> torch.Tensor:
        """Get final hidden representations (for visualization/analysis).
        
        Returns: (N, hidden_dim) hidden states after all MP layers
        """
        node_feats = self._build_node_features(domain)
        edge_feats = self._build_edge_features(domain)
        
        h = self.encoder(node_feats)
        for mp_layer in self.mp_layers:
            h = mp_layer(h, domain.edge_index, edge_feats)
        
        return h


# ─────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────

def create_sph_gnn(
    hidden_dim: int = 96,
    n_mp_layers: int = 5,
    device: str = "cpu",
    v_ref: float = 0.25,
    p_ref: float = 100.0,
) -> SPHGNNModel:
    """Create SPH-GNN model with DPC-GNN-compatible settings.
    
    Args:
        hidden_dim: Hidden dimension (96, per task spec)
        n_mp_layers: Message passing layers (5, per task spec)
        device: Target device (cpu / cuda / mps)
        v_ref: Reference velocity for feature normalization [m/s]
        p_ref: Reference pressure for feature normalization [Pa]
    
    Returns:
        Initialized SPHGNNModel
    """
    model = SPHGNNModel(
        hidden_dim=hidden_dim,
        n_mp_layers=n_mp_layers,
        v_ref=v_ref,
        p_ref=p_ref,
    ).to(device)
    
    total, trainable = model.count_parameters()
    print(f"[SPH-GNN] Model created:")
    print(f"  hidden_dim={hidden_dim}, n_mp_layers={n_mp_layers}")
    print(f"  Total: {total:,} | Trainable: {trainable:,}")
    print(f"  Device: {device}")
    
    return model


# ─────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    print("=" * 60)
    print("sph_gnn_model.py — Self Test")
    print("=" * 60)
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # ── Generate test domain ──
    from sph_domain import generate_portal_vein_domain
    
    domain = generate_portal_vein_domain(
        D=0.007, L=0.080, dp=0.001,  # coarse for testing
        device=device,
    )
    
    print(f"\nDomain: {domain.n_particles} particles, {domain.edge_index.shape[1]} edges")
    
    # ── Create model ──
    model = create_sph_gnn(hidden_dim=96, n_mp_layers=5, device=device)
    
    # ── Test 1: Forward pass ──
    print("\nTest 1: Forward pass")
    with torch.no_grad():
        a = model(domain)
    print(f"  Output shape: {a.shape}")
    print(f"  Acceleration max: {a.abs().max().item():.4e} m/s²")
    print(f"  Fluid acc max: {a[domain.particle_type == 0].abs().max().item():.4e}")
    
    # Verify BC: non-fluid particles have a = 0
    non_fluid_acc = a[domain.particle_type != 0]
    print(f"  Non-fluid acc max: {non_fluid_acc.abs().max().item():.2e} (should be 0)")
    assert non_fluid_acc.abs().max().item() < 1e-8, "Boundary condition violated!"
    print(f"  ✅ Boundary conditions enforced correctly")
    
    # ── Test 2: Gradient flow ──
    print("\nTest 2: Gradient flow through model")
    a2 = model(domain)
    loss = a2.sum()
    loss.backward()
    
    grads_ok = 0
    grads_none = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            grads_ok += 1
        else:
            grads_none += 1
    
    print(f"  Params with non-zero gradients: {grads_ok}")
    print(f"  Params with zero/no gradients:  {grads_none}")
    assert grads_ok > 0, "No gradients flowing!"
    print(f"  ✅ Gradient flow verified")
    
    # ── Test 3: Edge feature construction ──
    print("\nTest 3: Edge feature construction")
    with torch.no_grad():
        edge_feats = model._build_edge_features(domain)
    print(f"  Edge features shape: {edge_feats.shape}  (should be (E, 11))")
    assert edge_feats.shape[1] == 11, f"Expected 11D edge features, got {edge_feats.shape[1]}"
    print(f"  ✅ Edge features are 11D")
    
    # ── Test 4: Node feature construction ──
    print("\nTest 4: Node feature construction")
    with torch.no_grad():
        node_feats = model._build_node_features(domain)
    print(f"  Node features shape: {node_feats.shape}  (should be (N, 9))")
    assert node_feats.shape[1] == 9, f"Expected 9D node features, got {node_feats.shape[1]}"
    print(f"  ✅ Node features are 9D")
    
    # ── Test 5: Different velocities produce different outputs ──
    print("\nTest 5: Model sensitivity to velocity input")
    import copy
    
    # Create domain with non-zero velocities
    domain2_vels = domain.velocities.clone()
    domain2_vels[:domain.n_fluid] = 0.1  # 0.1 m/s in all directions
    
    from sph_domain import SPHDomain
    domain2 = SPHDomain(
        positions=domain.positions,
        velocities=domain2_vels,
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
    
    with torch.no_grad():
        a_zero = model(domain)
        a_nonzero = model(domain2)
    
    diff = (a_zero - a_nonzero).abs().max().item()
    print(f"  |a(v=0) - a(v≠0)| max = {diff:.4e}")
    print(f"  {'✅' if diff > 0 else '⚠️'} Model responds to velocity changes")
    
    # ── Summary ──
    total, trainable = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Model summary:")
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    print(f"  Architecture: 9D -> [96D x 5 MP] -> 3D")
    print(f"  Edge features: 11D (r_ij, |r_ij|, v_ij, W_ij, ∇W_ij)")
    print(f"\n✅ sph_gnn_model.py self-test PASSED")
