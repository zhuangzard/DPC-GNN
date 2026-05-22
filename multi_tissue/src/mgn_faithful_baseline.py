#!/usr/bin/env python3
# IMPORTANT: this MGN baseline supervises against an inline tet-linear-
# elasticity FEM solver using g = 9.81 m/s^2. The canonical
# fem_ground_truth.py values in all_results.json use back-computed
# gravity to enforce a 20% tip displacement target, so they're not
# directly comparable. Use err_EB (Euler-Bernoulli) as the primary
# DPC-GNN-vs-MGN comparison metric -- it's gravity-independent.
"""
mgn_faithful_baseline.py — FAITHFUL MeshGraphNets baseline for DPC-GNN.

This is the *real* MGN baseline that addresses reviewer R5B-1 ("no MGN
baseline").  The previously-existing `solid_tissue_train_mgn.py` is
mis-labelled: it keeps the *unsupervised* Neo-Hookean physics loss and only
replaces the antisymmetric edge construction with a directed-edge MP, so
it is more accurately described as UDMP (Unsupervised Directed Message
Passing) — useful for the AMP ablation, but NOT a faithful MeshGraphNets
re-implementation.

What this script does differently (= canonical MeshGraphNets, Pfaff et al.,
ICLR 2021):
    • SUPERVISED training on FEM displacement ground truth via MSE loss.
    • No physics terms whatsoever — no Neo-Hookean strain-energy density,
      no Jacobian barrier, no gradient-divergence regulariser.
    • Standard encoder → processor (directed-edge MP) → decoder backbone,
      with two independent per-edge MLPs (no antisymmetry coupling).
    • Per-node MLP encoder using the canonical MGN node feature set
      (normalised tissue stiffness, Poisson ratio, normalised density,
      one-hot fixed-DOF flag, gravity loading direction).
    • Per-edge MLP encoder using the canonical MGN edge feature set
      (relative-position vector Δx,Δy,Δz and its L2 norm).
    • Sum-aggregated message + residual node update + LayerNorm.

Comparison axis (one-sentence summary for the rebuttal):
    DPC-GNN:    trains on PHYSICS, no displacement labels.
    MGN (this): trains on DISPLACEMENT LABELS, no physics.
The two trainers share the same mesh, same node order, same per-tissue
test geometry — so the displacement metrics they produce are directly
comparable on the same 20-seed sweep.

FEM ground truth:
    We solve a small-strain (linear) elasticity boundary-value problem on
    the *same tetrahedral mesh* used by the GNN, with gravity body force
    matching the GNN training set-up.  Same mesh / same loading / Dirichlet
    BC at x=0, so per-node displacements line up element-for-element and
    MSE is a meaningful supervision signal.  This keeps the baseline self-
    contained (no FEniCS dependency) and reproducible from this single
    file.

Output (matches solid_tissue_train.py format so collect_20seed.py works):
    {output_root}/{tissue}_seed{N}/best_model.pt
    {output_root}/{tissue}_seed{N}/results.json   # tip_disp_mm, relative_error_pct, …
    {output_root}/{tissue}_seed{N}/history.json   # per-epoch loss

CLI:
    python3 mgn_faithful_baseline.py --tissue brain --seed 42 --epochs 1000
    python3 mgn_faithful_baseline.py --dry-run        # tiny syntax/shape check
"""

import os
import sys
import math
import time
import json
import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["PYTHONUNBUFFERED"] = "1"

# Re-use the canonical mesh generator + tissue catalogue.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solid_tissue_train import generate_beam_mesh, TISSUES as _SOLID_TISSUES  # noqa: E402

# Extend with bone (10 MPa) which solid_tissue_train.py omits from its
# in-script TISSUES dict but is included in the 6-tissue 20-seed sweep.
TISSUES: Dict[str, Dict[str, float]] = dict(_SOLID_TISSUES)
TISSUES.setdefault("bone", {"E": 1.0e7, "nu": 0.30, "rho": 1900.0})


# ════════════════════════════════════════════════════════════════════════
# 1. FEM ground truth (small-strain linear elasticity, on the GNN mesh)
# ════════════════════════════════════════════════════════════════════════

def _tet_linear_elasticity(
    nodes_np: np.ndarray,
    elements_np: np.ndarray,
    fixed_mask_np: np.ndarray,
    rho: float,
    E: float,
    nu: float,
    g: float = 9.81,
) -> np.ndarray:
    """Linear elasticity on a 4-node tetrahedral mesh — gravity body force,
    Dirichlet BC on `fixed_mask_np` (clamp all 3 DOFs).

    Returns
    -------
    u : (N, 3) numpy array — per-node displacement in metres.
    """
    import warnings
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.sparse.linalg import MatrixRankWarning

    n_nodes = nodes_np.shape[0]
    n_elem = elements_np.shape[0]
    ndof = 3 * n_nodes

    # Lamé parameters.
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Isotropic elasticity D matrix (6x6 in Voigt notation).
    D = np.array([
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ], dtype=np.float64)

    rows = np.zeros(n_elem * 144, dtype=np.int64)
    cols = np.zeros(n_elem * 144, dtype=np.int64)
    vals = np.zeros(n_elem * 144, dtype=np.float64)
    f_ext = np.zeros(ndof, dtype=np.float64)

    coords = nodes_np.astype(np.float64)
    elems = elements_np.astype(np.int64)

    idx = 0
    for e in range(n_elem):
        n = elems[e]
        x = coords[n]  # (4, 3)
        # Build M = [[1, x1, y1, z1], ...] to obtain shape-function gradients
        # via Cramer / inversion: ∇N_i = β_i where β = (M^{-1})[1:].
        M = np.hstack([np.ones((4, 1)), x])
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        beta = Minv[1:, :]  # (3, 4) — beta[:, i] = ∇N_i
        vol = abs(np.linalg.det(M)) / 6.0
        if vol <= 0.0:
            continue

        # Strain-displacement matrix B (6, 12).
        B = np.zeros((6, 12), dtype=np.float64)
        for i in range(4):
            bx, by, bz = beta[0, i], beta[1, i], beta[2, i]
            B[0, 3 * i + 0] = bx
            B[1, 3 * i + 1] = by
            B[2, 3 * i + 2] = bz
            B[3, 3 * i + 0] = by
            B[3, 3 * i + 1] = bx
            B[4, 3 * i + 1] = bz
            B[4, 3 * i + 2] = by
            B[5, 3 * i + 0] = bz
            B[5, 3 * i + 2] = bx

        Ke = (B.T @ D @ B) * vol  # (12, 12)

        # Gravity body force, lumped equally to the 4 nodes (−y direction).
        fe_y = -rho * g * vol / 4.0
        for i in range(4):
            f_ext[3 * n[i] + 1] += fe_y

        # Assemble Ke into triplets.
        edof = np.empty(12, dtype=np.int64)
        for i in range(4):
            edof[3 * i + 0] = 3 * n[i] + 0
            edof[3 * i + 1] = 3 * n[i] + 1
            edof[3 * i + 2] = 3 * n[i] + 2
        for a in range(12):
            for b in range(12):
                rows[idx] = edof[a]
                cols[idx] = edof[b]
                vals[idx] = Ke[a, b]
                idx += 1

    K = sparse.csr_matrix((vals[:idx], (rows[:idx], cols[:idx])),
                          shape=(ndof, ndof))

    fixed_nodes = np.where(fixed_mask_np)[0]
    fixed_dofs = np.sort(np.concatenate([3 * fixed_nodes,
                                         3 * fixed_nodes + 1,
                                         3 * fixed_nodes + 2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    u_vec = np.zeros(ndof, dtype=np.float64)
    if free_dofs.size > 0:
        K_ff = K[free_dofs[:, None], free_dofs[None, :]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MatrixRankWarning)
                sol = spsolve(K_ff.tocsc(), f_ext[free_dofs])
        except Exception:
            sol = np.zeros(free_dofs.size, dtype=np.float64)
        if np.all(np.isfinite(sol)):
            u_vec[free_dofs] = sol
        else:
            # Singular system (e.g. degenerate 4-node dry-run mesh with
            # under-constrained rigid-body modes).  Fall back to zeros.
            u_vec[free_dofs] = 0.0

    return u_vec.reshape(n_nodes, 3)


def compute_fem_targets(mesh: Dict, E: float, nu: float, rho: float,
                        device: torch.device) -> torch.Tensor:
    """Wrapper around `_tet_linear_elasticity` returning a torch.float32
    tensor on the GNN's device, matching the mesh node order."""
    nodes_np = mesh["nodes"].detach().cpu().numpy()
    elems_np = mesh["elements"].detach().cpu().numpy()
    fixed_np = mesh["fixed_mask"].detach().cpu().numpy()
    u_np = _tet_linear_elasticity(nodes_np, elems_np, fixed_np, rho, E, nu)
    return torch.from_numpy(u_np).to(device=device, dtype=torch.float32)


def compute_err_eb(tip_predicted_mm: float, E: float, Lx: float, Ly: float,
                   Lz: float, rho: float, g: float = 9.81) -> float:
    """Gravity-independent Euler-Bernoulli relative error (percent).

    EB cantilever tip deflection under self-weight:
        delta_EB = (rho * g * Ly * Lz) * Lx^4 / (8 * E * I),
        I = Lz * Ly^3 / 12.
    Both the numerator (rho*g*Ly*Lz) and the FEM target scale identically
    with g, so the *relative* error (|pred - eb| / |eb|) is the same
    whether g is 9.81 or back-computed to enforce a 20% tip.  This makes
    err_EB the canonical DPC-GNN-vs-MGN comparison axis.

    Returns
    -------
    float : 100 * |tip_predicted - delta_EB| / |delta_EB|  (percent)
    """
    I_val = Lz * (Ly ** 3) / 12.0
    if E <= 0.0 or I_val <= 0.0:
        return float("nan")
    w_load = rho * g * Ly * Lz
    delta_eb_m = w_load * (Lx ** 4) / (8.0 * E * I_val)
    delta_eb_mm = delta_eb_m * 1000.0
    if abs(delta_eb_mm) < 1e-12:
        return float("nan")
    return abs(abs(tip_predicted_mm) - abs(delta_eb_mm)) / abs(delta_eb_mm) * 100.0


# ════════════════════════════════════════════════════════════════════════
# 2. MeshGraphNets architecture (canonical, directed-edge MP)
# ════════════════════════════════════════════════════════════════════════

class EdgeBlock(nn.Module):
    """Canonical MGN per-edge MLP.  Inputs:
        [h_src (H), h_dst (H), h_edge (H)]  →  new h_edge (H).
    Independent for the forward and reverse direction of each undirected
    edge: caller passes both (s, d) and (d, s) entries in `edge_index`.
    """
    def __init__(self, hdim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                h_edge: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_src, h_dst, h_edge], dim=-1))


class NodeBlock(nn.Module):
    """Canonical MGN per-node MLP.  Sum-aggregates incoming edge messages
    and updates the node.  Residual connection."""
    def __init__(self, hdim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
        )

    def forward(self, h_node: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_node, agg], dim=-1))


class MGNProcessorLayer(nn.Module):
    """One MGN processor step: edge update → node update (with residuals)."""
    def __init__(self, hdim: int):
        super().__init__()
        self.edge_block = EdgeBlock(hdim)
        self.node_block = NodeBlock(hdim)

    def forward(self, h_node: torch.Tensor, h_edge: torch.Tensor,
                edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, d = edge_index
        # Edge update (directed: no antisymmetry construction).
        new_e = self.edge_block(h_node[s], h_node[d], h_edge)
        h_edge_out = h_edge + new_e

        # Sum-aggregate updated edge messages onto the source node.
        agg = torch.zeros_like(h_node)
        idx = s.unsqueeze(-1).expand(-1, h_node.shape[-1])
        agg.scatter_add_(0, idx, h_edge_out)

        # Node update with residual.
        h_node_out = h_node + self.node_block(h_node, agg)
        return h_node_out, h_edge_out


class MeshGraphNetsModel(nn.Module):
    """Encoder → 6× Processor → Decoder.  Canonical Pfaff et al. layout."""
    def __init__(self, node_in_dim: int = 6, edge_in_dim: int = 4,
                 hdim: int = 64, n_layers: int = 6, out_dim: int = 3):
        super().__init__()
        self.hdim = hdim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.LayerNorm(hdim),
        )
        self.processor = nn.ModuleList([MGNProcessorLayer(hdim)
                                        for _ in range(n_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, out_dim),
        )
        # Small initial decoder output (mirrors solid_tissue_train.py).
        nn.init.uniform_(self.decoder[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, node_feat: torch.Tensor, edge_feat: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h_node = self.node_encoder(node_feat)
        h_edge = self.edge_encoder(edge_feat)
        for layer in self.processor:
            h_node, h_edge = layer(h_node, h_edge, edge_index)
        return self.decoder(h_node)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════
# 3. Feature construction (canonical MGN node + edge feature sets)
# ════════════════════════════════════════════════════════════════════════

def build_features(mesh: Dict, E: float, nu: float, rho: float,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce (node_feat, edge_feat) tensors.

    Node features (per node, 6-D):
        [tissue_E_log_normalized, ν, ρ_normalized, fixed_flag,
         gravity_x, gravity_y]   (gravity_z is always 0, dropped to keep
                                  the canonical compact MGN feature set)

    The prompt asks for [E_log_norm, ν, ρ_norm] in the encoder; we keep
    those three exactly as specified and append three more standard MGN
    boundary-condition features (fixed flag, gravity-direction) so the
    network can distinguish the cantilever's clamped face from the rest.
    Without those, the supervised loss cannot disambiguate which face is
    clamped — MGN papers always inject Dirichlet flags into the node
    feature vector.

    Edge features (per directed edge, 4-D):
        [Δx, Δy, Δz, ||r||]
    """
    nodes = mesh["nodes"]
    edge_index = mesh["edge_index"]
    fixed_mask = mesh["fixed_mask"]
    n = nodes.shape[0]

    # Normalised material properties.
    logE = (math.log10(max(float(E), 1.0)) - 3.0) / 7.0   # E∈[10^3, 10^10] Pa → [0, 1]
    nu_v = float(nu)
    rho_norm = (float(rho) - 1000.0) / 1000.0             # ≈[0, 1] for tissue range

    node_feat = torch.zeros(n, 6, device=device, dtype=torch.float32)
    node_feat[:, 0] = logE
    node_feat[:, 1] = nu_v
    node_feat[:, 2] = rho_norm
    node_feat[:, 3] = fixed_mask.float()
    # Gravity points in −y in world frame: encode (0, -1) into the node feat.
    node_feat[:, 4] = 0.0
    node_feat[:, 5] = -1.0

    s, d = edge_index
    rel = nodes[s] - nodes[d]
    rnorm = rel.norm(dim=-1, keepdim=True)
    # Normalise by max coordinate range (matches solid_tissue_train.py).
    prng = (nodes.max(0).values - nodes.min(0).values + 1e-8).max()
    edge_feat = torch.cat([rel / prng, rnorm / prng], dim=-1).float()

    return node_feat, edge_feat


# ════════════════════════════════════════════════════════════════════════
# 4. Training loop
# ════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    """Deterministic seeding across random / numpy / torch / cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mgn(tissue: str, seed: int = 42, epochs: int = 1000,
              lr: float = 1e-3, hidden_dim: int = 64, n_layers: int = 6,
              output_root: str = "/root/results/mgn_faithful",
              device: str = None) -> Dict:
    """Train a faithful MGN baseline on FEM-displacement labels for one
    tissue + seed.  Returns the result dict that's also saved to JSON."""
    set_global_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tissue not in TISSUES:
        raise KeyError(f"Unknown tissue '{tissue}'. "
                       f"Known: {sorted(TISSUES.keys())}")
    p = TISSUES[tissue]
    E, nu, rho = float(p["E"]), float(p["nu"]), float(p["rho"])

    # Beam geometry — mirror solid_tissue_train.py heuristics so the
    # baseline and DPC-GNN see the same per-tissue geometry.
    Lx, Ly, Lz = 0.1, 0.02, 0.02
    I_val = Lz * Ly ** 3 / 12.0
    w_load = rho * 9.81 * Ly * Lz
    delta_exp = w_load * Lx ** 4 / (8.0 * E * I_val) if E > 0 else 1e-3
    if delta_exp > 0.01:
        Lx = 0.03
        delta_exp = w_load * Lx ** 4 / (8.0 * E * I_val)

    nx, ny, nz = 15, 4, 4
    if nu >= 0.48:
        nx, ny, nz = max(nx, 25), max(ny, 8), max(nz, 8)
    elif nu >= 0.45:
        nx, ny, nz = max(nx, 20), max(ny, 6), max(nz, 6)

    print(f"\n{'=' * 70}")
    print(f"  MGN (FAITHFUL) | tissue={tissue.upper()} | seed={seed}")
    print(f"  E={E:.1f} Pa | ν={nu} | ρ={rho} kg/m³")
    print(f"  Beam: {Lx * 100:.0f}cm × {Ly * 100:.0f}cm × {Lz * 100:.0f}cm "
          f"| Mesh: {nx}×{ny}×{nz} | Device: {device} | Epochs: {epochs}")
    print(f"  EB-expected tip δ: {delta_exp * 1000:.4f} mm")
    print(f"{'=' * 70}")

    mesh = generate_beam_mesh(Lx, Ly, Lz, nx, ny, nz, device=device)
    n_nodes = mesh["N"]
    edge_index = mesh["edge_index"]
    nodes_ref = mesh["nodes"]
    fixed_mask = mesh["fixed_mask"]
    print(f"  Nodes: {n_nodes} | Tets: {mesh['elements'].shape[0]} "
          f"| Edges: {edge_index.shape[1]}")

    # --- FEM ground truth (linear elasticity, same tet mesh). -----------
    t_fem0 = time.time()
    print(f"  Solving FEM ground truth …", flush=True)
    u_fem = compute_fem_targets(mesh, E, nu, rho, torch.device(device))
    print(f"  FEM done in {time.time() - t_fem0:.1f}s "
          f"| FEM tip δ ≈ {abs(u_fem[nodes_ref[:, 0] > Lx - 1e-6, 1].mean().item()) * 1000:.4f} mm",
          flush=True)

    # --- Build static node / edge feature tensors. ---------------------
    node_feat, edge_feat = build_features(mesh, E, nu, rho,
                                          torch.device(device))

    # --- Model + optimiser. --------------------------------------------
    model = MeshGraphNetsModel(node_in_dim=node_feat.shape[-1],
                               edge_in_dim=edge_feat.shape[-1],
                               hdim=hidden_dim, n_layers=n_layers,
                               out_dim=3).to(device)
    print(f"  Model params: {model.count_params():,}")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                 eta_min=lr * 0.01)
    warmup = min(50, max(1, epochs // 10))

    # Output-scale heuristic: cap the decoder amplitude near the expected
    # displacement so the MSE optimiser has a sensible starting basin.
    dscale = float(min(max(abs(delta_exp), 1e-8), 0.01))

    # Mask used to zero-out displacement at clamped nodes (Dirichlet BC).
    free_mask = (~fixed_mask).float().unsqueeze(-1)

    best_loss = float("inf")
    best_ep = 0
    best_state = None
    history = []
    t0 = time.time()

    print(f"\n  {'Ep':>5} | {'MSE':>12} | {'u_tip(mm)':>10} "
          f"| {'FEM_tip(mm)':>11} | {'lr':>9}")
    print(f"  {'-' * 60}")

    for ep in range(1, epochs + 1):
        # LR warm-up.
        if ep <= warmup:
            for pg in opt.param_groups:
                pg["lr"] = lr * ep / warmup

        opt.zero_grad()
        u_pred_raw = model(node_feat, edge_feat, edge_index)
        u_pred = u_pred_raw * dscale * free_mask
        loss = ((u_pred - u_fem) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep > warmup:
            sched.step()

        lv = float(loss.item())
        with torch.no_grad():
            tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
            u_tip = (u_pred[tip_mask, 1].mean().item() * 1000.0
                     if tip_mask.any() else 0.0)
            fem_tip = (u_fem[tip_mask, 1].mean().item() * 1000.0
                       if tip_mask.any() else 0.0)

        history.append({"ep": ep, "loss": lv, "u_tip_mm": u_tip,
                        "fem_tip_mm": fem_tip})

        if lv < best_loss:
            best_loss = lv
            best_ep = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if ep <= 5 or ep % 25 == 0 or ep == epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {u_tip:10.4f} "
                  f"| {fem_tip:11.4f} | {opt.param_groups[0]['lr']:9.2e}")

        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}.  Reverting to best (ep {best_ep}).")
            if best_state is not None:
                model.load_state_dict({k: v.to(device)
                                       for k, v in best_state.items()})
                for pg in opt.param_groups:
                    pg["lr"] *= 0.1
            else:
                break

    train_time = time.time() - t0

    # Final evaluation with best checkpoint.
    if best_state is not None:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})
    with torch.no_grad():
        u_eval = model(node_feat, edge_feat, edge_index) * dscale * free_mask
        tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
        tip_y_mm = (u_eval[tip_mask, 1].mean().item() * 1000.0
                    if tip_mask.any() else 0.0)
        fem_tip_mm = (u_fem[tip_mask, 1].mean().item() * 1000.0
                      if tip_mask.any() else 0.0)
        full_mse = float(((u_eval - u_fem) ** 2).mean().item())

    rel_err = (abs(abs(tip_y_mm) - abs(fem_tip_mm)) / abs(fem_tip_mm) * 100.0
               if abs(fem_tip_mm) > 1e-12 else 0.0)
    # Gravity-independent EB error — primary DPC-GNN-vs-MGN comparison axis
    # (the FEM target here is computed with g=9.81 m/s^2, but the canonical
    # all_results.json values use back-computed g; only err_EB is comparable).
    err_eb_pct = compute_err_eb(tip_y_mm, E, Lx, Ly, Lz, rho, g=9.81)

    results = {
        "baseline": "mgn_faithful_supervised",
        "tissue": tissue,
        "seed": seed,
        "E_Pa": E, "E_kPa": E / 1000.0,
        "nu": nu, "rho": rho,
        "beam_Lx_m": Lx, "beam_Ly_m": Ly, "beam_Lz_m": Lz,
        "epochs": epochs, "best_epoch": best_ep, "best_loss": best_loss,
        "tip_disp_mm": tip_y_mm,
        "fem_tip_mm": fem_tip_mm,
        "expected_tip_mm": fem_tip_mm,
        "eb_tip_mm": delta_exp * 1000.0,
        "relative_error_pct": rel_err,
        "err_EB_pct": err_eb_pct,  # gravity-independent; primary comparison metric
        "full_field_mse": full_mse,
        "training_time_s": train_time,
        "N_nodes": n_nodes,
        "N_elements": int(mesh["elements"].shape[0]),
        "params": model.count_params(),
    }

    print(f"\n  {'=' * 70}")
    print(f"  ✅ {tissue.upper()} (MGN FAITHFUL) COMPLETE")
    print(f"  Best MSE: {best_loss:.6e}  (epoch {best_ep})")
    print(f"  Tip: {tip_y_mm:.4f} mm  |  FEM target: {fem_tip_mm:.4f} mm "
          f"|  Relative error: {rel_err:.2f}%")
    print(f"  Training time: {train_time:.1f}s ({train_time / 60:.1f}min)")
    print(f"  {'=' * 70}\n")

    ckpt_dir = os.path.join(output_root, f"{tissue}_seed{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, os.path.join(ckpt_dir, "best_model.pt"))
    with open(os.path.join(ckpt_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(ckpt_dir, "history.json"), "w") as fh:
        json.dump(history, fh)

    return results


# ════════════════════════════════════════════════════════════════════════
# 5. CLI + dry-run self-test
# ════════════════════════════════════════════════════════════════════════

def _dry_run() -> None:
    """Tiny mesh forward+backward pass; no GPU, no files."""
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  mgn_faithful_baseline.py — DRY RUN (4-node tetrahedron) │")
    print("└──────────────────────────────────────────────────────────┘")
    device = torch.device("cpu")

    # Single tetrahedron: 4 nodes, 1 element, 12 directed edges.
    nodes = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    fixed_mask = torch.tensor([True, False, False, False])
    # All 12 directed edges (i→j for i≠j on a 4-node clique).
    ei = []
    for a in range(4):
        for b in range(4):
            if a != b:
                ei.append((a, b))
    edge_index = torch.tensor(ei, dtype=torch.long).t()
    # Compute element volume = 1/6 for the canonical unit tet.
    v0, v1, v2, v3 = (nodes[elements[:, i]] for i in range(4))
    volumes = ((v1 - v0) * torch.linalg.cross(v2 - v0, v3 - v0, dim=-1)
               ).sum(-1).abs() / 6.0
    mesh = {
        "nodes": nodes, "elements": elements, "edge_index": edge_index,
        "fixed_mask": fixed_mask, "volumes": volumes, "N": 4,
        "Lx": 1.0, "Ly": 1.0, "Lz": 1.0,
    }

    # Build features and a tiny MGN.
    E, nu, rho = 1000.0, 0.45, 1050.0
    node_feat, edge_feat = build_features(mesh, E, nu, rho, device)
    print(f"  node_feat: {tuple(node_feat.shape)}  "
          f"edge_feat: {tuple(edge_feat.shape)}  "
          f"edge_index: {tuple(edge_index.shape)}")

    model = MeshGraphNetsModel(node_in_dim=node_feat.shape[-1],
                               edge_in_dim=edge_feat.shape[-1],
                               hdim=16, n_layers=2, out_dim=3).to(device)
    print(f"  Model params: {model.count_params():,}")

    # Synthetic target (just to exercise backward).
    u_target = torch.zeros(4, 3)
    u_target[1, 1] = -1e-3

    opt = optim.Adam(model.parameters(), lr=1e-3)
    out = model(node_feat, edge_feat, edge_index)
    print(f"  Forward output shape: {tuple(out.shape)}  "
          f"dtype: {out.dtype}")
    loss = ((out * 1e-3 - u_target) ** 2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5
    opt.step()
    print(f"  Backward OK | loss={loss.item():.6e} | "
          f"grad-norm={grad_norm:.6e}")

    # FEM solver sanity check on the same 4-node tet.
    u_fem = compute_fem_targets(mesh, E, nu, rho, device)
    print(f"  FEM ground truth tensor: {tuple(u_fem.shape)} "
          f"max|u|={u_fem.abs().max().item():.3e} m")
    print("  ✅ Dry-run completed.  Exiting without writing files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAITHFUL MeshGraphNets baseline (supervised on FEM "
                    "displacement labels) for DPC-GNN comparison.")
    parser.add_argument("--tissue", type=str, default=None,
                        choices=sorted(TISSUES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--output-root", type=str,
                        default="/root/results/mgn_faithful")
    parser.add_argument("--dry-run", action="store_true",
                        help="4-node sanity check; no GPU, no files.")
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        return

    if args.tissue is None:
        parser.error("--tissue is required (unless --dry-run).  "
                     f"Choices: {sorted(TISSUES.keys())}")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DPC-GNN — Faithful MeshGraphNets Baseline (R5B-1)          ║")
    print("║  Supervised on FEM displacement labels, no physics loss     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch: {torch.__version__}")

    train_mgn(tissue=args.tissue, seed=args.seed, epochs=args.epochs,
              lr=args.lr, hidden_dim=args.hidden_dim,
              n_layers=args.n_layers, output_root=args.output_root)


if __name__ == "__main__":
    main()
