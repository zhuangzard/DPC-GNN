#!/usr/bin/env python3
"""
fem_fbar_at_nu049.py — P1-linear-tetrahedron F-bar FEM reference
that does NOT lock at ν → 0.49.

Used to give DPC-GNN a fair benchmark in the matched-ill-conditioning
regime (brain ν=0.49, vessel ν=0.49, kidney ν=0.45, cartilage ν=0.45).

Algorithm (Hughes 1980 B-bar / de Souza Neto 1996 F-bar, adapted to P1-tet):
    1. Construct the *exact same* P1-tet mesh that solid_tissue_train.py uses,
       via generate_beam_mesh().
    2. Compute the element-wise Jacobian J_e at the (unique) integration point.
    3. Nodal patch averaging: for each node a, compute
            J_node[a] = sum_e (J_e * V_e * I[a in e]) / sum_e (V_e * I[a in e])
       Then per-element J_bar = mean over the 4 incident node-J values.
       (This is the standard nodal F-bar of de Souza Neto / Andrade-Pires).
    4. Replace F_e with F_bar_e = (J_bar_e / J_e)^(1/3) * F_e.
    5. Assemble residual using F_bar with the consistent Neo-Hookean stress.
    6. Use the F-bar-modified material tangent for Newton-Raphson.
       (We deliberately ignore the inter-element coupling that the exact
       consistent F-bar tangent introduces; the symmetric "material" tangent
       on F_bar is sufficient for convergence in this load-stepping regime,
       matches the approach in fem_fbar.py, and keeps the system sparse.)
    7. scipy.sparse direct solver, no GPU needed.

CLI:
    python3 fem_fbar_at_nu049.py --tissue brain --output /tmp/out.json
    python3 fem_fbar_at_nu049.py --dry-run

Output JSON fields:
    tip_displacement_mm, displacement_field, tissue, nu, iteration_count,
    solver, compared_against.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------------ #
# Tissue table — same numbers solid_tissue_train.py uses.            #
# ------------------------------------------------------------------ #
TISSUES = {
    "brain":     {"E": 1000.0,    "nu": 0.49, "rho": 1040.0},
    "kidney":    {"E": 10000.0,   "nu": 0.45, "rho": 1050.0},
    "vessel":    {"E": 400000.0,  "nu": 0.49, "rho": 1050.0},
    "cartilage": {"E": 500000.0,  "nu": 0.45, "rho": 1100.0},  # corrected E3 retrain (ν=0.45)
    # myocardium / bone provided for completeness; not the main targets.
    "myocardium": {"E": 30000.0,  "nu": 0.40, "rho": 1060.0},
}

# DPC-GNN canonical mesh resolution (must match solid_tissue_train.py defaults
# for near-incompressible regime; see lines 186-194 of solid_tissue_train.py).
def _mesh_res_for(nu):
    if nu >= 0.48:
        return 25, 8, 8
    if nu >= 0.45:
        return 20, 6, 6
    return 15, 4, 4


# ------------------------------------------------------------------ #
# 1. Mesh: identical to generate_beam_mesh() in solid_tissue_train.py
#    but in pure numpy.                                              #
# ------------------------------------------------------------------ #
def generate_beam_mesh(Lx, Ly, Lz, nx, ny, nz):
    """P1-linear tetrahedron mesh of a cuboid, fixed at x=0.

    Each (i,j,k) hex cell is split into 5 tetrahedra exactly as in
    solid_tissue_train.py::generate_beam_mesh.
    Returns coords (N,3) and elems (E,4) in int64.
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    z = np.linspace(0.0, Lz, nz + 1)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float64)

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n = [
                    nid(i,   j,   k),
                    nid(i+1, j,   k),
                    nid(i+1, j+1, k),
                    nid(i,   j+1, k),
                    nid(i,   j,   k+1),
                    nid(i+1, j,   k+1),
                    nid(i+1, j+1, k+1),
                    nid(i,   j+1, k+1),
                ]
                tets.extend([
                    [n[0], n[1], n[3], n[4]],
                    [n[1], n[2], n[3], n[6]],
                    [n[1], n[4], n[5], n[6]],
                    [n[3], n[4], n[6], n[7]],
                    [n[1], n[3], n[4], n[6]],
                ])
    elems = np.asarray(tets, dtype=np.int64)
    return coords, elems


# ------------------------------------------------------------------ #
# 2. P1-tet kinematics                                                #
# ------------------------------------------------------------------ #
def _precompute_p1_tet(coords, elems):
    """Precompute dN/dX (constant per tet) and reference volumes.

    For a P1 tet with nodes (x0,x1,x2,x3), let
        Dm = [x1-x0 | x2-x0 | x3-x0]   (3x3)
    Then the gradients of the four barycentric shape functions w.r.t. X are
        dN_a/dX = inv(Dm)^T @ e_a       (a = 1,2,3),  dN_0/dX = -sum_{a>0} dN_a/dX
    Element volume = |det(Dm)| / 6.
    """
    xe = coords[elems]                                # (E, 4, 3)
    Dm = np.stack([xe[:, 1] - xe[:, 0],
                   xe[:, 2] - xe[:, 0],
                   xe[:, 3] - xe[:, 0]], axis=-1)     # (E, 3, 3)
    detDm = np.linalg.det(Dm)
    Vol = np.abs(detDm) / 6.0                         # (E,)
    DmInv = np.linalg.inv(Dm)                         # (E, 3, 3)
    # dN_a/dX for a=1,2,3 are rows of DmInv (cols of DmInv^T):
    #   actually for a P1 tet,  Nabla N_a = DmInv^T @ e_a, so the
    #   gradient vector for node a is DmInv[:, a-1] (column a-1).
    # We'll build (E, 4, 3) dNdX directly.
    dNdX = np.zeros((elems.shape[0], 4, 3), dtype=np.float64)
    # Column-major: DmInv has shape (E,3,3); column a-1 gives Nabla N_a.
    for a in range(1, 4):
        dNdX[:, a, :] = DmInv[:, :, a - 1]
    dNdX[:, 0, :] = -(dNdX[:, 1, :] + dNdX[:, 2, :] + dNdX[:, 3, :])
    return dNdX, Vol


def _compute_F(u_vec, elems, dNdX):
    """F = I + sum_a u_a (x) dN_a/dX, constant on each tet."""
    ue = u_vec.reshape(-1, 3)[elems]                  # (E, 4, 3)
    # F_ij = delta_ij + sum_a u_{a,i} * dN_{a,j}
    F = np.einsum("eai,eaj->eij", ue, dNdX)
    F += np.eye(3)[None, :, :]
    return F                                          # (E, 3, 3)


def _nodal_J_bar(J_e, Vol, elems, n_nodes):
    """Per-element J_bar from nodal patch averaging (standard P1-tet F-bar).

    J_node[a] = sum_{e ∋ a} J_e * V_e / sum_{e ∋ a} V_e
    J_bar_e   = mean of J_node[a] over the 4 nodes of e
    """
    num = np.zeros(n_nodes, dtype=np.float64)
    den = np.zeros(n_nodes, dtype=np.float64)
    for a in range(4):
        np.add.at(num, elems[:, a], J_e * Vol)
        np.add.at(den, elems[:, a], Vol)
    J_node = num / np.maximum(den, 1e-30)
    # mean over the 4 corners
    J_bar = 0.25 * (J_node[elems[:, 0]] + J_node[elems[:, 1]]
                    + J_node[elems[:, 2]] + J_node[elems[:, 3]])
    return J_bar, J_node


# ------------------------------------------------------------------ #
# 3. Assembly with F-bar                                              #
# ------------------------------------------------------------------ #
def assemble_fbar_p1tet(coords, elems, dNdX, Vol, u_vec, mu, lam):
    """Internal force + tangent for compressible Neo-Hookean with nodal F-bar.

    Neo-Hookean energy: psi = 0.5*mu*(I1 - 3) - mu*ln(J) + 0.5*lam*(ln J)^2
    1st-PK stress:       P = mu*(F - F^{-T}) + lam*ln(J) F^{-T}

    F is replaced by F_bar = (J_bar / J)^{1/3} * F, where J_bar is the nodal
    patch-averaged Jacobian.  The same DmInv that gives dN/dX gives B for the
    internal-force virtual-work integral.
    """
    n_nodes = coords.shape[0]
    ndof = 3 * n_nodes
    ne = elems.shape[0]

    F_e = _compute_F(u_vec, elems, dNdX)              # (E, 3, 3)
    J_e = np.linalg.det(F_e)
    J_e_safe = np.maximum(J_e, 1e-10)

    J_bar, _ = _nodal_J_bar(J_e_safe, Vol, elems, n_nodes)
    J_bar_safe = np.maximum(J_bar, 1e-10)

    # F_bar
    scale = (J_bar_safe / J_e_safe) ** (1.0 / 3.0)
    F = scale[:, None, None] * F_e                    # (E, 3, 3)

    J = np.linalg.det(F)
    J_safe = np.maximum(J, 1e-10)
    lnJ = np.log(J_safe)
    Finv = np.linalg.inv(F)
    FinvT = np.transpose(Finv, (0, 2, 1))

    # 1st Piola-Kirchhoff w.r.t. F_bar (we use as if w.r.t. F for the residual,
    # then weight by reference volume; same approximation as fem_fbar.py).
    P = mu * (F - FinvT) + lam * lnJ[:, None, None] * FinvT

    # Internal force: f_e[a, i] = sum_j P_ij * dN_a/dX_j * V_e
    PdN = np.einsum("eij,eaj->eai", P, dNdX) * Vol[:, None, None]   # (E,4,3)

    # DOF map: 12 dofs per tet (3 per node, 4 nodes)
    edof = np.zeros((ne, 12), dtype=np.int64)
    for a in range(4):
        edof[:, 3 * a]     = 3 * elems[:, a]
        edof[:, 3 * a + 1] = 3 * elems[:, a] + 1
        edof[:, 3 * a + 2] = 3 * elems[:, a] + 2

    f_int = np.zeros(ndof)
    np.add.at(f_int, edof.ravel(), PdN.reshape(ne, 12).ravel())

    # -------------------------------------------------------------- #
    # Material tangent (symmetric, sufficient for NR convergence):
    #   dP_ij/dF_kl = mu (delta_ik delta_jl  +  F^{-T}_il F^{-T}_kj)
    #               + lam ln(J) ( - F^{-T}_il F^{-T}_kj )
    #               + lam F^{-T}_ij F^{-T}_kl
    # Then k_e[a*3+i, b*3+k] = sum_{j,l} dP_ij/dF_kl * dN_a/dX_l * dN_b/dX_j * V_e
    # -------------------------------------------------------------- #
    # Build (E,4,4) helper matrices
    BtB = np.einsum("eaj,ebj->eab", dNdX, dNdX)            # B_a · B_b
    C_il = Finv                                            # F^{-T} columns: (Finv)_il = F^{-T}_il ... actually:
    # F^{-T}_ij = (Finv)[j, i]; for clarity use FinvT directly.
    # We want term Fi^T_il * Fi^T_kj contracted with dN_b/dX_j dN_a/dX_l:
    #   sum_j FinvT_kj dN_b/dX_j = (FinvT @ dN_b)_k
    # so build h_a[e, a, i] = sum_j FinvT_ij * dN_a/dX_j
    h_a = np.einsum("eij,eaj->eai", FinvT, dNdX)            # (E,4,3)
    # And g_a[e, a, i] = sum_l FinvT_il * dN_a/dX_l   (same shape)
    # (identical contraction because FinvT_il and FinvT_ij are the same tensor;
    #  the two factors come from different summations.)
    g_a = h_a  # alias

    # Allocate per-element 12x12 stiffness
    ke = np.zeros((ne, 12, 12), dtype=np.float64)

    mu_V    = (mu * Vol)[:, None, None]
    coef_FT = ((mu - lam * lnJ) * Vol)[:, None, None]      # multiplies g_il h_kj
    coef_FF = (lam * Vol)[:, None, None]                   # multiplies F^{-T}_ij F^{-T}_kl

    # First term: mu * delta_ik * (dN_a · dN_b) * V
    for i in range(3):
        ke[:, i::3, i::3] += mu_V * BtB

    # Second + third terms
    # mu*(F^{-T}_il F^{-T}_kj) - lam*ln(J)*F^{-T}_il F^{-T}_kj
    # = (mu - lam ln J) * F^{-T}_il F^{-T}_kj
    # Contracted: (mu - lam lnJ) * g_a[i] * h_b[k]   summed correctly:
    #   term[a,i,b,k] = (mu - lam lnJ) * V * (sum_l F^{-T}_il dN_a/dX_l)
    #                                       * (sum_j F^{-T}_kj dN_b/dX_j)
    #                = (mu - lam lnJ) * V * g_a[a,i] * h_a[b,k]
    for i in range(3):
        gi = g_a[:, :, i]                                  # (E, 4)
        for k in range(3):
            hk = h_a[:, :, k]                              # (E, 4)
            outer = gi[:, :, None] * hk[:, None, :]        # (E, 4, 4)
            ke[:, i::3, k::3] += coef_FT * outer

    # Fourth term: lam * F^{-T}_ij F^{-T}_kl * dN_a/dX_l * dN_b/dX_j * V
    # Define p_a[a, i] = sum_j F^{-T}_ij dN_a/dX_j = h_a[a, i]
    # and    q_b[b, k] = sum_l F^{-T}_kl dN_b/dX_l = h_a[b, k]
    # (same expression because we sum each tensor over the matching axis once)
    for i in range(3):
        pi = h_a[:, :, i]                                  # (E, 4)
        for k in range(3):
            qk = h_a[:, :, k]                              # (E, 4)
            outer = pi[:, :, None] * qk[:, None, :]        # (E, 4, 4)
            ke[:, i::3, k::3] += coef_FF * outer

    # Sparse assembly
    row_idx = np.repeat(edof, 12, axis=1).ravel()
    col_idx = np.tile(edof, (1, 12)).ravel()
    data = ke.ravel()
    K = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(ndof, ndof))

    return f_int, K, {"J_e_min": float(J_e.min()),
                      "J_bar_min": float(J_bar.min()),
                      "J_min": float(J.min())}


# ------------------------------------------------------------------ #
# 4. Body force (gravity) — lumped to nodes via P1 mass               #
# ------------------------------------------------------------------ #
def compute_body_force(coords, elems, Vol, rho, g_val):
    """Distribute -rho * g * V_e onto the 4 corners equally (lumped P1)."""
    ndof = 3 * coords.shape[0]
    f_ext = np.zeros(ndof)
    per_node = -rho * g_val * Vol * 0.25                  # negative y direction
    for a in range(4):
        np.add.at(f_ext, 3 * elems[:, a] + 1, per_node)
    return f_ext


# ------------------------------------------------------------------ #
# 5. Newton-Raphson with load stepping                                #
# ------------------------------------------------------------------ #
def solve_tissue(tissue_name, E, nu, rho, n_steps=5, tol=1e-7, max_nr=30, verbose=True):
    """Returns dict matching the requested output JSON format."""
    # Domain dimensions — MUST match solid_tissue_train.py::train_tissue
    # exactly (lines 172, 182-186): start at 10cm × 2cm × 2cm; if the soft-
    # tissue EB tip exceeds 1cm, shrink ONLY Lx → 3cm.  Ly,Lz stay 2cm.
    # The previous version of this script also shrank Ly,Lz to 1cm, which
    # made the brain ν=0.49 reference run on a *different* mesh than the
    # DPC-GNN prediction it's being compared against (CR-B #1).
    Lx, Ly, Lz = 0.1, 0.02, 0.02
    I_val = Lz * Ly**3 / 12.0
    w_load = rho * 9.81 * Ly * Lz
    delta_exp = w_load * Lx**4 / (8.0 * E * I_val) if E > 0 else 1e-3
    if delta_exp > 0.01:
        Lx = 0.03  # ONLY Lx changes; Ly,Lz remain 0.02 (canonical mesh)

    nx, ny, nz = _mesh_res_for(nu)
    if verbose:
        print(f"\n{'='*70}", flush=True)
        print(f"  {tissue_name}: E={E:g} Pa  nu={nu}  rho={rho}  "
              f"beam {Lx*100:.0f}x{Ly*100:.0f}x{Lz*100:.0f} cm  "
              f"mesh {nx}x{ny}x{nz}", flush=True)

    coords, elems = generate_beam_mesh(Lx, Ly, Lz, nx, ny, nz)
    n_nodes = coords.shape[0]
    ndof = 3 * n_nodes

    dNdX, Vol = _precompute_p1_tet(coords, elems)

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # BCs: fix all 3 DOFs at x=0 nodes
    fixed_nodes = np.where(coords[:, 0] < 1e-10)[0]
    fixed_dofs = np.sort(np.concatenate([3 * fixed_nodes,
                                         3 * fixed_nodes + 1,
                                         3 * fixed_nodes + 2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)

    g_target = 9.81
    f_ext_full = compute_body_force(coords, elems, Vol, rho, g_target)

    u = np.zeros(ndof)
    total_iters = 0
    converged_all = True
    t0 = time.time()

    if verbose:
        print(f"  n_nodes={n_nodes}  n_tets={elems.shape[0]}  "
              f"n_load_steps={n_steps}", flush=True)

    for step in range(1, n_steps + 1):
        lf = step / n_steps
        f_step = lf * f_ext_full
        step_converged = False
        for nit in range(max_nr):
            f_int, K, diag = assemble_fbar_p1tet(coords, elems, dNdX, Vol, u, mu, lam)
            R = f_int - f_step
            R[fixed_dofs] = 0.0
            res = np.linalg.norm(R[free_dofs])
            fnorm = max(np.linalg.norm(f_step[free_dofs]), 1e-12)
            if res / fnorm < tol:
                total_iters += nit
                step_converged = True
                break
            try:
                du_f = spsolve(K[np.ix_(free_dofs, free_dofs)].tocsc(),
                               -R[free_dofs])
            except Exception as ex:
                print(f"  ! spsolve failed at step {step} nit {nit}: {ex}", flush=True)
                converged_all = False
                break
            du = np.zeros(ndof)
            du[free_dofs] = du_f
            # simple line search to keep J>0
            alpha = 1.0
            for _ls in range(8):
                u_try = u + alpha * du
                F_try = _compute_F(u_try, elems, dNdX)
                if np.linalg.det(F_try).min() > 1e-6:
                    break
                alpha *= 0.5
            u = u + alpha * du
            total_iters += 1
        if not step_converged:
            converged_all = False
        if verbose:
            tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
            tip_y = -np.mean(u[3 * tip_nodes + 1])
            print(f"   step {step}/{n_steps}: tip = {tip_y*1e3:8.4f} mm  "
                  f"nr_iters={total_iters}  J_min={diag['J_min']:.4f}", flush=True)

    tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
    tip_disp_mm = float(abs(np.mean(u[3 * tip_nodes + 1])) * 1e3)
    elapsed = time.time() - t0
    if verbose:
        print(f"  done: tip = {tip_disp_mm:.4f} mm  "
              f"converged={converged_all}  iters={total_iters}  "
              f"{elapsed:.1f}s", flush=True)

    disp_field = u.reshape(-1, 3).tolist()

    return {
        "tip_displacement_mm": tip_disp_mm,
        "fem_tip_mm": tip_disp_mm,           # alias for downstream compatibility
        "tip_disp_mm": tip_disp_mm,          # additional alias (used by other scripts)
        "displacement_field": disp_field,
        "tissue": tissue_name,
        "nu": nu,
        "E": E,
        "rho": rho,
        "iteration_count": int(total_iters),
        "n_load_steps": int(n_steps),
        "converged": bool(converged_all),
        "n_nodes": int(n_nodes),
        "n_tets": int(elems.shape[0]),
        "Lx_m": float(Lx),
        "Ly_m": float(Ly),
        "Lz_m": float(Lz),
        "mesh_nx": int(nx),
        "mesh_ny": int(ny),
        "mesh_nz": int(nz),
        "solver": "F-bar P1-linear-tet nodal patch averaging",
        "compared_against": "DPC-GNN canonical mesh (brain/vessel)",
        "compute_time_s": round(elapsed, 2),
    }


# ------------------------------------------------------------------ #
# 6. Self-test (dry-run): single tet, 1 Newton step                   #
# ------------------------------------------------------------------ #
def dry_run():
    """Build a 5-node / 1-tet assembly, do one Newton iteration, print
    residual norm and tangent matrix condition number.
    """
    print("[dry-run] building 5-node / 1-tet test problem...", flush=True)

    # 5 nodes: a single tetrahedron (4 corners) + 1 extra free-floating
    # node to verify the assembly path handles disconnected DOFs.
    # Only node 0 sits on the x=0 clamp plane; nodes 1-3 are free.
    coords = np.array([
        [0.000, 0.0, 0.0],   # 0  fixed (x=0)
        [0.010, 0.0, 0.0],   # 1
        [0.005, 0.01, 0.0],  # 2
        [0.005, 0.0, 0.01],  # 3
        [0.020, 0.02, 0.02], # 4 — extra, not in any element
    ], dtype=np.float64)
    elems = np.array([[0, 1, 2, 3]], dtype=np.int64)

    dNdX, Vol = _precompute_p1_tet(coords, elems)
    print(f"[dry-run] element volume = {Vol[0]:.6e} m^3")

    # Material: brain-ish
    E, nu, rho = 1000.0, 0.49, 1040.0
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    print(f"[dry-run] mu = {mu:.4f}, lambda = {lam:.4f}")

    n_nodes = coords.shape[0]
    ndof = 3 * n_nodes
    u = np.zeros(ndof)

    f_ext = compute_body_force(coords, elems, Vol, rho, 9.81)
    f_int, K, diag = assemble_fbar_p1tet(coords, elems, dNdX, Vol, u, mu, lam)

    print(f"[dry-run] J_e_min={diag['J_e_min']:.6f}, "
          f"J_bar_min={diag['J_bar_min']:.6f}, "
          f"J_min={diag['J_min']:.6f}")
    print(f"[dry-run] |f_ext| = {np.linalg.norm(f_ext):.6e}")
    print(f"[dry-run] |f_int| = {np.linalg.norm(f_int):.6e}")

    fixed_nodes = np.where(coords[:, 0] < 1e-10)[0]
    fixed_dofs = np.sort(np.concatenate([3 * fixed_nodes,
                                         3 * fixed_nodes + 1,
                                         3 * fixed_nodes + 2]))
    # also fix the floating node 4 (no element references it)
    floating = np.array([4])
    fixed_dofs = np.unique(np.concatenate([fixed_dofs,
                                           3 * floating,
                                           3 * floating + 1,
                                           3 * floating + 2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    print(f"[dry-run] fixed dofs: {fixed_dofs.tolist()}")
    print(f"[dry-run] free  dofs: {free_dofs.tolist()}  "
          f"(should be DOFs of nodes 1,2,3)")

    R = f_int - f_ext
    R[fixed_dofs] = 0.0
    print(f"[dry-run] |R_free| (initial)  = {np.linalg.norm(R[free_dofs]):.6e}")

    Kff = K[np.ix_(free_dofs, free_dofs)].toarray()
    cond = np.linalg.cond(Kff)
    print(f"[dry-run] tangent K_ff shape: {Kff.shape}")
    print(f"[dry-run] tangent K_ff cond  : {cond:.4e}")
    print(f"[dry-run] tangent K_ff sym err: "
          f"{np.linalg.norm(Kff - Kff.T) / max(np.linalg.norm(Kff), 1e-12):.3e}")

    # One Newton step
    du_f = spsolve(K[np.ix_(free_dofs, free_dofs)].tocsc(), -R[free_dofs])
    du = np.zeros(ndof)
    du[free_dofs] = du_f
    u_new = u + du

    f_int2, _, _ = assemble_fbar_p1tet(coords, elems, dNdX, Vol, u_new, mu, lam)
    R2 = f_int2 - f_ext
    R2[fixed_dofs] = 0.0
    print(f"[dry-run] |R_free| (after 1 NR step) = "
          f"{np.linalg.norm(R2[free_dofs]):.6e}")
    print(f"[dry-run] |du_free|                  = "
          f"{np.linalg.norm(du_f):.6e}")

    # Print the per-tissue canonical mesh dimensions so CI / reviewers
    # can verify these match solid_tissue_train.py::train_tissue exactly.
    # Brain (soft) → Lx=3cm, Ly=Lz=2cm (only Lx shrinks).
    print("[dry-run] canonical mesh dimensions (per-tissue):")
    print("[dry-run]   tissue       Lx(m)  Ly(m)  Lz(m)   nx ny nz")
    for tname, info in TISSUES.items():
        E_t, nu_t, rho_t = info["E"], info["nu"], info["rho"]
        Lx_t, Ly_t, Lz_t = 0.1, 0.02, 0.02
        I_val_t = Lz_t * Ly_t**3 / 12.0
        w_load_t = rho_t * 9.81 * Ly_t * Lz_t
        delta_t = w_load_t * Lx_t**4 / (8.0 * E_t * I_val_t) if E_t > 0 else 1e-3
        if delta_t > 0.01:
            Lx_t = 0.03  # Ly,Lz unchanged
        nx_t, ny_t, nz_t = _mesh_res_for(nu_t)
        print(f"[dry-run]   {tname:<10s} {Lx_t:6.3f} {Ly_t:6.3f} {Lz_t:6.3f}   "
              f"{nx_t:2d} {ny_t:2d} {nz_t:2d}")
    print("[dry-run] OK")


# ------------------------------------------------------------------ #
# 7. CLI                                                              #
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser(
        description="P1-tet F-bar FEM reference for ν->0.49 (no locking).")
    p.add_argument("--tissue", type=str, default="brain",
                   choices=sorted(TISSUES.keys()),
                   help="Which tissue to solve (brain/vessel/kidney/cartilage).")
    p.add_argument("--output", type=str, default=None,
                   help="Path to write the result JSON.")
    p.add_argument("--n-steps", type=int, default=5,
                   help="Number of load steps (default 5).  NOTE: ν≥0.48 "
                        "auto-bumps to ≥25 (matches fem_fbar.py heuristic) "
                        "because the symmetric F-bar tangent will not "
                        "converge in 5 steps at near-incompressibility.")
    p.add_argument("--tol", type=float, default=1e-7,
                   help="Newton relative residual tolerance (default 1e-7).")
    p.add_argument("--max-nr", type=int, default=30,
                   help="Max Newton iterations per load step (default 30).  "
                        "Auto-bumps to ≥40 when ν≥0.48.")
    p.add_argument("--dry-run", action="store_true",
                   help="Self-test: 5-node/1-tet assembly + 1 NR step, then exit.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress prints.")
    args = p.parse_args()

    if args.dry_run:
        dry_run()
        return 0

    info = TISSUES[args.tissue]

    # Auto-bump load steps + max NR iters for near-incompressible tissues.
    # The symmetric (inconsistent) F-bar tangent in this file converges
    # slowly when ν → 0.5; mirror the heuristic in fem_fbar.py so users
    # don't get spuriously diverged runs with the default --n-steps 5.
    nu = float(info["nu"])
    if nu >= 0.48:
        n_steps_auto = max(args.n_steps, 25)
        if n_steps_auto > args.n_steps:
            print(f"[auto] bumping load steps from {args.n_steps} to "
                  f"{n_steps_auto} for ν={nu}", flush=True)
            args.n_steps = n_steps_auto
        if args.max_nr < 40:
            print(f"[auto] bumping max-nr from {args.max_nr} to 40 for "
                  f"ν={nu}", flush=True)
            args.max_nr = 40

    res = solve_tissue(args.tissue, info["E"], info["nu"], info["rho"],
                       n_steps=args.n_steps, tol=args.tol,
                       max_nr=args.max_nr, verbose=not args.quiet)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(res, f, indent=2)
        if not args.quiet:
            print(f"\n[wrote] {args.output}", flush=True)
    else:
        # Print a slim summary if no output requested
        summary = {k: v for k, v in res.items() if k != "displacement_field"}
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
