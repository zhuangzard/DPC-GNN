#!/usr/bin/env python3
"""
3D FEM Ground Truth for DPC-GNN multi-tissue validation.
Compressible Neo-Hookean, Hex8 elements, vectorized assembly.
Newton-Raphson with load stepping.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import json, os, time, sys

TISSUES = {
    "brain":      {"E": 1000,     "nu": 0.49, "rho": 1040, "L": 0.03, "W": 0.01, "H": 0.01},
    "kidney":     {"E": 10000,    "nu": 0.45, "rho": 1050, "L": 0.03, "W": 0.01, "H": 0.01},
    "myocardium": {"E": 30000,    "nu": 0.40, "rho": 1060, "L": 0.03, "W": 0.01, "H": 0.01},
    "cartilage":  {"E": 500000,   "nu": 0.30, "rho": 1100, "L": 0.10, "W": 0.02, "H": 0.02},
    "vessel":     {"E": 400000,   "nu": 0.49, "rho": 1050, "L": 0.10, "W": 0.02, "H": 0.02},
    "bone":       {"E": 10000000, "nu": 0.30, "rho": 1900, "L": 0.10, "W": 0.02, "H": 0.02},
}

_g = 1.0 / np.sqrt(3.0)
GAUSS_PTS = np.array([[-_g,-_g,-_g],[_g,-_g,-_g],[_g,_g,-_g],[-_g,_g,-_g],
                       [-_g,-_g, _g],[_g,-_g, _g],[_g,_g, _g],[-_g,_g, _g]])

def precompute_shape():
    """Precompute shape functions and derivatives at all Gauss points."""
    dN_all = np.zeros((8, 8, 3))
    N_all = np.zeros((8, 8))
    for g in range(8):
        xi, eta, zeta = GAUSS_PTS[g]
        N_all[g] = np.array([
            (1-xi)*(1-eta)*(1-zeta), (1+xi)*(1-eta)*(1-zeta),
            (1+xi)*(1+eta)*(1-zeta), (1-xi)*(1+eta)*(1-zeta),
            (1-xi)*(1-eta)*(1+zeta), (1+xi)*(1-eta)*(1+zeta),
            (1+xi)*(1+eta)*(1+zeta), (1-xi)*(1+eta)*(1+zeta),
        ]) / 8.0
        dN_all[g] = np.array([
            [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
            [ (1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
            [ (1+eta)*(1-zeta),  (1+xi)*(1-zeta), -(1+xi)*(1+eta)],
            [-(1+eta)*(1-zeta),  (1-xi)*(1-zeta), -(1-xi)*(1+eta)],
            [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta),  (1-xi)*(1-eta)],
            [ (1-eta)*(1+zeta), -(1+xi)*(1+zeta),  (1+xi)*(1-eta)],
            [ (1+eta)*(1+zeta),  (1+xi)*(1+zeta),  (1+xi)*(1+eta)],
            [-(1+eta)*(1+zeta),  (1-xi)*(1+zeta),  (1-xi)*(1+eta)],
        ]) / 8.0
    return N_all, dN_all

N_ref, dN_ref = precompute_shape()

def generate_hex_mesh(Lx, Ly, Lz, nx, ny, nz):
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    z = np.linspace(0, Lz, nz+1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    elems = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = i*(ny+1)*(nz+1) + j*(nz+1) + k
                n1 = (i+1)*(ny+1)*(nz+1) + j*(nz+1) + k
                n2 = (i+1)*(ny+1)*(nz+1) + (j+1)*(nz+1) + k
                n3 = i*(ny+1)*(nz+1) + (j+1)*(nz+1) + k
                elems.append([n0, n1, n2, n3, n0+1, n1+1, n2+1, n3+1])
    return coords, np.array(elems, dtype=np.int64)

def assemble(coords, elems, u_vec, mu, lam):
    """Vectorized assembly of internal force + tangent stiffness."""
    ne = elems.shape[0]
    nn = coords.shape[0]
    ndof = 3 * nn
    
    xe = coords[elems]     # (ne, 8, 3)
    ue = u_vec.reshape(-1, 3)[elems]  # (ne, 8, 3)
    
    # DOF mapping
    edof = np.zeros((ne, 24), dtype=np.int64)
    for a in range(8):
        edof[:, 3*a]   = 3*elems[:, a]
        edof[:, 3*a+1] = 3*elems[:, a] + 1
        edof[:, 3*a+2] = 3*elems[:, a] + 2
    
    fe_all = np.zeros((ne, 24))
    ke_all = np.zeros((ne, 24, 24))
    
    for g in range(8):
        dN = dN_ref[g]  # (8, 3)
        
        # Reference Jacobian
        J0 = np.einsum('ai,eaj->eij', dN, xe)      # (ne, 3, 3)
        detJ0 = np.linalg.det(J0)                   # (ne,)
        J0inv = np.linalg.inv(J0)                    # (ne, 3, 3)
        dNdX = np.einsum('ai,eij->eaj', dN, J0inv)  # (ne, 8, 3)
        
        # Deformation gradient
        F = np.eye(3)[None,:,:] + np.einsum('eai,eaj->eij', ue, dNdX)  # (ne,3,3)
        J = np.linalg.det(F)
        J_safe = np.maximum(J, 1e-10)
        lnJ = np.log(J_safe)
        Finv = np.linalg.inv(F)
        FinvT = Finv.transpose(0, 2, 1)
        
        # 1st Piola-Kirchhoff stress
        P = mu * (F - FinvT) + lam * lnJ[:, None, None] * FinvT
        
        # Internal force
        PdN = np.einsum('eij,eaj->eai', P, dNdX) * detJ0[:, None, None]
        for a in range(8):
            fe_all[:, 3*a:3*a+3] += PdN[:, a, :]
        
        # Tangent: use B matrix approach
        # B[e, a, :] = dNdX[e, a, :] mapped through Finv
        # C2[e, a, k] = sum_j dNdX[e,a,j] * Finv[e,j,k]
        C2 = np.einsum('eaj,ejk->eak', dNdX, Finv)  # (ne, 8, 3)
        BtB = np.einsum('eaj,ebj->eab', dNdX, dNdX) # (ne, 8, 8)
        
        coeff2 = ((mu - lam * lnJ) * detJ0)[:, None, None]
        coeff3 = (lam * detJ0)[:, None, None]
        mu_detJ = (mu * detJ0)[:, None, None]
        
        # Term 1: mu * delta_ik * BtB
        for i in range(3):
            ke_all[:, i::3, i::3] += mu_detJ * BtB
        
        # Terms 2+3 combined: loop over 3x3 spatial indices
        for i in range(3):
            Ci = C2[:, :, i:i+1]    # (ne, 8, 1)
            CiT = Ci.transpose(0, 2, 1)  # (ne, 1, 8)
            for k in range(3):
                Ck = C2[:, :, k:k+1]  # (ne, 8, 1)
                CkT = Ck.transpose(0, 2, 1)  # (ne, 1, 8)
                
                # Term 2: (mu-lam*lnJ) * C2[:,a,k] * C2[:,b,i]
                # Term 3: lam * C2[:,a,i] * C2[:,b,k]
                ke_all[:, i::3, k::3] += coeff2 * (Ck @ CiT) + coeff3 * (Ci @ CkT)
    
    # Global assembly
    f_int = np.zeros(ndof)
    np.add.at(f_int, edof.ravel(), fe_all.ravel())
    
    row_idx = np.repeat(edof, 24, axis=1).ravel()
    col_idx = np.tile(edof, (1, 24)).ravel()
    K = sparse.csr_matrix((ke_all.ravel(), (row_idx, col_idx)), shape=(ndof, ndof))
    
    return f_int, K

def compute_body_force(coords, elems, rho, g_val):
    nn = coords.shape[0]
    ndof = 3 * nn
    xe = coords[elems]
    f_ext = np.zeros(ndof)
    
    for g in range(8):
        dN = dN_ref[g]
        N = N_ref[g]
        J0 = np.einsum('ai,eaj->eij', dN, xe)
        detJ0 = np.linalg.det(J0)
        for a in range(8):
            np.add.at(f_ext, 3*elems[:, a] + 1, -rho * g_val * N[a] * detJ0)
    return f_ext

def solve_tissue(name, params):
    E, nu, rho = params["E"], params["nu"], params["rho"]
    Lx, Ly, Lz = params["L"], params["W"], params["H"]
    mu = E / (2*(1+nu))
    lam = E * nu / ((1+nu)*(1-2*nu))
    
    # Use moderate mesh - enough for convergence, not too slow
    if Lx <= 0.05:
        nx, ny, nz = 24, 6, 6
    else:
        nx, ny, nz = 30, 8, 8
    
    print(f"\n{'='*60}", flush=True)
    print(f"{name}: E={E}, nu={nu}, mesh={nx}x{ny}x{nz}", flush=True)
    
    coords, elems = generate_hex_mesh(Lx, Ly, Lz, nx, ny, nz)
    nn, ne = coords.shape[0], elems.shape[0]
    ndof = 3 * nn
    print(f"  Nodes={nn}, Elems={ne}, DOFs={ndof}", flush=True)
    
    fixed_nodes = np.where(coords[:, 0] < 1e-10)[0]
    fixed_dofs = np.sort(np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    
    I_val = Ly * Lz**3 / 12
    A_val = Ly * Lz
    target_tip = 0.20 * Lx
    g_needed = target_tip * 8 * E * I_val / (rho * A_val * Lx**4)
    
    print(f"  target_tip={target_tip*1e3:.2f}mm, g={g_needed:.2f}", flush=True)
    
    f_ext = compute_body_force(coords, elems, rho, g_needed)
    
    n_steps = 20 if nu >= 0.48 else (15 if E < 5000 else 10)
    u = np.zeros(ndof)
    total_iters = 0
    converged = True
    t0 = time.time()
    
    for step in range(1, n_steps+1):
        lf = step / n_steps
        f_step = lf * f_ext
        
        for nit in range(25):
            f_int, K = assemble(coords, elems, u, mu, lam)
            R = f_int - f_step
            R[fixed_dofs] = 0.0
            
            res = np.linalg.norm(R[free_dofs])
            fnorm = max(np.linalg.norm(f_step[free_dofs]), 1e-10)
            
            if res / fnorm < 1e-7:
                total_iters += nit
                break
            
            try:
                du_f = spsolve(K[np.ix_(free_dofs, free_dofs)], -R[free_dofs])
            except:
                converged = False
                break
            
            du = np.zeros(ndof)
            du[free_dofs] = du_f
            u += du
            total_iters += 1
        else:
            total_iters += 25
        
        if not converged:
            break
        
        if step % max(1, n_steps//3) == 0 or step == n_steps:
            tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
            tip_y = np.mean(u[3*tip_nodes+1])
            print(f"  Step {step}/{n_steps}: tip={tip_y*1e3:.4f}mm ({time.time()-t0:.1f}s)", flush=True)
    
    tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
    fem_tip = abs(np.mean(u[3*tip_nodes+1])) * 1e3
    eb_tip = rho * g_needed * A_val * Lx**4 / (8*E*I_val) * 1e3
    
    print(f"  RESULT: FEM={fem_tip:.4f}mm, EB={eb_tip:.4f}mm, ratio={fem_tip/max(eb_tip,1e-10):.4f}", flush=True)
    
    return {
        "fem_tip_mm": round(float(fem_tip), 4),
        "eb_tip_mm": round(float(eb_tip), 4),
        "method": "3D-Hex8-NeoHookean-NR",
        "mesh": f"{nx}x{ny}x{nz}",
        "n_nodes": int(nn), "n_elements": int(ne),
        "n_load_steps": n_steps,
        "total_nr_iterations": int(total_iters),
        "converged": converged,
        "gravity_m_s2": round(float(g_needed), 4),
        "fem_eb_ratio": round(float(fem_tip / max(eb_tip, 1e-10)), 4),
        "compute_time_s": round(time.time()-t0, 1),
    }

def main():
    os.makedirs("/root/results", exist_ok=True)
    results = {}
    
    for name, params in TISSUES.items():
        try:
            results[name] = solve_tissue(name, params)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
    
    with open("/root/results/fem_ground_truth.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'Tissue':<12} {'FEM':>8} {'EB':>8} {'Ratio':>7}", flush=True)
    for n, r in results.items():
        if "error" in r:
            print(f"{n:<12} ERROR", flush=True)
        else:
            print(f"{n:<12} {r['fem_tip_mm']:>8.4f} {r['eb_tip_mm']:>8.4f} {r['fem_eb_ratio']:>7.4f}", flush=True)

if __name__ == "__main__":
    main()
