#!/usr/bin/env python3
"""
FEM with SAME loading as GNN training (matched gravity from EB expected tips).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import json, os, time, sys

# GNN results (from summary.json)
GNN_RESULTS = {
    "brain":      {"gnn_tip_mm": 26.3012, "eb_tip_mm": 30.9898, "E": 1000,     "nu": 0.49, "rho": 1040, "L": 0.03, "W": 0.01, "H": 0.01},
    "kidney":     {"gnn_tip_mm": 3.4920,  "eb_tip_mm": 3.1288,  "E": 10000,    "nu": 0.45, "rho": 1050, "L": 0.03, "W": 0.01, "H": 0.01},
    "myocardium": {"gnn_tip_mm": 1.1838,  "eb_tip_mm": 1.0529,  "E": 30000,    "nu": 0.40, "rho": 1060, "L": 0.03, "W": 0.01, "H": 0.01},
    "cartilage":  {"gnn_tip_mm": 6.3094,  "eb_tip_mm": 8.0932,  "E": 500000,   "nu": 0.30, "rho": 1100, "L": 0.10, "W": 0.02, "H": 0.02},
    "vessel":     {"gnn_tip_mm": 2.4497,  "eb_tip_mm": 9.6567,  "E": 400000,   "nu": 0.49, "rho": 1050, "L": 0.10, "W": 0.02, "H": 0.02},
    "bone":       {"gnn_tip_mm": 0.5882,  "eb_tip_mm": 0.6990,  "E": 10000000, "nu": 0.30, "rho": 1900, "L": 0.10, "W": 0.02, "H": 0.02},
}

_g = 1.0 / np.sqrt(3.0)
GAUSS_PTS = np.array([[-_g,-_g,-_g],[_g,-_g,-_g],[_g,_g,-_g],[-_g,_g,-_g],
                       [-_g,-_g, _g],[_g,-_g, _g],[_g,_g, _g],[-_g,_g, _g]])

def precompute_shape():
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
    ne = elems.shape[0]
    nn = coords.shape[0]
    ndof = 3 * nn
    xe = coords[elems]
    ue = u_vec.reshape(-1, 3)[elems]
    edof = np.zeros((ne, 24), dtype=np.int64)
    for a in range(8):
        edof[:, 3*a] = 3*elems[:, a]; edof[:, 3*a+1] = 3*elems[:, a]+1; edof[:, 3*a+2] = 3*elems[:, a]+2
    fe_all = np.zeros((ne, 24))
    ke_all = np.zeros((ne, 24, 24))
    
    for g in range(8):
        dN = dN_ref[g]
        J0 = np.einsum('ai,eaj->eij', dN, xe)
        detJ0 = np.linalg.det(J0)
        J0inv = np.linalg.inv(J0)
        dNdX = np.einsum('ai,eij->eaj', dN, J0inv)
        F = np.eye(3)[None,:,:] + np.einsum('eai,eaj->eij', ue, dNdX)
        J = np.linalg.det(F)
        J_safe = np.maximum(J, 1e-10)
        lnJ = np.log(J_safe)
        Finv = np.linalg.inv(F)
        FinvT = Finv.transpose(0, 2, 1)
        P = mu * (F - FinvT) + lam * lnJ[:, None, None] * FinvT
        PdN = np.einsum('eij,eaj->eai', P, dNdX) * detJ0[:, None, None]
        for a in range(8):
            fe_all[:, 3*a:3*a+3] += PdN[:, a, :]
        
        C2 = np.einsum('eaj,ejk->eak', dNdX, Finv)
        BtB = np.einsum('eaj,ebj->eab', dNdX, dNdX)
        coeff2 = ((mu - lam * lnJ) * detJ0)[:, None, None]
        coeff3 = (lam * detJ0)[:, None, None]
        mu_detJ = (mu * detJ0)[:, None, None]
        for i in range(3):
            ke_all[:, i::3, i::3] += mu_detJ * BtB
        for i in range(3):
            Ci = C2[:, :, i:i+1]; CiT = Ci.transpose(0, 2, 1)
            for k in range(3):
                Ck = C2[:, :, k:k+1]; CkT = Ck.transpose(0, 2, 1)
                ke_all[:, i::3, k::3] += coeff2 * (Ck @ CiT) + coeff3 * (Ci @ CkT)
    
    f_int = np.zeros(ndof)
    np.add.at(f_int, edof.ravel(), fe_all.ravel())
    row_idx = np.repeat(edof, 24, axis=1).ravel()
    col_idx = np.tile(edof, (1, 24)).ravel()
    K = sparse.csr_matrix((ke_all.ravel(), (row_idx, col_idx)), shape=(ndof, ndof))
    return f_int, K

def compute_body_force(coords, elems, rho, g_val):
    nn = coords.shape[0]; ndof = 3 * nn; xe = coords[elems]
    f_ext = np.zeros(ndof)
    for g in range(8):
        dN = dN_ref[g]; N = N_ref[g]
        J0 = np.einsum('ai,eaj->eij', dN, xe)
        detJ0 = np.linalg.det(J0)
        for a in range(8):
            np.add.at(f_ext, 3*elems[:, a] + 1, -rho * g_val * N[a] * detJ0)
    return f_ext

def solve_tissue(name, p, g_target):
    E, nu, rho = p["E"], p["nu"], p["rho"]
    Lx, Ly, Lz = p["L"], p["W"], p["H"]
    mu = E / (2*(1+nu)); lam = E * nu / ((1+nu)*(1-2*nu))
    
    if Lx <= 0.05:
        nx, ny, nz = 24, 6, 6
    else:
        nx, ny, nz = 30, 8, 8
    
    print(f"\n{'='*60}", flush=True)
    print(f"{name}: E={E}, nu={nu}, g={g_target:.4f}, mesh={nx}x{ny}x{nz}", flush=True)
    
    coords, elems = generate_hex_mesh(Lx, Ly, Lz, nx, ny, nz)
    nn = coords.shape[0]; ndof = 3 * nn
    
    fixed_nodes = np.where(coords[:, 0] < 1e-10)[0]
    fixed_dofs = np.sort(np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    
    f_ext = compute_body_force(coords, elems, rho, g_target)
    
    # More steps for near-incompressible or large deformation
    eb_tip_m = p["eb_tip_mm"] / 1000.0
    ratio = eb_tip_m / Lx  # expected tip/length ratio
    if nu >= 0.48 and ratio > 0.5:
        n_steps = 30
    elif nu >= 0.48:
        n_steps = 20
    elif ratio > 0.3:
        n_steps = 15
    else:
        n_steps = 10
    
    u = np.zeros(ndof); total_iters = 0; converged = True; t0 = time.time()
    
    for step in range(1, n_steps+1):
        lf = step / n_steps; f_step = lf * f_ext
        for nit in range(30):
            f_int, K = assemble(coords, elems, u, mu, lam)
            R = f_int - f_step; R[fixed_dofs] = 0.0
            res = np.linalg.norm(R[free_dofs])
            fnorm = max(np.linalg.norm(f_step[free_dofs]), 1e-10)
            if res / fnorm < 1e-7:
                total_iters += nit; break
            try:
                du_f = spsolve(K[np.ix_(free_dofs, free_dofs)], -R[free_dofs])
            except:
                converged = False; break
            du = np.zeros(ndof); du[free_dofs] = du_f; u += du; total_iters += 1
        else:
            total_iters += 30
        if not converged: break
        if step % max(1, n_steps//3) == 0 or step == n_steps:
            tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
            tip_y = np.mean(u[3*tip_nodes+1])
            print(f"  Step {step}/{n_steps}: tip={tip_y*1e3:.4f}mm ({time.time()-t0:.1f}s)", flush=True)
    
    tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
    fem_tip = abs(np.mean(u[3*tip_nodes+1])) * 1e3
    elapsed = time.time() - t0
    
    print(f"  FEM tip: {fem_tip:.4f} mm (converged={converged}, iters={total_iters}, {elapsed:.1f}s)", flush=True)
    
    return {
        "fem_tip_mm": round(float(fem_tip), 4),
        "method": "3D-Hex8-NeoHookean-NR",
        "mesh": f"{nx}x{ny}x{nz}",
        "n_nodes": int(nn), "n_elements": int(elems.shape[0]),
        "n_load_steps": n_steps,
        "total_nr_iterations": int(total_iters),
        "converged": converged,
        "gravity_m_s2": round(float(g_target), 4),
        "compute_time_s": round(elapsed, 1),
    }

def main():
    os.makedirs("/root/results", exist_ok=True)
    results = {}
    
    for name, info in GNN_RESULTS.items():
        E, nu, rho = info["E"], info["nu"], info["rho"]
        Lx, Ly, Lz = info["L"], info["W"], info["H"]
        eb_mm = info["eb_tip_mm"]
        gnn_mm = info["gnn_tip_mm"]
        
        # Back-calculate gravity from EB expected tip
        I_val = Ly * Lz**3 / 12
        A_val = Ly * Lz
        eb_m = eb_mm / 1000.0
        g_used = eb_m * 8 * E * I_val / (rho * A_val * Lx**4)
        
        print(f"\n>>> {name}: EB={eb_mm:.4f}mm → g={g_used:.4f} m/s²", flush=True)
        
        try:
            r = solve_tissue(name, info, g_used)
            r["gnn_tip_mm"] = gnn_mm
            r["eb_tip_mm"] = eb_mm
            r["old_error_vs_eb_pct"] = round(abs(gnn_mm - eb_mm) / eb_mm * 100, 2)
            r["new_error_vs_fem_pct"] = round(abs(gnn_mm - r["fem_tip_mm"]) / r["fem_tip_mm"] * 100, 2)
            results[name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}
    
    with open("/root/results/fem_ground_truth.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}", flush=True)
    print("FINAL COMPARISON TABLE", flush=True)
    print(f"{'Tissue':<12} {'GNN':>8} {'FEM':>8} {'EB':>8} {'Old%':>7} {'New%':>7} {'FEM/EB':>7}", flush=True)
    print("-"*60, flush=True)
    for n, r in results.items():
        if "error" in r:
            print(f"{n:<12} ERROR", flush=True)
        else:
            print(f"{n:<12} {r['gnn_tip_mm']:>8.3f} {r['fem_tip_mm']:>8.3f} {r['eb_tip_mm']:>8.3f} "
                  f"{r['old_error_vs_eb_pct']:>6.1f}% {r['new_error_vs_fem_pct']:>6.1f}% {r['fem_tip_mm']/r['eb_tip_mm']:>7.4f}", flush=True)
    
    print(f"\nSaved: /root/results/fem_ground_truth.json", flush=True)

if __name__ == "__main__":
    main()
