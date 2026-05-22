#!/usr/bin/env python3
"""
FEM with UNIFORM gravity g=9.81 m/s² — cross-validation with FEniCS.
Standard Neo-Hookean (no F-bar) with incremental loading + adaptive stepping.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import json, os, time

G_UNIFORM = 9.81

TISSUES = {
    "brain":      {"E": 1000,     "nu": 0.49, "rho": 1040, "L": 0.03, "W": 0.01, "H": 0.01, "mesh": (20,6,6)},
    "kidney":     {"E": 10000,    "nu": 0.45, "rho": 1050, "L": 0.03, "W": 0.01, "H": 0.01, "mesh": (20,6,6)},
    "myocardium": {"E": 30000,    "nu": 0.40, "rho": 1060, "L": 0.03, "W": 0.01, "H": 0.01, "mesh": (20,6,6)},
    "cartilage":  {"E": 500000,   "nu": 0.30, "rho": 1100, "L": 0.10, "W": 0.02, "H": 0.02, "mesh": (24,6,6)},
    "vessel":     {"E": 400000,   "nu": 0.49, "rho": 1050, "L": 0.10, "W": 0.02, "H": 0.02, "mesh": (24,6,6)},
    "bone":       {"E": 10000000, "nu": 0.30, "rho": 1900, "L": 0.10, "W": 0.02, "H": 0.02, "mesh": (24,6,6)},
}

GNN_TIPS = {"brain": 26.3012, "kidney": 3.4920, "myocardium": 1.1838,
            "cartilage": 6.3094, "vessel": 2.4497, "bone": 0.5882}
FENICS_TIPS = {"brain": 34.6136, "kidney": 9.6285, "myocardium": 3.6926,
               "cartilage": 7.0446, "vessel": 4.2876, "bone": 0.6101}

_g = 1.0 / np.sqrt(3.0)
GAUSS_PTS = np.array([[-_g,-_g,-_g],[_g,-_g,-_g],[_g,_g,-_g],[-_g,_g,-_g],
                       [-_g,-_g, _g],[_g,-_g, _g],[_g,_g, _g],[-_g,_g, _g]])

def precompute_shape():
    dN_all = np.zeros((8, 8, 3)); N_all = np.zeros((8, 8))
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

def assemble_standard(coords, elems, u_vec, mu, lam):
    """Standard Neo-Hookean assembly without F-bar."""
    ne = elems.shape[0]; nn = coords.shape[0]; ndof = 3*nn
    xe = coords[elems]; ue = u_vec.reshape(-1, 3)[elems]
    edof = np.zeros((ne, 24), dtype=np.int64)
    for a in range(8):
        edof[:, 3*a] = 3*elems[:, a]; edof[:, 3*a+1] = 3*elems[:, a]+1; edof[:, 3*a+2] = 3*elems[:, a]+2
    fe_all = np.zeros((ne, 24)); ke_all = np.zeros((ne, 24, 24))
    
    for g in range(8):
        dN = dN_ref[g]
        J0 = np.einsum('ai,eaj->eij', dN, xe); detJ0 = np.linalg.det(J0)
        J0inv = np.linalg.inv(J0); dNdX = np.einsum('ai,eij->eaj', dN, J0inv)
        F = np.eye(3)[None,:,:] + np.einsum('eai,eaj->eij', ue, dNdX)
        J = np.linalg.det(F)
        J_safe = np.maximum(J, 1e-10); lnJ = np.log(J_safe)
        Finv = np.linalg.inv(F); FinvT = Finv.transpose(0, 2, 1)
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
            Ci = C2[:, :, i:i+1]
            for k in range(3):
                Ck = C2[:, :, k:k+1]
                ke_all[:, i::3, k::3] += coeff2 * (Ck @ Ci.transpose(0,2,1)) + coeff3 * (Ci @ Ck.transpose(0,2,1))
    
    f_int = np.zeros(ndof)
    np.add.at(f_int, edof.ravel(), fe_all.ravel())
    row = np.repeat(edof, 24, axis=1).ravel()
    col = np.tile(edof, (1, 24)).ravel()
    K = sparse.csr_matrix((ke_all.ravel(), (row, col)), shape=(ndof, ndof))
    return f_int, K

def compute_body_force(coords, elems, rho, g_val):
    nn = coords.shape[0]; ndof = 3*nn; xe = coords[elems]
    f_ext = np.zeros(ndof)
    for g in range(8):
        dN = dN_ref[g]; N = N_ref[g]
        J0 = np.einsum('ai,eaj->eij', dN, xe); detJ0 = np.linalg.det(J0)
        for a in range(8):
            np.add.at(f_ext, 3*elems[:, a] + 1, -rho * g_val * N[a] * detJ0)
    return f_ext

def solve_tissue(name, p):
    E, nu, rho = p["E"], p["nu"], p["rho"]
    Lx, Ly, Lz = p["L"], p["W"], p["H"]
    nx, ny, nz = p["mesh"]
    mu = E / (2*(1+nu)); lam = E*nu / ((1+nu)*(1-2*nu))
    
    I_val = Ly * Lz**3 / 12; A_val = Ly * Lz
    eb_tip = rho * G_UNIFORM * A_val * Lx**4 / (8 * E * I_val)
    ratio = eb_tip / Lx
    
    # More steps for harder problems
    if ratio > 1.0: n_steps = 200
    elif ratio > 0.3: n_steps = 40
    elif ratio > 0.1: n_steps = 20
    else: n_steps = 10
    
    print(f"\n{'='*60}", flush=True)
    print(f"{name}: E={E}, nu={nu}, g={G_UNIFORM}, mesh={nx}x{ny}x{nz}", flush=True)
    print(f"  EB={eb_tip*1e3:.2f}mm, ratio={ratio:.3f}, steps={n_steps}", flush=True)
    
    coords, elems = generate_hex_mesh(Lx, Ly, Lz, nx, ny, nz)
    nn = coords.shape[0]; ndof = 3*nn
    fixed_nodes = np.where(coords[:, 0] < 1e-10)[0]
    fixed_dofs = np.sort(np.concatenate([3*fixed_nodes, 3*fixed_nodes+1, 3*fixed_nodes+2]))
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    f_ext = compute_body_force(coords, elems, rho, G_UNIFORM)
    
    u = np.zeros(ndof); total_iters = 0; converged = True; t0 = time.time()
    n_unconverged = 0
    
    for step in range(1, n_steps+1):
        lf = step / n_steps; f_step = lf * f_ext
        step_ok = False
        for nit in range(30):
            f_int, K = assemble_standard(coords, elems, u, mu, lam)
            R = f_int - f_step; R[fixed_dofs] = 0.0
            res = np.linalg.norm(R[free_dofs])
            fnorm = max(np.linalg.norm(f_step[free_dofs]), 1e-10)
            rel_res = res / fnorm
            if np.isnan(res):
                converged = False; break
            if rel_res < 1e-4:
                total_iters += nit; step_ok = True; break
            try:
                du_f = spsolve(K[np.ix_(free_dofs, free_dofs)], -R[free_dofs])
            except:
                converged = False; break
            if np.any(np.isnan(du_f)):
                converged = False; break
            u[free_dofs] += du_f; total_iters += 1
        
        if not converged: 
            print(f"  FAILED at step {step}", flush=True)
            break
        if not step_ok:
            n_unconverged += 1
        
        if step % max(1, n_steps//5) == 0 or step == n_steps:
            tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
            tip_y = np.mean(u[3*tip_nodes+1])
            print(f"  Step {step}/{n_steps}: tip={tip_y*1e3:.4f}mm (res={rel_res:.2e}, {time.time()-t0:.1f}s)", flush=True)
    
    tip_nodes = np.where(coords[:, 0] > Lx - 1e-10)[0]
    fem_tip = abs(np.mean(u[3*tip_nodes+1])) * 1e3
    elapsed = time.time() - t0
    print(f"  RESULT: {fem_tip:.4f}mm (conv={converged}, unconverged_steps={n_unconverged}, iters={total_iters}, {elapsed:.1f}s)", flush=True)
    
    return {
        "tissue": name, "fem_tip_mm": round(float(fem_tip), 4),
        "method": "3D-Hex8-NeoHookean-NR", "mesh": f"{nx}x{ny}x{nz}",
        "n_nodes": int(nn), "n_elements": int(elems.shape[0]),
        "n_load_steps": n_steps, "total_nr_iterations": int(total_iters),
        "converged": converged, "gravity_m_s2": G_UNIFORM,
        "E_Pa": E, "nu": nu, "rho": rho,
        "beam_mm": [Lx*1e3, Ly*1e3, Lz*1e3], "compute_time_s": round(elapsed, 1),
    }

def main():
    os.makedirs("/root/results", exist_ok=True)
    results = {"metadata": {
        "gravity": G_UNIFORM, "purpose": "Cross-validation with FEniCS (g=9.81)",
        "constitutive": "Neo-Hookean compressible", "element": "Hex8 trilinear",
        "note": "Standard assembly (no F-bar). Expect stiffer response for nu=0.49 due to volumetric locking.",
    }, "tissues": {}}
    
    for name, params in TISSUES.items():
        try:
            r = solve_tissue(name, params)
            r["gnn_tip_mm"] = GNN_TIPS.get(name)
            r["fenics_tip_mm"] = FENICS_TIPS.get(name)
            if r["fenics_tip_mm"]:
                r["vs_fenics_pct"] = round(abs(r["fem_tip_mm"] - r["fenics_tip_mm"]) / r["fenics_tip_mm"] * 100, 2)
            if r["gnn_tip_mm"] and r["fenics_tip_mm"]:
                r["gnn_vs_fenics_pct"] = round(abs(r["gnn_tip_mm"] - r["fenics_tip_mm"]) / r["fenics_tip_mm"] * 100, 2)
            results["tissues"][name] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            results["tissues"][name] = {"error": str(e), "traceback": traceback.format_exc()}
    
    with open("/root/results/fem_aligned.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}", flush=True)
    print("CROSS-VALIDATION (all g=9.81 m/s²)", flush=True)
    print(f"{'Tissue':<12} {'NumPyFEM':>10} {'FEniCS':>10} {'GNN':>8} {'NP/FE diff':>10} {'GNN/FE err':>10}", flush=True)
    print("-"*62, flush=True)
    diffs = []
    for n, r in results["tissues"].items():
        if "error" in r:
            print(f"{n:<12} ERROR: {r['error'][:40]}", flush=True)
        else:
            diff = r.get("vs_fenics_pct", 0); gnn_diff = r.get("gnn_vs_fenics_pct", 0)
            diffs.append(diff)
            print(f"{n:<12} {r['fem_tip_mm']:>10.4f} {r.get('fenics_tip_mm',0):>10.4f} {r.get('gnn_tip_mm',0):>8.4f} {diff:>9.1f}% {gnn_diff:>9.1f}%", flush=True)
    if diffs:
        print(f"\nAvg NumPy-FEM vs FEniCS diff: {np.mean(diffs):.1f}%", flush=True)
    print(f"\nSaved: /root/results/fem_aligned.json", flush=True)

if __name__ == "__main__":
    main()
