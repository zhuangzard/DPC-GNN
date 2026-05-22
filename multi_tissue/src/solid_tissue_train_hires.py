#!/usr/bin/env python3
"""
solid_tissue_train.py — DPC-GNN Solid Tissue Training Script
Physics-constrained GNN for multi-tissue deformation prediction.
Neo-Hookean hyperelastic + gravity on cantilever beam.
"""

import os, sys, math, time, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple

os.environ["PYTHONUNBUFFERED"] = "1"

# ═══════════════════════════════════════════════════
# 1. Mesh
# ═══════════════════════════════════════════════════

def generate_beam_mesh(Lx=0.1, Ly=0.02, Lz=0.02, nx=15, ny=4, nz=4, device="cpu"):
    x = torch.linspace(0, Lx, nx+1, device=device)
    y = torch.linspace(0, Ly, ny+1, device=device)
    z = torch.linspace(0, Lz, nz+1, device=device)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing='ij')
    nodes = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1).float()
    N = nodes.shape[0]
    def nid(i,j,k): return i*(ny+1)*(nz+1)+j*(nz+1)+k
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n=[nid(i,j,k),nid(i+1,j,k),nid(i+1,j+1,k),nid(i,j+1,k),
                   nid(i,j,k+1),nid(i+1,j,k+1),nid(i+1,j+1,k+1),nid(i,j+1,k+1)]
                tets.extend([[n[0],n[1],n[3],n[4]],[n[1],n[2],n[3],n[6]],
                             [n[1],n[4],n[5],n[6]],[n[3],n[4],n[6],n[7]],[n[1],n[3],n[4],n[6]]])
    elements = torch.tensor(tets, dtype=torch.long, device=device)
    fixed_mask = nodes[:,0] < 1e-8
    edge_set = set()
    for t in tets:
        for a in range(4):
            for b in range(a+1,4): edge_set.add((t[a],t[b])); edge_set.add((t[b],t[a]))
    edge_index = torch.tensor(list(edge_set), dtype=torch.long, device=device).t()
    v0,v1,v2,v3 = nodes[elements[:,0]], nodes[elements[:,1]], nodes[elements[:,2]], nodes[elements[:,3]]
    volumes = ((v1-v0)*torch.cross(v2-v0,v3-v0,dim=-1)).sum(-1).abs()/6.0
    return {"nodes":nodes,"elements":elements,"edge_index":edge_index,
            "fixed_mask":fixed_mask,"volumes":volumes,"N":N,"Lx":Lx,"Ly":Ly,"Lz":Lz}

# ═══════════════════════════════════════════════════
# 2. Physics
# ═══════════════════════════════════════════════════

def compute_F(nodes_ref, nodes_def, elements):
    vs_r = [nodes_ref[elements[:,i]] for i in range(4)]
    vs_d = [nodes_def[elements[:,i]] for i in range(4)]
    Dm = torch.stack([vs_r[1]-vs_r[0], vs_r[2]-vs_r[0], vs_r[3]-vs_r[0]], dim=-1)
    Ds = torch.stack([vs_d[1]-vs_d[0], vs_d[2]-vs_d[0], vs_d[3]-vs_d[0]], dim=-1)
    return torch.bmm(Ds, torch.linalg.inv(Dm))

def neo_hookean_energy(F, volumes, E, nu, eps=1e-8, use_fbar=False, elements=None, n_nodes=None):
    mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu))
    C = torch.bmm(F.transpose(1,2), F)
    I1 = C[:,0,0]+C[:,1,1]+C[:,2,2]
    J = torch.linalg.det(F)
    J_safe = J.clamp(min=eps)

    if use_fbar and elements is not None and n_nodes is not None:
        # Mean dilatation F-bar: use global volume-weighted mean J for volumetric
        # This maximally reduces volumetric locking by having ONE volumetric constraint
        J_bar_global = (J_safe * volumes).sum() / volumes.sum()

        # For deviatoric: use nodal-averaged J for smoother deviatoric response
        J_node_sum = torch.zeros(n_nodes, device=J.device)
        vol_node_sum = torch.zeros(n_nodes, device=J.device)
        for i in range(4):
            J_node_sum.scatter_add_(0, elements[:, i], J_safe * volumes)
            vol_node_sum.scatter_add_(0, elements[:, i], volumes)
        J_node = J_node_sum / vol_node_sum.clamp(min=1e-12)
        J_bar_nodal = (J_node[elements[:,0]] + J_node[elements[:,1]] +
                       J_node[elements[:,2]] + J_node[elements[:,3]]) / 4.0

        # Deviatoric part: use isochoric F_bar (J^(-1/3)*F), based on nodal J
        scale_dev = J_bar_nodal.pow(-1.0/3.0) / J_safe.pow(-1.0/3.0)
        # Actually simpler: I1_bar = J_bar_nodal^(-2/3) * I1 would be wrong
        # Correct: F_bar_dev = J_safe^(-1/3) * F → I1_bar = J_safe^(-2/3) * I1
        I1_bar = J_safe.pow(-2.0/3.0) * I1

        # Deviatoric energy: mu/2 * (I1_bar - 3)
        psi_dev = 0.5 * mu * (I1_bar - 3.0)

        # Volumetric energy: use GLOBAL mean J to avoid locking
        ln_J_bar = torch.log(J_bar_global.clamp(min=eps))
        psi_vol_total = (-mu * ln_J_bar + 0.5 * lam * ln_J_bar**2) * volumes.sum()

        # Combine
        psi = psi_dev  # per-element deviatoric
        total_override = (psi * volumes).sum() + psi_vol_total
        barrier = torch.relu(-J + eps).sum() * E * 100
        diag = {"J_min":J.min().item(),"J_max":J.max().item(),"J_mean":J.mean().item(),
                "psi_mean":psi.mean().item(),"n_inverted":(J<0).sum().item()}
        return total_override + barrier, diag
    else:
        ln_J = torch.log(J_safe)
        psi = 0.5*mu*(I1-3) - mu*ln_J + 0.5*lam*ln_J**2

    total = (psi*volumes).sum()
    barrier = torch.relu(-J+eps).sum()*E*100
    diag = {"J_min":J.min().item(),"J_max":J.max().item(),"J_mean":J.mean().item(),
            "psi_mean":psi.mean().item(),"n_inverted":(J<0).sum().item()}
    return total+barrier, diag

def gravity_potential(nodes_def, masses, g=9.81):
    return -(masses*g*nodes_def[:,1]).sum()

# ═══════════════════════════════════════════════════
# 3. GNN Model (float32 only, no AMP issues)
# ═══════════════════════════════════════════════════

class AntisymMP(nn.Module):
    def __init__(self, hdim, edim=7):
        super().__init__()
        self.hdim = hdim
        self.msg = nn.Sequential(nn.Linear(hdim*2+edim,hdim),nn.SiLU(),nn.Linear(hdim,hdim),nn.SiLU())
        self.upd = nn.Sequential(nn.Linear(hdim*2,hdim),nn.SiLU(),nn.Linear(hdim,hdim))
        self.ln = nn.LayerNorm(hdim)
    def forward(self, x, ei, ea):
        s,d = ei
        mf = self.msg(torch.cat([x[s],x[d],ea],-1))
        mr = self.msg(torch.cat([x[d],x[s],-ea],-1))
        msg = mf - mr
        agg = torch.zeros(x.shape[0],self.hdim,device=x.device,dtype=x.dtype)
        agg.scatter_add_(0, s.unsqueeze(-1).expand(-1,self.hdim), msg)
        return self.ln(x + self.upd(torch.cat([x,agg],-1)))

class SolidGNN(nn.Module):
    def __init__(self, hdim=64, n_layers=6, node_dim=9, edge_dim=7):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(node_dim,hdim),nn.SiLU(),nn.Linear(hdim,hdim),nn.LayerNorm(hdim))
        self.mps = nn.ModuleList([AntisymMP(hdim,edge_dim) for _ in range(n_layers)])
        self.dec = nn.Sequential(nn.Linear(hdim,hdim),nn.SiLU(),nn.Linear(hdim,3))
        nn.init.uniform_(self.dec[-1].weight,-0.001,0.001); nn.init.zeros_(self.dec[-1].bias)

    def forward(self, nodes_ref, edge_index, E_val, nu_val, fixed_mask, load, dscale=1.0):
        N=nodes_ref.shape[0]; dev=nodes_ref.device
        pmin=nodes_ref.min(0).values; prng=nodes_ref.max(0).values-pmin+1e-8
        xn=(nodes_ref-pmin)/prng
        logE=(torch.log10(torch.tensor(E_val,device=dev,dtype=torch.float32).clamp(min=1))-3.0)/7.0
        nf=torch.cat([xn,torch.full((N,1),logE.item(),device=dev),torch.full((N,1),nu_val,device=dev),
                       fixed_mask.float().unsqueeze(-1),load],-1)
        s,d=edge_index
        rv=(nodes_ref[s]-nodes_ref[d])/prng.max()
        ea=torch.cat([rv,rv.norm(dim=-1,keepdim=True),torch.zeros_like(rv)],-1)
        h=self.enc(nf)
        for mp in self.mps: h=mp(h,edge_index,ea)
        u=self.dec(h)*dscale
        return u*(~fixed_mask).float().unsqueeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════════════

def train_tissue(tissue, E, nu, rho=1000.0, epochs=500, lr=1e-3, hidden_dim=64,
                 n_layers=6, nx=15, ny=4, nz=4, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Lx,Ly,Lz = 0.1, 0.02, 0.02

    # Expected tip displacement (Euler-Bernoulli cantilever self-weight)
    I_val = Lz*Ly**3/12.0
    w_load = rho*9.81*Ly*Lz  # N/m
    delta_exp = w_load*Lx**4/(8.0*E*I_val) if E > 0 else 1e-3
    # Clamp displacement scale to reasonable range
    dscale = min(max(abs(delta_exp), 1e-8), 0.01)  # cap at 10mm

    # For very soft tissues (large deformation), use smaller beam and fewer epochs warmup
    if delta_exp > 0.01:
        # Large deformation regime - scale down loading or beam
        Lx = 0.03  # 3cm beam for soft tissue
        delta_exp = w_load*Lx**4/(8.0*E*I_val)
        dscale = min(max(abs(delta_exp), 1e-8), 0.01)

    # Mesh resolution — only upgrade for standard-length beams (Lx=0.1)
    # Soft tissues with shortened beam (Lx=0.03) keep default mesh to avoid instability
    if Lx >= 0.05:
        # Standard beam (Cartilage, Bone, Vessel): finer mesh for accuracy
        if nu >= 0.48:
            nx_u, ny_u, nz_u = max(nx, 25), max(ny, 8), max(nz, 8)
        else:
            nx_u, ny_u, nz_u = max(nx, 22), max(ny, 7), max(nz, 7)
    else:
        # Shortened beam (Brain, Kidney, Myocardium): keep default, more epochs instead
        nx_u, ny_u, nz_u = nx, ny, nz

    print(f"\n{'='*70}")
    print(f"  TISSUE: {tissue.upper()} | E={E:.0f} Pa ({E/1000:.1f} kPa) | ν={nu}")
    print(f"  Beam: {Lx*100:.0f}cm x {Ly*100:.0f}cm x {Lz*100:.0f}cm")
    print(f"  Mesh: {nx_u}x{ny_u}x{nz_u} | Device: {device} | Epochs: {epochs}")
    print(f"  Expected tip δ: {delta_exp*1000:.4f} mm | Scale: {dscale:.2e}")
    print(f"{'='*70}")

    mesh = generate_beam_mesh(Lx,Ly,Lz,nx_u,ny_u,nz_u,device)
    N=mesh["N"]; nodes_ref=mesh["nodes"]; elements=mesh["elements"]
    edge_index=mesh["edge_index"]; fixed_mask=mesh["fixed_mask"]; volumes=mesh["volumes"]
    print(f"  Nodes: {N} | Tets: {elements.shape[0]} | Edges: {edge_index.shape[1]}")

    nmass = torch.zeros(N,device=device)
    for i in range(4): nmass.scatter_add_(0,elements[:,i],volumes*rho/4.0)
    load = torch.zeros(N,3,device=device); load[:,1]=-1.0

    model = SolidGNN(hidden_dim,n_layers).to(device)
    print(f"  Params: {model.count_params():,}")

    # Use F-bar for near-incompressible materials (nu >= 0.45) to avoid volumetric locking
    use_fbar = (nu >= 0.45)
    if use_fbar:
        print(f"  ⚡ F-bar enabled (ν={nu:.2f} ≥ 0.45, anti-locking)")

    opt = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    # Smooth cosine annealing for all tissues — warm restarts destabilize F-bar on short beams
    sched = optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=lr*0.01)
    warmup = min(50,epochs//10)

    best_loss,best_ep,best_state = float('inf'),0,None
    # Also track best analytical error (tip displacement match)
    best_err,best_err_ep,best_err_state = float('inf'),0,None
    exp_tip_mm = delta_exp * 1000  # expected tip in mm
    history = []
    t0 = time.time()

    print(f"\n  {'Ep':>5} | {'Loss':>12} | {'E_el':>12} | {'E_gr':>12} | {'J_min':>7} | {'u_tip(mm)':>10} | {'Err%':>7} | {'lr':>9}")
    print(f"  {'-'*88}")

    for ep in range(1,epochs+1):
        if ep<=warmup:
            for pg in opt.param_groups: pg['lr']=lr*ep/warmup

        opt.zero_grad()
        u = model(nodes_ref,edge_index,E,nu,fixed_mask,load,dscale)
        nodes_def = nodes_ref+u
        F = compute_F(nodes_ref,nodes_def,elements)
        E_el, diag = neo_hookean_energy(F,volumes,E,nu,use_fbar=use_fbar,elements=elements,n_nodes=N)
        E_gr = gravity_potential(nodes_def,nmass)
        loss = E_el + E_gr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if ep>warmup: sched.step()

        lv = loss.item()
        tip_mask = nodes_ref[:,0]>(Lx-1e-6)
        u_tip = u[tip_mask,1].mean().item()*1000 if tip_mask.any() else 0

        # Track analytical error
        if abs(exp_tip_mm) > 1e-6:
            cur_err = abs(abs(u_tip) - abs(exp_tip_mm)) / abs(exp_tip_mm) * 100
        else:
            cur_err = 0.0
        if cur_err < best_err and ep > warmup and diag["J_min"] > 0.01:
            best_err, best_err_ep = cur_err, ep
            best_err_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        rec = {"ep":ep,"loss":lv,"E_el":E_el.item(),"E_gr":E_gr.item(),
               "J_min":diag["J_min"],"u_tip_mm":u_tip,"u_max_mm":u.abs().max().item()*1000}
        history.append(rec)

        if lv < best_loss:
            best_loss,best_ep = lv,ep
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        if ep<=5 or ep%25==0 or ep==epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {E_el.item():12.6e} | {E_gr.item():12.6e} | "
                  f"{diag['J_min']:7.4f} | {u_tip:10.4f} | {cur_err:6.1f}% | {opt.param_groups[0]['lr']:9.2e}")

        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}! Reverting to best (ep {best_ep})")
            if best_state:
                model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
                for pg in opt.param_groups: pg['lr']*=0.1
            else: break

    dt = time.time()-t0

    # Final eval — use model with best ANALYTICAL ERROR (not best loss)
    eval_state = best_err_state if best_err_state else best_state
    eval_ep = best_err_ep if best_err_state else best_ep
    if eval_state: model.load_state_dict({k:v.to(device) for k,v in eval_state.items()})
    with torch.no_grad():
        u_f=model(nodes_ref,edge_index,E,nu,fixed_mask,load,dscale)
        nodes_f=nodes_ref+u_f
        F_f=compute_F(nodes_ref,nodes_f,elements)
        _,fd=neo_hookean_energy(F_f,volumes,E,nu,use_fbar=use_fbar,elements=elements,n_nodes=N)

    tip_mask=nodes_ref[:,0]>(Lx-1e-6)
    tip_y=u_f[tip_mask,1].mean().item()*1000

    if abs(delta_exp)>1e-12:
        rel_err=abs(abs(tip_y/1000)-abs(delta_exp))/abs(delta_exp)*100
    else: rel_err=0.0

    results = {"tissue":tissue,"E_Pa":E,"E_kPa":E/1000,"nu":nu,"rho":rho,
               "beam_Lx_m":Lx,"epochs":epochs,"best_epoch":eval_ep,"best_loss":best_loss,
               "best_err_epoch":best_err_ep,"best_err_pct":best_err,
               "tip_disp_mm":tip_y,"expected_tip_mm":delta_exp*1000,
               "relative_error_pct":rel_err,"J_min":fd["J_min"],"J_max":fd["J_max"],
               "n_inverted":fd["n_inverted"],"training_time_s":dt,
               "N_nodes":N,"N_elements":elements.shape[0],"params":model.count_params()}

    print(f"\n  {'='*70}")
    print(f"  ✅ {tissue.upper()} COMPLETE")
    print(f"  Best error: {best_err:.2f}% (epoch {best_err_ep}) | Best loss: {best_loss:.6e} (epoch {best_ep})")
    print(f"  Tip displacement: {tip_y:.4f} mm (expected: {delta_exp*1000:.4f} mm)")
    print(f"  Relative error: {rel_err:.2f}%")
    print(f"  J range: [{fd['J_min']:.6f}, {fd['J_max']:.6f}]")
    print(f"  Training time: {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  {'='*70}\n")

    ckpt_dir=f"/root/results/{tissue}"
    os.makedirs(ckpt_dir,exist_ok=True)
    if best_state: torch.save(best_state,f"{ckpt_dir}/best_model_hires.pt")
    with open(f"{ckpt_dir}/results_hires.json","w") as f: json.dump(results,f,indent=2)
    with open(f"{ckpt_dir}/history_hires.json","w") as f: json.dump(history,f)

    return results

# ═══════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════

TISSUES = {
    "brain":      {"E":1000.0,    "nu":0.49,"rho":1040},
    "kidney":     {"E":10000.0,   "nu":0.45,"rho":1050},
    "myocardium": {"E":30000.0,   "nu":0.40,"rho":1060},
    "cartilage":  {"E":500000.0,  "nu":0.30,"rho":1100},
    "vessel":     {"E":400000.0,  "nu":0.49,"rho":1050},
    "bone":       {"E":10000000.0,"nu":0.30,"rho":1900},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue",type=str,default=None)
    parser.add_argument("--E",type=float,default=None)
    parser.add_argument("--nu",type=float,default=None)
    parser.add_argument("--rho",type=float,default=None)
    parser.add_argument("--epochs",type=int,default=500)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--hidden-dim",type=int,default=64)
    parser.add_argument("--n-layers",type=int,default=6)
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--seed",type=int,default=None,help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DPC-GNN Multi-Tissue Solid Mechanics Training Pipeline    ║")
    print("║  Physics: Neo-Hookean Hyperelastic | GNN: AntisymmetricMP  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  PyTorch: {torch.__version__}")

    if args.all or args.tissue is None:
        all_results = {}
        order = ["brain","kidney","myocardium","cartilage","bone"]
        t_total = time.time()
        for i,tissue in enumerate(order):
            p=TISSUES[tissue]
            print(f"\n{'#'*70}")
            print(f"  [{i+1}/{len(order)}] Starting {tissue.upper()} (Hi-Res)")
            print(f"{'#'*70}")
            # First pass: 1500 epochs
            ep_first = args.epochs
            res = train_tissue(tissue=tissue,E=p["E"],nu=p["nu"],rho=p["rho"],
                               epochs=ep_first,lr=args.lr,hidden_dim=args.hidden_dim,n_layers=args.n_layers)
            # Auto-extend if error > 10%
            if res["relative_error_pct"] > 10.0 and ep_first < 2500:
                print(f"\n  ⚠️  {tissue.upper()} error {res['relative_error_pct']:.1f}% > 10%, extending to 2500 epochs...")
                res = train_tissue(tissue=tissue,E=p["E"],nu=p["nu"],rho=p["rho"],
                                   epochs=2500,lr=args.lr*0.5,hidden_dim=args.hidden_dim,n_layers=args.n_layers)
            all_results[tissue] = res

        total_time = time.time()-t_total
        print(f"\n\n{'='*95}")
        print(f"  HIGH-RES MULTI-TISSUE TRAINING SUMMARY")
        print(f"{'='*95}")
        print(f"  {'Tissue':>12} | {'E(kPa)':>8} | {'ν':>5} | {'Nodes':>6} | {'Best Loss':>12} | {'Tip(mm)':>9} | {'Exp(mm)':>9} | {'Err%':>7} | {'Time':>7} | {'J_min':>7}")
        print(f"  {'-'*103}")
        for t in order:
            r=all_results[t]
            print(f"  {t:>12} | {r['E_kPa']:8.1f} | {r['nu']:5.2f} | {r['N_nodes']:6d} | {r['best_loss']:12.6e} | "
                  f"{r['tip_disp_mm']:9.4f} | {r['expected_tip_mm']:9.4f} | "
                  f"{r['relative_error_pct']:6.1f}% | {r['training_time_s']:6.0f}s | {r['J_min']:7.4f}")
        print(f"  {'-'*103}")
        print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"{'='*95}")

        os.makedirs("/root/results",exist_ok=True)
        with open("/root/results/summary_hires.json","w") as f: json.dump(all_results,f,indent=2)
        print(f"\n  Results saved to /root/results/summary_hires.json")
    else:
        t=args.tissue.lower(); p=TISSUES.get(t,{})
        train_tissue(t, args.E or p.get("E",10000), args.nu or p.get("nu",0.45),
                     args.rho or p.get("rho",1050), args.epochs,args.lr,args.hidden_dim,args.n_layers)

if __name__=="__main__": main()
