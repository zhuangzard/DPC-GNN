#!/usr/bin/env python3
"""
vessel_ramp_ablation.py — Method B Hyperparameter Ablation
Three ablations:
  1. Ramp start force amplitude: F_start = 0.1 / 0.3 / 0.5
  2. LR warmup strength: 0% / 5% / 15%
  3. Ramp speed: fast (1000ep) / normal (2000ep) / slow (3000ep)

Each ablation: 3 seeds (42, 456, 789)
Based on vessel_improved_B.py
"""

import os, sys, math, time, json, argparse, random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        J_bar_global = (J_safe * volumes).sum() / volumes.sum()
        I1_bar = J_safe.pow(-2.0/3.0) * I1
        psi_dev = 0.5 * mu * (I1_bar - 3.0)
        ln_J_bar = torch.log(J_bar_global.clamp(min=eps))
        psi_vol_total = (-mu * ln_J_bar + 0.5 * lam * ln_J_bar**2) * volumes.sum()
        psi = psi_dev
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

def gravity_potential(nodes_def, masses, g=9.81, f_scale=1.0):
    return -(masses * g * f_scale * nodes_def[:,1]).sum()

# ═══════════════════════════════════════════════════
# 3. GNN Model
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
# 4. Training with configurable ramp parameters
# ═══════════════════════════════════════════════════

def train_vessel_ramp(seed, epochs=2000, lr=1e-3, hidden_dim=64, n_layers=6,
                      f_start=0.3, f_end=1.0, warmup_boost=0.05, device=None):
    """
    Train vessel GNN with configurable ramp parameters.
    
    Args:
        f_start: starting F_scale (0.1, 0.3, or 0.5)
        f_end: ending F_scale (always 1.0)
        warmup_boost: micro warmup LR boost (0.0 = no warmup, 0.05 = 5%, 0.15 = 15%)
    """
    tissue = "vessel"
    E, nu, rho = 400000.0, 0.49, 1050.0
    Lx, Ly, Lz = 0.03, 0.02, 0.02
    nx, ny, nz = 25, 8, 8

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    I_val = Lz*Ly**3/12.0
    w_load = rho*9.81*Ly*Lz
    delta_exp = w_load*Lx**4/(8.0*E*I_val)
    dscale = min(max(abs(delta_exp), 1e-8), 0.01)

    print(f"\n{'='*70}")
    print(f"  RAMP ABLATION | Seed={seed} | F: {f_start}→{f_end} | warmup={warmup_boost*100:.0f}%")
    print(f"  VESSEL: E={E:.0f} Pa | ν={nu} | Epochs: {epochs}")
    print(f"  Expected tip δ (full): {delta_exp*1000:.4f} mm")
    print(f"{'='*70}")

    mesh = generate_beam_mesh(Lx,Ly,Lz,nx,ny,nz,device)
    N=mesh["N"]; nodes_ref=mesh["nodes"]; elements=mesh["elements"]
    edge_index=mesh["edge_index"]; fixed_mask=mesh["fixed_mask"]; volumes=mesh["volumes"]
    print(f"  Nodes: {N} | Tets: {elements.shape[0]}")

    nmass = torch.zeros(N,device=device)
    for i in range(4): nmass.scatter_add_(0,elements[:,i],volumes*rho/4.0)

    model = SolidGNN(hidden_dim,n_layers).to(device)
    print(f"  Params: {model.count_params():,}")

    opt = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-6)
    warmup_init = 50

    best_loss, best_ep, best_state = float('inf'), 0, None
    history = []
    t0 = time.time()

    micro_warmup_start = -1000
    micro_warmup_duration = 100
    prev_f_scale = f_start

    f_range = f_end - f_start

    for ep in range(1, epochs+1):
        # Linear ramp from f_start to f_end
        f_scale = f_start + f_range * min(ep / epochs, 1.0)

        # Base LR: cosine decay
        if ep <= warmup_init:
            base_lr_now = lr * ep / warmup_init
        else:
            base_lr_now = lr * 0.01 + 0.5 * lr * (1 + math.cos(math.pi * ep / epochs))

        # Check for 0.1 boundary crossing → trigger micro warmup (if warmup_boost > 0)
        if warmup_boost > 0 and int(f_scale * 10) > int(prev_f_scale * 10) and ep > warmup_init:
            micro_warmup_start = ep

        # Apply micro warmup if active
        if warmup_boost > 0 and 0 <= (ep - micro_warmup_start) < micro_warmup_duration:
            progress = (ep - micro_warmup_start) / micro_warmup_duration
            if progress < 0.5:
                boost = warmup_boost * (progress / 0.5)
            else:
                boost = warmup_boost * (1.0 - (progress - 0.5) / 0.5)
            actual_lr = base_lr_now * (1.0 + boost)
        else:
            actual_lr = base_lr_now

        for pg in opt.param_groups: pg['lr'] = actual_lr

        load = torch.zeros(N, 3, device=device)
        load[:, 1] = -f_scale

        opt.zero_grad()
        u = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_def = nodes_ref + u
        F = compute_F(nodes_ref, nodes_def, elements)
        E_el, diag = neo_hookean_energy(F, volumes, E, nu, use_fbar=True, elements=elements, n_nodes=N)
        E_gr = gravity_potential(nodes_def, nmass, f_scale=f_scale)
        loss = E_el + E_gr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        lv = loss.item()
        tip_mask = nodes_ref[:,0] > (Lx - 1e-6)
        u_tip = u[tip_mask, 1].mean().item() * 1000 if tip_mask.any() else 0

        # Evaluate at full load periodically
        if ep % 100 == 0 or ep == epochs:
            with torch.no_grad():
                load_full = torch.zeros(N, 3, device=device)
                load_full[:, 1] = -1.0
                u_eval = model(nodes_ref, edge_index, E, nu, fixed_mask, load_full, dscale)
                nodes_eval = nodes_ref + u_eval
                F_eval = compute_F(nodes_ref, nodes_eval, elements)
                E_el_eval, _ = neo_hookean_energy(F_eval, volumes, E, nu, use_fbar=True, elements=elements, n_nodes=N)
                E_gr_eval = gravity_potential(nodes_eval, nmass, f_scale=1.0)
                eval_loss = (E_el_eval + E_gr_eval).item()
            if eval_loss < best_loss:
                best_loss, best_ep = eval_loss, ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        rec = {"ep":ep,"f_scale":f_scale,"loss":lv,"J_min":diag["J_min"],"u_tip_mm":u_tip}
        history.append(rec)
        prev_f_scale = f_scale

        if ep % 500 == 0 or ep == epochs:
            print(f"  {ep:5d} | F={f_scale:.3f} | loss={lv:12.6e} | J_min={diag['J_min']:7.4f} | tip={u_tip:10.4f}mm | lr={actual_lr:9.2e}")

        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}!")
            if best_state:
                model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
                for pg in opt.param_groups: pg['lr'] *= 0.1
            else: break

    dt = time.time() - t0

    # Final eval at F_scale=1.0
    if best_state: model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    with torch.no_grad():
        load_full = torch.zeros(N, 3, device=device)
        load_full[:, 1] = -1.0
        u_f = model(nodes_ref, edge_index, E, nu, fixed_mask, load_full, dscale)
        nodes_f = nodes_ref + u_f
        F_f = compute_F(nodes_ref, nodes_f, elements)
        _, fd = neo_hookean_energy(F_f, volumes, E, nu, use_fbar=True, elements=elements, n_nodes=N)

    tip_mask = nodes_ref[:,0] > (Lx - 1e-6)
    tip_y = u_f[tip_mask, 1].mean().item() * 1000
    rel_err = abs(abs(tip_y/1000) - abs(delta_exp)) / abs(delta_exp) * 100 if abs(delta_exp) > 1e-12 else 0.0

    results = {"seed":seed,"epochs":epochs,"f_start":f_start,"f_end":f_end,
               "warmup_boost":warmup_boost,
               "best_epoch":best_ep,"best_loss":best_loss,
               "tip_disp_mm":tip_y,"expected_tip_mm":delta_exp*1000,
               "relative_error_pct":rel_err,"J_min":fd["J_min"],"J_max":fd["J_max"],
               "n_inverted":fd["n_inverted"],"training_time_s":dt}

    print(f"\n  ✅ DONE | Seed={seed} | Error={rel_err:.2f}% | Time={dt:.1f}s")
    return results


def run_ablation():
    """Run all three ablations."""
    seeds = [42, 456, 789]
    base_dir = "/root/results/vessel_ramp_ablation"
    os.makedirs(base_dir, exist_ok=True)
    
    all_results = {}
    
    # ═══════════════════════════════════════════════════
    # Ablation 1: Ramp start (0.1 vs 0.3 vs 0.5), 2000ep, warmup=5%
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  ABLATION 1: Ramp Start Force Amplitude")
    print("="*70)
    
    abl1 = {}
    for f_start in [0.1, 0.3, 0.5]:
        key = f"fstart_{f_start}"
        abl1[key] = []
        for seed in seeds:
            r = train_vessel_ramp(seed=seed, epochs=2000, f_start=f_start, 
                                  warmup_boost=0.05)
            abl1[key].append(r)
            
        errors = [r["relative_error_pct"] for r in abl1[key]]
        mean_err = sum(errors) / len(errors)
        print(f"\n  >>> F_start={f_start}: mean={mean_err:.2f}% | per-seed: {[f'{e:.2f}%' for e in errors]}")
    
    all_results["ablation1_ramp_start"] = abl1
    
    # ═══════════════════════════════════════════════════
    # Ablation 2: Warmup boost (0% vs 5% vs 15%), 2000ep, f_start=0.3
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  ABLATION 2: LR Warmup Strength")
    print("="*70)
    
    abl2 = {}
    for warmup in [0.0, 0.05, 0.15]:
        key = f"warmup_{int(warmup*100)}pct"
        abl2[key] = []
        for seed in seeds:
            r = train_vessel_ramp(seed=seed, epochs=2000, f_start=0.3,
                                  warmup_boost=warmup)
            abl2[key].append(r)
            
        errors = [r["relative_error_pct"] for r in abl2[key]]
        mean_err = sum(errors) / len(errors)
        print(f"\n  >>> Warmup={warmup*100:.0f}%: mean={mean_err:.2f}% | per-seed: {[f'{e:.2f}%' for e in errors]}")
    
    all_results["ablation2_warmup"] = abl2
    
    # ═══════════════════════════════════════════════════
    # Ablation 3: Ramp speed (1000 vs 2000 vs 3000 epochs), f_start=0.3, warmup=5%
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  ABLATION 3: Ramp Speed (Total Epochs)")
    print("="*70)
    
    abl3 = {}
    for ep in [1000, 2000, 3000]:
        key = f"epochs_{ep}"
        abl3[key] = []
        for seed in seeds:
            r = train_vessel_ramp(seed=seed, epochs=ep, f_start=0.3,
                                  warmup_boost=0.05)
            abl3[key].append(r)
            
        errors = [r["relative_error_pct"] for r in abl3[key]]
        mean_err = sum(errors) / len(errors)
        print(f"\n  >>> Epochs={ep}: mean={mean_err:.2f}% | per-seed: {[f'{e:.2f}%' for e in errors]}")
    
    all_results["ablation3_ramp_speed"] = abl3
    
    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    summary = {"ablation1": {}, "ablation2": {}, "ablation3": {}}
    
    print("\n  Ablation 1: Ramp Start")
    for key, runs in abl1.items():
        errors = [r["relative_error_pct"] for r in runs]
        m = sum(errors)/len(errors)
        summary["ablation1"][key] = {"mean": m, "errors": errors}
        print(f"    {key}: mean={m:.2f}% | {errors}")
    
    print("\n  Ablation 2: Warmup Strength")
    for key, runs in abl2.items():
        errors = [r["relative_error_pct"] for r in runs]
        m = sum(errors)/len(errors)
        summary["ablation2"][key] = {"mean": m, "errors": errors}
        print(f"    {key}: mean={m:.2f}% | {errors}")
    
    print("\n  Ablation 3: Ramp Speed")
    for key, runs in abl3.items():
        errors = [r["relative_error_pct"] for r in runs]
        m = sum(errors)/len(errors)
        summary["ablation3"][key] = {"mean": m, "errors": errors}
        print(f"    {key}: mean={m:.2f}% | {errors}")
    
    # Save all
    with open(f"{base_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    with open(f"{base_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    
    # Find best config
    best_mean = float('inf')
    best_config = ""
    for abl_name, abl_data in [("abl1", abl1), ("abl2", abl2), ("abl3", abl3)]:
        for key, runs in abl_data.items():
            errors = [r["relative_error_pct"] for r in runs]
            m = sum(errors)/len(errors)
            if m < best_mean:
                best_mean = m
                best_config = f"{abl_name}/{key}"
    
    print(f"\n  🏆 BEST CONFIG: {best_config} | mean={best_mean:.2f}%")
    print(f"  Results saved to {base_dir}/")
    
    return summary, best_config, best_mean


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Vessel Method B — Ramp Hyperparameter Ablation Study      ║")
    print("║  3 ablations × 3 configs × 3 seeds = 27 training runs      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    summary, best_config, best_mean = run_ablation()
    
    print(f"\n{'='*70}")
    print(f"  ALL ABLATIONS COMPLETE")
    print(f"  Best: {best_config} @ {best_mean:.2f}%")
    print(f"{'='*70}")
