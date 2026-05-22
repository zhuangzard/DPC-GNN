#!/usr/bin/env python3
"""
vessel_curriculum_v2.py — Fixed Curriculum Learning for Vessel (ν=0.49)

Root Cause Analysis of v1 failure (mean 18.5% vs baseline 10.3%):
  1. Optimizer RESET at each stage → lost Adam momentum/variance
  2. CosineAnnealingWarmRestarts in Stage 3 → LR spike destroyed converged solution
  3. Stage 1 (f_scale=0.1) learned near-zero deformation, wasting 500 epochs

Fixes in v2:
  - Single optimizer throughout (no reset)
  - Smooth f_scale ramp (linear interpolation, no step jumps)
  - Single cosine LR schedule across all epochs (no warm restarts)
  - Total 2500 epochs with smooth ramp from f_scale=0.3→1.0 over first 800 epochs
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
        J_node_sum = torch.zeros(n_nodes, device=J.device)
        vol_node_sum = torch.zeros(n_nodes, device=J.device)
        for i in range(4):
            J_node_sum.scatter_add_(0, elements[:, i], J_safe * volumes)
            vol_node_sum.scatter_add_(0, elements[:, i], volumes)
        J_node = J_node_sum / vol_node_sum.clamp(min=1e-12)
        J_bar_nodal = (J_node[elements[:,0]] + J_node[elements[:,1]] +
                       J_node[elements[:,2]] + J_node[elements[:,3]]) / 4.0
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

def gravity_potential(nodes_def, masses, g=9.81):
    return -(masses*g*nodes_def[:,1]).sum()

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
# 4. Smooth Curriculum Training (v2)
# ═══════════════════════════════════════════════════

def get_f_scale(epoch, total_epochs, ramp_epochs=800, f_start=0.3, f_end=1.0):
    """Smooth linear ramp from f_start to f_end over ramp_epochs, then hold at f_end."""
    if epoch <= ramp_epochs:
        return f_start + (f_end - f_start) * (epoch / ramp_epochs)
    return f_end


def train_curriculum_v2(seed, device=None, save_dir="/root/results/vessel_curriculum_v2",
                        total_epochs=2500, lr_start=5e-4, lr_end=1e-5,
                        ramp_epochs=800, f_start=0.3, f_end=1.0):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Vessel parameters
    E, nu, rho = 400000.0, 0.49, 1050.0
    Lx, Ly, Lz = 0.1, 0.02, 0.02
    nx, ny, nz = 25, 8, 8
    hidden_dim, n_layers = 64, 6

    I_val = Lz*Ly**3/12.0
    w_load = rho*9.81*Ly*Lz
    delta_exp = w_load*Lx**4/(8.0*E*I_val)
    dscale = min(max(abs(delta_exp), 1e-8), 0.01)

    print(f"\n{'='*70}")
    print(f"  VESSEL CURRICULUM v2 (FIXED) | seed={seed}")
    print(f"  E={E:.0f} Pa | ν={nu} | ρ={rho}")
    print(f"  Mesh: {nx}x{ny}x{nz} | Device: {device}")
    print(f"  Expected tip δ: {delta_exp*1000:.4f} mm | dscale: {dscale:.2e}")
    print(f"  Epochs: {total_epochs} | LR: {lr_start}→{lr_end} (single cosine)")
    print(f"  F_scale ramp: {f_start}→{f_end} over {ramp_epochs} epochs")
    print(f"  FIXES: single optimizer, no warm restarts, smooth ramp")
    print(f"{'='*70}")

    mesh = generate_beam_mesh(Lx,Ly,Lz,nx,ny,nz,device)
    N=mesh["N"]; nodes_ref=mesh["nodes"]; elements=mesh["elements"]
    edge_index=mesh["edge_index"]; fixed_mask=mesh["fixed_mask"]; volumes=mesh["volumes"]
    print(f"  Nodes: {N} | Tets: {elements.shape[0]} | Edges: {edge_index.shape[1]}")

    nmass = torch.zeros(N,device=device)
    for i in range(4): nmass.scatter_add_(0,elements[:,i],volumes*rho/4.0)
    load = torch.zeros(N,3,device=device); load[:,1]=-1.0
    tip_mask = nodes_ref[:,0] > (Lx - 1e-6)

    model = SolidGNN(hidden_dim, n_layers).to(device)
    print(f"  Params: {model.count_params():,}")
    print(f"  ⚡ F-bar enabled (ν=0.49)")

    # ★ SINGLE optimizer — no reset between stages
    opt = optim.Adam(model.parameters(), lr=lr_start, weight_decay=1e-6)
    # ★ SINGLE cosine schedule — no warm restarts
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=lr_end)

    warmup_ep = 50
    best_loss, best_ep, best_state = float('inf'), 0, None
    history = []
    t0 = time.time()

    print(f"\n  {'Ep':>5} | {'Loss':>12} | {'E_el':>12} | {'E_gr':>12} | {'J_min':>7} | {'u_tip(mm)':>10} | {'lr':>9} | {'f_sc':>5}")
    print(f"  {'-'*90}")

    for ep in range(1, total_epochs + 1):
        # ★ Smooth LR warmup
        if ep <= warmup_ep:
            for pg in opt.param_groups:
                pg['lr'] = lr_start * ep / warmup_ep

        # ★ Smooth f_scale ramp
        f_scale = get_f_scale(ep, total_epochs, ramp_epochs, f_start, f_end)

        opt.zero_grad()
        u = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_def = nodes_ref + u
        F_tensor = compute_F(nodes_ref, nodes_def, elements)
        E_el, diag = neo_hookean_energy(F_tensor, volumes, E, nu,
                                         use_fbar=True, elements=elements, n_nodes=N)
        E_gr = gravity_potential(nodes_def, nmass)

        loss = E_el + f_scale * E_gr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep > warmup_ep:
            sched.step()

        lv = loss.item()
        u_tip = u[tip_mask, 1].mean().item() * 1000 if tip_mask.any() else 0

        rec = {"ep": ep, "loss": lv, "E_el": E_el.item(), "E_gr": E_gr.item(),
               "J_min": diag["J_min"], "u_tip_mm": u_tip, "f_scale": f_scale,
               "lr": opt.param_groups[0]['lr']}
        history.append(rec)

        if lv < best_loss:
            best_loss, best_ep = lv, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep <= 3 or ep % 100 == 0 or ep == total_epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {E_el.item():12.6e} | {E_gr.item()*f_scale:12.6e} | "
                  f"{diag['J_min']:7.4f} | {u_tip:10.4f} | {opt.param_groups[0]['lr']:9.2e} | {f_scale:.3f}")

        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}! Reverting to best (ep {best_ep})")
            if best_state:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in opt.param_groups: pg['lr'] *= 0.5

    dt = time.time() - t0

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Final evaluation at full force
    with torch.no_grad():
        u_f = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_f = nodes_ref + u_f
        F_f = compute_F(nodes_ref, nodes_f, elements)
        _, fd = neo_hookean_energy(F_f, volumes, E, nu, use_fbar=True, elements=elements, n_nodes=N)

    tip_y = u_f[tip_mask, 1].mean().item() * 1000
    rel_err = abs(abs(tip_y/1000) - abs(delta_exp)) / abs(delta_exp) * 100 if abs(delta_exp) > 1e-12 else 0.0

    results = {
        "seed": seed,
        "tissue": "vessel",
        "method": "curriculum_v2_smooth",
        "E_Pa": E, "nu": nu, "rho": rho,
        "tip_disp_mm": tip_y,
        "expected_tip_mm": delta_exp * 1000,
        "relative_error_pct": rel_err,
        "J_min": fd["J_min"], "J_max": fd["J_max"],
        "n_inverted": fd["n_inverted"],
        "training_time_s": dt,
        "total_epochs": total_epochs,
        "best_epoch": best_ep,
        "ramp_epochs": ramp_epochs,
        "f_start": f_start, "f_end": f_end,
        "lr_start": lr_start, "lr_end": lr_end,
    }

    print(f"\n  {'='*70}")
    print(f"  ✅ VESSEL CURRICULUM v2 COMPLETE (seed={seed})")
    print(f"  Tip displacement: {tip_y:.4f} mm (expected: {delta_exp*1000:.4f} mm)")
    print(f"  Relative error: {rel_err:.2f}%")
    print(f"  J range: [{fd['J_min']:.6f}, {fd['J_max']:.6f}]")
    print(f"  Training time: {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  Best epoch: {best_ep}")
    print(f"  {'='*70}\n")

    ckpt_dir = f"{save_dir}/seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, f"{ckpt_dir}/model.pt")
    with open(f"{ckpt_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{ckpt_dir}/history.json", "w") as f:
        json.dump(history, f)

    return results


def main():
    parser = argparse.ArgumentParser(description="Vessel Curriculum v2 (Fixed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 42,456,2026")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--save-dir", type=str, default="/root/results/vessel_curriculum_v2")
    parser.add_argument("--epochs", type=int, default=2500)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Vessel Curriculum v2 — FIXED (smooth ramp, single optim)  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB" if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    if args.all_seeds:
        seeds = [42, 123, 456, 789, 2026]
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    all_results = {}
    t_total = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"  [{i+1}/{len(seeds)}] Seed {seed}")
        print(f"{'#'*70}")
        res = train_curriculum_v2(seed, device=args.device, save_dir=args.save_dir,
                                   total_epochs=args.epochs)
        all_results[str(seed)] = res

    total_time = time.time() - t_total

    if len(seeds) > 1:
        errors = [all_results[str(s)]["relative_error_pct"] for s in seeds]
        mean_err = sum(errors) / len(errors)
        print(f"\n\n{'='*80}")
        print(f"  CURRICULUM v2 — {len(seeds)}-SEED SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Seed':>6} | {'Tip(mm)':>9} | {'Exp(mm)':>9} | {'Err%':>7} | {'J_min':>7} | {'BestEp':>7}")
        print(f"  {'-'*60}")

        for seed in seeds:
            r = all_results[str(seed)]
            print(f"  {seed:6d} | {r['tip_disp_mm']:9.4f} | {r['expected_tip_mm']:9.4f} | "
                  f"{r['relative_error_pct']:6.2f}% | {r['J_min']:7.4f} | {r['best_epoch']:7d}")

        success = sum(1 for e in errors if e < 15.0)
        print(f"  {'-'*60}")
        print(f"  Mean error: {mean_err:.2f}%")
        print(f"  Success (<15%): {success}/{len(seeds)}")
        print(f"  Total time: {total_time:.0f}s")

        # Baseline comparison
        baseline = {"42": 8.86, "123": 20.33, "456": 0.68, "789": 8.11, "2026": 13.38}
        print(f"\n  vs BASELINE:")
        print(f"  {'Seed':>6} | {'Base%':>7} | {'CurrV2%':>8} | {'Δ':>8}")
        print(f"  {'-'*40}")
        for seed in seeds:
            if str(seed) in baseline:
                b = baseline[str(seed)]
                c = all_results[str(seed)]["relative_error_pct"]
                print(f"  {seed:6d} | {b:6.2f}% | {c:7.2f}% | {c-b:+7.2f}%")

        summary = {
            "method": "curriculum_v2_smooth",
            "seeds": seeds,
            "results": all_results,
            "mean_error_pct": mean_err,
            "success_rate": success / len(seeds),
            "total_time_s": total_time,
        }
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f"{args.save_dir}/sweep_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to {args.save_dir}/sweep_summary.json")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
