#!/usr/bin/env python3
"""
solid_tissue_train_mgn.py — MGN-style baseline for AMP ablation.

This baseline isolates the contribution of Antisymmetric Message Passing
(AMP) by replacing the antisymmetric construction
    m_ij = phi(h_i, h_j, r_ij) - phi(h_j, h_i, -r_ij)
with a standard MeshGraphNets-style directed-edge MP
    m_ij = phi_fwd(h_i, h_j, r_ij)    (no antisymmetry coupling)
The two messages on a single undirected edge are now independent learned
functions; the network has NO structural guarantee that m_ji = -m_ij.

Everything else (mesh, physics loss, hidden dim, F-bar, barrier, optimiser,
LR schedule, seeds, evaluation, output format) is identical to the canonical
DPC-GNN trainer (solid_tissue_train.py).  Side-by-side comparison of the
two trainers' 20-seed outputs answers R1 Minor 4 ("does AMP actually
matter?") with an apples-to-apples ablation: same mesh, same loss, same
hyperparameters, same seeds — only the message-passing primitive differs.

Output:
  /root/results/{tissue}_mgn/best_model.pt
  /root/results/{tissue}_mgn/results.json
  /root/results/{tissue}_mgn/history.json
"""

import os, sys, time, json, argparse, math
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse all shared infrastructure from the canonical trainer.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solid_tissue_train import (
    generate_beam_mesh, compute_F,
    neo_hookean_energy, gravity_potential,
    TISSUES,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. MGN-style directed message passing (the AMP ablation)
# ═══════════════════════════════════════════════════════════════════════

class DirectedMP(nn.Module):
    """
    Standard MeshGraphNets-style directed-edge message passing.
    Two independent edge MLPs (forward and reverse direction); the
    aggregation does NOT subtract the reverse from the forward, so
    Newton's third law is not structurally enforced.

    This is identical in parameter count to two MLPs in a MGN edge-block,
    and differs from AntisymMP only in the absence of the subtraction
    at the message-combination step.
    """
    def __init__(self, hdim, edim=7):
        super().__init__()
        self.hdim = hdim
        self.msg_fwd = nn.Sequential(
            nn.Linear(hdim * 2 + edim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
        )
        self.msg_rev = nn.Sequential(
            nn.Linear(hdim * 2 + edim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.SiLU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(hdim * 2, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim),
        )
        self.ln = nn.LayerNorm(hdim)

    def forward(self, x, ei, ea):
        s, d = ei
        # Independent forward and reverse messages (no antisymmetry).
        mf = self.msg_fwd(torch.cat([x[s], x[d], ea], -1))
        mr = self.msg_rev(torch.cat([x[d], x[s], -ea], -1))
        msg = mf + mr           # standard MGN-style SUM (not subtraction!)
        agg = torch.zeros(x.shape[0], self.hdim, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, s.unsqueeze(-1).expand(-1, self.hdim), msg)
        return self.ln(x + self.upd(torch.cat([x, agg], -1)))


class MGNSolidGNN(nn.Module):
    """Identical to SolidGNN except DirectedMP replaces AntisymMP."""
    def __init__(self, hdim=64, n_layers=6, node_dim=9, edge_dim=7):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(node_dim, hdim), nn.SiLU(),
            nn.Linear(hdim, hdim), nn.LayerNorm(hdim),
        )
        self.mps = nn.ModuleList([DirectedMP(hdim, edge_dim) for _ in range(n_layers)])
        self.dec = nn.Sequential(
            nn.Linear(hdim, hdim), nn.SiLU(),
            nn.Linear(hdim, 3),
        )
        nn.init.uniform_(self.dec[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, nodes_ref, edge_index, E_val, nu_val, fixed_mask, load, dscale=1.0):
        N = nodes_ref.shape[0]
        dev = nodes_ref.device
        pmin = nodes_ref.min(0).values
        prng = nodes_ref.max(0).values - pmin + 1e-8
        xn = (nodes_ref - pmin) / prng
        logE = (torch.log10(torch.tensor(E_val, device=dev, dtype=torch.float32).clamp(min=1)) - 3.0) / 7.0
        nf = torch.cat([
            xn,
            torch.full((N, 1), logE.item(), device=dev),
            torch.full((N, 1), nu_val, device=dev),
            fixed_mask.float().unsqueeze(-1),
            load,
        ], -1)
        s, d = edge_index
        rv = (nodes_ref[s] - nodes_ref[d]) / prng.max()
        ea = torch.cat([rv, rv.norm(dim=-1, keepdim=True), torch.zeros_like(rv)], -1)
        h = self.enc(nf)
        for mp in self.mps:
            h = mp(h, edge_index, ea)
        u = self.dec(h) * dscale
        return u * (~fixed_mask).float().unsqueeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
# 2. Training loop (clone of train_tissue, MGN model substituted)
# ═══════════════════════════════════════════════════════════════════════

def train_tissue_mgn(tissue, E, nu, rho=1000.0, epochs=500, lr=1e-3,
                     hidden_dim=64, n_layers=6,
                     nx=15, ny=4, nz=4, device=None,
                     output_root="/root/results", output_suffix=""):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Lx, Ly, Lz = 0.1, 0.02, 0.02
    I_val = Lz * Ly**3 / 12.0
    w_load = rho * 9.81 * Ly * Lz
    delta_exp = w_load * Lx**4 / (8.0 * E * I_val) if E > 0 else 1e-3
    dscale = min(max(abs(delta_exp), 1e-8), 0.01)

    if delta_exp > 0.01:
        Lx = 0.03
        delta_exp = w_load * Lx**4 / (8.0 * E * I_val)
        dscale = min(max(abs(delta_exp), 1e-8), 0.01)

    if nu >= 0.48:
        nx_u, ny_u, nz_u = max(nx, 25), max(ny, 8), max(nz, 8)
    elif nu >= 0.45:
        nx_u, ny_u, nz_u = max(nx, 20), max(ny, 6), max(nz, 6)
    else:
        nx_u, ny_u, nz_u = nx, ny, nz

    print(f"\n{'='*70}")
    print(f"  TISSUE: {tissue.upper()} (MGN-baseline) | E={E:.0f} Pa ({E/1000:.1f} kPa) | ν={nu}")
    print(f"  Beam: {Lx*100:.0f}cm x {Ly*100:.0f}cm x {Lz*100:.0f}cm")
    print(f"  Mesh: {nx_u}x{ny_u}x{nz_u} | Device: {device} | Epochs: {epochs}")
    print(f"  Expected tip δ: {delta_exp*1000:.4f} mm | Scale: {dscale:.2e}")
    print(f"  AMP: OFF (MGN-style directed MP, two independent edge MLPs)")
    print(f"{'='*70}")

    mesh = generate_beam_mesh(Lx, Ly, Lz, nx_u, ny_u, nz_u, device)
    N = mesh["N"]; nodes_ref = mesh["nodes"]; elements = mesh["elements"]
    edge_index = mesh["edge_index"]; fixed_mask = mesh["fixed_mask"]; volumes = mesh["volumes"]
    print(f"  Nodes: {N} | Tets: {elements.shape[0]} | Edges: {edge_index.shape[1]}")

    nmass = torch.zeros(N, device=device)
    for i in range(4):
        nmass.scatter_add_(0, elements[:, i], volumes * rho / 4.0)
    load = torch.zeros(N, 3, device=device)
    load[:, 1] = -1.0

    model = MGNSolidGNN(hidden_dim, n_layers).to(device)
    print(f"  Params: {model.count_params():,}  (~2× SolidGNN edge-MLP params)")

    use_fbar = (nu >= 0.45)
    if use_fbar:
        print(f"  ⚡ F-bar enabled (ν={nu:.2f} ≥ 0.45, anti-locking)")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if use_fbar:
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2, eta_min=lr*0.05)
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    warmup = min(50, epochs // 10)

    best_loss, best_ep, best_state = float('inf'), 0, None
    history = []
    t0 = time.time()

    print(f"\n  {'Ep':>5} | {'Loss':>12} | {'E_el':>12} | {'E_gr':>12} | {'J_min':>7} | {'u_tip(mm)':>10} | {'lr':>9}")
    print(f"  {'-'*78}")

    for ep in range(1, epochs+1):
        if ep <= warmup:
            for pg in opt.param_groups: pg['lr'] = lr * ep / warmup
        opt.zero_grad()
        u = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_def = nodes_ref + u
        F = compute_F(nodes_ref, nodes_def, elements)
        E_el, diag = neo_hookean_energy(F, volumes, E, nu, use_fbar=use_fbar, elements=elements, n_nodes=N)
        E_gr = gravity_potential(nodes_def, nmass)
        loss = E_el + E_gr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep > warmup: sched.step()
        lv = loss.item()
        tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
        u_tip = u[tip_mask, 1].mean().item()*1000 if tip_mask.any() else 0
        rec = {"ep": ep, "loss": lv, "E_el": E_el.item(), "E_gr": E_gr.item(),
               "J_min": diag["J_min"], "u_tip_mm": u_tip,
               "u_max_mm": u.abs().max().item()*1000}
        history.append(rec)
        if lv < best_loss:
            best_loss, best_ep = lv, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep <= 5 or ep % 25 == 0 or ep == epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {E_el.item():12.6e} | {E_gr.item():12.6e} | "
                  f"{diag['J_min']:7.4f} | {u_tip:10.4f} | {opt.param_groups[0]['lr']:9.2e}")
        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep}! Reverting to best (ep {best_ep})")
            if best_state:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                for pg in opt.param_groups: pg['lr'] *= 0.1
            else:
                break

    dt = time.time() - t0

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    with torch.no_grad():
        u_f = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_f = nodes_ref + u_f
        F_f = compute_F(nodes_ref, nodes_f, elements)
        _, fd = neo_hookean_energy(F_f, volumes, E, nu, use_fbar=use_fbar, elements=elements, n_nodes=N)

    tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
    tip_y = u_f[tip_mask, 1].mean().item() * 1000
    rel_err = abs(abs(tip_y/1000) - abs(delta_exp)) / abs(delta_exp) * 100 if abs(delta_exp) > 1e-12 else 0.0

    results = {"tissue": tissue, "ablation": "mgn_baseline", "E_Pa": E, "E_kPa": E/1000,
               "nu": nu, "rho": rho, "beam_Lx_m": Lx, "epochs": epochs,
               "best_epoch": best_ep, "best_loss": best_loss,
               "tip_disp_mm": tip_y, "expected_tip_mm": delta_exp*1000,
               "relative_error_pct": rel_err, "J_min": fd["J_min"],
               "J_max": fd["J_max"], "n_inverted": fd["n_inverted"],
               "training_time_s": dt, "N_nodes": N,
               "N_elements": elements.shape[0], "params": model.count_params()}

    print(f"\n  {'='*70}")
    print(f"  ✅ {tissue.upper()} (MGN-baseline) COMPLETE")
    print(f"  Best loss: {best_loss:.6e} (epoch {best_ep})")
    print(f"  Tip: {tip_y:.4f} mm (expected: {delta_exp*1000:.4f} mm)  Err: {rel_err:.2f}%")
    print(f"  Training time: {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  {'='*70}\n")

    # Allow callers (e.g. sweep_udmp_fairness.py) to write to a unique
    # per-cell directory so concurrent invocations don't clobber each
    # other's results.json.  Empty suffix == legacy "<tissue>_mgn" path.
    suffix = f"_{output_suffix}" if output_suffix else ""
    ckpt_dir = f"{output_root}/{tissue}_mgn{suffix}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state: torch.save(best_state, f"{ckpt_dir}/best_model.pt")
    with open(f"{ckpt_dir}/results.json", "w") as f: json.dump(results, f, indent=2)
    with open(f"{ckpt_dir}/history.json", "w") as f: json.dump(history, f)
    return results


def main():
    parser = argparse.ArgumentParser(description="MGN-baseline trainer for AMP ablation")
    parser.add_argument("--tissue", type=str, default=None,
                        choices=list(TISSUES.keys()) + [None])
    parser.add_argument("--E", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--all", action="store_true",
                        help="Train all canonical tissues sequentially.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="/root/results")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Optional suffix appended to <tissue>_mgn so "
                             "concurrent sweep cells (different lr/lam/seed) "
                             "do not clobber the same results.json.")
    args = parser.parse_args()

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DPC-GNN — MGN-style Baseline (AMP ablation)                ║")
    print("║  Replaces AntisymMP with directed-edge MP (no antisymmetry) ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

    if args.all or args.tissue is None:
        order = ["brain", "kidney", "myocardium", "vessel", "cartilage"]
        for i, tissue in enumerate(order):
            p = TISSUES[tissue]
            print(f"\n[{i+1}/{len(order)}] {tissue.upper()}")
            train_tissue_mgn(tissue=tissue, E=p["E"], nu=p["nu"], rho=p["rho"],
                             epochs=args.epochs, lr=args.lr,
                             hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                             output_root=args.output_root,
                             output_suffix=args.output_suffix)
    else:
        t = args.tissue.lower(); p = TISSUES.get(t, {})
        train_tissue_mgn(t, args.E or p.get("E", 10000), args.nu or p.get("nu", 0.45),
                         args.rho or p.get("rho", 1050), args.epochs, args.lr,
                         args.hidden_dim, args.n_layers,
                         output_root=args.output_root,
                         output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()
