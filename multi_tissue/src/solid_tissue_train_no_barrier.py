#!/usr/bin/env python3
"""
solid_tissue_train_no_barrier.py — Barrier-on/off ablation for EBC.

The abstract and §3.4 of the paper claim that the linear ReLU barrier
W_barrier = k_E · ReLU(eps - J) prevents element inversion ("zero
inverted elements across all 20-seed runs"), and §1 claims that without
the barrier the standard Neo-Hookean energy alone produces "hundreds of
inverted elements per simulation step under aggressive compression".

This script provides the missing ablation experiment (F10 D-14 / F13)
by training the canonical DPC-GNN with the barrier coefficient set to
zero (k_E = 0), keeping everything else identical (architecture,
F-bar, optimiser, mesh, seeds, evaluation).  The per-epoch n_inverted
diagnostic recorded in history.json then quantifies the empirical
barrier benefit.

Recommended tissues for the ablation:
  - cartilage (ν = 0.30) — moderate locking risk, force-on F-bar
  - bone      (ν = 0.30) — stiffest tissue, force-on F-bar
  - vessel    (ν = 0.49) — near-incompressible, auto-on F-bar
These three are the cases where element inversion under compressive
loading is most likely; soft tissues (brain, kidney, myo) rarely invert
even without a barrier.

Output:
  /root/results/{tissue}_no_barrier/best_model.pt
  /root/results/{tissue}_no_barrier/results.json
  /root/results/{tissue}_no_barrier/history.json   (n_inverted per epoch)
"""

import os, sys, time, json, argparse, math
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solid_tissue_train import (
    generate_beam_mesh, compute_F, gravity_potential,
    SolidGNN, TISSUES,
)


def neo_hookean_energy_nobar(F, volumes, E, nu, eps=1e-8, use_fbar=False,
                             elements=None, n_nodes=None,
                             barrier_coef=0.0):
    """
    Neo-Hookean energy with optional ReLU barrier.  Setting
    barrier_coef=0 disables the barrier entirely (k_E·ReLU(ε−J) → 0),
    which is the EBC ablation.  Everything else matches the canonical
    neo_hookean_energy() in solid_tissue_train.py.
    """
    mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    C = torch.bmm(F.transpose(1, 2), F)
    I1 = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]
    J = torch.linalg.det(F)
    J_safe = J.clamp(min=eps)

    if use_fbar and elements is not None and n_nodes is not None:
        J_bar_global = (J_safe * volumes).sum() / volumes.sum()
        I1_bar = J_safe.pow(-2.0 / 3.0) * I1
        psi_dev = 0.5 * mu * (I1_bar - 3.0)
        ln_J_bar = torch.log(J_bar_global.clamp(min=eps))
        psi_vol_total = (-mu * ln_J_bar + 0.5 * lam * ln_J_bar**2) * volumes.sum()
        total_psi = (psi_dev * volumes).sum() + psi_vol_total
    else:
        ln_J = torch.log(J_safe)
        psi = 0.5 * mu * (I1 - 3) - mu * ln_J + 0.5 * lam * ln_J**2
        total_psi = (psi * volumes).sum()

    barrier = torch.relu(-J + eps).sum() * E * barrier_coef
    diag = {"J_min": J.min().item(), "J_max": J.max().item(),
            "J_mean": J.mean().item(),
            "n_inverted": (J < 0).sum().item()}
    return total_psi + barrier, diag


def train_tissue_no_barrier(tissue, E, nu, rho=1000.0, epochs=500, lr=1e-3,
                             hidden_dim=64, n_layers=6,
                             nx=15, ny=4, nz=4, device=None,
                             barrier_coef=0.0,
                             output_root="/root/results"):
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
    print(f"  TISSUE: {tissue.upper()} (BARRIER ABLATION, k_E={barrier_coef}) "
          f"| E={E:.0f} Pa | ν={nu}")
    print(f"  Mesh: {nx_u}x{ny_u}x{nz_u} | Device: {device} | Epochs: {epochs}")
    print(f"  Expected tip δ: {delta_exp*1000:.4f} mm | Scale: {dscale:.2e}")
    print(f"{'='*70}")

    mesh = generate_beam_mesh(Lx, Ly, Lz, nx_u, ny_u, nz_u, device)
    N = mesh["N"]; nodes_ref = mesh["nodes"]; elements = mesh["elements"]
    edge_index = mesh["edge_index"]; fixed_mask = mesh["fixed_mask"]; volumes = mesh["volumes"]
    print(f"  Nodes: {N} | Tets: {elements.shape[0]} | Edges: {edge_index.shape[1]}")

    nmass = torch.zeros(N, device=device)
    for i in range(4):
        nmass.scatter_add_(0, elements[:, i], volumes * rho / 4.0)
    load = torch.zeros(N, 3, device=device); load[:, 1] = -1.0

    model = SolidGNN(hidden_dim, n_layers).to(device)
    print(f"  Params: {model.count_params():,}")

    use_fbar = (nu >= 0.45)
    if use_fbar: print(f"  ⚡ F-bar enabled (ν≥0.45)")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    if use_fbar:
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2, eta_min=lr*0.05)
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    warmup = min(50, epochs // 10)

    best_loss, best_ep, best_state = float('inf'), 0, None
    history = []
    t0 = time.time()

    # Cumulative max inversions across training — load-bearing diagnostic
    # for the abstract claim.
    max_n_inverted = 0
    epoch_first_inversion = None

    print(f"\n  {'Ep':>5} | {'Loss':>12} | {'J_min':>7} | {'n_inv':>7} | {'u_tip(mm)':>10} | {'lr':>9}")
    print(f"  {'-'*78}")

    for ep in range(1, epochs+1):
        if ep <= warmup:
            for pg in opt.param_groups: pg['lr'] = lr * ep / warmup
        opt.zero_grad()
        u = model(nodes_ref, edge_index, E, nu, fixed_mask, load, dscale)
        nodes_def = nodes_ref + u
        F = compute_F(nodes_ref, nodes_def, elements)
        E_el, diag = neo_hookean_energy_nobar(
            F, volumes, E, nu,
            use_fbar=use_fbar, elements=elements, n_nodes=N,
            barrier_coef=barrier_coef,
        )
        E_gr = gravity_potential(nodes_def, nmass)
        loss = E_el + E_gr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep > warmup: sched.step()

        lv = loss.item()
        n_inv = diag["n_inverted"]
        if n_inv > max_n_inverted:
            max_n_inverted = n_inv
            if epoch_first_inversion is None and n_inv > 0:
                epoch_first_inversion = ep
        tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
        u_tip = u[tip_mask, 1].mean().item() * 1000 if tip_mask.any() else 0
        rec = {"ep": ep, "loss": lv, "E_el": E_el.item(), "E_gr": E_gr.item(),
               "J_min": diag["J_min"], "n_inverted": n_inv,
               "u_tip_mm": u_tip, "u_max_mm": u.abs().max().item()*1000}
        history.append(rec)
        if lv < best_loss:
            best_loss, best_ep = lv, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep <= 5 or ep % 25 == 0 or ep == epochs:
            print(f"  {ep:5d} | {lv:12.6e} | {diag['J_min']:7.4f} | {n_inv:7d} | "
                  f"{u_tip:10.4f} | {opt.param_groups[0]['lr']:9.2e}")
        if math.isnan(lv) or math.isinf(lv):
            print(f"  ❌ Diverged at ep {ep} (n_inv={n_inv})")
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
        _, fd = neo_hookean_energy_nobar(F_f, volumes, E, nu,
                                         use_fbar=use_fbar, elements=elements, n_nodes=N,
                                         barrier_coef=barrier_coef)

    tip_mask = nodes_ref[:, 0] > (Lx - 1e-6)
    tip_y = u_f[tip_mask, 1].mean().item() * 1000
    rel_err = abs(abs(tip_y/1000) - abs(delta_exp)) / abs(delta_exp) * 100 if abs(delta_exp) > 1e-12 else 0.0

    results = {"tissue": tissue, "ablation": "no_barrier",
               "barrier_coef": barrier_coef,
               "E_Pa": E, "E_kPa": E/1000, "nu": nu, "rho": rho,
               "beam_Lx_m": Lx, "epochs": epochs,
               "best_epoch": best_ep, "best_loss": best_loss,
               "tip_disp_mm": tip_y, "expected_tip_mm": delta_exp*1000,
               "relative_error_pct": rel_err,
               "J_min_final": fd["J_min"], "J_max_final": fd["J_max"],
               "n_inverted_final": fd["n_inverted"],
               "n_inverted_max_during_training": max_n_inverted,
               "epoch_first_inversion": epoch_first_inversion,
               "training_time_s": dt, "N_nodes": N,
               "N_elements": elements.shape[0], "params": model.count_params()}

    print(f"\n  {'='*70}")
    print(f"  ✅ {tissue.upper()} (barrier_coef={barrier_coef}) COMPLETE")
    print(f"  Best loss: {best_loss:.6e} (epoch {best_ep})")
    print(f"  Tip: {tip_y:.4f} mm (expected: {delta_exp*1000:.4f} mm)  Err: {rel_err:.2f}%")
    print(f"  n_inverted: final={fd['n_inverted']}, max_during_training={max_n_inverted}")
    if epoch_first_inversion:
        print(f"  First inversion at epoch {epoch_first_inversion}")
    print(f"  Training time: {dt:.1f}s ({dt/60:.1f}min)")
    print(f"  {'='*70}\n")

    tag = "no_barrier" if barrier_coef == 0.0 else f"barrier_{barrier_coef:g}"
    ckpt_dir = f"{output_root}/{tissue}_{tag}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if best_state: torch.save(best_state, f"{ckpt_dir}/best_model.pt")
    with open(f"{ckpt_dir}/results.json", "w") as f: json.dump(results, f, indent=2)
    with open(f"{ckpt_dir}/history.json", "w") as f: json.dump(history, f)
    return results


def main():
    parser = argparse.ArgumentParser(description="EBC (barrier) ablation trainer")
    parser.add_argument("--tissue", type=str, default=None)
    parser.add_argument("--E", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--all", action="store_true",
                        help="Train cartilage + bone + vessel (the inversion-prone trio).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--barrier-coef", type=float, default=0.0,
                        help="Barrier coefficient k_E multiplier. 0 = ablation (off), "
                             "100 = canonical DPC-GNN value. Defaults to 0.")
    parser.add_argument("--output-root", type=str, default="/root/results")
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
    print("║  DPC-GNN — EBC (barrier) ablation                           ║")
    print(f"║  barrier_coef = {args.barrier_coef:.3f}                                       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Cartilage / bone / vessel are the inversion-prone trio; soft tissues
    # almost never invert even without a barrier.
    BONE = {"E": 10_000_000.0, "nu": 0.30, "rho": 1900.0}
    TISSUES_LOCAL = dict(TISSUES); TISSUES_LOCAL["bone"] = BONE

    if args.all or args.tissue is None:
        order = ["cartilage", "bone", "vessel"]
        for i, tissue in enumerate(order):
            p = TISSUES_LOCAL[tissue]
            print(f"\n[{i+1}/{len(order)}] {tissue.upper()}")
            train_tissue_no_barrier(tissue=tissue, E=p["E"], nu=p["nu"], rho=p["rho"],
                                    epochs=args.epochs, lr=args.lr,
                                    hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                                    barrier_coef=args.barrier_coef,
                                    output_root=args.output_root)
    else:
        t = args.tissue.lower(); p = TISSUES_LOCAL.get(t, {})
        train_tissue_no_barrier(t, args.E or p.get("E", 10000),
                                args.nu or p.get("nu", 0.30),
                                args.rho or p.get("rho", 1100),
                                args.epochs, args.lr,
                                args.hidden_dim, args.n_layers,
                                barrier_coef=args.barrier_coef,
                                output_root=args.output_root)


if __name__ == "__main__":
    main()
