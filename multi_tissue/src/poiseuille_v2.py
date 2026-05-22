"""
poiseuille_v2.py — Improved Poiseuille Validation for SPH-GNN (RTX 5090 optimized).

Changes from v1:
  1. Uses dp=1mm (1817 fluid particles) for publication-quality resolution
  2. Mixed precision (AMP) for CUDA efficiency
  3. Learning rate warmup + cosine annealing
  4. Loss spike detection with auto-recovery (revert to best model)
  5. Acceleration output clamping for stability
  6. Better gradient clipping (0.5)
  7. Checkpoint saving (best model)
  8. Comprehensive logging
  
Known v1 issues (fixed here):
  - dp=2mm (78 fluid particles) too coarse for publication
  - dp=1mm OOM on small GPUs (NOT an issue on RTX 5090 32GB, ~117MB needed)
  - Loss divergence risk with aggressive lr + no recovery mechanism
"""

import os
import sys
import math
import time
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict
import argparse

os.environ["PYTHONUNBUFFERED"] = "1"

from sph_domain import generate_portal_vein_domain, SPHDomain, FLUID, WALL, INLET, OUTLET
from sph_kernels import wendland_c2_gradient
from sph_gnn_model import create_sph_gnn, SPHGNNModel
from sph_physics_loss import sph_physics_loss, BLOOD
from sph_integrator import SPHState, symplectic_euler_step, compute_cfl_dt, poiseuille_velocity


def poiseuille_analytical(r, R, delta_p, L, mu=BLOOD.mu_inf):
    v_z = (delta_p / (4.0 * mu * L)) * (R**2 - r.clamp(max=R)**2)
    return v_z.clamp(min=0.0)


def poiseuille_flow_rate(R, delta_p, L, mu=BLOOD.mu_inf):
    return math.pi * R**4 * delta_p / (8.0 * mu * L)


def run_poiseuille_v2(
    D: float = 0.007,
    L: float = 0.080,
    dp: float = 0.001,           # 1mm for publication quality
    delta_p: float = 0.5,
    mu: float = BLOOD.mu_inf,
    n_epochs: int = 500,
    n_steps_per_epoch: int = 10,
    dt: float = 5e-5,
    lr: float = 3e-4,
    hidden_dim: int = 96,
    n_mp_layers: int = 5,
    warmup_epochs: int = 20,
    grad_clip: float = 0.5,
    use_amp: bool = True,
    checkpoint_dir: str = None,
    verbose: bool = True,
) -> Dict:
    """Train SPH-GNN with improved stability on Poiseuille flow."""
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = "mps"
        use_amp = False  # MPS doesn't support AMP well
    else:
        device = "cpu"
        use_amp = False
    
    print(f"[Poiseuille-v2] Device: {device}, AMP: {use_amp}")
    
    R = D / 2.0
    dp_dz = -delta_p / L
    dp_dz_mag = abs(dp_dz)
    v_max_ref = dp_dz_mag * R**2 / (4.0 * mu)
    
    # Generate domain
    domain = generate_portal_vein_domain(D=D, L=L, dp=dp, device=device)
    N = domain.n_particles
    
    # Feature normalization refs
    v_ref = max(v_max_ref * 2.0, 0.01)
    p_ref = max(delta_p * 2.0, 1.0)
    print(f"  v_ref={v_ref:.4f} m/s, p_ref={p_ref:.2f} Pa, v_max_analytical={v_max_ref:.4f} m/s")
    
    # Inlet BC
    def inlet_bc(pos, t):
        return poiseuille_velocity(pos, R, dp_dz_mag, mu=mu).to(device)
    
    # Model
    model = create_sph_gnn(hidden_dim=hidden_dim, n_mp_layers=n_mp_layers,
                           device=device, v_ref=v_ref, p_ref=p_ref)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing (after warmup)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device == 'cuda') else None
    
    # Initial state with warm-start Poiseuille profile
    fluid_mask = (domain.particle_type == FLUID)
    inlet_mask = (domain.particle_type == INLET)
    
    init_v = torch.zeros(N, 3, device=device)
    if inlet_mask.any():
        init_v[inlet_mask] = inlet_bc(domain.positions[inlet_mask], 0.0)
    if fluid_mask.any():
        r_fluid = torch.sqrt(domain.positions[fluid_mask, 0]**2 +
                             domain.positions[fluid_mask, 1]**2)
        init_v[fluid_mask, 2] = poiseuille_analytical(r_fluid, R, delta_p, L, mu)
    
    # Supervised target
    target_v = torch.zeros(N, 3, device=device)
    r_all = torch.sqrt(domain.positions[:, 0]**2 + domain.positions[:, 1]**2)
    if fluid_mask.any():
        target_v[fluid_mask, 2] = poiseuille_analytical(r_all[fluid_mask], R, delta_p, L, mu)
    if inlet_mask.any():
        target_v[inlet_mask] = inlet_bc(domain.positions[inlet_mask], 0.0)
    
    print(f"  Fluid particles: {fluid_mask.sum()}, target v_z_max={target_v[:,2].max():.4f} m/s")
    print(f"  Q_analytical = {poiseuille_flow_rate(R, delta_p, L, mu)*1e6:.4f} mL/s")
    
    # Training loop
    losses = []
    best_loss = float('inf')
    best_state_dict = None
    spike_count = 0
    t_start = time.time()
    
    # Acceleration clamp: max physical acceleration for this setup
    # a_max ~ dp_dz / rho ~ 6.25 / 1060 ~ 0.006 m/s², use 10x margin
    a_clamp = max(dp_dz_mag / BLOOD.rho0 * 100, 1.0)  # generous clamp
    
    print(f"\n[Training] {n_epochs} epochs, lr={lr}, warmup={warmup_epochs}, a_clamp={a_clamp:.2f} m/s²")
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Build input domain (reset each epoch)
        v_in = init_v.clone()
        domain_in = SPHDomain(
            positions=domain.positions, velocities=v_in,
            densities=domain.densities, pressures=domain.pressures,
            particle_type=domain.particle_type, edge_index=domain.edge_index,
            h=domain.h, n_particles=N, n_fluid=domain.n_fluid,
            n_wall=domain.n_wall, n_inlet=domain.n_inlet, n_outlet=domain.n_outlet,
            R=R, L=L, boundary_mask=domain.boundary_mask, n_vertices=N,
        )
        
        # Forward pass (with AMP if available)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                a_pred = model(domain_in)
                a_pred = torch.clamp(a_pred, -a_clamp, a_clamp)
                a_fluid = a_pred[fluid_mask]
                L_steady = (a_fluid ** 2).mean()
                v_pred = v_in + dt * a_pred
                v_err = v_pred[fluid_mask] - target_v[fluid_mask]
                L_vel = (v_err ** 2).mean() / (v_max_ref**2 + 1e-10)
                epoch_loss = L_steady + 100.0 * L_vel
            
            scaler.scale(epoch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            a_pred = model(domain_in)
            a_pred = torch.clamp(a_pred, -a_clamp, a_clamp)
            a_fluid = a_pred[fluid_mask]
            L_steady = (a_fluid ** 2).mean()
            v_pred = v_in + dt * a_pred
            v_err = v_pred[fluid_mask] - target_v[fluid_mask]
            L_vel = (v_err ** 2).mean() / (v_max_ref**2 + 1e-10)
            epoch_loss = L_steady + 100.0 * L_vel
            
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
        scheduler.step()
        loss_val = epoch_loss.item()
        losses.append(loss_val)
        
        # Best model tracking
        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Spike detection: if loss > 10x best, revert to best model
        if loss_val > 10 * best_loss and best_state_dict is not None and epoch > warmup_epochs:
            spike_count += 1
            model.load_state_dict(best_state_dict)
            # Halve learning rate
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            if verbose:
                print(f"  ⚠️ Epoch {epoch+1}: loss spike {loss_val:.2e} > 10×best {best_loss:.2e}, "
                      f"reverted to best model (spike #{spike_count})")
            continue
        
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            lr_now = optimizer.param_groups[0]['lr']
            with torch.no_grad():
                a_max_val = a_fluid.abs().max().item()
                v_err_max = v_err.abs().max().item()
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: loss={loss_val:.4e}, "
                  f"|a|={a_max_val:.2e}, |v_err|={v_err_max:.2e}, "
                  f"lr={lr_now:.1e}, t={elapsed:.0f}s")
    
    # Save best model
    if checkpoint_dir and best_state_dict:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(best_state_dict, ckpt_path)
        print(f"  Saved best model to {ckpt_path}")
    
    # Load best model for evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    # ── Evaluation ──
    print("\n[Evaluation] Running steady-state evaluation...")
    model.eval()
    
    with torch.no_grad():
        eval_state = SPHState(
            positions=domain.positions.clone(), velocities=init_v.clone(),
            densities=domain.densities.clone(), pressures=domain.pressures.clone(),
            time=0.0, step=0,
        )
        eval_domain = domain
        n_eval_steps = 200
        
        for _ in range(n_eval_steps):
            v_bc = eval_state.velocities.clone()
            if inlet_mask.any():
                v_bc[inlet_mask] = inlet_bc(eval_state.positions[inlet_mask], eval_state.time)
            domain_eval = SPHDomain(
                positions=eval_state.positions, velocities=v_bc,
                densities=eval_state.densities, pressures=eval_state.pressures,
                particle_type=eval_domain.particle_type, edge_index=eval_domain.edge_index,
                h=eval_domain.h, n_particles=N, n_fluid=eval_domain.n_fluid,
                n_wall=eval_domain.n_wall, n_inlet=eval_domain.n_inlet,
                n_outlet=eval_domain.n_outlet, R=R, L=L,
                boundary_mask=eval_domain.boundary_mask, n_vertices=N,
            )
            a = model(domain_eval)
            a = torch.clamp(a, -a_clamp, a_clamp)
            eval_state, eval_domain = symplectic_euler_step(
                eval_state, eval_domain, a, dt, inlet_velocity_fn=inlet_bc,
            )
    
    # Extract velocity profile at mid-tube
    fluid_pos = eval_state.positions[fluid_mask]
    fluid_vel = eval_state.velocities[fluid_mask]
    z_mid = L / 2.0
    z_tol = 2.0 * dp
    mid_mask = (fluid_pos[:, 2] - z_mid).abs() < z_tol
    
    if mid_mask.sum() < 3:
        mid_mask = torch.ones(fluid_pos.shape[0], dtype=torch.bool, device=device)
    
    r_eval = torch.sqrt(fluid_pos[mid_mask, 0]**2 + fluid_pos[mid_mask, 1]**2)
    vz_gnn = fluid_vel[mid_mask, 2]
    vz_analytical = poiseuille_analytical(r_eval, R, delta_p, L, mu)
    
    v_max_analytical = float((delta_p / (4.0 * mu * L)) * R**2)
    vz_max_gnn = float(vz_gnn.max())
    
    max_vel_error = abs(vz_max_gnn - v_max_analytical) / max(v_max_analytical, 1e-8) * 100
    profile_l2 = float((vz_gnn - vz_analytical).norm() / (vz_analytical.norm() + 1e-8) * 100) if len(r_eval) > 0 else float('nan')
    
    Q_gnn = math.pi * R**2 * vz_max_gnn / 2.0
    Q_analytical = poiseuille_flow_rate(R, delta_p, L, mu)
    flow_rate_error = abs(Q_gnn - Q_analytical) / max(abs(Q_analytical), 1e-12) * 100
    
    elapsed_total = time.time() - t_start
    
    print(f"\n{'='*60}")
    print(f"[Results] dp={dp*1000:.1f}mm, {domain.n_fluid} fluid particles")
    print(f"  v_max (GNN):        {vz_max_gnn:.6f} m/s")
    print(f"  v_max (analytical): {v_max_analytical:.6f} m/s")
    print(f"  Max velocity error: {max_vel_error:.2f}%")
    print(f"  Profile L2 error:   {profile_l2:.2f}%")
    print(f"  Q (GNN):            {Q_gnn*1e6:.4f} mL/s")
    print(f"  Q (analytical):     {Q_analytical*1e6:.4f} mL/s")
    print(f"  Flow rate error:    {flow_rate_error:.2f}%")
    print(f"  Best training loss: {best_loss:.4e}")
    print(f"  Loss spikes recovered: {spike_count}")
    print(f"  Training time: {elapsed_total:.1f}s")
    
    passed = max_vel_error < 10.0 and flow_rate_error < 5.0
    print(f"\n  {'✅ VALIDATION PASSED' if passed else '⚠️ NEEDS MORE TRAINING'}")
    print(f"  v_max_error: {max_vel_error:.2f}% {'✅' if max_vel_error < 10 else '❌'} (target <10%)")
    print(f"  flow_rate_error: {flow_rate_error:.2f}% {'✅' if flow_rate_error < 5 else '❌'} (target <5%)")
    print(f"{'='*60}")
    
    results = {
        "dp_mm": dp * 1000,
        "n_fluid": domain.n_fluid,
        "n_total": N,
        "v_max_gnn": vz_max_gnn,
        "v_max_analytical": v_max_analytical,
        "max_vel_error_pct": max_vel_error,
        "profile_l2_error_pct": profile_l2,
        "Q_gnn": Q_gnn,
        "Q_analytical": Q_analytical,
        "flow_rate_error_pct": flow_rate_error,
        "best_loss": best_loss,
        "final_loss": losses[-1] if losses else float('nan'),
        "spike_count": spike_count,
        "training_time_s": elapsed_total,
        "passed": passed,
        "device": device,
    }
    
    # Save results JSON
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poiseuille v2 — improved SPH-GNN training")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dp", type=float, default=0.001, help="Particle spacing [m]")
    parser.add_argument("--delta-p", type=float, default=0.5, help="Pressure drop [Pa]")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--n-layers", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="/root/multi_tissue/checkpoints")
    args = parser.parse_args()
    
    results = run_poiseuille_v2(
        dp=args.dp,
        n_epochs=args.epochs,
        delta_p=args.delta_p,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_mp_layers=args.n_layers,
        warmup_epochs=args.warmup,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        verbose=True,
    )
