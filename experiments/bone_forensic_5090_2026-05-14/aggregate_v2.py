#!/usr/bin/env python3
"""Re-aggregate forensic 5090 results using BOTH "Best error" (paper convention)
and "Relative error" (final epoch) metrics, for transparent comparison.

R2 forensic predicted that paper's `0.87% mean / 0.07% median / 19/20 < 5%`
comes from best-checkpoint selection against the Euler-Bernoulli analytical
tip displacement — confirmed: every bone log emits two numbers:
  - Best error: X% (epoch Y)     <- best-during-training (paper uses this)
  - Relative error: Z%           <- final epoch (often much worse on bone)
"""

import os, re, json, statistics
from pathlib import Path

ROOT = Path("./")

SEEDS = [42, 123, 456, 789, 2026, 1337, 999, 31415, 271828, 1000,
         2024, 3407, 7777, 8888, 12345, 54321, 99999, 11111, 66666, 100]


def parse_log(path):
    """Return (best_err, best_epoch, relative_err) tuple."""
    txt = open(path).read()
    m_best = re.search(r"Best error:\s+([\d.]+)%\s+\(epoch\s+(\d+)\)", txt)
    m_rel = re.search(r"Relative error:\s+([\d.]+)%", txt)
    best = float(m_best.group(1)) if m_best else None
    best_ep = int(m_best.group(2)) if m_best else None
    rel = float(m_rel.group(1)) if m_rel else None
    return best, best_ep, rel


def parse_group(dir_):
    results = {}
    for seed in SEEDS:
        p = dir_ / f"bone_seed{seed}.log"
        if p.exists():
            b, ep, r = parse_log(p)
            results[seed] = {"best": b, "best_epoch": ep, "relative": r}
    return results


def stats(name, errs_dict, key, paper_mean, paper_median, paper_succ):
    if not errs_dict:
        print(f"  {name}: no data"); return None
    seeds = sorted(errs_dict.keys())
    vals = [errs_dict[s][key] for s in seeds]
    vals_s = sorted(vals)
    mean = statistics.mean(vals)
    stdv = statistics.stdev(vals) if len(vals) > 1 else 0.0
    med = statistics.median(vals)
    mn, mx = min(vals), max(vals)
    succ5 = sum(1 for v in vals if v < 5)
    print(f"=== {name} (metric: {key}) ===")
    print(f"  n={len(vals)}, mean={mean:.4f}%, std={stdv:.4f}%, median={med:.4f}%")
    print(f"  min={mn:.4f}%, max={mx:.4f}%, success<5%={succ5}/{len(vals)}")
    print(f"  Paper claim: mean {paper_mean}, median {paper_median}, success {paper_succ}")
    print(f"  Sorted errors: {[round(v,3) for v in vals_s]}")
    return {
        "name": name, "metric": key, "n": len(vals),
        "mean_pct": round(mean, 4), "std_pct": round(stdv, 4),
        "median_pct": round(med, 4), "min_pct": round(mn, 4), "max_pct": round(mx, 4),
        "success_below_5pct": succ5,
        "seeds_run_order": SEEDS, "errors_run_order": [errs_dict[s][key] for s in SEEDS if s in errs_dict],
        "errors_sorted": vals_s,
    }


def main():
    A_dir = ROOT / "logs" / "group_A_baseline"
    B_dir = ROOT / "logs" / "group_B_fbar"
    A = parse_group(A_dir)
    B = parse_group(B_dir)

    print("=" * 70)
    print("Group A baseline (no F-bar)")
    print("=" * 70)
    A_best = stats("Group A: BEST error", A, "best", "—", "—", "—")
    print()
    A_rel = stats("Group A: FINAL relative error", A, "relative", "7.36 ± 0.22%", "?", "0/20")

    print()
    print("=" * 70)
    print("Group B F-bar fix")
    print("=" * 70)
    B_best = stats("Group B: BEST error (paper convention)", B, "best",
                   "0.87 ± 1.70%", "0.07%", "19/20")
    print()
    B_rel = stats("Group B: FINAL relative error", B, "relative", "—", "—", "—")

    print()
    print("=" * 70)
    print("Paper-vs-forensic comparison (BEST error metric)")
    print("=" * 70)
    if A_best and B_best:
        imp_mean = A_best["mean_pct"] / B_best["mean_pct"] if B_best["mean_pct"] > 0 else None
        imp_med = A_best["median_pct"] / B_best["median_pct"] if B_best["median_pct"] > 0 else None
        print(f"  Group A best:  mean={A_best['mean_pct']:.4f}%  median={A_best['median_pct']:.4f}%")
        print(f"  Group B best:  mean={B_best['mean_pct']:.4f}%  median={B_best['median_pct']:.4f}%")
        if imp_mean: print(f"  Mean improvement: {imp_mean:.2f}x  (paper: 8.5x)")
        if imp_med: print(f"  Median improvement: {imp_med:.2f}x  (paper: 105x)")
        print(f"  19/20 < 5%? Group B: {B_best['success_below_5pct']}/20  (paper: 19/20)")

    print()
    print("Seed 7777 in Group B (paper outlier at 6.50%):")
    if 7777 in B:
        print(f"  Forensic best  = {B[7777]['best']:.4f}%  (epoch {B[7777]['best_epoch']})")
        print(f"  Forensic final = {B[7777]['relative']:.4f}%")
        print(f"  Paper          = 6.50%")

    print()
    print("Per-seed Group B BEST error (run order):")
    for seed in SEEDS:
        if seed in B:
            print(f"  seed {seed:6d}: best={B[seed]['best']:.4f}%  (epoch {B[seed]['best_epoch']:4d})  final={B[seed]['relative']:.4f}%")

    out = {
        "date": "2026-05-14",
        "device": "RTX 5090 (GPUHub, container 11b84db1d2)",
        "tissue": "bone", "E_Pa": 10000000.0, "nu": 0.30, "rho": 1900,
        "mesh": "22x7x7", "params": 161731, "epochs": 3000,
        "seeds_canonical": SEEDS,
        "note": ("Group A uses unmodified solid_tissue_train_hires.py (use_fbar = nu>=0.45 → False for bone). "
                 "Group B uses same script with line 217 sed-patched to use_fbar = True. "
                 "Two metrics reported per log: Best error (peak-vs-Euler-Bernoulli during training), "
                 "and Relative error (final epoch). Paper's published numbers correspond to Best error."),
        "group_A_baseline_no_fbar": {"best": A_best, "final": A_rel},
        "group_B_fbar_fix": {"best": B_best, "final": B_rel},
        "per_seed_group_A": {str(s): A[s] for s in SEEDS if s in A},
        "per_seed_group_B": {str(s): B[s] for s in SEEDS if s in B},
    }
    out_path = ROOT / "bone_forensic_aggregated_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
