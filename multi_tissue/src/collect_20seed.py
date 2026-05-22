#!/usr/bin/env python3
"""Collect 20-seed results into summary JSON."""
import os, json, re, statistics

SEEDS = [42, 123, 456, 789, 2026, 1337, 999, 31415, 271828, 1000, 2024, 3407, 7777, 8888, 12345, 54321, 99999, 11111, 66666, 100]
TISSUES = ["brain", "kidney", "myocardium", "cartilage", "vessel", "bone"]
OUTDIR = "/root/results/twentyseed"

summary = {}
for tissue in TISSUES:
    seeds_done = []
    errors = []
    tip_disps = []
    exp_tips = []
    for seed in SEEDS:
        logfile = os.path.join(OUTDIR, f"{tissue}_seed{seed}.log")
        if not os.path.exists(logfile):
            continue
        with open(logfile) as f:
            content = f.read()
        # Extract relative error
        m = re.findall(r"Relative error:\s*([\d.]+)%", content)
        if m:
            err = float(m[-1])
            seeds_done.append(seed)
            errors.append(err)
            # Extract tip displacement
            mt = re.findall(r"tip_disp_mm.*?:\s*([\d.]+)", content)
            if mt: tip_disps.append(float(mt[-1]))
            me = re.findall(r"expected_tip_mm.*?:\s*([\d.]+)", content)
            if me: exp_tips.append(float(me[-1]))

    if errors:
        summary[tissue] = {
            "seeds": seeds_done,
            "errors": errors,
            "mean": round(statistics.mean(errors), 2),
            "std": round(statistics.stdev(errors) if len(errors) > 1 else 0.0, 2),
            "min": round(min(errors), 2),
            "max": round(max(errors), 2),
            "median": round(statistics.median(errors), 2),
            "n_completed": len(errors),
            "n_total": 20,
            "success_rate_15pct": f"{sum(1 for e in errors if e < 15)}/{len(errors)}",
            "success_rate_10pct": f"{sum(1 for e in errors if e < 10)}/{len(errors)}",
            "success_rate_5pct": f"{sum(1 for e in errors if e < 5)}/{len(errors)}",
        }
    else:
        summary[tissue] = {"n_completed": 0, "n_total": 20, "errors": []}

# Overall
all_errors = []
for t in TISSUES:
    all_errors.extend(summary[t].get("errors", []))
if all_errors:
    summary["_overall"] = {
        "mean": round(statistics.mean(all_errors), 2),
        "std": round(statistics.stdev(all_errors) if len(all_errors) > 1 else 0, 2),
        "min": round(min(all_errors), 2),
        "max": round(max(all_errors), 2),
        "n_completed": len(all_errors),
        "n_total": 120,
        "success_rate_15pct": f"{sum(1 for e in all_errors if e < 15)}/{len(all_errors)}",
    }

with open(os.path.join(OUTDIR, "twentyseed_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
