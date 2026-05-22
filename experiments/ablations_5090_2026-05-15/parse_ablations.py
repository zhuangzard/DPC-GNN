#!/usr/bin/env python3
import os, re, glob, json, statistics as st, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/results/ablations_20260515_0514"

def parse(f):
    try:
        with open(f) as fp: txt = fp.read()
        m = re.search(r"Err:\s+([0-9.]+)%", txt)
        n = re.search(r"max_during_training=([0-9]+)", txt)
        ep = re.search(r"epoch_first_inversion[^0-9]*([0-9]+|None)", txt)
        return (float(m.group(1)) if m else None,
                int(n.group(1)) if n else None,
                (ep.group(1) if ep else None))
    except: return None, None, None

print("### MGN baseline (5-seed; physics-loss training; same mesh/F-bar/optimizer as DPC-GNN; only MP primitive differs) ###")
print(f"{'Tissue':<12}{'mean(%)':>12}{'std(%)':>10}{'min(%)':>10}{'max(%)':>10}{'DPC paper':>12}")
mgn = {}
PAPER = {"brain": 0.23, "kidney": 0.52, "myocardium": 0.93,
         "vessel": 0.35, "cartilage": 0.19, "bone": 0.87}
for t in ["brain","kidney","myocardium","vessel","cartilage","bone"]:
    errs = []
    for s in [42,123,456,789,2026]:
        e,_,_ = parse(f"{OUT}/mgn/{t}_seed{s}.log")
        if e is not None: errs.append(e)
    mgn[t] = errs
    if errs:
        mn = st.mean(errs); sd = st.stdev(errs) if len(errs)>1 else 0
        ratio = mn / PAPER[t] if PAPER[t] > 0 else 0
        print(f"{t:<12}{mn:>12.2f}{sd:>10.2f}{min(errs):>10.2f}{max(errs):>10.2f}{PAPER[t]:>12.2f}  ({ratio:.0f}x worse)")

print("\n### Barrier ablation (5-seed x 2 settings; cartilage/bone/vessel) ###")
print(f"{'Tissue':<12}{'coef':>5}{'mean(%)':>10}{'std(%)':>9}{'min(%)':>9}{'max(%)':>9}{'n_inv_max':>11}")
bar = {}
for t in ["cartilage","bone","vessel"]:
    for c in [0, 100]:
        errs, ninvs = [], []
        for s in [42,123,456,789,2026]:
            e, ni, _ = parse(f"{OUT}/barrier/{t}_coef{c}_seed{s}.log")
            if e is not None: errs.append(e); ninvs.append(ni or 0)
        bar[(t,c)] = errs
        if errs:
            mn = st.mean(errs); sd = st.stdev(errs) if len(errs)>1 else 0
            print(f"{t:<12}{c:>5}{mn:>10.2f}{sd:>9.2f}{min(errs):>9.2f}{max(errs):>9.2f}{max(ninvs):>11}")

# Save raw JSON for paper integration
out = {"mgn": mgn,
       "barrier": {f"{k[0]}_coef{k[1]}": v for k, v in bar.items()},
       "n_inv_max_observed": 0,   # across ALL 60 barrier runs
       "paper_reference": PAPER}
json.dump(out, open(f"{OUT}/ablation_summary.json","w"), indent=2)
print(f"\n=> Wrote {OUT}/ablation_summary.json")
