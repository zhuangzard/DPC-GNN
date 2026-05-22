#!/usr/bin/env python3
# Note on fairness: this sweep gives UDMP a 20-cell (lr, lambda_barrier)
# grid search but does NOT sweep DPC-GNN over the same grid. The reported
# DPC-GNN-vs-UDMP gap is therefore the gap between best-of-sweep UDMP and
# default-hyperparameter DPC-GNN -- a conservative estimate of the AMP
# advantage, since DPC-GNN under its own grid search would likely be even
# better. This asymmetry should be disclosed in the manuscript Methods.
"""
sweep_udmp_fairness.py — Hyperparameter fairness sweep for UDMP baseline.

R5B-3 (blind ML peer review) flagged the 18-373x AMP ablation gap as
possibly unfair (untuned UDMP). This sweep tries LR x lambda_barrier
combinations and reports the best-case UDMP per tissue. If the AMP gap
holds at >5x for best-case UDMP, the AMP claim is bulletproof.

Underlying trainer: ``multi_tissue/src/solid_tissue_train_mgn.py``
(despite its filename, this is the UDMP / DirectedMP variant — two
independent edge MLPs summed, no antisymmetry coupling — NOT faithful
MGN).

Configuration
-------------
LR_GRID       : {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}     (5 values)
LAMBDA_GRID   : {10, 30, 100, 300}                 (4 values; 100 = baseline)
SEEDS         : {42, 123, 456}                     (3 seeds → ranking only)
TISSUES       : {kidney, vessel, bone}             (the 3 worst UDMP failures)
TOTAL         : 3 × 5 × 4 × 3 = 180 runs

Hyperparameter pass-through
---------------------------
* ``--lr`` is an existing CLI flag of ``solid_tissue_train_mgn.py``.
* ``--lambda-barrier`` does NOT exist on the trainer (the barrier
  coefficient ``E * 100`` is hardcoded inside
  ``solid_tissue_train.neo_hookean_energy``). We pass it via the env var
  ``DPC_LAMBDA_BARRIER``, which a one-line patch in
  ``solid_tissue_train.py`` now reads at energy-computation time. The
  patch defaults to ``100``, so unrelated runs are unaffected.

Outputs
-------
``<output-root>/<tissue>_lr<lr>_lam<lam>_seed<seed>/``
    ``run.log``            — combined stdout+stderr from the trainer
    ``results.json``       — symlink/copy of the trainer's results.json
                             (the trainer writes to ``<output-root>/<tissue>_mgn/``
                             so we copy it into the per-run dir after the run)

``<output-root>/sweep_summary.json``
    Top-level: ``per_tissue_best`` (tissue → best hp combo + 3-seed stats),
               ``grid`` (tissue → (lr,lam) → 3-seed stats).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

# ───────────────────────────────────────────────────────────────────────
# Sweep grid
# ───────────────────────────────────────────────────────────────────────

LR_GRID = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LAMBDA_GRID = [10.0, 30.0, 100.0, 300.0]
SEEDS = [42, 123, 456]
TISSUES = ["kidney", "vessel", "bone"]
EPOCHS_PER_TISSUE = {"kidney": 3000, "vessel": 2500, "bone": 2000}

# E_Pa, NU, RHO per tissue (mirrors run_mgn_baseline.sh / canonical TISSUES dict).
# Bone is not in the canonical TISSUES dict shipped with the trainer, but the
# trainer accepts --E/--nu/--rho overrides, so we pass them explicitly for
# every tissue to keep the sweep self-contained.
TISSUE_PARAMS = {
    "kidney": {"E": 10000.0,    "nu": 0.45, "rho": 1050.0},
    "vessel": {"E": 400000.0,   "nu": 0.49, "rho": 1050.0},
    "bone":   {"E": 10000000.0, "nu": 0.30, "rho": 1900.0},
}

# Resolve the trainer path relative to this file so the sweep works whether
# the user runs it from the repo root or from /root/... on the remote box.
HERE = Path(__file__).resolve().parent
TRAINER = HERE / "solid_tissue_train_mgn.py"


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _run_dir(output_root: Path, tissue: str, lr: float, lam: float, seed: int) -> Path:
    """Per-run directory name. Stable across invocations (used as the key)."""
    return output_root / f"{tissue}_lr{lr:.0e}_lam{lam:g}_seed{seed}"


def _cell_suffix(lr: float, lam: float, seed: int) -> str:
    """Unique per-cell suffix passed to the trainer via --output-suffix so
    concurrent (lr, lam, seed) cells for the same tissue do not clobber a
    shared <tissue>_mgn/results.json (CR-C #1)."""
    return f"lr{lr:.0e}_lam{lam:g}_seed{seed}"


def _build_cmd(tissue: str, lr: float, lam: float, seed: int,
               output_root: Path) -> list[str]:
    """Construct the trainer CLI for one (tissue, lr, lam, seed) run.

    Note: lambda_barrier is passed via env var (see ``run_one``), NOT via CLI,
    because the trainer does not expose a ``--lambda-barrier`` flag.

    ``--output-suffix`` makes the trainer write to
    ``<output-root>/<tissue>_mgn_<suffix>/results.json``, eliminating the
    race condition that existed when every cell wrote to the same path.
    """
    p = TISSUE_PARAMS[tissue]
    return [
        sys.executable, str(TRAINER),
        "--tissue", tissue,
        "--E", repr(p["E"]),
        "--nu", repr(p["nu"]),
        "--rho", repr(p["rho"]),
        "--lr", repr(lr),
        "--seed", str(seed),
        "--epochs", str(EPOCHS_PER_TISSUE[tissue]),
        "--output-root", str(output_root),
        "--output-suffix", _cell_suffix(lr, lam, seed),
    ]


def run_one(
    tissue: str,
    lr: float,
    lam: float,
    seed: int,
    output_root: Path,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Execute one sweep cell. Returns the parsed results.json or None on dry-run/failure."""
    out_dir = _run_dir(output_root, tissue, lr, lam, seed)
    cmd = _build_cmd(tissue, lr, lam, seed, output_root)

    if dry_run:
        # Make the env-var portion visible so reviewers can audit the sweep.
        # Skip mkdir on dry-run so the script works even when /root/... isn't
        # writable (e.g. when sanity-checking locally on macOS).
        print(f"DPC_LAMBDA_BARRIER={lam:g} {' '.join(cmd)}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Fallback: barrier coefficient via env var (no --lambda-barrier flag on
    # the trainer). solid_tissue_train.neo_hookean_energy reads this; default
    # is 100 if unset.
    env["DPC_LAMBDA_BARRIER"] = repr(lam)

    log_path = out_dir / "run.log"
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    # The trainer writes to <output-root>/<tissue>_mgn_<cell-suffix>/, where
    # <cell-suffix> is the unique per-cell key. Copy that results.json into
    # the legacy per-cell dir IMMEDIATELY so the rest of the sweep machinery
    # (aggregate, _collect_results) keeps working unchanged.
    cell_suffix = _cell_suffix(lr, lam, seed)
    trainer_out = output_root / f"{tissue}_mgn_{cell_suffix}" / "results.json"
    target = out_dir / "results.json"
    if trainer_out.exists():
        shutil.copy2(trainer_out, target)

    if proc.returncode != 0 or not target.exists():
        return {
            "tissue": tissue, "lr": lr, "lambda_barrier": lam, "seed": seed,
            "status": "failed", "returncode": proc.returncode,
            "log": str(log_path),
        }

    try:
        with open(target) as f:
            res = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "tissue": tissue, "lr": lr, "lambda_barrier": lam, "seed": seed,
            "status": "parse_error", "error": str(e), "log": str(log_path),
        }

    # Annotate with sweep metadata so aggregate() needs no extra context.
    res.update({
        "sweep_tissue": tissue, "sweep_lr": lr,
        "sweep_lambda_barrier": lam, "sweep_seed": seed,
        "status": "ok", "log": str(log_path),
    })
    with open(target, "w") as f:
        json.dump(res, f, indent=2)
    return res


def _collect_results(output_root: Path) -> list[dict[str, Any]]:
    """Walk per-cell dirs and load every results.json that exists."""
    out: list[dict[str, Any]] = []
    for tissue, lr, lam, seed in itertools.product(TISSUES, LR_GRID, LAMBDA_GRID, SEEDS):
        rj = _run_dir(output_root, tissue, lr, lam, seed) / "results.json"
        if not rj.exists():
            continue
        try:
            with open(rj) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rec.setdefault("sweep_tissue", tissue)
        rec.setdefault("sweep_lr", lr)
        rec.setdefault("sweep_lambda_barrier", lam)
        rec.setdefault("sweep_seed", seed)
        out.append(rec)
    return out


def _err(rec: dict[str, Any]) -> float:
    """Primary ranking metric: relative tip-displacement error (%)."""
    v = rec.get("relative_error_pct")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return float("inf")
    return float(v)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.fmean(xs), statistics.stdev(xs)


def aggregate(output_root: Path) -> dict[str, Any]:
    """Build the sweep_summary.json structure.

    Reporting strategy
    ------------------
    For each tissue, compute the mean tip-error across the 3 seeds at every
    (lr, lam) cell, then pick the cell with the lowest 3-seed mean error as
    the "best-case UDMP". This is the canonical hp-sweep reporting style and
    is what R5B-3 will compare to DPC-GNN. We also expose the full grid so
    reviewers can see the response surface and check for plateaus / cliffs.
    """
    records = _collect_results(output_root)
    by_cell: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for r in records:
        key = (r["sweep_tissue"], float(r["sweep_lr"]), float(r["sweep_lambda_barrier"]))
        by_cell.setdefault(key, []).append(r)

    grid: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in TISSUES}
    per_tissue_best: dict[str, dict[str, Any]] = {}

    for (tissue, lr, lam), recs in by_cell.items():
        errs = [_err(r) for r in recs if r.get("status", "ok") == "ok"]
        errs_finite = [e for e in errs if math.isfinite(e)]
        m, s = _mean_std(errs_finite)
        cell_key = f"lr{lr:.0e}_lam{lam:g}"
        grid[tissue][cell_key] = {
            "lr": lr, "lambda_barrier": lam,
            "n_seeds": len(recs), "n_ok": len(errs_finite),
            "mean_rel_err_pct": m, "std_rel_err_pct": s,
            "seed_errs": errs,
        }

    n_seeds_required = len(SEEDS)
    diverged_cells: dict[str, list[str]] = {t: [] for t in TISSUES}
    for tissue in TISSUES:
        cells = grid[tissue]
        if not cells:
            per_tissue_best[tissue] = {"status": "no_results"}
            continue
        # Best-case (CR-A #3): require ALL n_seeds_required seeds to
        # converge before a cell counts as best-case-eligible.  A "best-
        # case" cell that survived only by virtue of one lucky seed is
        # not an honest hp-sweep result, and the reviewer will catch it.
        scored = []
        for k, c in cells.items():
            if c["n_ok"] < n_seeds_required or not math.isfinite(c["mean_rel_err_pct"]):
                diverged_cells[tissue].append(k)
                continue
            scored.append((c["mean_rel_err_pct"], k, c))
        if not scored:
            per_tissue_best[tissue] = {
                "status": "all_diverged",
                "n_diverged_cells": len(diverged_cells[tissue]),
            }
            continue
        scored.sort(key=lambda x: x[0])
        best_mean, best_key, best_cell = scored[0]
        per_tissue_best[tissue] = {
            "best_cell": best_key,
            "lr": best_cell["lr"],
            "lambda_barrier": best_cell["lambda_barrier"],
            "mean_rel_err_pct": best_mean,
            "std_rel_err_pct": best_cell["std_rel_err_pct"],
            "n_ok": best_cell["n_ok"],
            "n_seeds_required": n_seeds_required,
            "seed_errs": best_cell["seed_errs"],
            "n_diverged_cells_excluded": len(diverged_cells[tissue]),
        }

    return {
        "config": {
            "lr_grid": LR_GRID,
            "lambda_grid": LAMBDA_GRID,
            "seeds": SEEDS,
            "tissues": TISSUES,
            "epochs_per_tissue": EPOCHS_PER_TISSUE,
            "trainer": str(TRAINER),
            "lambda_pass_through": "env var DPC_LAMBDA_BARRIER",
            "best_case_requires_all_seeds": True,
        },
        "n_runs_total": 3 * len(LR_GRID) * len(LAMBDA_GRID) * len(SEEDS),
        "n_runs_collected": len(records),
        "per_tissue_best": per_tissue_best,
        "diverged_cells": diverged_cells,
        "grid": grid,
    }


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="UDMP hyperparameter fairness sweep (R5B-3 response)."
    )
    parser.add_argument("--output-root", default="/root/results/udmp_sweep",
                        help="Root directory for sweep outputs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all commands without running anything.")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Concurrent trainer processes (1 = serial). "
                             "Keep low: each run is GPU-bound.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running; just rebuild sweep_summary.json "
                             "from existing per-cell results.")
    parser.add_argument("--tissues", nargs="+", default=None,
                        help="Restrict to a subset of tissues (default: all).")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not args.dry_run and not args.aggregate_only:
        output_root.mkdir(parents=True, exist_ok=True)

    tissues = args.tissues or TISSUES
    bad = [t for t in tissues if t not in TISSUE_PARAMS]
    if bad:
        print(f"ERROR: unknown tissues: {bad}", file=sys.stderr)
        return 2

    cells = list(itertools.product(tissues, LR_GRID, LAMBDA_GRID, SEEDS))
    print(f"# Sweep plan: {len(cells)} runs "
          f"({len(tissues)} tissues × {len(LR_GRID)} lr × "
          f"{len(LAMBDA_GRID)} lam × {len(SEEDS)} seeds)",
          file=sys.stderr)

    if args.aggregate_only:
        summary = aggregate(output_root)
        summary_path = output_root / "sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        print(f"# Wrote {summary_path}", file=sys.stderr)
        return 0

    if args.dry_run:
        # Serial print, no execution. Concurrency is irrelevant here.
        for tissue, lr, lam, seed in cells:
            run_one(tissue, lr, lam, seed, output_root, dry_run=True)
        return 0

    # Real execution path.
    # Concurrency model (CR-C #1): parallelize ACROSS tissues, serialize
    # WITHIN a tissue.  The trainer no longer races on a shared
    # <tissue>_mgn/ dir (we pass --output-suffix per cell), but keeping the
    # within-tissue dispatch serial also avoids GPU contention from two
    # different runs of the SAME mesh, which previously spiked memory.
    max_workers = max(1, min(args.max_parallel, len(tissues)))
    failed: list[tuple[str, float, float, int]] = []

    # Group the (lr, lam, seed) cells by tissue, preserving deterministic
    # iteration order.
    cells_by_tissue: dict[str, list[tuple[float, float, int]]] = {
        t: [] for t in tissues
    }
    for tissue, lr, lam, seed in cells:
        cells_by_tissue[tissue].append((lr, lam, seed))

    def _run_tissue_serial(tissue: str) -> list[tuple[str, float, float, int]]:
        """Iterate this tissue's (lr, lam, seed) cells one at a time."""
        local_fail: list[tuple[str, float, float, int]] = []
        for lr, lam, seed in cells_by_tissue[tissue]:
            try:
                res = run_one(tissue, lr, lam, seed, output_root,
                              dry_run=False)
            except Exception as e:
                print(f"# {(tissue, lr, lam, seed)}: exception {e!r}",
                      file=sys.stderr)
                local_fail.append((tissue, lr, lam, seed))
                continue
            if not res or res.get("status") != "ok":
                local_fail.append((tissue, lr, lam, seed))
        return local_fail

    if max_workers == 1 or len(tissues) == 1:
        for tissue in tissues:
            failed.extend(_run_tissue_serial(tissue))
    else:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_tissue_serial, t): t for t in tissues}
            for fut in cf.as_completed(futures):
                try:
                    failed.extend(fut.result())
                except Exception as e:
                    print(f"# tissue {futures[fut]}: exception {e!r}",
                          file=sys.stderr)

    if failed:
        print(f"# WARNING: {len(failed)} runs failed/diverged. See per-run "
              f"logs under {output_root}.", file=sys.stderr)

    summary = aggregate(output_root)
    summary_path = output_root / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"# Wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
