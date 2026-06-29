"""Extract the architecture that each method WOULD have selected at intermediate
budgets (cutoff = 10, 20) so their true held-out TEST performance can be measured.

Motivation
----------
Table 1 only reports the test AUPRC of the architecture selected at the full
budget (30 evals). To show an "anytime" view (test performance if search had
stopped at 10 or 20 evals), we must identify the best-by-validation architecture
within the first k evaluations and evaluate IT on the test set.

Because `*_search.csv` stores only validation metrics (test is never touched
during search, by design), the intermediate selections have no test value yet.
This script finds those selections and emits a deduplicated job list for a
one-off test re-evaluation.

Selection rule (identical to the main pipeline): within the first k evaluations,
pick the architecture with the lowest composite validation rank (mean of the
per-metric descending ranks over {accuracy, f1, auroc, auprc}).

Dedup logic
-----------
  • cutoff = 30  → reuse the test value already in `*_best.csv` (no re-run).
  • cutoff = 10/20:
       - if the selected arch == the final (30) selection → reuse best.csv test.
       - else a re-run is needed. Test depends only on (hospital, task, seed,
         arch) — NOT on which method/cutoff selected it — so re-runs are
         deduplicated on that key.

Outputs (under --out_dir)
  • anytime_selection_map.csv : (hospital, method, task, seed, cutoff,
                                 embed_dim, depth, mlp_ratio, num_heads,
                                 source)  where source ∈ {best_csv, rerun}
  • anytime_rerun_jobs.tsv    : unique (hospital, task, seed, embed_dim, depth,
                                 mlp_ratio, num_heads) needing a test re-run

Usage:
    python analyze/extract_anytime_jobs.py \
        --results_root /blue/mei.liu/lideyi/MAS-NAS/results \
        --hospitals source_15 MIMIC-IV \
        --out_dir /blue/mei.liu/lideyi/MAS-NAS/analyze/anytime
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["baseline0", "baseline1", "baseline2", "baseline4", "mas"]
TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
CUTOFFS = [5, 10, 20, 30]
VAL_COLS = ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]
ARCH_COLS = ["embed_dim", "depth", "mlp_ratio", "num_heads"]


def best_arch_in_prefix(df: pd.DataFrame, cutoff: int) -> dict | None:
    """Return the arch (dict) with the lowest composite val rank within the
    first `cutoff` evaluations, or None if the prefix is empty."""
    if "iteration" not in df.columns:
        df = df.copy()
        df["iteration"] = range(1, len(df) + 1)
    prefix = df[df["iteration"] <= cutoff].dropna(subset=VAL_COLS)
    if len(prefix) == 0:
        return None
    ranks = pd.DataFrame()
    for col in VAL_COLS:
        ranks[col] = prefix[col].rank(ascending=False, method="average")
    avg_rank = ranks.mean(axis=1)
    best_row = prefix.loc[avg_rank.idxmin()]
    return {c: int(best_row[c]) for c in ARCH_COLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--hospitals", required=True, nargs="+")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--skip_existing_dir", type=Path, default=None,
                    help="Directory of completed retest JSONs. Archs already "
                         "evaluated there are excluded from the rerun list (their "
                         "test value is reused at table-build time).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    rerun_keys = set()   # (hospital, task, seed, embed_dim, depth, mlp_ratio, num_heads)

    # Pre-load already-completed retest arch keys to avoid re-running them
    existing = set()
    if args.skip_existing_dir and args.skip_existing_dir.exists():
        for jf in args.skip_existing_dir.glob("*.json"):
            try:
                d = __import__("json").load(open(jf))
                existing.add((d["hospital"], d["task"], int(d["seed"]),
                              int(d["embed_dim"]), int(d["depth"]),
                              int(d["mlp_ratio"]), int(d["num_heads"])))
            except Exception:
                pass
        print(f"  [skip] {len(existing)} archs already in {args.skip_existing_dir}")

    seed_dirs = sorted(args.results_root.glob("seed_*"))
    seeds = [int(d.name.split("_", 1)[1]) for d in seed_dirs
             if d.name.split("_", 1)[1].isdigit()]

    for hospital in args.hospitals:
        for method in METHODS:
            for task in TASKS:
                for seed in seeds:
                    base = (args.results_root / f"seed_{seed}" / hospital
                            / "search" / method / task)
                    search_csv = base / f"{method}_search.csv"
                    best_csv = base / f"{method}_best.csv"
                    if not search_csv.exists():
                        continue
                    try:
                        sdf = pd.read_csv(search_csv)
                    except Exception:
                        continue
                    if "val_auprc" not in sdf.columns:
                        continue

                    # Final (30) selection from best.csv (already test-evaluated)
                    final_arch = None
                    if best_csv.exists():
                        try:
                            bdf = pd.read_csv(best_csv)
                            if len(bdf) > 0:
                                final_arch = {c: int(bdf.iloc[0][c]) for c in ARCH_COLS}
                        except Exception:
                            final_arch = None

                    for cutoff in CUTOFFS:
                        if cutoff >= 30 and final_arch is not None:
                            arch = final_arch
                            source = "best_csv"
                        else:
                            arch = best_arch_in_prefix(sdf, cutoff)
                            if arch is None:
                                continue
                            # If intermediate selection == final selection, reuse best.csv test
                            if final_arch is not None and arch == final_arch:
                                source = "best_csv"
                            else:
                                source = "rerun"
                                akey = (hospital, task, seed,
                                        arch["embed_dim"], arch["depth"],
                                        arch["mlp_ratio"], arch["num_heads"])
                                # Only queue a re-run if not already test-evaluated
                                if akey not in existing:
                                    rerun_keys.add(akey)
                        selection_rows.append({
                            "hospital": hospital, "method": method, "task": task,
                            "seed": seed, "cutoff": cutoff,
                            **arch, "source": source,
                        })

    sel_df = pd.DataFrame(selection_rows)
    sel_path = args.out_dir / "anytime_selection_map.csv"
    sel_df.to_csv(sel_path, index=False)

    rerun_df = pd.DataFrame(
        sorted(rerun_keys),
        columns=["hospital", "task", "seed",
                 "embed_dim", "depth", "mlp_ratio", "num_heads"],
    )
    rerun_path = args.out_dir / "anytime_rerun_jobs.tsv"
    rerun_df.to_csv(rerun_path, sep="\t", index=False)

    print(f"✓ Selection map: {len(sel_df)} rows  →  {sel_path}")
    print(f"✓ Unique test re-runs needed: {len(rerun_df)}  →  {rerun_path}")
    if len(sel_df) > 0:
        n_reuse = (sel_df["source"] == "best_csv").sum()
        n_rerun = (sel_df["source"] == "rerun").sum()
        print(f"  selections reusing best.csv test: {n_reuse}")
        print(f"  selections needing re-run:        {n_rerun}")
        print(f"  (deduped to {len(rerun_df)} unique finetune jobs)")


if __name__ == "__main__":
    main()
