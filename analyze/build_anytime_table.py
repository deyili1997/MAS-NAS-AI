"""Assemble the 3-panel anytime main table: test AUPRC of the architecture each
method would have selected at budget = 10, 20, 30.

Joins:
  • anytime_selection_map.csv  (which arch each method/cutoff selected)
  • retest/*.json              (test metrics from the re-run finetunes)
  • results/.../*_best.csv      (test metrics for selections reusing the final arch)

Output (per hospital, suffixed _<hospital>):
  • main_table_anytime_<H>.csv : method × task × cutoff → test AUPRC mean ± std
                                 over seeds (long format, mirrors main_table)

Usage:
    python analyze/build_anytime_table.py \
        --results_root /blue/.../results \
        --anytime_dir  /blue/.../analyze/anytime \
        --hospitals source_15 MIMIC-IV \
        --out_dir /blue/.../analyze/<DATE>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_DISPLAY = {
    "baseline0": "Random", "baseline1": "EA", "baseline2": "GENIUS",
    "baseline3": "LLMatic", "baseline4": "CoLLM-NAS",
    "baseline5": "ENAS", "baseline6": "TPE", "mas": "ATHENA",
}
METHOD_ORDER = ["baseline0", "baseline1", "baseline2", "baseline4", "mas"]  # baseline3 excluded by DEFAULT — override with --methods
TASK_DISPLAY = {
    "death": "Death", "stay": "Stay>7d", "readmission": "Readmission (3M)",
    "next_diag_6m_pheno": "Phenotype (6M)", "next_diag_12m_pheno": "Phenotype (12M)",
    "med_rec": "Drug Recommendation",
}
TASK_ORDER = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
ARCH_COLS = ["embed_dim", "depth", "mlp_ratio", "num_heads"]


def _retest_lookup(anytime_dir: Path) -> dict:
    """Map (hospital, task, seed, arch_tuple) -> test_auprc from retest JSONs."""
    out = {}
    retest_dir = anytime_dir / "retest"
    if not retest_dir.exists():
        return out
    for jf in retest_dir.glob("*.json"):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        key = (d["hospital"], d["task"], int(d["seed"]),
               int(d["embed_dim"]), int(d["depth"]), int(d["mlp_ratio"]), int(d["num_heads"]))
        out[key] = float(d["test_auprc"])
    return out


def _bestcsv_lookup(results_root: Path, hospital: str) -> dict:
    """Map (method, task, seed) -> (arch_tuple, test_auprc) from *_best.csv."""
    out = {}
    for seed_dir in results_root.glob("seed_*"):
        if not seed_dir.name.split("_", 1)[1].isdigit():
            continue
        seed = int(seed_dir.name.split("_", 1)[1])
        sr = seed_dir / hospital / "search"
        if not sr.exists():
            continue
        for method in METHOD_ORDER:
            for task in TASK_ORDER:
                bc = sr / method / task / f"{method}_best.csv"
                if not bc.exists():
                    continue
                try:
                    df = pd.read_csv(bc)
                except Exception:
                    continue
                if len(df) == 0 or "test_auprc" not in df.columns:
                    continue
                r = df.iloc[0]
                arch = tuple(int(r[c]) for c in ARCH_COLS)
                out[(method, task, seed)] = (arch, float(r["test_auprc"]))
    return out


def main():
    global TASK_ORDER, METHOD_ORDER   # so --tasks / --methods can override
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--anytime_dir", required=True, type=Path)
    ap.add_argument("--hospitals", required=True, nargs="+")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--tasks", type=str, nargs="+", default=TASK_ORDER,
                    help="Tasks to include (default: the main 5). Use --tasks "
                         "med_rec with --results_root results_medrec.")
    ap.add_argument("--methods", type=str, nargs="+", default=METHOD_ORDER,
                    help=f"Method rows (default: {METHOD_ORDER}; baseline3/LLMatic "
                         "excluded). Pass e.g. --methods baseline0 baseline1 baseline2 "
                         "baseline3 baseline4 mas to include LLMatic.")
    args = ap.parse_args()
    TASK_ORDER = args.tasks
    METHOD_ORDER = args.methods
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(args.anytime_dir / "anytime_selection_map.csv")
    retest = _retest_lookup(args.anytime_dir)

    for hospital in args.hospitals:
        bestcsv = _bestcsv_lookup(args.results_root, hospital)
        hsel = sel[sel["hospital"] == hospital]
        rows = []
        for method in METHOD_ORDER:
            for task in TASK_ORDER:
                for cutoff in [5, 10, 20, 30]:
                    sub = hsel[(hsel["method"] == method) &
                               (hsel["task"] == task) &
                               (hsel["cutoff"] == cutoff)]
                    test_vals = []
                    for _, r in sub.iterrows():
                        seed = int(r["seed"])
                        arch = tuple(int(r[c]) for c in ARCH_COLS)
                        if r["source"] == "best_csv":
                            bc = bestcsv.get((method, task, seed))
                            if bc is not None:
                                test_vals.append(bc[1])
                        else:  # rerun
                            tv = retest.get((hospital, task, seed, *arch))
                            if tv is not None:
                                test_vals.append(tv)
                    if not test_vals:
                        continue
                    rows.append({
                        "method": METHOD_DISPLAY[method],
                        "task": TASK_DISPLAY[task],
                        "metric": "AUPRC",
                        "cutoff": cutoff,
                        "mean_pct": round(float(np.mean(test_vals)) * 100, 2),
                        "std_pct": round(float(np.std(test_vals, ddof=1)) * 100, 2)
                                   if len(test_vals) > 1 else 0.0,
                        "n_seeds": len(test_vals),
                    })
        out_df = pd.DataFrame(rows)
        out_path = args.out_dir / f"main_table_anytime_{hospital}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"✓ {hospital}: {len(out_df)} rows  →  {out_path}")
        # Wide pivot for quick inspection
        if len(out_df) > 0:
            piv = out_df.pivot_table(index=["method", "task"], columns="cutoff",
                                     values="mean_pct")
            print(piv.to_string())


if __name__ == "__main__":
    main()
