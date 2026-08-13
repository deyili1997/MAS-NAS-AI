"""Bootstrap 95% CI significance for the anytime main table: is ATHENA (mas)
significantly better than EACH baseline at search budget = 5, 10, 20, 30?

Mirrors analyze/build_anytime_table.py for data assembly, but instead of
aggregating to mean/std it collects per-seed paired test-AUPRC values and runs
the SAME bootstrap procedure as build_significance_table() in aggregate_results.py:

  1. per-seed paired difference  d_i = ATHENA_i - baseline_i   (AUPRC, paired by seed)
  2. resample d with replacement n_boot times; record bootstrap mean each time
  3. 95% CI = [2.5th, 97.5th] percentile of the bootstrap distribution
  4. significant_ci>0 = True  iff  ci_lower > 0   (ATHENA clearly better)

Identical knobs to the @30 main significance: n_boot=1000, rng_seed=42, metric=AUPRC.
The cutoff=30 panel here MUST reproduce significance_<H>.csv (AUPRC rows) exactly
— a built-in consistency check.

Data sources (same as build_anytime_table.py):
  • anytime_selection_map.csv  (which arch each method/cutoff selected, + source)
  • retest/*.json              (test metrics from the re-run finetunes)
  • results/.../*_best.csv      (test metrics for selections reusing the final arch)

Output (per hospital):
  • anytime_significance_<H>.csv :
      task, baseline, cutoff, delta_pct, delta_std_pct,
      ci_lower_pct, ci_upper_pct, significant_ci>0, n_seeds

Usage:
    python analyze/build_anytime_significance.py \
        --results_root /blue/mei.liu/lideyi/MAS-NAS/results \
        --anytime_dir  /blue/mei.liu/lideyi/MAS-NAS/analyze/anytime \
        --hospitals source_15 MIMIC-IV \
        --out_dir /blue/mei.liu/lideyi/MAS-NAS/analyze/<DATE>
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
BASELINES = ["baseline0", "baseline1", "baseline2", "baseline4"]  # ATHENA vs each; re-derived from --methods in main()
TASK_DISPLAY = {
    "death": "Death", "stay": "Stay>7d", "readmission": "Readmission (3M)",
    "next_diag_6m_pheno": "Phenotype (6M)", "next_diag_12m_pheno": "Phenotype (12M)",
    "med_rec": "Drug Recommendation",
}
TASK_ORDER = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
CUTOFFS = [5, 10, 20, 30]
ARCH_COLS = ["embed_dim", "depth", "mlp_ratio", "num_heads"]
METRIC = "test_auprc"  # table metric


def _retest_lookup(anytime_dir: Path) -> dict:
    """(hospital, task, seed, arch_tuple) -> test_auprc from retest JSONs."""
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
        out[key] = float(d[METRIC])
    return out


def _bestcsv_lookup(results_root: Path, hospital: str) -> dict:
    """(method, task, seed) -> (arch_tuple, test_auprc) from *_best.csv."""
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
                if len(df) == 0 or METRIC not in df.columns:
                    continue
                r = df.iloc[0]
                arch = tuple(int(r[c]) for c in ARCH_COLS)
                out[(method, task, seed)] = (arch, float(r[METRIC]))
    return out


def main():
    global TASK_ORDER, METHOD_ORDER, BASELINES   # so --tasks / --methods can override
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--anytime_dir", required=True, type=Path)
    ap.add_argument("--hospitals", required=True, nargs="+")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--rng_seed", type=int, default=42)
    ap.add_argument("--tasks", type=str, nargs="+", default=TASK_ORDER,
                    help="Tasks to include (default: the main 5). Use --tasks "
                         "med_rec with --results_root results_medrec.")
    ap.add_argument("--methods", type=str, nargs="+", default=METHOD_ORDER,
                    help=f"Methods (default: {METHOD_ORDER}; baseline3/LLMatic "
                         "excluded). ATHENA is compared against each non-mas method. "
                         "Pass a set including baseline3 to add LLMatic.")
    args = ap.parse_args()
    TASK_ORDER = args.tasks
    METHOD_ORDER = args.methods
    BASELINES = [m for m in METHOD_ORDER if m != "mas"]   # ATHENA vs each non-mas method
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(args.anytime_dir / "anytime_selection_map.csv")
    retest = _retest_lookup(args.anytime_dir)

    for hospital in args.hospitals:
        bestcsv = _bestcsv_lookup(args.results_root, hospital)
        hsel = sel[sel["hospital"] == hospital]

        # vals[(method, task, cutoff)][seed] = test_auprc
        vals: dict = {}
        for _, r in hsel.iterrows():
            method, task, cutoff = r["method"], r["task"], int(r["cutoff"])
            seed = int(r["seed"])
            arch = tuple(int(r[c]) for c in ARCH_COLS)
            if r["source"] == "best_csv":
                bc = bestcsv.get((method, task, seed))
                v = bc[1] if bc is not None else None
            else:
                v = retest.get((hospital, task, seed, *arch))
            if v is None:
                continue
            vals.setdefault((method, task, cutoff), {})[seed] = v

        # Fresh RNG per hospital so the cutoff=30 panel matches the standalone run
        rng = np.random.default_rng(args.rng_seed)
        rows = []
        for task in TASK_ORDER:
            for cutoff in CUTOFFS:
                mas = vals.get(("mas", task, cutoff), {})
                if len(mas) < 2:
                    continue
                for baseline in BASELINES:
                    bl = vals.get((baseline, task, cutoff), {})
                    diffs = [mas[s] - bl[s] for s in mas if s in bl]
                    if len(diffs) < 2:
                        continue
                    diffs = np.array(diffs)
                    obs_mean = float(np.mean(diffs))
                    obs_std = float(np.std(diffs, ddof=1))
                    boot_means = np.array([
                        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                        for _ in range(args.n_boot)
                    ])
                    ci_lo = float(np.percentile(boot_means, 2.5))
                    ci_hi = float(np.percentile(boot_means, 97.5))
                    rows.append({
                        "task": TASK_DISPLAY[task],
                        "baseline": METHOD_DISPLAY[baseline],
                        "cutoff": cutoff,
                        "delta_pct": round(obs_mean * 100, 3),
                        "delta_std_pct": round(obs_std * 100, 3),
                        "ci_lower_pct": round(ci_lo * 100, 3),
                        "ci_upper_pct": round(ci_hi * 100, 3),
                        "significant_ci>0": bool(ci_lo > 0),
                        "n_seeds": len(diffs),
                    })

        out_df = pd.DataFrame(rows)
        out_path = args.out_dir / f"anytime_significance_{hospital}.csv"
        out_df.to_csv(out_path, index=False)
        n_sig = int(out_df["significant_ci>0"].sum()) if len(out_df) else 0
        print(f"✓ {hospital}: {len(out_df)} rows, {n_sig} significant  →  {out_path}")
        if len(out_df) > 0:
            piv = out_df.pivot_table(index=["task", "baseline"], columns="cutoff",
                                     values="significant_ci>0")
            print(piv.to_string())


if __name__ == "__main__":
    main()
