"""Aggregate baseline + MAS-NAS results across seeds into paper-ready tables.

Reads from `results/seed_<N>/<hospital>/search/<method>/<task>/`:
  - `<method>_best.csv`     — single-row CSV with selected architecture's
                              val + test metrics + params/FLOPs
  - `<method>_search.csv`   — all evaluated architectures (with iteration col)
  - `search_meta.json`      — wall-clock, LLM calls, budget, seed, ...

Produces 5 CSVs under `analyze/`:
  - main_table.csv         Table 1 (paper main): best AUPRC per method × task,
                           mean ± std over seeds
  - supp_table.csv         Supplementary: all 4 metrics
  - arch_table.csv         Selected architectures' params/FLOPs (verifies
                           --max_params constraint met across all methods)
  - cost_table.csv         Wall-clock + LLM call counts (compute-fairness audit)
  - significance.csv       Paired Wilcoxon: MAS-NAS vs each baseline, with
                           Holm-Bonferroni correction across 5 baselines

Usage:
    python analyze/aggregate_results.py --hospital MIMIC-IV
    python analyze/aggregate_results.py --hospital MIMIC-IV --results_root /blue/.../results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Display labels (paper-friendly)
# ---------------------------------------------------------------------------
METHODS = ["baseline0", "baseline1", "baseline2", "baseline3", "baseline4", "mas"]
METHOD_DISPLAY = {
    "baseline0": "Random",
    "baseline1": "EA",
    "baseline2": "LLM-1shot",
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
    "mas": "MAS-NAS",
}

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
TASK_DISPLAY = {
    "death": "Death",
    "stay": "Stay>7d",
    "readmission": "Readmit-3M",
    "next_diag_6m_pheno": "Pheno-6M",
    "next_diag_12m_pheno": "Pheno-12M",
}

METRICS = ["test_accuracy", "test_f1", "test_auroc", "test_auprc"]
METRIC_DISPLAY = {
    "test_accuracy": "Accuracy",
    "test_f1": "F1",
    "test_auroc": "AUROC",
    "test_auprc": "AUPRC",
}


# ---------------------------------------------------------------------------
# Walk filesystem, return dict keyed by (method, task, seed)
# ---------------------------------------------------------------------------
def collect_results(results_root: Path, hospital: str) -> dict:
    """Glob over results/seed_*/hospital/search/method/task/ and load
    {*_best.csv first row, search_meta.json}. Missing files are tolerated;
    partial seeds just produce smaller mean estimates."""
    records: dict = {}
    seed_dirs = sorted(results_root.glob("seed_*"))
    if not seed_dirs:
        # Fall back: also try results_root directly (no seed subdir, single-seed legacy)
        if (results_root / hospital / "search").exists():
            seed_dirs = [results_root]   # treat as seed=None

    for seed_dir in seed_dirs:
        # Parse seed number, or None for single-seed legacy
        if seed_dir.name.startswith("seed_"):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                print(f"  ⚠ Cannot parse seed from dir name: {seed_dir.name}, skipping.")
                continue
        else:
            seed = None  # legacy single-seed run

        search_root = seed_dir / hospital / "search"
        if not search_root.exists():
            continue

        for method_dir in search_root.iterdir():
            method = method_dir.name
            if method not in METHODS:
                continue
            for task_dir in method_dir.iterdir():
                task = task_dir.name
                if task not in TASKS:
                    continue
                # best CSV
                best_files = list(task_dir.glob("*_best.csv"))
                if not best_files:
                    continue
                try:
                    best_df = pd.read_csv(best_files[0])
                except Exception as e:
                    print(f"  ⚠ Cannot read {best_files[0]}: {e}")
                    continue
                if len(best_df) == 0:
                    continue
                best_row = best_df.iloc[0].to_dict()
                # meta JSON
                meta_path = task_dir / "search_meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.load(open(meta_path))
                    except Exception:
                        pass
                records[(method, task, seed)] = {"best": best_row, "meta": meta}
    return records


def get_seeds(records: dict, method: str, task: str) -> list:
    """All seeds available for a (method, task) combination."""
    return sorted([s for (m, t, s) in records if m == method and t == task and s is not None],
                  key=lambda x: x if x is not None else -1)


# ---------------------------------------------------------------------------
# Main table — AUPRC × method × task
# ---------------------------------------------------------------------------
def build_main_table(records: dict, output_path: Path) -> pd.DataFrame:
    """Best test AUPRC per (method, task), mean ± std over seeds. The cell
    values match Table 1 of the paper. Bold-the-max in LaTeX downstream."""
    rows = []
    for method in METHODS:
        for task in TASKS:
            seeds = get_seeds(records, method, task)
            scores = [records[(method, task, s)]["best"].get("test_auprc") for s in seeds]
            scores = [x for x in scores if x is not None and not pd.isna(x)]
            if not scores:
                rows.append({
                    "method": METHOD_DISPLAY[method], "task": TASK_DISPLAY[task],
                    "metric": "AUPRC",
                    "mean_pct": None, "std_pct": None, "n_seeds": 0, "scores_pct": [],
                })
                continue
            rows.append({
                "method": METHOD_DISPLAY[method],
                "task": TASK_DISPLAY[task],
                "metric": "AUPRC",
                "mean_pct": round(float(np.mean(scores)) * 100, 2),
                "std_pct": round(float(np.std(scores, ddof=1)) * 100, 2) if len(scores) > 1 else 0.0,
                "n_seeds": len(scores),
                "scores_pct": [round(s * 100, 2) for s in scores],
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Supplementary table — 4 metrics × method × task
# ---------------------------------------------------------------------------
def build_supp_table(records: dict, output_path: Path) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for task in TASKS:
            seeds = get_seeds(records, method, task)
            for metric in METRICS:
                scores = [records[(method, task, s)]["best"].get(metric) for s in seeds]
                scores = [x for x in scores if x is not None and not pd.isna(x)]
                if not scores:
                    continue
                rows.append({
                    "method": METHOD_DISPLAY[method],
                    "task": TASK_DISPLAY[task],
                    "metric": METRIC_DISPLAY[metric],
                    "mean_pct": round(float(np.mean(scores)) * 100, 2),
                    "std_pct": round(float(np.std(scores, ddof=1)) * 100, 2) if len(scores) > 1 else 0.0,
                    "n_seeds": len(scores),
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Architecture table — params + FLOPs of selected models
# ---------------------------------------------------------------------------
def build_arch_table(records: dict, output_path: Path) -> pd.DataFrame:
    """Verifies all methods picked architectures within --max_params constraint.
    Reports mean params / FLOPs of the chosen architecture per (method, task)."""
    rows = []
    for method in METHODS:
        for task in TASKS:
            seeds = get_seeds(records, method, task)
            params = [records[(method, task, s)]["best"].get("num_params") for s in seeds]
            flops = [records[(method, task, s)]["best"].get("flops") for s in seeds]
            params = [p for p in params if p is not None and not pd.isna(p)]
            flops = [f for f in flops if f is not None and not pd.isna(f)]
            if not params:
                continue
            rows.append({
                "method": METHOD_DISPLAY[method],
                "task": TASK_DISPLAY[task],
                "params_mean": int(np.mean(params)),
                "params_std": int(np.std(params, ddof=1)) if len(params) > 1 else 0,
                "params_min": int(np.min(params)),
                "params_max": int(np.max(params)),
                "flops_mean": int(np.mean(flops)) if flops else None,
                "n_seeds": len(params),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Cost table — wall-clock + LLM calls
# ---------------------------------------------------------------------------
def build_cost_table(records: dict, output_path: Path) -> pd.DataFrame:
    """Per-method total compute cost averaged across (task × seed) runs."""
    rows = []
    for method in METHODS:
        wall_clocks: list = []
        llm_calls: list = []
        for task in TASKS:
            for seed in get_seeds(records, method, task):
                meta = records[(method, task, seed)]["meta"]
                if "wall_clock_sec" in meta:
                    wall_clocks.append(meta["wall_clock_sec"])
                if "llm_calls" in meta:
                    llm_calls.append(meta["llm_calls"])
        if not wall_clocks:
            continue
        rows.append({
            "method": METHOD_DISPLAY[method],
            "wall_clock_min_mean": round(float(np.mean(wall_clocks)) / 60, 1),
            "wall_clock_min_std": round(float(np.std(wall_clocks, ddof=1)) / 60, 1) if len(wall_clocks) > 1 else 0,
            "wall_clock_min_total": round(float(np.sum(wall_clocks)) / 60, 1),
            "llm_calls_mean": round(float(np.mean(llm_calls)), 1) if llm_calls else 0,
            "llm_calls_total": int(np.sum(llm_calls)) if llm_calls else 0,
            "n_runs": len(wall_clocks),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Significance table — paired Wilcoxon: MAS-NAS vs each baseline
# ---------------------------------------------------------------------------
def build_significance_table(records: dict, output_path: Path) -> pd.DataFrame:
    """For each (task, metric):
      - Pair MAS-NAS scores with each baseline by seed
      - Wilcoxon signed-rank test (one-sided: MAS > baseline)
      - Holm-Bonferroni correction across 5 baselines per (task, metric)
    """
    rows = []
    for task in TASKS:
        for metric in METRICS:
            mas_seeds = get_seeds(records, "mas", task)
            if len(mas_seeds) < 2:
                continue

            # Collect per-baseline comparisons for this (task, metric)
            comparisons = []
            for baseline in [m for m in METHODS if m != "mas"]:
                bl_seeds = set(get_seeds(records, baseline, task))
                paired = []
                for s in mas_seeds:
                    if s not in bl_seeds:
                        continue
                    x = records[("mas", task, s)]["best"].get(metric)
                    y = records[(baseline, task, s)]["best"].get(metric)
                    if x is None or y is None or pd.isna(x) or pd.isna(y):
                        continue
                    paired.append((x, y))
                if len(paired) < 2:
                    continue
                xs = np.array([x for x, _ in paired])
                ys = np.array([y for _, y in paired])
                try:
                    if np.allclose(xs, ys):
                        # Wilcoxon undefined when all differences are 0
                        p_raw = 1.0
                    else:
                        _, p_raw = wilcoxon(xs, ys, alternative="greater")
                except ValueError:
                    p_raw = float("nan")
                comparisons.append({
                    "baseline": baseline,
                    "p_raw": float(p_raw),
                    "delta_mean": float(np.mean(xs - ys)),
                    "delta_std": float(np.std(xs - ys, ddof=1)) if len(xs) > 1 else 0.0,
                    "n_pairs": len(paired),
                })

            # Holm-Bonferroni correction across the 5 baselines
            ps = [c["p_raw"] for c in comparisons]
            valid_ps = [p for p in ps if not (p is None or pd.isna(p))]
            if len(valid_ps) >= 1:
                _, p_adj_arr, _, _ = multipletests(ps, method="holm")
                for c, p_adj in zip(comparisons, p_adj_arr):
                    c["p_adj_holm"] = float(p_adj)
            else:
                for c in comparisons:
                    c["p_adj_holm"] = c["p_raw"]

            for c in comparisons:
                rows.append({
                    "task": TASK_DISPLAY[task],
                    "metric": METRIC_DISPLAY[metric],
                    "baseline": METHOD_DISPLAY[c["baseline"]],
                    "delta_pct": round(c["delta_mean"] * 100, 3),
                    "delta_std_pct": round(c["delta_std"] * 100, 3),
                    "p_raw": round(c["p_raw"], 4) if not pd.isna(c["p_raw"]) else None,
                    "p_adj_holm": round(c["p_adj_holm"], 4) if not pd.isna(c["p_adj_holm"]) else None,
                    "significant_p<0.05": (c["p_adj_holm"] < 0.05) if not pd.isna(c["p_adj_holm"]) else None,
                    "n_pairs": c["n_pairs"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results_root", type=str, default="./results",
                   help="Root containing seed_<N>/ subdirectories (or single seed root).")
    p.add_argument("--hospital", type=str, required=True,
                   help="Hospital name, e.g. MIMIC-IV.")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Where to write 5 output CSVs.")
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Aggregate] results_root={results_root}")
    print(f"[Aggregate] hospital={args.hospital}")
    print(f"[Aggregate] out_dir={out_dir}\n")

    records = collect_results(results_root, args.hospital)
    print(f"Found {len(records)} (method, task, seed) records.")
    if not records:
        print("⚠ No records. Check --results_root + glob layout (results/seed_*/hospital/search/method/task/).")
        return

    print("\n[1/5] Main table — AUPRC mean ± std")
    df_main = build_main_table(records, out_dir / "main_table.csv")
    print(df_main.to_string(index=False))

    print("\n[2/5] Supplementary table — all 4 metrics")
    df_supp = build_supp_table(records, out_dir / "supp_table.csv")
    print(f"  Rows: {len(df_supp)}  →  {out_dir / 'supp_table.csv'}")

    print("\n[3/5] Architecture table — selected models' size/FLOPs")
    df_arch = build_arch_table(records, out_dir / "arch_table.csv")
    print(df_arch.to_string(index=False))

    print("\n[4/5] Cost table — wall-clock + LLM calls")
    df_cost = build_cost_table(records, out_dir / "cost_table.csv")
    print(df_cost.to_string(index=False))

    print("\n[5/5] Significance table — Wilcoxon w/ Holm-Bonferroni")
    df_sig = build_significance_table(records, out_dir / "significance.csv")
    print(f"  Rows: {len(df_sig)}  →  {out_dir / 'significance.csv'}")
    if len(df_sig) > 0:
        print("  Sample rows:")
        print(df_sig.head(10).to_string(index=False))

    print(f"\n✓ All 5 tables saved to {out_dir}/")
    print("  - main_table.csv      (Table 1, paper main)")
    print("  - supp_table.csv      (Supplementary)")
    print("  - arch_table.csv      (size constraint audit)")
    print("  - cost_table.csv      (compute fairness audit)")
    print("  - significance.csv    (Wilcoxon + Holm-Bonferroni)")


if __name__ == "__main__":
    main()
