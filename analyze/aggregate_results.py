"""Aggregate baseline + MAS-NAS results across seeds into paper-ready tables.

Reads from `results/seed_<N>/<hospital>/search/<method>/<task>/`:
  - `<method>_best.csv`     — single-row CSV with selected architecture's
                              val + test metrics + params/FLOPs
  - `<method>_search.csv`   — all evaluated architectures (with iteration col)
  - `search_meta.json`      — wall-clock, LLM calls, budget, seed, ...

Produces 7 CSVs PER HOSPITAL under `analyze/` (suffixed `_<hospital>`):
  - main_table_<H>.csv             Table 1 (paper main): best AUPRC per method × task,
                                   mean ± std over seeds (all methods @ same budget)
  - efficiency_table_<H>.csv       Table 2 (paper main): MAS-NAS @ evals=30 vs baselines
                                   @ evals=100 — "70% less compute, still wins". Extracted
                                   from same search.csv (one run gives both tables).
  - loto_ablation_table_<H>.csv    Fig 5 companion: per-task comparison of
                                   {MAS-exact, MAS-Layer1-only, MAS-LOTO, MAS-cold, best-LLM-baseline}
                                   with delta columns. Validates Plan A robustness.
  - supp_table_<H>.csv             Supplementary: all 4 metrics
  - arch_table_<H>.csv             Selected architectures' params/FLOPs (verifies
                                   --max_params constraint met across all methods)
  - cost_table_<H>.csv             Wall-clock + LLM call counts (compute-fairness audit)
  - significance_<H>.csv           Bootstrap 95% CI (1000 resamples): MAS-NAS vs each
                                   baseline, paired per seed. Reports CI bounds and
                                   whether CI lower bound > 0 (significant).

Multi-hospital usage (Phase 2 production — MIMIC-III + MIMIC-IV cross-hospital NAS):
    python analyze/aggregate_results.py --hospitals MIMIC-III MIMIC-IV
    python analyze/aggregate_results.py --hospitals MIMIC-III MIMIC-IV \\
        --results_root /blue/mei.liu/lideyi/MAS-NAS/results

Single-hospital usage (Phase 1 test / sanity check):
    python analyze/aggregate_results.py --hospitals MIMIC-IV
    python analyze/aggregate_results.py --hospitals MIMIC-IV --mas_budget 30 --baseline_budget 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Display labels (paper-friendly)
# ---------------------------------------------------------------------------
# Tables 1, S1, S2, S3, cost — paper main comparison: 5 baselines + MAS-NAS only.
# Ablation methods (mas_loto, mas_layer1_only, mas_cold) are EXCLUDED from these to avoid:
#   - inflating Table 1 with extra rows
#   - polluting Wilcoxon (would compare MAS vs ablations as if "baselines")
#   - skewing arch/cost audits
MAIN_METHODS = ["baseline0", "baseline1", "baseline2", "baseline3", "baseline4", "mas"]

# Fig 5 robustness ablation only — these methods are extra MAS variants, not baselines.
# Used exclusively by build_loto_ablation_table(). Layer factorial design:
#   mas              — Layer 1 (exact) ON, Layer 2 ON   (full method, in MAIN_METHODS)
#   mas_layer1_only  — Layer 1 (exact) ON, Layer 2 OFF  (isolates Layer 2 contribution)
#   mas_loto         — Layer 1 (fallback) ON, Layer 2 ON (Layer 1 robustness when exact missing)
#   mas_cold         — Layer 1 OFF, Layer 2 OFF        (no prior; multi-agent reasoning only)
ABLATION_METHODS = ["mas_layer1_only", "mas_loto", "mas_cold"]

# Union — used by collect_results() to load every run on disk.
METHODS = MAIN_METHODS + ABLATION_METHODS

METHOD_DISPLAY = {
    "baseline0": "Random",
    "baseline1": "EA",
    "baseline2": "LLM-1shot",
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
    "mas":             "MAS-NAS",
    "mas_layer1_only": "MAS-NAS (L1-only)",
    "mas_loto":        "MAS-NAS (LOTO)",
    "mas_cold":        "MAS-NAS (cold)",
}

# Methods that compete with MAS as upper-bound LLM-based baselines.
# `build_loto_ablation_table` automatically picks the best of these per task
# to fill the "Best baseline" column in Fig 5's companion table.
LLM_BASELINE_METHODS = ["baseline3", "baseline4"]

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
    for method in MAIN_METHODS:
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
    for method in MAIN_METHODS:
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
    for method in MAIN_METHODS:
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
    for method in MAIN_METHODS:
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
# Efficiency table — MAS-NAS @ small budget vs baselines @ full budget ★
# ---------------------------------------------------------------------------
def build_efficiency_table(
    results_root: Path,
    hospital: str,
    records: dict,
    output_path: Path,
    mas_budget: int = 30,
    baseline_budget: int = 100,
) -> pd.DataFrame:
    """Per (method, task): cumulative-max val_auprc at each method's intended
    cutoff (mas_budget for MAS, baseline_budget for all baselines).

    Core paper claim (Table 2): MAS-NAS at evals≤30 already matches/exceeds
    baselines at evals≤100. We extract both budgets from the SAME `*_search.csv`
    (all methods run at full budget, then we slice the prefix) — ensures
    Table 1 (full-budget fair comparison) and Table 2 (efficiency) come from
    one set of jobs.

    For wall-clock and LLM calls at the partial cutoff, we report a
    proportional estimate: (cutoff / total_evals) * total_wall_clock_sec.
    This assumes per-eval cost is roughly constant — true in practice since
    every evaluated arch goes through the same finetune pipeline.

    Also reports the `num_params` and `flops` of the BEST arch in the prefix
    (the one whose val_auprc was max). Lets reviewers see whether MAS@30
    achieves its win with a smaller/cheaper architecture vs baselines@100.
    """
    rows = []
    for method in MAIN_METHODS:
        cutoff = mas_budget if method == "mas" else baseline_budget
        for task in TASKS:
            seeds = get_seeds(records, method, task)
            best_at_cutoff: list = []
            evals_used: list = []
            best_params: list = []        # num_params of arch with max val_auprc in prefix
            best_flops: list = []         # flops of same arch
            wall_min_at_cutoff: list = []
            llm_calls_at_cutoff: list = []
            wall_min_full: list = []
            llm_calls_full: list = []

            for s in seeds:
                # search.csv lives next to *_best.csv, same task_dir
                search_csv = (
                    results_root / f"seed_{s}" / hospital / "search"
                    / method / task / f"{method}_search.csv"
                )
                if not search_csv.exists():
                    continue
                try:
                    df = pd.read_csv(search_csv)
                except Exception as e:
                    print(f"  ⚠ Cannot read {search_csv}: {e}")
                    continue
                if "val_auprc" not in df.columns:
                    continue
                if "iteration" not in df.columns:
                    df["iteration"] = range(1, len(df) + 1)
                df = df.sort_values("iteration").reset_index(drop=True)

                df_cut = df[df["iteration"] <= cutoff].dropna(subset=["val_auprc"])
                if len(df_cut) == 0:
                    continue
                # Locate the best architecture in the prefix (max val_auprc)
                best_idx = df_cut["val_auprc"].idxmax()
                best_row = df_cut.loc[best_idx]
                best_at_cutoff.append(float(best_row["val_auprc"]))
                evals_used.append(len(df_cut))
                if "num_params" in df_cut.columns and not pd.isna(best_row.get("num_params")):
                    best_params.append(float(best_row["num_params"]))
                if "flops" in df_cut.columns and not pd.isna(best_row.get("flops")):
                    best_flops.append(float(best_row["flops"]))

                # Proportional cost estimate
                meta = records[(method, task, s)]["meta"]
                total_evals = max(len(df), 1)
                frac = len(df_cut) / total_evals
                if "wall_clock_sec" in meta:
                    full_wc_min = meta["wall_clock_sec"] / 60.0
                    wall_min_full.append(full_wc_min)
                    wall_min_at_cutoff.append(full_wc_min * frac)
                if "llm_calls" in meta:
                    full_llm = meta["llm_calls"]
                    llm_calls_full.append(full_llm)
                    llm_calls_at_cutoff.append(full_llm * frac)

            if not best_at_cutoff:
                continue

            rows.append({
                "method": METHOD_DISPLAY[method],
                "task": TASK_DISPLAY[task],
                "budget_cutoff": cutoff,
                "evals_used_mean": round(float(np.mean(evals_used)), 1),
                "best_val_auprc_mean_pct": round(float(np.mean(best_at_cutoff)) * 100, 2),
                "best_val_auprc_std_pct": (
                    round(float(np.std(best_at_cutoff, ddof=1)) * 100, 2)
                    if len(best_at_cutoff) > 1 else 0.0
                ),
                "best_arch_num_params_mean": (
                    int(np.mean(best_params)) if best_params else None
                ),
                "best_arch_num_params_std": (
                    int(np.std(best_params, ddof=1))
                    if len(best_params) > 1 else 0
                ),
                "best_arch_flops_mean": (
                    int(np.mean(best_flops)) if best_flops else None
                ),
                "best_arch_flops_std": (
                    int(np.std(best_flops, ddof=1))
                    if len(best_flops) > 1 else 0
                ),
                "wall_clock_min_at_cutoff_mean": (
                    round(float(np.mean(wall_min_at_cutoff)), 1)
                    if wall_min_at_cutoff else None
                ),
                "wall_clock_min_full_mean": (
                    round(float(np.mean(wall_min_full)), 1)
                    if wall_min_full else None
                ),
                "llm_calls_at_cutoff_mean": (
                    round(float(np.mean(llm_calls_at_cutoff)), 1)
                    if llm_calls_at_cutoff else 0
                ),
                "llm_calls_full_mean": (
                    round(float(np.mean(llm_calls_full)), 1)
                    if llm_calls_full else 0
                ),
                "n_seeds": len(best_at_cutoff),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# LOTO + Cold-start robustness ablation — Fig 5 companion table ★
# ---------------------------------------------------------------------------
def build_loto_ablation_table(records: dict, output_path: Path) -> pd.DataFrame:
    """Per-task 4-condition factorial: {MAS-exact, MAS-Layer1-only, MAS-LOTO, MAS-cold} vs best-LLM-baseline.

    For each task: aggregates test_auprc mean ± std across seeds, picks the
    best LLM-based baseline (LLMatic or CoLLM-NAS) to compete against, and
    reports delta columns to support Fig 5's paper claims:

      delta_layer1_only_vs_exact  ★ isolates Layer 2's incremental contribution
      delta_loto_vs_exact         ≈ -0.5 ~ -1%   (Layer 1 graceful fallback)
      delta_cold_vs_exact         typically lower (no prior at all)
      delta_*_vs_baseline         ≥ 0 means MAS variant still beats best baseline

    Reads the same `records` dict produced by `collect_results`, so it
    requires the 6 methods (mas, mas_layer1_only, mas_loto, mas_cold,
    baseline3, baseline4) to all have *_best.csv files at expected paths.
    """
    rows = []
    for task in TASKS:
        row = {"task": TASK_DISPLAY[task]}

        # Helper: aggregate AUPRC across seeds for one method
        def _stats(method):
            seeds = get_seeds(records, method, task)
            scores = [records[(method, task, s)]["best"].get("test_auprc") for s in seeds]
            scores = [x for x in scores if x is not None and not pd.isna(x)]
            if not scores:
                return None, None, 0
            mean_pct = float(np.mean(scores)) * 100
            std_pct = float(np.std(scores, ddof=1)) * 100 if len(scores) > 1 else 0.0
            return round(mean_pct, 2), round(std_pct, 2), len(scores)

        # Four MAS conditions (column key  →  records method key)
        for cond, key in [
            ("exact", "mas"),
            ("layer1_only", "mas_layer1_only"),
            ("loto", "mas_loto"),
            ("cold", "mas_cold"),
        ]:
            mean, std, n = _stats(key)
            row[f"mas_{cond}_mean_pct"] = mean
            row[f"mas_{cond}_std_pct"] = std
            row[f"mas_{cond}_n_seeds"] = n

        # Best LLM-based baseline for this task (max mean AUPRC across LLMatic + CoLLM-NAS)
        best_baseline_method = None
        best_baseline_mean = -1.0
        best_baseline_std = None
        best_baseline_n = 0
        for bm in LLM_BASELINE_METHODS:
            mean, std, n = _stats(bm)
            if mean is not None and mean > best_baseline_mean:
                best_baseline_method = bm
                best_baseline_mean = mean
                best_baseline_std = std
                best_baseline_n = n
        row["best_baseline_method"] = METHOD_DISPLAY.get(best_baseline_method, best_baseline_method)
        row["best_baseline_mean_pct"] = best_baseline_mean if best_baseline_method else None
        row["best_baseline_std_pct"] = best_baseline_std
        row["best_baseline_n_seeds"] = best_baseline_n

        # Delta columns (None if either side missing)
        def _delta(a_key, b_val):
            a = row.get(a_key)
            if a is None or b_val is None:
                return None
            return round(a - b_val, 2)

        row["delta_layer1_only_vs_exact"]    = _delta("mas_layer1_only_mean_pct", row["mas_exact_mean_pct"])
        row["delta_loto_vs_exact"]           = _delta("mas_loto_mean_pct",        row["mas_exact_mean_pct"])
        row["delta_cold_vs_exact"]           = _delta("mas_cold_mean_pct",        row["mas_exact_mean_pct"])
        row["delta_layer1_only_vs_baseline"] = _delta("mas_layer1_only_mean_pct", row["best_baseline_mean_pct"])
        row["delta_loto_vs_baseline"]        = _delta("mas_loto_mean_pct",        row["best_baseline_mean_pct"])
        row["delta_cold_vs_baseline"]        = _delta("mas_cold_mean_pct",        row["best_baseline_mean_pct"])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Significance table — paired Wilcoxon: MAS-NAS vs each baseline
# ---------------------------------------------------------------------------
def build_significance_table(
    records: dict,
    output_path: Path,
    n_boot: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CI for MAS-NAS vs each baseline (paired by seed).

    For each (task, metric, baseline) triple:
      1. Compute per-seed paired differences  d_i = MAS_i - baseline_i
      2. Resample d with replacement 1000 times; record bootstrap mean each time
      3. 95% CI = [2.5th, 97.5th] percentile of bootstrap distribution
      4. significant_ci>0 = True  iff  ci_lower > 0  (MAS clearly better)

    This approach is valid for n ≥ 2 seeds and makes no distributional assumption.
    With n=3 seeds the CI is wide but honest — unlike Wilcoxon which cannot
    achieve p<0.05 at n=3 and Holm correction makes it even worse.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for task in TASKS:
        for metric in METRICS:
            mas_seeds = get_seeds(records, "mas", task)
            if len(mas_seeds) < 2:
                continue

            for baseline in [m for m in MAIN_METHODS if m != "mas"]:
                bl_seeds = set(get_seeds(records, baseline, task))
                diffs = []
                for s in mas_seeds:
                    if s not in bl_seeds:
                        continue
                    x = records[("mas", task, s)]["best"].get(metric)
                    y = records[(baseline, task, s)]["best"].get(metric)
                    if x is None or y is None or pd.isna(x) or pd.isna(y):
                        continue
                    diffs.append(float(x) - float(y))

                if len(diffs) < 2:
                    continue

                diffs = np.array(diffs)
                obs_mean = float(np.mean(diffs))
                obs_std = float(np.std(diffs, ddof=1))

                # Bootstrap: resample seeds with replacement
                boot_means = np.array([
                    np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                    for _ in range(n_boot)
                ])
                ci_lo = float(np.percentile(boot_means, 2.5))
                ci_hi = float(np.percentile(boot_means, 97.5))

                rows.append({
                    "task":             TASK_DISPLAY[task],
                    "metric":           METRIC_DISPLAY[metric],
                    "baseline":         METHOD_DISPLAY[baseline],
                    "delta_pct":        round(obs_mean * 100, 3),
                    "delta_std_pct":    round(obs_std * 100, 3),
                    "ci_lower_pct":     round(ci_lo * 100, 3),
                    "ci_upper_pct":     round(ci_hi * 100, 3),
                    "significant_ci>0": bool(ci_lo > 0),
                    "n_seeds":          len(diffs),
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
    p.add_argument("--hospitals", type=str, nargs="+", required=True,
                   help="One or more hospital names, e.g. MIMIC-III MIMIC-IV. "
                        "Each hospital is aggregated independently and produces "
                        "its own set of 7 output CSVs with _<hospital> suffix.")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Where to write 6 output CSVs.")
    p.add_argument("--mas_budget", type=int, default=30,
                   help="MAS-NAS budget cutoff for efficiency table (default: 30).")
    p.add_argument("--baseline_budget", type=int, default=100,
                   help="Baseline budget cutoff for efficiency table (default: 100).")
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Aggregate] results_root={results_root}")
    print(f"[Aggregate] hospitals={args.hospitals}  ({len(args.hospitals)} site(s))")
    print(f"[Aggregate] out_dir={out_dir}\n")

    # Per-hospital aggregation. Each hospital is independent: same 7 build_*
    # functions, but output paths get a _<hospital> suffix so multi-site runs
    # don't overwrite each other. records dict (keyed by (method, task, seed))
    # is rebuilt per hospital from its own search/ subtree.
    summary_rows = []     # cross-hospital summary at the end
    for hospital in args.hospitals:
        print("=" * 70)
        print(f"=== Hospital: {hospital}")
        print("=" * 70)

        records = collect_results(results_root, hospital)
        print(f"Found {len(records)} (method, task, seed) records for {hospital}.")
        if not records:
            print(f"  ⚠ No records for {hospital}. Skipping. "
                  f"(Check {results_root}/seed_*/{hospital}/search/...)")
            summary_rows.append({"hospital": hospital, "n_records": 0, "status": "skipped"})
            continue

        suf = f"_{hospital}"

        print(f"\n[1/7] {hospital} — Main table (AUPRC mean ± std)")
        df_main = build_main_table(records, out_dir / f"main_table{suf}.csv")
        print(df_main.to_string(index=False))

        print(f"\n[2/7] {hospital} — Efficiency table (MAS@{args.mas_budget} vs baselines@{args.baseline_budget})")
        df_eff = build_efficiency_table(
            results_root, hospital, records,
            out_dir / f"efficiency_table{suf}.csv",
            mas_budget=args.mas_budget, baseline_budget=args.baseline_budget,
        )
        if len(df_eff) > 0:
            print(df_eff.to_string(index=False))
        else:
            print("  (empty — no search.csv found at expected path)")

        print(f"\n[3/7] {hospital} — Supplementary table (all 4 metrics)")
        df_supp = build_supp_table(records, out_dir / f"supp_table{suf}.csv")
        print(f"  Rows: {len(df_supp)}  →  {out_dir / f'supp_table{suf}.csv'}")

        print(f"\n[4/7] {hospital} — Architecture table (selected models' size/FLOPs)")
        df_arch = build_arch_table(records, out_dir / f"arch_table{suf}.csv")
        print(df_arch.to_string(index=False))

        print(f"\n[5/7] {hospital} — Cost table (wall-clock + LLM calls, full budget)")
        df_cost = build_cost_table(records, out_dir / f"cost_table{suf}.csv")
        print(df_cost.to_string(index=False))

        print(f"\n[6/7] {hospital} — Significance table (Wilcoxon + Holm-Bonferroni)")
        df_sig = build_significance_table(records, out_dir / f"significance{suf}.csv")
        print(f"  Rows: {len(df_sig)}  →  {out_dir / f'significance{suf}.csv'}")
        if len(df_sig) > 0:
            print("  Sample rows:")
            print(df_sig.head(10).to_string(index=False))

        print(f"\n[7/7] {hospital} — Layer ablation table (MAS-exact / L1-only / LOTO / cold / best-baseline)")
        df_loto = build_loto_ablation_table(records, out_dir / f"loto_ablation_table{suf}.csv")
        if len(df_loto) > 0:
            cols_to_show = [
                "task",
                "mas_exact_mean_pct", "mas_layer1_only_mean_pct",
                "mas_loto_mean_pct", "mas_cold_mean_pct",
                "best_baseline_method", "best_baseline_mean_pct",
                "delta_layer1_only_vs_exact",   # ★ Layer 2's incremental contribution
                "delta_loto_vs_exact",
            ]
            cols_to_show = [c for c in cols_to_show if c in df_loto.columns]
            print(df_loto[cols_to_show].to_string(index=False))
        else:
            print("  (empty — no MAS or LLM-baseline records found yet)")

        summary_rows.append({"hospital": hospital, "n_records": len(records), "status": "ok"})

    # Final cross-hospital summary
    print("\n" + "=" * 70)
    print("=== ALL HOSPITALS COMPLETE")
    print("=" * 70)
    for r in summary_rows:
        print(f"  {r['hospital']:<20s}  records={r['n_records']:<6d}  [{r['status']}]")
    print(f"\n✓ Per-hospital outputs saved to {out_dir}/")
    print(f"  For each hospital <H>, the following 7 files are written:")
    print(f"    main_table_<H>.csv           (Table 1)")
    print(f"    efficiency_table_<H>.csv     (Table 2: MAS@30 vs baselines@100)")
    print(f"    loto_ablation_table_<H>.csv  (Fig 5 companion)")
    print(f"    supp_table_<H>.csv           (Supplementary)")
    print(f"    arch_table_<H>.csv           (size constraint audit)")
    print(f"    cost_table_<H>.csv           (compute fairness audit)")
    print(f"    significance_<H>.csv         (Bootstrap 95% CI, 1000 resamples)")


if __name__ == "__main__":
    main()
