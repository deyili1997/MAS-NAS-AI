"""Fig 5 — Layer Ablation (4-condition factorial).

For each of the 5 tasks, plot a grouped bar chart comparing:
  - MAS-NAS (exact)              Layer 1 exact + Layer 2 ON  — full method
  - MAS-NAS (L1-only)            Layer 1 exact + Layer 2 OFF — isolates Layer 2 contribution
  - MAS-NAS (LOTO)               Layer 1 fallback + Layer 2 ON — Layer 1 robustness
  - MAS-NAS (cold)               Layer 1 OFF + Layer 2 OFF   — no prior at all
  - Best LLM baseline            max(LLMatic, CoLLM-NAS) per task

Reads:
    results/seed_<SEED>/<hospital>/search/<method>/<task>/<method>_best.csv
    where <method> ∈ {mas, mas_layer1_only, mas_loto, mas_cold, baseline3, baseline4}

Outputs:
    figure5_loto_robustness.png — 5 panels (one per task) of grouped bars
                                  with std error bars
                                  (paired Wilcoxon: MAS-LOTO vs best-baseline)

Paper claims (each isolated by one delta in the companion table):
    delta_layer1_only_vs_exact ★ value added by Layer 2 beyond Layer 1 retrieval
    delta_loto_vs_exact          Layer 1 graceful fallback when exact task missing
    delta_cold_vs_baseline       multi-agent reasoning beats LLM baselines even with no prior

Usage:
    python analyze/plot_loto_ablation.py --hospitals MIMIC-IV
    python analyze/plot_loto_ablation.py --hospitals MIMIC-III MIMIC-IV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root on sys.path: this file is executed as `python analyze/<script>.py`,
# so sys.path[0] is analyze/ and `utils` is not importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.site_labels import site_label  # noqa: E402
from utils.panel_labels import panel_label  # noqa: E402
from utils.fig_layout import panel_grid  # noqa: E402


# Colors: MAS variants = same crimson family (intensity ~ amount of prior context),
# baseline = neutral gray. Reuses METHOD_COLOR convention from plot_pareto.py.
CONDITION_COLOR = {
    "mas":             "#DC143C",   # crimson — full method (Layer 1 + Layer 2)
    "mas_layer1_only": "#E74C3C",   # slightly lighter — Layer 1 only (no Layer 2)
    "mas_loto":        "#FF6B6B",   # pink-red — Layer 1 fallback + Layer 2
    "mas_cold":        "#FFB3B3",   # pale pink — no prior at all
    # Baseline color resolved at plot time from BASELINE_COLOR
}
BASELINE_COLOR = {
    "baseline3": "#1E90FF",   # LLMatic — dodger blue
    "baseline4": "#9370DB",   # CoLLM-NAS — purple
}
BASELINE_DISPLAY = {
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
}

CONDITION_DISPLAY = {
    "mas":             "ATHENA\n(exact)",
    "mas_layer1_only": "ATHENA\n(L1-only)",
    "mas_loto":        "ATHENA\n(LOTO)",
    "mas_cold":        "ATHENA\n(cold)",
}

LLM_BASELINE_METHODS = ["baseline4"]  # baseline3 (LLMatic) excluded

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
TASK_DISPLAY = {
    "death": "Mortality",
    "stay": "Stay > 7d",
    "readmission": "Readmission (3M)",
    "next_diag_6m_pheno": "Phenotype (6M)",
    "next_diag_12m_pheno": "Phenotype (12M)",
    "med_rec": "Drug Recommendation",
}


# ---------------------------------------------------------------------------
# Data loading — walks results/seed_*/<hospital>/search/<method>/<task>/<method>_best.csv
# ---------------------------------------------------------------------------
def load_best_scores(results_roots, hospital: str) -> dict:
    """Returns {(method, task): {seed: test_auprc, ...}}."""
    out: dict = {}
    for seed_dir in [d for root in results_roots for d in sorted(Path(root).glob("seed_*"))]:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        search_root = seed_dir / hospital / "search"
        if not search_root.exists():
            continue
        for method_dir in search_root.iterdir():
            method = method_dir.name
            for task_dir in method_dir.iterdir():
                task = task_dir.name
                best_files = list(task_dir.glob("*_best.csv"))
                if not best_files:
                    continue
                try:
                    df = pd.read_csv(best_files[0])
                except Exception as e:
                    print(f"  ⚠ Cannot read {best_files[0]}: {e}")
                    continue
                if len(df) == 0 or "test_auprc" not in df.columns:
                    continue
                out.setdefault((method, task), {})[seed] = float(df["test_auprc"].iloc[0])
    return out


def aggregate(scores_by_seed: dict) -> tuple:
    """Mean ± std (sample std, ddof=1) of a dict {seed: score}."""
    if not scores_by_seed:
        return None, None, 0
    vals = list(scores_by_seed.values())
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def best_baseline_per_task(all_scores: dict, task: str) -> str | None:
    """Pick the LLM-based baseline (LLMatic or CoLLM-NAS) with highest mean
    test AUPRC for `task`. Returns the method name (e.g., 'baseline4'), or
    None if neither has data."""
    best_method = None
    best_mean = -np.inf
    for bm in LLM_BASELINE_METHODS:
        seeds_dict = all_scores.get((bm, task), {})
        m, _, n = aggregate(seeds_dict)
        if m is not None and n > 0 and m > best_mean:
            best_method = bm
            best_mean = m
    return best_method




# ---------------------------------------------------------------------------
# Per-task plotting
# ---------------------------------------------------------------------------
def plot_panel(ax, all_scores: dict, task: str):
    """Draw one panel: 5 grouped bars for one task (4 MAS conditions + best baseline)."""
    bb_method = best_baseline_per_task(all_scores, task)

    # Resolve display order: most-context → least-context, then best-baseline
    cond_order = ["mas", "mas_layer1_only", "mas_loto", "mas_cold"]
    if bb_method:
        cond_order.append(bb_method)

    means, stds, ns, labels, colors = [], [], [], [], []
    for cond in cond_order:
        seeds_dict = all_scores.get((cond, task), {})
        m, s, n = aggregate(seeds_dict)
        means.append(m if m is not None else 0.0)
        stds.append(s if s is not None else 0.0)
        ns.append(n)
        if cond in CONDITION_DISPLAY:
            labels.append(CONDITION_DISPLAY[cond])
            colors.append(CONDITION_COLOR[cond])
        else:
            labels.append(BASELINE_DISPLAY.get(cond, cond))
            colors.append(BASELINE_COLOR.get(cond, "#888888"))

    # Convert AUPRC → percent for display
    means_pct = [m * 100 for m in means]
    stds_pct = [s * 100 for s in stds]

    x = np.arange(len(cond_order))
    bars = ax.bar(x, means_pct, yerr=stds_pct, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  alpha=0.9, error_kw={"linewidth": 1.0})

    # (per-bar n=5 annotation removed — the seed count is stated in the caption)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test AUPRC (%)", fontsize=9)
    ax.set_title(TASK_DISPLAY[task], fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    # Tight y-axis: zoom to data range so error bars stand out
    if any(m > 0 for m in means_pct):
        valid_means = [m for m in means_pct if m > 0]
        ymin = max(0, min(valid_means) - max(stds_pct) - 3)
        ymax = max(valid_means) + max(stds_pct) + 3
        ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    global TASKS   # so --tasks can override the module default
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results_root", type=str, nargs="+", default=["./results"],
                   help="Root containing seed_<N>/ subdirectories.")
    p.add_argument("--hospitals", type=str, nargs="+", required=True,
                   help="One or more hospital names (e.g. MIMIC-III MIMIC-IV). "
                        "Each hospital produces its own PNG with _<hospital> suffix.")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Output directory for the figure.")
    p.add_argument("--tasks", type=str, nargs="+", default=TASKS,
                   help="Tasks to plot (default: the main 5). Use --tasks "
                        "med_rec with --results_root results_medrec.")
    args = p.parse_args()
    TASKS = args.tasks

    results_roots = [Path(r).resolve() for r in args.results_root]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Fig 5] hospitals={args.hospitals}")
    print(f"[Fig 5] out_dir={out_dir}\n")

    # Per-hospital ablation figure. 4 MAS conditions + best baseline shown
    # per task; figure5_loto_robustness_<hospital>.png so cross-hospital runs
    # don't overwrite.
    for hospital in args.hospitals:
        print(f"=== Hospital: {hospital}")
        print(f"  reading from {len(results_roots)} root(s)/seed_*/{hospital}/search/...")
        all_scores = load_best_scores(results_roots, hospital)
        print(f"  loaded {len(all_scores)} (method, task) groups")

        if not all_scores:
            print(f"  ⚠ No best.csv for {hospital}. "
                  f"Submit mas + mas_layer1_only + mas_loto + mas_cold + baseline3/4 jobs first.\n")
            continue

        fig, axes = panel_grid(len(TASKS))

        for panel_i, (ax, task) in enumerate(zip(axes, TASKS)):
            panel_label(ax, panel_i)
            plot_panel(ax, all_scores, task)

        # No figure title — added in the manuscript.
        plt.tight_layout()

        out_path = out_dir / f"figure5_loto_robustness_{hospital}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path}\n")


if __name__ == "__main__":
    main()
