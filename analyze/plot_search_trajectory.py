"""Plot search trajectory: best validation metric vs # architectures evaluated.

For each (method, task) combination, plots the cumulative-max of val_auprc as
the search progresses. Shows that MAS-NAS reaches high AUPRC earlier (at
fewer evaluations) than baselines — the core "compute efficiency" claim.

Reads:
    results/seed_<SEED>/<hospital>/search/<method>/<task>/<method>_search.csv
    (each row = 1 evaluated architecture, columns include `iteration`, `val_auprc`, ...)

Outputs to <out_dir>:
    figure1_search_trajectory_val_auprc.png  — 5 panels (one per task), 6 method
                                                 curves with mean ± std over seeds.

Usage:
    python analyze/plot_search_trajectory.py --hospitals MIMIC-IV
    python analyze/plot_search_trajectory.py --hospitals MIMIC-III MIMIC-IV --metric val_auroc
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


METHODS = ["baseline0", "baseline1", "baseline2", "baseline4", "mas"]  # baseline3 (LLMatic) excluded by DEFAULT — override with --methods
METHOD_DISPLAY = {
    "baseline0": "Random",
    "baseline1": "EA",
    "baseline2": "GENIUS",
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
    "baseline5": "ENAS",
    "baseline6": "TPE",
    "mas": "ATHENA",
}
METHOD_COLOR = {
    "baseline0": "#888888",   # gray
    "baseline1": "#FF8C00",   # orange
    "baseline2": "#2E8B57",   # sea green
    "baseline3": "#1E90FF",   # dodger blue
    "baseline4": "#9370DB",   # purple
    "baseline5": "#17BECF",   # cyan — ENAS (RL)
    "baseline6": "#BCBD22",   # olive — TPE (BO)
    "mas":       "#DC143C",   # crimson — MAS stands out
}
METHOD_LINESTYLE = {
    "baseline0": "-",
    "baseline1": "-",
    "baseline2": "--",
    "baseline3": "--",
    "baseline4": "--",
    "baseline5": "--",
    "baseline6": "--",
    "mas":       "-",
}

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
TASK_DISPLAY = {
    "death": "Mortality",
    "stay": "Stay > 7d",
    "readmission": "Readmission (3M)",
    "next_diag_6m_pheno": "Phenotype (6M)",
    "next_diag_12m_pheno": "Phenotype (12M)",
    "med_rec": "Drug Recommendation",
}


def load_search_records(results_roots, hospital: str) -> dict:
    """Walk results/seed_*/<hospital>/search/<method>/<task>/*_search.csv.
    Returns dict: (method, task, seed) -> sorted-by-iteration DataFrame."""
    records: dict = {}
    seed_dirs = [d for root in results_roots for d in sorted(Path(root).glob("seed_*"))]

    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

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
                csv_files = list(task_dir.glob("*_search.csv"))
                if not csv_files:
                    continue
                try:
                    df = pd.read_csv(csv_files[0])
                except Exception as e:
                    print(f"  ⚠ Cannot read {csv_files[0]}: {e}")
                    continue
                if "iteration" not in df.columns:
                    # Fallback: use row order as iteration
                    df["iteration"] = range(1, len(df) + 1)
                df = df.sort_values("iteration").reset_index(drop=True)
                records[(method, task, seed)] = df
    return records


def cumulative_max(values: np.ndarray) -> np.ndarray:
    """Best-so-far at each step."""
    return np.maximum.accumulate(values)


def plot_trajectory_panel(ax, records: dict, task: str, metric: str, max_iter_global: int):
    """Plot all 6 method curves on a single axes for one task."""
    for method in METHODS:
        seeds = sorted([s for (m, t, s) in records if m == method and t == task])
        if not seeds:
            continue

        # Each method may have a different budget; pad shorter trajectories
        # with NaN past their actual end so they don't artificially flatten
        # the mean curve.
        per_seed_curves = []
        method_max_iter = 0
        for s in seeds:
            df = records[(method, task, s)]
            vals = df[metric].values
            if len(vals) == 0:
                continue
            curve = cumulative_max(vals)
            method_max_iter = max(method_max_iter, len(curve))
            per_seed_curves.append(curve)

        if not per_seed_curves:
            continue

        # Pad to method_max_iter
        padded = np.full((len(per_seed_curves), method_max_iter), np.nan)
        for i, c in enumerate(per_seed_curves):
            padded[i, :len(c)] = c

        x = np.arange(1, method_max_iter + 1)
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0, ddof=1) if padded.shape[0] > 1 else np.zeros_like(mean)

        ax.plot(
            x, mean,
            color=METHOD_COLOR[method],
            linestyle=METHOD_LINESTYLE[method],
            label=f"{METHOD_DISPLAY[method]} (n={method_max_iter})",
            linewidth=2.0 if method == "mas" else 1.4,
            zorder=10 if method == "mas" else 5,
        )
        if padded.shape[0] > 1:
            ax.fill_between(
                x, mean - std, mean + std,
                color=METHOD_COLOR[method], alpha=0.15, linewidth=0,
            )

    ax.set_xlabel("# architectures evaluated", fontsize=10)
    ax.set_ylabel(f"Best {metric.replace('val_', '').upper()} so far", fontsize=10)
    ax.set_title(TASK_DISPLAY[task], fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)


def main():
    global TASKS, METHODS   # --tasks / --methods override the module defaults
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results_root", type=str, nargs="+", default=["./results"],
                   help="Root containing seed_<N>/ subdirectories.")
    p.add_argument("--hospitals", type=str, nargs="+", required=True,
                   help="One or more hospital names (e.g. MIMIC-III MIMIC-IV). "
                        "Each hospital produces its own PNG with _<hospital> suffix.")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Output directory for the figure.")
    p.add_argument("--metric", type=str, default="val_auprc",
                   choices=["val_auprc", "val_auroc", "val_f1", "val_accuracy"],
                   help="Metric to track (Y-axis).")
    p.add_argument("--tasks", type=str, nargs="+", default=TASKS,
                   help="Tasks to plot (default: the main 5). Use --tasks "
                        "med_rec with --results_root results_medrec.")
    p.add_argument("--methods", type=str, nargs="+", default=METHODS,
                   help=f"Method curves to plot (default: {METHODS}; baseline3/"
                        "LLMatic excluded). Pass e.g. --methods baseline0 baseline1 "
                        "baseline2 baseline3 baseline4 mas to include LLMatic.")
    args = p.parse_args()
    TASKS = args.tasks
    METHODS = args.methods

    results_roots = [Path(r).resolve() for r in args.results_root]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Trajectory] hospitals={args.hospitals}  metric={args.metric}")
    print(f"[Trajectory] out_dir={out_dir}\n")

    # Per-hospital trajectory figure. Each hospital is independent — its own
    # records dict (loaded from seed_*/<H>/search/), own panel grid, own PNG.
    # Suffix _<hospital> on output so cross-hospital runs don't overwrite.
    for hospital in args.hospitals:
        print(f"=== Hospital: {hospital}")
        print(f"  reading from {len(results_roots)} root(s)/seed_*/{hospital}/search/...")
        records = load_search_records(results_roots, hospital)
        print(f"  loaded {len(records)} (method, task, seed) records")

        if not records:
            print(f"  ⚠ No records for {hospital}. Skipping.\n")
            continue

        max_iter_global = max(len(df) for df in records.values())

        fig, axes = panel_grid(len(TASKS))

        for panel_i, (ax, task) in enumerate(zip(axes, TASKS)):
            panel_label(ax, panel_i)
            plot_trajectory_panel(ax, records, task, args.metric, max_iter_global)

        # No figure title — added in the manuscript. site_label kept imported
        # for the output filename / logs only.
        plt.tight_layout()

        out_path = out_dir / f"figure1_search_trajectory_{args.metric}_{hospital}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path}\n")


if __name__ == "__main__":
    main()
