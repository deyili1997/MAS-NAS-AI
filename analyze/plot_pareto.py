"""Plot Pareto front: architecture parameters vs validation AUPRC.

For each (method, task), scatters all evaluated architectures (params, AUPRC)
and overlays the Pareto front (architectures not dominated by any other on the
small-params + high-AUPRC frontier).

Reads:
    results/seed_<SEED>/<hospital>/search/<method>/<task>/<method>_search.csv
    (each row = 1 evaluated architecture, columns include `num_params`,
    `val_auprc`, ...)

Outputs to <out_dir>:
    figure2_pareto.png  — 5 panels (one per task), all 6 methods overlaid
                          with distinct colors. Pareto front drawn as black
                          dashed line.

Usage:
    python analyze/plot_pareto.py --hospitals MIMIC-IV
    python analyze/plot_pareto.py --hospitals MIMIC-III MIMIC-IV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd


def _human_params(x, _pos=None):
    """Log-axis tick label as a human-readable count: 5e5→'0.5M', 3e6→'3M'."""
    if x <= 0:
        return ""
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}K"
    return f"{x:g}"

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
    "mas": "ATHENA",
}
METHOD_COLOR = {
    "baseline0": "#888888",
    "baseline1": "#FF8C00",
    "baseline2": "#2E8B57",
    "baseline3": "#1E90FF",
    "baseline4": "#9370DB",
    "mas":       "#DC143C",
}
METHOD_MARKER = {
    "baseline0": "o",
    "baseline1": "^",
    "baseline2": "s",
    "baseline3": "D",
    "baseline4": "v",
    "mas":       "*",
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
    Returns dict: (method, task, seed) -> DataFrame."""
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
                records[(method, task, seed)] = df
    return records


def is_pareto_optimal(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points.

    `points` is shape (N, 2) with columns [params, auprc].
    Goal: minimize params, maximize auprc.
    A point is Pareto-optimal if no other point has both <= params and
    >= auprc, with at least one strict inequality.

    O(N^2) — fine for our scale (≤ 600 archs total per task).
    """
    n = len(points)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        if not optimal[i]:
            continue
        x_i, y_i = points[i]
        for j in range(n):
            if i == j:
                continue
            x_j, y_j = points[j]
            # j dominates i if: params_j <= params_i AND auprc_j >= auprc_i,
            # with at least one strict inequality
            if x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                optimal[i] = False
                break
    return optimal


def plot_pareto_panel(ax, records: dict, task: str):
    """Scatter all evaluated archs colored by method + Pareto front overlay."""
    all_points = []          # (params, auprc, method) accumulated for Pareto
    method_handles = []

    for method in METHODS:
        seeds = sorted([s for (m, t, s) in records if m == method and t == task])
        if not seeds:
            continue

        params_all = []
        auprc_all = []
        for s in seeds:
            df = records[(method, task, s)]
            if "num_params" not in df.columns or "val_auprc" not in df.columns:
                continue
            params_all.extend(df["num_params"].dropna().values)
            auprc_all.extend(df["val_auprc"].dropna().values)

        if not params_all:
            continue

        sc = ax.scatter(
            params_all, auprc_all,
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            alpha=0.5, s=35 if method == "mas" else 20,
            edgecolors="black" if method == "mas" else "none",
            linewidths=0.5 if method == "mas" else 0,
            label=METHOD_DISPLAY[method],
            zorder=10 if method == "mas" else 5,
        )
        method_handles.append(sc)

        for p, a in zip(params_all, auprc_all):
            all_points.append((p, a))

    # Compute and overlay Pareto front (across ALL methods combined)
    if len(all_points) >= 3:
        pts = np.array(all_points)
        is_pf = is_pareto_optimal(pts)
        if is_pf.sum() >= 2:
            pf = pts[is_pf]
            pf = pf[pf[:, 0].argsort()]   # sort by params for the line plot
            ax.plot(
                pf[:, 0], pf[:, 1],
                color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                label="Pareto front",
                zorder=15,
            )

    ax.set_xlabel("# parameters (log scale)", fontsize=10)
    ax.set_ylabel("Val AUPRC", fontsize=10)
    ax.set_title(TASK_DISPLAY[task], fontsize=11)
    ax.set_xscale("log")
    # Human-readable ticks (0.5M, 1M, 3M, ...) on both decade and mid-decade
    # positions so a narrow ~1–2 decade range still gets several labels.
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(0.2, 0.3, 0.5, 0.7), numticks=12))
    ax.xaxis.set_major_formatter(FuncFormatter(_human_params))
    ax.xaxis.set_minor_formatter(FuncFormatter(_human_params))
    ax.tick_params(axis="x", which="both", labelsize=7)
    plt.setp(ax.get_xticklabels(which="both"), rotation=30, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85, ncol=2)


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
    p.add_argument("--tasks", type=str, nargs="+", default=TASKS,
                   help="Tasks to plot (default: the main 5). Use --tasks "
                        "med_rec with --results_root results_medrec.")
    p.add_argument("--methods", type=str, nargs="+", default=METHODS,
                   help=f"Methods to scatter (default: {METHODS}; baseline3/LLMatic "
                        "excluded). Pass e.g. --methods baseline0 baseline1 baseline2 "
                        "baseline3 baseline4 mas to include LLMatic.")
    args = p.parse_args()
    TASKS = args.tasks
    METHODS = args.methods

    results_roots = [Path(r).resolve() for r in args.results_root]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pareto] hospitals={args.hospitals}")
    print(f"[Pareto] out_dir={out_dir}\n")

    # Per-hospital Pareto front. Same panel grid per hospital, separate PNG
    # (figure2_pareto_<hospital>.png) so cross-hospital runs don't overwrite.
    for hospital in args.hospitals:
        print(f"=== Hospital: {hospital}")
        print(f"  reading from {len(results_roots)} root(s)/seed_*/{hospital}/search/...")
        records = load_search_records(results_roots, hospital)
        print(f"  loaded {len(records)} (method, task, seed) records")

        if not records:
            print(f"  ⚠ No records for {hospital}. Skipping.\n")
            continue

        fig, axes = panel_grid(len(TASKS))

        for panel_i, (ax, task) in enumerate(zip(axes, TASKS)):
            panel_label(ax, panel_i)
            plot_pareto_panel(ax, records, task)

        # No figure title — added in the manuscript.
        plt.tight_layout()

        out_path = out_dir / f"figure2_pareto_{hospital}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {out_path}\n")


if __name__ == "__main__":
    main()
