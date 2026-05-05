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
    python analyze/plot_pareto.py --hospital MIMIC-IV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["baseline0", "baseline1", "baseline2", "baseline3", "baseline4", "mas"]
METHOD_DISPLAY = {
    "baseline0": "Random",
    "baseline1": "EA",
    "baseline2": "LLM-1shot",
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
    "mas": "MAS-NAS",
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
    "readmission": "Readmit (3M)",
    "next_diag_6m_pheno": "Phenotype 6M",
    "next_diag_12m_pheno": "Phenotype 12M",
}


def load_search_records(results_root: Path, hospital: str) -> dict:
    """Walk results/seed_*/<hospital>/search/<method>/<task>/*_search.csv.
    Returns dict: (method, task, seed) -> DataFrame."""
    records: dict = {}
    seed_dirs = sorted(results_root.glob("seed_*"))

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

    ax.set_xlabel("# parameters", fontsize=10)
    ax.set_ylabel("Val AUPRC", fontsize=10)
    ax.set_title(TASK_DISPLAY[task], fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85, ncol=2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results_root", type=str, default="./results",
                   help="Root containing seed_<N>/ subdirectories.")
    p.add_argument("--hospital", type=str, required=True,
                   help="Hospital name (e.g. MIMIC-IV).")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Output directory for the figure.")
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Pareto] hospital={args.hospital}")
    print(f"[Pareto] reading from {results_root}/seed_*/{args.hospital}/search/...")
    records = load_search_records(results_root, args.hospital)
    print(f"[Pareto] loaded {len(records)} (method, task, seed) records")

    if not records:
        print("⚠ No records found. Check --results_root and that baseline/mas jobs have produced search.csv.")
        return

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 4.5), sharey=False)
    if len(TASKS) == 1:
        axes = [axes]

    for ax, task in zip(axes, TASKS):
        plot_pareto_panel(ax, records, task)

    fig.suptitle(
        f"Pareto front (params vs val AUPRC) — {args.hospital}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out_path = out_dir / "figure2_pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
