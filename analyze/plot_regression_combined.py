"""Fig 3 — AutoFormer vs Traditional ranking validity (combined 5×2 panel).

For each of 5 downstream tasks, plot two scatter subplots (AUROC, AUPRC) of
AutoFormer-supernet finetune metrics vs Traditional from-scratch pretrain+finetune
metrics, with Spearman ρ + p-value annotated in each subplot.

Why Spearman (not Pearson)
--------------------------
The Fig 3 claim is *ranking validity* — does the AutoFormer supernet rank
architectures the same way independent from-scratch training would?
Spearman ρ measures rank-order agreement directly and is invariant to monotone
transforms of either axis. Pearson r conflates rank with absolute scale and
penalises us for AutoFormer's known systematic offset on imbalanced binary
tasks; Spearman ρ separates those two effects cleanly.

Why only AUROC + AUPRC
----------------------
Accuracy and F1 are threshold-dependent and degenerate on heavily imbalanced
binary tasks (death ~7% prevalence on MIMIC-IV — predicting all-negative gives
93% accuracy). AUROC and AUPRC are threshold-free and the metrics actually
optimized + reported throughout the paper, so the validity argument should rest
on them alone.

Reads (per task)
----------------
    results/<hospital>/regression/<task>/results.csv
        columns: arch_idx, traditional_<metric>, autoformer_<metric>
        where <metric> ∈ {accuracy, f1, auroc, auprc}

Outputs
-------
    analyze/figure3_regression_combined.png    — 2 rows × 5 cols composite figure
                                                  (rows = metrics, cols = tasks)
    analyze/figure3_spearman_summary.csv       — per-(task, metric) ρ + p + n table

Usage
-----
    python analyze/plot_regression_combined.py --hospital MIMIC-IV
    python analyze/plot_regression_combined.py \\
        --results_dir /blue/.../results --hospital MIMIC-IV \\
        --output analyze/figure3_regression_combined.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# 5-task column order (left-to-right). Display labels are paper-friendly and
# kept short since they're now rendered as column headers above each pair of
# stacked subplots.
TASKS: list[tuple[str, str]] = [
    ("death",               "Mortality"),
    ("stay",                "Stay > 7d"),
    ("readmission",         "Readmission (90d)"),
    ("next_diag_6m_pheno",  "Phenotype (6m)"),
    ("next_diag_12m_pheno", "Phenotype (12m)"),
]

# 2-metric row order (top-to-bottom). Both threshold-free, both reported in
# the main tables.
METRICS: list[tuple[str, str]] = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
]


def _annotate_spearman(ax, x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Compute Spearman ρ and render the corner annotation box.

    Returns (rho, p_value, n) for downstream CSV logging.
    Returns NaN ρ / p if either array has zero variance (constant column) —
    spearmanr would otherwise raise; we silently skip and the annotation reads
    'ρ = nan'.
    """
    n = len(x)
    if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
        rho, pv = spearmanr(x, y)
    else:
        rho, pv = float("nan"), float("nan")

    ax.text(
        0.05, 0.95,
        f"Spearman ρ = {rho:.3f}\n(p = {pv:.3g}, n = {n})",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9),
    )
    return rho, pv, n


def _draw_subplot(ax, df: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    """One scatter subplot: traditional_<metric> on x, autoformer_<metric> on y.

    Adds the y=x reference line and an OLS linear fit. Returns Spearman stats.
    """
    x = df[f"traditional_{metric}"].to_numpy(dtype=float)
    y = df[f"autoformer_{metric}"].to_numpy(dtype=float)

    ax.scatter(
        x, y,
        s=55, alpha=0.8, edgecolor="black", linewidth=0.5,
        color="#4C72B0", zorder=3,
    )

    # y=x identity reference. Padded so the dashed line extends past data range.
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 0.01
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        color="gray", ls="--", lw=1, label="y = x", zorder=2,
    )

    # OLS linear fit — visualises the slope/intercept disagreement separately
    # from the Spearman rank statistic (which ignores both).
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(
            xs, slope * xs + intercept,
            color="tab:red", lw=1.8, label="linear fit", zorder=4,
        )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    return _annotate_spearman(ax, x, y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", type=Path, default=Path("results"),
                    help="Root dir holding <hospital>/regression/<task>/results.csv (default: results/)")
    ap.add_argument("--hospital", type=str, default="MIMIC-IV",
                    help="Hospital whose regression results to plot (default: MIMIC-IV)")
    ap.add_argument("--output", type=Path, default=Path("analyze/figure3_regression_combined.png"),
                    help="Output PNG path (default: analyze/figure3_regression_combined.png)")
    ap.add_argument("--summary_csv", type=Path, default=Path("analyze/figure3_spearman_summary.csv"),
                    help="Output Spearman summary CSV (default: analyze/figure3_spearman_summary.csv)")
    args = ap.parse_args()

    # Load every (task) regression CSV up-front so we can fail fast if anything
    # is missing rather than producing a half-rendered figure.
    per_task: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for task_key, _ in TASKS:
        csv_path = args.results_dir / args.hospital / "regression" / task_key / "results.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        per_task[task_key] = pd.read_csv(csv_path)

    if missing:
        raise FileNotFoundError(
            "Missing regression results for one or more tasks:\n  "
            + "\n  ".join(missing)
            + f"\nExpected layout: {args.results_dir}/{args.hospital}/regression/<task>/results.csv"
        )

    # 2 rows (AUROC, AUPRC) × 5 cols (tasks). Wider aspect ratio fits a paper
    # double-column figure better than the 5×2 portrait layout.
    n_rows = len(METRICS)
    n_cols = len(TASKS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    spearman_rows: list[dict] = []
    for r, (metric_key, metric_label) in enumerate(METRICS):
        for c, (task_key, task_label) in enumerate(TASKS):
            ax = axes[r][c]
            df = per_task[task_key]
            rho, pv, n = _draw_subplot(ax, df, metric_key)
            spearman_rows.append({
                "task": task_key,
                "metric": metric_key,
                "spearman_rho": rho,
                "p_value": pv,
                "n": n,
            })

            # Top row → task name as column header. Left column → metric name
            # in the y-axis label (combined with "AutoFormer" axis label).
            if r == 0:
                ax.set_title(task_label, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(
                    f"{metric_label}\n\nAutoFormer",
                    fontsize=11, fontweight="bold",
                )
            else:
                ax.set_ylabel("AutoFormer", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Traditional pretrain+finetune", fontsize=10)

    fig.suptitle(
        f"AutoFormer-supernet ranking validity vs traditional pretrain+finetune "
        f"({args.hospital})",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(spearman_rows)
    summary.to_csv(args.summary_csv, index=False)

    print(f"\n→ Figure:   {args.output}")
    print(f"→ Spearman: {args.summary_csv}")
    print("\nSpearman ρ table:")
    pivot = summary.pivot(index="task", columns="metric", values="spearman_rho")
    pivot = pivot.reindex(index=[t for t, _ in TASKS], columns=[m for m, _ in METRICS])
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
