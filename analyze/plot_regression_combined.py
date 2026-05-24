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
# stacked subplots. `task_type` drives the scatter-point colour and the
# column-group header at the top of the figure.
TASKS: list[tuple[str, str, str]] = [
    # (task_key, display_label, task_type)
    ("death",               "Mortality",          "binary"),
    ("stay",                "Stay > 7d",          "binary"),
    ("readmission",         "Readmission (90d)",  "binary"),
    ("next_diag_6m_pheno",  "Phenotype (6m)",     "multilabel"),
    ("next_diag_12m_pheno", "Phenotype (12m)",    "multilabel"),
]

# 2-metric row order (top-to-bottom). Both threshold-free, both reported in
# the main tables.
METRICS: list[tuple[str, str]] = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
]

# Task-type colour scheme: binary = steel blue, multilabel = warm orange.
# Chosen for colour-blind-safe contrast (Wong palette, equivalent in
# luminance for B/W printing).
TASK_TYPE_COLOR = {
    "binary":     "#4C72B0",
    "multilabel": "#DD8452",
}
TASK_TYPE_LABEL = {
    "binary":     "Binary tasks",
    "multilabel": "Multi-label tasks",
}

# Spearman ρ strength thresholds drive the annotation-box border colour.
# `>= 0.70` = strong (green), `>= 0.50` = moderate (yellow), else = weak (gray).
# Threshold cutoffs match the standard "Mukaka 2012 — A guide to appropriate
# use of correlation coefficient in medical research" scale.
def _rho_box_color(rho: float) -> str:
    if not np.isfinite(rho):
        return "gray"
    if rho >= 0.70:
        return "#2E7D32"      # forest green — strong (rank-validity holds)
    if rho >= 0.50:
        return "#F9A825"      # amber — moderate (acceptable for proxy use)
    return "#9E9E9E"           # gray — weak (caveat-worthy)


def _annotate_spearman(ax, x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Compute Spearman ρ and render the corner annotation box.

    The annotation's border + line colour reflects the ρ-strength threshold
    (see `_rho_box_color`), giving the reader a fast at-a-glance read of which
    cells pass the rank-validity bar (green ≥0.7, amber ≥0.5, gray <0.5).

    Returns (rho, p_value, n) for downstream CSV logging. NaN ρ / p when either
    axis has zero variance (constant column) — spearmanr would otherwise raise;
    we silently skip and the annotation reads 'ρ = nan'.
    """
    n = len(x)
    if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
        rho, pv = spearmanr(x, y)
    else:
        rho, pv = float("nan"), float("nan")

    box_color = _rho_box_color(rho)
    ax.text(
        0.05, 0.95,
        f"Spearman ρ = {rho:.3f}\n(p = {pv:.3g}, n = {n})",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        color=box_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=box_color,
                  linewidth=1.8, alpha=0.95),
    )
    return rho, pv, n


def _draw_subplot(ax, df: pd.DataFrame, metric: str,
                  task_type: str) -> tuple[float, float, int]:
    """One scatter subplot: traditional_<metric> on x, autoformer_<metric> on y.

    Scatter points are coloured by `task_type` (binary vs multilabel) so the
    figure's two task-type groups are immediately distinguishable across rows.
    Adds the y=x reference line and an OLS linear fit. Returns Spearman stats.
    """
    x = df[f"traditional_{metric}"].to_numpy(dtype=float)
    y = df[f"autoformer_{metric}"].to_numpy(dtype=float)

    ax.scatter(
        x, y,
        s=55, alpha=0.8, edgecolor="black", linewidth=0.5,
        color=TASK_TYPE_COLOR[task_type], zorder=3,
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
    for task_key, _, _ in TASKS:
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
        for c, (task_key, task_label, task_type) in enumerate(TASKS):
            ax = axes[r][c]
            df = per_task[task_key]
            rho, pv, n = _draw_subplot(ax, df, metric_key, task_type)
            spearman_rows.append({
                "task": task_key,
                "task_type": task_type,
                "metric": metric_key,
                "spearman_rho": rho,
                "p_value": pv,
                "n": n,
            })

            # Top row → task name as column header, recoloured to match task
            # type. Left column → metric name in the y-axis label (combined
            # with "AutoFormer" axis label).
            if r == 0:
                ax.set_title(
                    task_label, fontsize=12, fontweight="bold",
                    color=TASK_TYPE_COLOR[task_type],
                )
            if c == 0:
                ax.set_ylabel(
                    f"{metric_label}\n\nAutoFormer",
                    fontsize=11, fontweight="bold",
                )
            else:
                ax.set_ylabel("AutoFormer", fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("Traditional pretrain+finetune", fontsize=10)

    # ----- Task-type group header bands above the column titles -----
    # We render group headers as horizontal coloured bars spanning the column
    # ranges of each task type, placed above the top row of subplots. Using
    # figure-fraction coords (`fig.text`) keeps them aligned regardless of
    # individual subplot positioning.
    fig.canvas.draw()  # populate axis positions for `get_position()`
    # Compute group spans by task_type
    group_ranges: dict[str, tuple[int, int]] = {}
    for c, (_, _, ttype) in enumerate(TASKS):
        lo, hi = group_ranges.get(ttype, (c, c))
        group_ranges[ttype] = (min(lo, c), max(hi, c))

    for ttype, (lo, hi) in group_ranges.items():
        # Figure-fraction extent: span from left edge of col `lo` to right
        # edge of col `hi`, at a y just above the subplot titles.
        left = axes[0][lo].get_position().x0
        right = axes[0][hi].get_position().x1
        y_band = 0.955
        fig.add_artist(plt.Rectangle(
            (left, y_band - 0.018), right - left, 0.022,
            transform=fig.transFigure,
            facecolor=TASK_TYPE_COLOR[ttype], alpha=0.20,
            edgecolor=TASK_TYPE_COLOR[ttype], linewidth=0,
        ))
        fig.text(
            (left + right) / 2, y_band - 0.007,
            f"{TASK_TYPE_LABEL[ttype]} (n = {hi - lo + 1})",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=TASK_TYPE_COLOR[ttype],
        )

    fig.suptitle(
        f"AutoFormer-supernet ranking validity vs traditional pretrain+finetune "
        f"({args.hospital})",
        fontsize=15, y=0.995,
    )
    # Leave headroom for both the suptitle (top) and the task-type group bands.
    fig.tight_layout(rect=[0, 0, 1, 0.93])

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
    pivot = pivot.reindex(index=[t for t, _, _ in TASKS], columns=[m for m, _ in METRICS])
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
