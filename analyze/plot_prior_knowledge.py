"""Visualize the Layer-2 architecture prior learned from the source pool.

The meta-regression (run_meta_regression.py) fits, per task, a mixed-effect model
over the source hospitals' NAS metadata and estimates each architecture choice's
effect on performance — the "prior knowledge" the agents are handed. This script
renders that as a forest plot: one subplot per hyperparameter, one row per level,
showing the estimated SHAP effect with its 95% CI. A level whose CI clears zero
is what the pool found reliably good (green) or bad (red); CIs spanning zero are
inconclusive (grey). Vertical line at zero.

Reads   <prior_root>/<task>/level_effects.csv
        columns: task, feature, level, est_mean_shap, ci_lo, ci_hi, n_hospitals
Writes  <out_dir>/figure_prior_<task>.png   (no title — added in the manuscript)

Because the prior is POOLED across the source hospitals, no site labels appear;
the pool size (n_hospitals) is annotated so the reader knows it is 4-site.

Usage:
    python analyze/plot_prior_knowledge.py --task med_rec \
        --prior_root /blue/mei.liu/lideyi/MAS-NAS/results_medrec/meta_regression \
        --out_dir /blue/mei.liu/lideyi/MAS-NAS/analyze/2026-07-23_final
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.panel_labels import panel_label  # noqa: E402
from utils.paths import get_final_dir  # noqa: E402

# Prior figures live in their own subfolder of the dated final-results folder.
PRIOR_SUBDIR = "prior_knowledge"

# Fixed hyperparameter order + display names (matches the search space).
FEATURE_ORDER = ["embed_dim", "depth", "num_heads", "mlp_ratio"]
FEATURE_LABEL = {"embed_dim": "Embedding dim", "depth": "Depth",
                 "num_heads": "# Heads", "mlp_ratio": "MLP ratio"}
# Row order + display names for the combined grid.
TASK_ORDER = ["death", "stay", "readmission",
              "next_diag_6m_pheno", "next_diag_12m_pheno", "med_rec"]
TASK_LABEL = {"death": "Mortality", "stay": "Stay > 7d", "readmission": "Readmission (3M)",
              "next_diag_6m_pheno": "Phenotype (6M)", "next_diag_12m_pheno": "Phenotype (12M)",
              "med_rec": "Drug Rec"}
POS = "#2E8B57"    # sea green — CI entirely > 0 (preferred)
NEG = "#DC143C"    # crimson   — CI entirely < 0 (discouraged)
NEU = "#9E9E9E"    # grey      — CI spans 0 (inconclusive)


def _forest(ax, sub):
    """Draw one hyperparameter's level effects (sorted by level) into `ax`."""
    sub = sub.sort_values("level")
    for y, (_, r) in enumerate(sub.iterrows()):
        c = _color(r["ci_lo"], r["ci_hi"])
        ax.errorbar(r["est_mean_shap"], y,
                    xerr=[[r["est_mean_shap"] - r["ci_lo"]],
                          [r["ci_hi"] - r["est_mean_shap"]]],
                    fmt="o", color=c, ecolor=c, elinewidth=1.6, capsize=3,
                    markersize=7, markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([str(int(v)) for v in sub["level"]], fontsize=7)
    ax.margins(y=0.25)
    ax.grid(True, axis="x", alpha=0.25)


def _legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                      markeredgecolor="black", markersize=9, label=lab)
               for col, lab in [(POS, "preferred (CI > 0)"),
                                (NEG, "discouraged (CI < 0)"),
                                (NEU, "inconclusive (CI spans 0)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))


def _color(lo, hi):
    if lo > 0:
        return POS
    if hi < 0:
        return NEG
    return NEU


def plot_task(df: pd.DataFrame, out_path: Path, task: str):
    """Single-task figure: 2x2, one subplot per hyperparameter."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    n_hosp = int(df["n_hospitals"].max()) if "n_hospitals" in df else None
    for panel_i, (ax, feat) in enumerate(zip(axes.flat, FEATURE_ORDER)):
        sub = df[df["feature"] == feat]
        if sub.empty:
            ax.set_visible(False)
            continue
        _forest(ax, sub)
        ax.set_ylabel(FEATURE_LABEL[feat])
        ax.set_xlabel("Effect on performance (SHAP, 95% CI)")
        panel_label(ax, panel_i)
    _legend(fig)
    if n_hosp:
        fig.text(0.99, 0.005, f"pooled over {n_hosp} source sites",
                 ha="right", va="bottom", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_combined_grid(task_dfs, out_path: Path):
    """Combined grid: rows = tasks, cols = the 4 hyperparameters.

    task_dfs: list of (task_key, df). Each cell is a mini forest plot with its
    own x-axis (effect scales differ wildly across tasks and hyperparameters).
    """
    n = len(task_dfs)
    fig, axes = plt.subplots(n, 4, figsize=(13, 1.9 * n + 0.8), squeeze=False)
    n_hosp = None
    for ri, (task, df) in enumerate(task_dfs):
        if n_hosp is None and "n_hospitals" in df:
            n_hosp = int(df["n_hospitals"].max())
        for ci, feat in enumerate(FEATURE_ORDER):
            ax = axes[ri][ci]
            sub = df[df["feature"] == feat]
            if sub.empty:
                ax.set_visible(False)
                continue
            _forest(ax, sub)
            if ri == 0:
                ax.set_title(FEATURE_LABEL[feat], fontsize=11)
            if ci == 0:
                ax.set_ylabel(TASK_LABEL.get(task, task), fontsize=10, fontweight="bold")
            if ri == n - 1:
                ax.set_xlabel("SHAP effect (95% CI)", fontsize=9)
    _legend(fig)
    if n_hosp:
        fig.text(0.99, 0.005, f"prior pooled over {n_hosp} source sites",
                 ha="right", va="bottom", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ combined grid ({n} tasks): {out_path}")


def _find_level_effects(roots, task):
    """Return the first <root>/<task>/level_effects.csv that exists, or None."""
    for root in roots:
        f = Path(root) / task / "level_effects.csv"
        if f.exists():
            return f
    return None
    print(f"  ✓ {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prior_root", nargs="+", required=True,
                   help="One or more dirs containing <task>/level_effects.csv. In "
                        "--combined mode each task is looked up across these in order, "
                        "so the 5-task prior (results_orig_256grid/meta_regression) and "
                        "med_rec (results_medrec/meta_regression) can be combined.")
    p.add_argument("--task", default="med_rec",
                   help="Single-task mode: task to plot (default: med_rec).")
    p.add_argument("--combined", action="store_true",
                   help="Combined mode: one grid, rows = --tasks, cols = the 4 "
                        "hyperparameters. Tasks with no level_effects.csv in any "
                        "--prior_root are skipped (so it works before med_rec exists).")
    p.add_argument("--tasks", nargs="+", default=TASK_ORDER,
                   help=f"Combined-mode row order. Default: {TASK_ORDER}")
    p.add_argument("--out_dir", default=None,
                   help="Output dir. Default: <analyze_root>/<--date>_final/"
                        f"{PRIOR_SUBDIR}/ — created if missing, so a standalone run "
                        "lands in today's final-results folder automatically.")
    p.add_argument("--date", default=None,
                   help="YYYY-MM-DD stamping the default final folder "
                        "(<date>_final). Default: today. Ignored if --out_dir is given.")
    args = p.parse_args()

    # Resolve output dir. Explicit --out_dir wins; otherwise default to the dated
    # final-results folder's prior_knowledge/ subdir and create it if absent.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        date = args.date or datetime.date.today().isoformat()
        out_dir = get_final_dir(date) / PRIOR_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prior] out_dir={out_dir}")

    if args.combined:
        task_dfs, missing = [], []
        for t in args.tasks:
            f = _find_level_effects(args.prior_root, t)
            if f is None:
                missing.append(t)
                continue
            task_dfs.append((t, pd.read_csv(f)))
        if missing:
            print(f"[prior] skipped (no level_effects.csv found): {missing}")
        if not task_dfs:
            raise SystemExit("❌ No tasks had level_effects.csv in any --prior_root.")
        print(f"[prior] combined grid rows: {[t for t, _ in task_dfs]}")
        plot_combined_grid(task_dfs, out_dir / "figure_prior_combined.png")
        return

    f = _find_level_effects(args.prior_root, args.task)
    if f is None:
        raise SystemExit(f"❌ {args.task}/level_effects.csv not found under any "
                         f"{args.prior_root} — has MODE=prior run for this task?")
    df = pd.read_csv(f)
    print(f"[prior] {args.task}: {len(df)} level-effect rows, "
          f"pool n_hospitals={df['n_hospitals'].max() if 'n_hospitals' in df else '?'}")
    plot_task(df, out_dir / f"figure_prior_{args.task}.png", args.task)


if __name__ == "__main__":
    main()
