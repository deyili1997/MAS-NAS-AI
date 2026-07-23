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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.panel_labels import panel_label  # noqa: E402

# Fixed hyperparameter order + display names (matches the search space).
FEATURE_ORDER = ["embed_dim", "depth", "num_heads", "mlp_ratio"]
FEATURE_LABEL = {"embed_dim": "Embedding dim", "depth": "Depth",
                 "num_heads": "# Heads", "mlp_ratio": "MLP ratio"}
POS = "#2E8B57"    # sea green — CI entirely > 0 (preferred)
NEG = "#DC143C"    # crimson   — CI entirely < 0 (discouraged)
NEU = "#9E9E9E"    # grey      — CI spans 0 (inconclusive)


def _color(lo, hi):
    if lo > 0:
        return POS
    if hi < 0:
        return NEG
    return NEU


def plot_task(df: pd.DataFrame, out_path: Path, task: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    n_hosp = int(df["n_hospitals"].max()) if "n_hospitals" in df else None

    for panel_i, (ax, feat) in enumerate(zip(axes.flat, FEATURE_ORDER)):
        sub = df[df["feature"] == feat].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("level")
        ys = range(len(sub))
        for y, (_, r) in zip(ys, sub.iterrows()):
            c = _color(r["ci_lo"], r["ci_hi"])
            ax.errorbar(r["est_mean_shap"], y,
                        xerr=[[r["est_mean_shap"] - r["ci_lo"]],
                              [r["ci_hi"] - r["est_mean_shap"]]],
                        fmt="o", color=c, ecolor=c, elinewidth=1.8,
                        capsize=4, markersize=8, markeredgecolor="black",
                        markeredgewidth=0.6, zorder=3)
        ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([str(int(v)) for v in sub["level"]])
        ax.set_ylabel(FEATURE_LABEL[feat])
        ax.set_xlabel("Effect on performance (SHAP, 95% CI)")
        ax.margins(y=0.25)
        ax.grid(True, axis="x", alpha=0.25)
        panel_label(ax, panel_i)

    # Legend (shared) + pool-size note, drawn once.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                      markeredgecolor="black", markersize=9, label=lab)
               for col, lab in [(POS, "preferred (CI > 0)"),
                                (NEG, "discouraged (CI < 0)"),
                                (NEU, "inconclusive (CI spans 0)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    if n_hosp:
        fig.text(0.99, 0.005, f"pooled over {n_hosp} source sites",
                 ha="right", va="bottom", fontsize=8, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prior_root", required=True,
                   help="Dir containing <task>/level_effects.csv "
                        "(e.g. .../results_medrec/meta_regression).")
    p.add_argument("--task", default="med_rec",
                   help="Task to plot (default: med_rec). One figure per task.")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    f = Path(args.prior_root) / args.task / "level_effects.csv"
    if not f.exists():
        raise SystemExit(f"❌ {f} not found — has MODE=prior run for this task?")
    df = pd.read_csv(f)
    print(f"[prior] {args.task}: {len(df)} level-effect rows, "
          f"pool n_hospitals={df['n_hospitals'].max() if 'n_hospitals' in df else '?'}")
    plot_task(df, Path(args.out_dir) / f"figure_prior_{args.task}.png", args.task)


if __name__ == "__main__":
    main()
