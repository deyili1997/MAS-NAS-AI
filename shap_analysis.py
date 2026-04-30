"""
SHAP Analysis of Architecture Features (per hospital × task)
=============================================================
For each (hospital, task) group:
1. Compute composite target = average rank across accuracy, f1, auroc, auprc
   (lower rank = better, rank 1 = best)
2. Train XGBoost on 4 architecture features: embed_dim, depth, mlp_ratio, num_heads
3. Compute SHAP values and save plots + CSV
"""

import argparse
import glob
import os
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path


METRIC_COLS = ["accuracy", "f1", "auroc", "auprc"]
ARCH_COLS = ["embed_dim", "depth", "mlp_ratio", "num_heads"]


def parse_args():
    p = argparse.ArgumentParser(description="Per-task SHAP analysis on NAS metadata")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Root results directory containing per-hospital metadata.csv files")
    p.add_argument("--output_dir", type=str, default="./results/shap_analysis")
    return p.parse_args()


def load_metadata(results_dir: str) -> pd.DataFrame:
    """Load and concatenate all metadata.csv files across hospitals."""
    pattern = os.path.join(results_dir, "*", "metadata.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No metadata.csv files found under {results_dir}/*/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} hospital(s): "
          f"{df['hospital'].unique().tolist()}")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure mlp_ratio and num_heads are numeric (they are already scalars)."""
    df = df.copy()
    df["mlp_ratio"] = pd.to_numeric(df["mlp_ratio"])
    df["num_heads"] = pd.to_numeric(df["num_heads"])
    return df


def compute_avg_rank(group: pd.DataFrame) -> np.ndarray:
    """
    For a (hospital, task) group, rank each architecture by each metric
    (ascending rank, rank 1 = best), then return the average rank.
    """
    ranks = pd.DataFrame()
    for col in METRIC_COLS:
        # ascending=False: higher metric value gets rank 1 (best)
        ranks[col] = group[col].rank(ascending=False, method="average")
    return ranks.mean(axis=1).values


def run_shap_for_group(X, y, group_name, output_dir):
    """Train XGBoost and compute SHAP for one (hospital, task) group."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(group_name)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150)
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(group_name)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=150)
    plt.close()

    # SHAP values CSV
    shap_df = pd.DataFrame(shap_values, columns=ARCH_COLS)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)

    # Mean |SHAP| per feature
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=ARCH_COLS)
    mean_abs = mean_abs.sort_values(ascending=False)
    mean_abs.to_csv(output_dir / "mean_abs_shap.csv", header=False)

    # Combined panel: SHAP left + 4 univariate regression panels right
    _save_shap_with_regression(X, y, shap_values, group_name, output_dir)

    print(f"  [{group_name}] saved to {output_dir}")
    print(f"    Mean |SHAP|: {mean_abs.to_dict()}")

    return mean_abs


def _save_shap_with_regression(X, y, shap_values, group_name, output_dir):
    """Save a side-by-side panel: SHAP bee swarm (left ~50%) + 2x2 grid of
    univariate rank-vs-feature scatter with linear fit + Pearson r (right ~50%).

    Why this combo: the SHAP plot answers 'how do features affect performance
    in a multivariate sense', while the 4 regression panels answer 'is the
    relationship monotonic / strong on the marginal level'. Together they give
    a complete picture per (hospital, task).

    Output: shap_with_regression.png in output_dir.
    """
    output_dir = Path(output_dir)

    # ---- Step 1: render SHAP bee swarm to a memory buffer (shap.summary_plot
    # creates its own figure and doesn't accept an `ax` param, so we capture
    # it as a PNG and re-embed via imshow() in our composite figure).
    buf = BytesIO()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    shap_img = plt.imread(buf)
    buf.close()

    # ---- Step 2: build composite figure (1×2 layout where right is 2×2 grid)
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1.5, 1.5, 1, 1],
        wspace=0.35, hspace=0.45,
    )

    # Left half: embedded SHAP plot
    ax_shap = fig.add_subplot(gs[:, 0:2])
    ax_shap.imshow(shap_img)
    ax_shap.axis("off")
    ax_shap.set_title(f"SHAP feature importance — {group_name}", fontsize=11)

    # Right half: 2×2 mini-scatter + linear fit per feature
    for i, feat in enumerate(ARCH_COLS):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        x_vals = X[feat].values.astype(float)
        ax.scatter(x_vals, y, alpha=0.4, s=20, color="steelblue", edgecolors="none")
        # Linear fit (require ≥ 2 distinct values to avoid ill-conditioned polyfit)
        if len(np.unique(x_vals)) >= 2:
            slope, intercept = np.polyfit(x_vals, y, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 60)
            ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5)
            r = float(np.corrcoef(x_vals, y)[0, 1])
            ax.set_title(f"{feat} (r={r:+.2f})", fontsize=10)
        else:
            ax.set_title(feat, fontsize=10)
        ax.set_xlabel(feat, fontsize=9)
        if i % 2 == 0:
            ax.set_ylabel("Performance\n(higher = better)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(group_name, fontsize=13, y=0.99)
    plt.savefig(output_dir / "shap_with_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = load_metadata(args.results_dir)
    df = prepare_features(df)

    all_shap_summary = []

    for (hospital, task), group in df.groupby(["hospital", "task"]):
        group = group.reset_index(drop=True)
        group_name = f"{hospital}/{task}"

        if len(group) < 5:
            print(f"  [{group_name}] skipped — only {len(group)} samples (need >= 5)")
            continue

        # Compute composite target: negate avg_rank so higher = better performance
        # This way positive SHAP = feature improves performance
        avg_rank = compute_avg_rank(group)
        y = -avg_rank

        X = group[ARCH_COLS].copy()

        group_output = Path(args.output_dir) / hospital / task
        mean_abs = run_shap_for_group(X, y, group_name, group_output)

        row = {"hospital": hospital, "task": task}
        for feat in ARCH_COLS:
            row[f"mean_abs_shap_{feat}"] = mean_abs[feat]
        all_shap_summary.append(row)

    # Save combined summary
    if all_shap_summary:
        summary_df = pd.DataFrame(all_shap_summary)
        out_path = Path(args.output_dir) / "shap_summary_all.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nCombined SHAP summary: {out_path}")
        print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
