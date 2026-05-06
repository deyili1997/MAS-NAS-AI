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
    p = argparse.ArgumentParser(
        description="Per-task SHAP analysis on NAS metadata, pooled across hospitals.")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Root results directory containing per-hospital metadata.csv files. "
                        "Globs <results_dir>/*/metadata.csv for source data.")
    p.add_argument("--output_dir", type=str, default="./results/shap_analysis",
                   help="Where to write per-task SHAP outputs. "
                        "Layout: <output_dir>/<task>/{shap_summary.png, shap_bar.png, "
                        "shap_with_regression.png, shap_interaction_*.png, "
                        "shap_values.csv, mean_abs_shap.csv}.")
    p.add_argument("--hospitals", type=str, nargs="+", default=None,
                   help="Filter to specific hospitals before pooling. "
                        "If omitted, all hospitals under --results_dir are pooled. "
                        "Phase 2 standard:  --hospitals OneFL_H1 OneFL_H2 ... OneFL_H10 "
                        "Phase 1 / single-hospital test:  --hospitals MIMIC-IV "
                        "(filter to 1 hospital → effectively per-hospital × per-task SHAP).")
    p.add_argument("--interaction_main", type=str, nargs="+",
                   default=["embed_dim"],
                   choices=ARCH_COLS,
                   help=("One or more architecture features to use as the 'main' "
                         "feature in interaction plots. For each, produces a "
                         "shap_interaction_<feat>.png with 3 sub-panels showing "
                         "interactions with the OTHER architecture features. "
                         "E.g. --interaction_main embed_dim depth → 2 PNGs per task. "
                         "Default: just embed_dim."))
    return p.parse_args()


def load_metadata(results_dir: str, hospitals_filter: list = None) -> pd.DataFrame:
    """Load and concatenate all metadata.csv files across hospitals.

    Args:
        results_dir: Glob root, expects <results_dir>/<hospital>/metadata.csv
        hospitals_filter: Optional list of hospital names to keep. If None,
            uses all hospitals found under results_dir. Hospitals not in
            this filter are dropped before SHAP analysis.
    """
    pattern = os.path.join(results_dir, "*", "metadata.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No metadata.csv files found under {results_dir}/*/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} metadata.csv file(s).")
    print(f"  Hospitals found: {df['hospital'].unique().tolist()}")

    if hospitals_filter:
        before = len(df)
        df = df[df["hospital"].isin(hospitals_filter)].reset_index(drop=True)
        print(f"  Filtered to {hospitals_filter}: {before} → {len(df)} rows.")
        if len(df) == 0:
            raise ValueError(
                f"No rows remain after filtering. Hospitals in data: "
                f"{df['hospital'].unique().tolist()}, requested: {hospitals_filter}.")
    print(f"  Final pool: {df['hospital'].nunique()} hospital(s) "
          f"({df['hospital'].unique().tolist()}), {len(df)} rows total.")
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


def run_shap_for_group(X, y, group_name, output_dir, interaction_mains=("embed_dim",),
                       hospitals=None):
    """Train XGBoost and compute SHAP for one task group (pooled across hospitals).

    Args:
        X: pd.DataFrame of architecture features (4 columns from ARCH_COLS).
        y: composite target (-avg_rank), higher = better.
        group_name: label for plot titles.
        output_dir: where to write outputs.
        interaction_mains: iterable of architecture feature names to use as
            the "main" feature in dependence/interaction plots. For each,
            produces shap_interaction_<feat>.png with sub-panels for
            interactions with the other 3 features. Default: only embed_dim.
        hospitals: Optional pd.Series with same index as X giving the
            hospital each row came from. Saved alongside shap_values so
            run_meta_regression.py can fit hospital-random-intercept models.
    """
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

    # SHAP values CSV — long-format with hospital + X cols + signed SHAP cols.
    # Layout enables run_meta_regression.py to fit categorical mixed-effect
    # models like `shap_<feat> ~ C(<feat>) + (1 | hospital)` directly.
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in ARCH_COLS])
    if hospitals is not None:
        shap_df["hospital"] = hospitals.values
    for c in ARCH_COLS:
        shap_df[c] = X[c].values
    # Reorder columns for readability: hospital, arch features, signed SHAPs.
    col_order = (["hospital"] if hospitals is not None else []) + \
                list(ARCH_COLS) + \
                [f"shap_{c}" for c in ARCH_COLS]
    shap_df = shap_df[col_order]
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)

    # Mean |SHAP| per feature (Fig 4 配套数值表; not loaded by mas_search anymore)
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=ARCH_COLS)
    mean_abs = mean_abs.sort_values(ascending=False)
    mean_abs.to_csv(output_dir / "mean_abs_shap.csv", header=False)

    # Combined panel: SHAP left + 4 univariate regression panels right
    _save_shap_with_regression(X, y, shap_values, group_name, output_dir)

    # Interaction plot(s): one PNG per main feature requested via CLI
    for main_feat in interaction_mains:
        _save_shap_interaction(X, shap_values, group_name, output_dir,
                               main_feature=main_feat)

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


def _save_shap_interaction(X, shap_values, group_name, output_dir, main_feature="embed_dim"):
    """SHAP dependence plots for `main_feature` × each other feature.

    For each pair (main, other):
      x = value of main_feature
      y = SHAP value contributed by main_feature
      color = value of other feature

    Visual cue:
      - Colors stack vertically by other-feature value → STRONG interaction
        (main_feature's effect on prediction depends on other's value)
      - Colors evenly mixed across y → no interaction (main effect only)
      - Slope flips sign across colors → strong qualitative interaction

    This is the canonical SHAP-style way to detect feature interactions.
    Output: shap_interaction_<main_feature>.png in output_dir.
    """
    output_dir = Path(output_dir)
    other_features = [f for f in ARCH_COLS if f != main_feature]
    if not other_features:
        return

    main_idx = list(X.columns).index(main_feature)
    main_vals = X[main_feature].values.astype(float)
    main_shap = shap_values[:, main_idx]

    n = len(other_features)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, other in zip(axes, other_features):
        other_vals = X[other].values.astype(float)
        # Manual scatter (avoids version-dependent shap.dependence_plot ax= API)
        sc = ax.scatter(
            main_vals, main_shap,
            c=other_vals, cmap="coolwarm",
            s=30, alpha=0.75, edgecolors="none",
        )
        ax.axhline(0, color="gray", lw=0.5, linestyle="--", alpha=0.7)
        ax.set_xlabel(main_feature, fontsize=10)
        ax.set_ylabel(f"SHAP value of {main_feature}", fontsize=10)
        ax.set_title(f"{main_feature} × {other}", fontsize=11)
        ax.grid(True, alpha=0.25)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(other, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{group_name}: how {main_feature}'s effect interacts with other features",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_interaction_{main_feature}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    df = load_metadata(args.results_dir, args.hospitals)
    df = prepare_features(df)

    all_shap_summary = []

    # Pool by task ACROSS all hospitals in the filtered set.
    # Output layout: <output_dir>/<task>/...   (no per-hospital subdir)
    # Sample size: ~K_archs × n_hospitals per task (vs K_archs alone in old per-hospital design).
    for task, group in df.groupby("task"):
        group = group.reset_index(drop=True)
        n_hospitals = group["hospital"].nunique()
        n_rows = len(group)
        group_name = f"task={task} (pooled across {n_hospitals} hospital(s), n={n_rows})"

        if n_rows < 5:
            print(f"  [{group_name}] skipped — only {n_rows} samples (need >= 5)")
            continue

        # Compute composite target: negate avg_rank so higher = better performance
        # This way positive SHAP = feature improves performance
        avg_rank = compute_avg_rank(group)
        y = -avg_rank

        X = group[ARCH_COLS].copy()
        hospitals = group["hospital"]   # for run_meta_regression mixed-effect models

        group_output = Path(args.output_dir) / task
        mean_abs = run_shap_for_group(
            X, y, group_name, group_output,
            interaction_mains=args.interaction_main,
            hospitals=hospitals,
        )

        row = {
            "task": task,
            "n_hospitals": n_hospitals,
            "n_rows": n_rows,
        }
        for feat in ARCH_COLS:
            row[f"mean_abs_shap_{feat}"] = mean_abs[feat]
        all_shap_summary.append(row)

    # Save combined summary (paper artifact + run_meta_regression input)
    if all_shap_summary:
        summary_df = pd.DataFrame(all_shap_summary)
        out_path = Path(args.output_dir) / "shap_summary_all.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nCombined SHAP summary: {out_path}")
        print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
