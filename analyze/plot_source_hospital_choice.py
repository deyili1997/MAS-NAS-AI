"""
Figure 6 (supplementary) — Source hospital choice distribution.

Aggregates `matched_source_hospital` decisions made by Layer 1's dataset-similarity
cosine matcher across all `mas`/`mas_layer1_only`/`mas_loto` runs. Demonstrates
the paper's "per-target source selection" claim: across (target × task × seed)
combinations, the matcher picks different OneFL+ source hospitals depending on
target & task — i.e. the prior is not a single dominant donor.

Reads from:
    results/seed_<N>/<target>/search/<method>/<task>/search_meta.json
where method ∈ {mas, mas_layer1_only, mas_loto}. mas_cold runs are EXCLUDED
because they bypass Layer 1 entirely (matched_source_hospital is None there).

Per (target_hospital × task) cell, the chosen `matched_source_hospital` is
aggregated across seeds × eligible methods. Output:
  - figure6_source_hospital_choice.png  Heatmap: rows = (target, task),
                                          cols = source hospitals, cell = pick
                                          frequency (normalized per row).
  - source_hospital_distribution.csv    Long-format table behind the heatmap
                                          for reviewers / supplementary report.

Usage:
    python analyze/plot_source_hospital_choice.py \\
        --target_hospitals MIMIC-III MIMIC-IV
    python analyze/plot_source_hospital_choice.py \\
        --target_hospitals MIMIC-III MIMIC-IV \\
        --results_root /blue/mei.liu/lideyi/MAS-NAS/results
"""
from __future__ import annotations

import argparse
import sys
import json
from collections import Counter
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


# Methods that actually exercise Layer 1 retrieval — the only runs where
# matched_source_hospital is meaningful. mas_cold is intentionally excluded.
LAYER1_METHODS = ["mas", "mas_layer1_only", "mas_loto"]

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
TASK_DISPLAY = {
    "death": "Mortality",
    "stay": "Stay > 7d",
    "readmission": "Readmit (3M)",
    "next_diag_6m_pheno": "Phenotype 6M",
    "next_diag_12m_pheno": "Phenotype 12M",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_meta_records(results_root: Path, target_hospitals: list[str]) -> pd.DataFrame:
    """Walk results/seed_*/<target>/search/<method>/<task>/search_meta.json.

    Returns long-format DataFrame with one row per (target, task, method, seed)
    that has a non-null matched_source_hospital (mas_cold and legacy runs
    without the field are skipped with a notice).
    """
    rows = []
    skipped_legacy = 0
    skipped_cold = 0
    skipped_no_match = 0

    for seed_dir in sorted(results_root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        for target in target_hospitals:
            search_root = seed_dir / target / "search"
            if not search_root.exists():
                continue
            for method in LAYER1_METHODS:
                method_dir = search_root / method
                if not method_dir.exists():
                    continue
                for task in TASKS:
                    meta_path = method_dir / task / "search_meta.json"
                    if not meta_path.exists():
                        continue
                    try:
                        meta = json.load(open(meta_path))
                    except Exception:
                        continue
                    if "matched_source_hospital" not in meta:
                        # Legacy run (pre-改5) — field doesn't exist yet
                        skipped_legacy += 1
                        continue
                    msrc = meta.get("matched_source_hospital")
                    if msrc is None:
                        # Cold-start branch where Layer 1 was off (shouldn't
                        # normally hit since we filter LAYER1_METHODS above,
                        # but possible if no_history was passed)
                        skipped_cold += 1
                        continue
                    rows.append({
                        "seed": seed,
                        "target_hospital": target,
                        "task": task,
                        "method": method,
                        "matched_source_hospital": msrc,
                        "matched_source_similarity": meta.get("matched_source_similarity"),
                        "matched_task": meta.get("matched_task"),
                        "matched_task_similarity": meta.get("matched_task_similarity"),
                    })

    df = pd.DataFrame(rows)
    if skipped_legacy or skipped_cold:
        print(f"  (skipped {skipped_legacy} legacy runs without matched_source_hospital field, "
              f"{skipped_cold} cold-start runs with None)")
    return df


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------
def render_heatmap(ax, df_target: pd.DataFrame, target: str, all_sources: list[str]):
    """One heatmap per target hospital.

    Y axis = tasks (in fixed TASKS order)
    X axis = source hospitals (all_sources, sorted for cross-target consistency)
    Cell value = fraction of seeds × methods that picked this source for this task.
    """
    n_tasks = len(TASKS)
    n_sources = len(all_sources)
    mat = np.zeros((n_tasks, n_sources))
    counts = []     # per-task total picks (for caption density)

    for i, task in enumerate(TASKS):
        sub = df_target[df_target["task"] == task]
        total = len(sub)
        counts.append(total)
        if total == 0:
            continue
        c = Counter(sub["matched_source_hospital"])
        for j, src in enumerate(all_sources):
            mat[i, j] = c.get(src, 0) / total

    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    # Cell annotations (only if cell has data)
    for i in range(n_tasks):
        for j in range(n_sources):
            if mat[i, j] > 0:
                color = "white" if mat[i, j] < 0.55 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xticks(range(n_sources))
    ax.set_xticklabels([site_label(s) for s in all_sources],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f"{TASK_DISPLAY[t]}\n(n={counts[i]})"
                        for i, t in enumerate(TASKS)], fontsize=8)
    ax.set_title(f"Target: {site_label(target)}", fontsize=11)
    return im


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results_root", type=str, default="./results",
                   help="Root containing seed_<N>/ subdirectories.")
    p.add_argument("--target_hospitals", type=str, nargs="+", required=True,
                   help="Target hospital names (e.g. MIMIC-III MIMIC-IV). "
                        "These are the NAS targets, not the OneFL+ sources.")
    p.add_argument("--out_dir", type=str, default="./analyze",
                   help="Output directory for the figure + CSV.")
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Fig 6] target_hospitals={args.target_hospitals}")
    print(f"[Fig 6] reading from {results_root}/seed_*/<target>/search/...\n")

    df = load_meta_records(results_root, args.target_hospitals)
    print(f"  Loaded {len(df)} (target, task, method, seed) records with matched_source field.")
    if len(df) == 0:
        print("⚠ No usable records. Either:")
        print("  - mas_search.py needs to be re-run after 改5 (matched_source_hospital field added).")
        print("  - --target_hospitals doesn't match any results/seed_*/<target>/ layout.")
        return

    # Long-format distribution CSV (for reviewers + supplementary table)
    dist_rows = []
    for (target, task), sub in df.groupby(["target_hospital", "task"]):
        total = len(sub)
        for src, n in Counter(sub["matched_source_hospital"]).items():
            dist_rows.append({
                "target_hospital": target,
                "task": task,
                "matched_source_hospital": src,
                "n_picks": n,
                "total_runs": total,
                "fraction": round(n / total, 4),
                "mean_cosine": round(
                    float(sub[sub["matched_source_hospital"] == src]
                          ["matched_source_similarity"].mean()), 4),
            })
    dist_df = pd.DataFrame(dist_rows).sort_values(
        ["target_hospital", "task", "fraction"], ascending=[True, True, False])
    dist_csv = out_dir / "source_hospital_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    print(f"\n  ✓ Long-format distribution: {dist_csv} ({len(dist_df)} rows)")
    print("\n  Per-(target, task) most-frequent matched source:")
    most_freq = dist_df.sort_values("fraction", ascending=False)\
                       .drop_duplicates(["target_hospital", "task"])
    print(most_freq[["target_hospital", "task",
                     "matched_source_hospital", "fraction", "mean_cosine"]].to_string(index=False))

    # Common source axis (union across all targets, sorted alphabetically)
    all_sources = sorted(df["matched_source_hospital"].unique())
    print(f"\n  Source hospital pool found: {all_sources}")

    # One heatmap per target hospital, stacked horizontally
    targets_with_data = [t for t in args.target_hospitals if (df["target_hospital"] == t).any()]
    if not targets_with_data:
        print("⚠ No target hospitals have any matched_source data. Nothing to plot.")
        return

    n_targets = len(targets_with_data)
    fig_width = max(6, 1.0 * len(all_sources)) * n_targets
    fig, axes = plt.subplots(1, n_targets, figsize=(fig_width, 4.5), sharey=True)
    if n_targets == 1:
        axes = [axes]

    last_im = None
    for ax, target in zip(axes, targets_with_data):
        last_im = render_heatmap(ax, df[df["target_hospital"] == target], target, all_sources)

    # Single colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.18, 0.012, 0.65])
    fig.colorbar(last_im, cax=cbar_ax, label="Pick fraction\n(over seeds × Layer-1 methods)")

    fig.suptitle(
        "Layer 1 source hospital selection across target × task\n"
        f"Pool: {len(all_sources)} OneFlorida+ source sites; "
        f"Layer-1 methods: {', '.join(LAYER1_METHODS)} (mas_cold excluded)",
        fontsize=11, y=1.04,
    )

    out_path = out_dir / "figure6_source_hospital_choice.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Saved {out_path}")


if __name__ == "__main__":
    main()
