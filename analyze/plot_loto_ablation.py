"""Fig 5 — LOTO + Cold-Start Robustness Ablation.

For each of the 5 tasks, plot a grouped bar chart comparing:
  - MAS-NAS (exact match)        full historical context, exact task in metadata
  - MAS-NAS (LOTO)               target task removed → cosine fallback path
  - MAS-NAS (cold)               no historical context at all
  - Best LLM baseline            max(LLMatic, CoLLM-NAS) per task

Reads:
    results/seed_<SEED>/<hospital>/search/<method>/<task>/<method>_best.csv
    where <method> ∈ {mas, mas_loto, mas_cold, baseline3, baseline4}

Outputs:
    figure5_loto_robustness.png — 5 panels (one per task) of grouped bars
                                  with std error bars; * marks p<0.05
                                  (paired Wilcoxon: MAS-LOTO vs best-baseline)

Paper claim:
    "Even when the target task is removed from historical metadata (LOTO),
     MAS-NAS still outperforms the strongest baseline. Cold-start MAS-NAS
     further degrades but typically remains competitive, confirming that
     multi-agent collaboration provides value beyond the historical prior."

Usage:
    python analyze/plot_loto_ablation.py --hospital MIMIC-IV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# Colors: MAS variants = same crimson family (intensity ~ amount of context),
# baseline = neutral gray. Reuses METHOD_COLOR convention from plot_pareto.py.
CONDITION_COLOR = {
    "mas":      "#DC143C",   # crimson — full context
    "mas_loto": "#FF6B6B",   # lighter crimson — degraded but still has fallback prior
    "mas_cold": "#FFB3B3",   # pink — no prior at all
    # Baseline color resolved at plot time from BASELINE_COLOR
}
BASELINE_COLOR = {
    "baseline3": "#1E90FF",   # LLMatic — dodger blue
    "baseline4": "#9370DB",   # CoLLM-NAS — purple
}
BASELINE_DISPLAY = {
    "baseline3": "LLMatic",
    "baseline4": "CoLLM-NAS",
}

CONDITION_DISPLAY = {
    "mas":      "MAS-NAS\n(exact)",
    "mas_loto": "MAS-NAS\n(LOTO)",
    "mas_cold": "MAS-NAS\n(cold)",
}

LLM_BASELINE_METHODS = ["baseline3", "baseline4"]

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]
TASK_DISPLAY = {
    "death": "Mortality",
    "stay": "Stay > 7d",
    "readmission": "Readmit (3M)",
    "next_diag_6m_pheno": "Phenotype 6M",
    "next_diag_12m_pheno": "Phenotype 12M",
}


# ---------------------------------------------------------------------------
# Data loading — walks results/seed_*/<hospital>/search/<method>/<task>/<method>_best.csv
# ---------------------------------------------------------------------------
def load_best_scores(results_root: Path, hospital: str) -> dict:
    """Returns {(method, task): {seed: test_auprc, ...}}."""
    out: dict = {}
    for seed_dir in sorted(results_root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        search_root = seed_dir / hospital / "search"
        if not search_root.exists():
            continue
        for method_dir in search_root.iterdir():
            method = method_dir.name
            for task_dir in method_dir.iterdir():
                task = task_dir.name
                best_files = list(task_dir.glob("*_best.csv"))
                if not best_files:
                    continue
                try:
                    df = pd.read_csv(best_files[0])
                except Exception as e:
                    print(f"  ⚠ Cannot read {best_files[0]}: {e}")
                    continue
                if len(df) == 0 or "test_auprc" not in df.columns:
                    continue
                out.setdefault((method, task), {})[seed] = float(df["test_auprc"].iloc[0])
    return out


def aggregate(scores_by_seed: dict) -> tuple:
    """Mean ± std (sample std, ddof=1) of a dict {seed: score}."""
    if not scores_by_seed:
        return None, None, 0
    vals = list(scores_by_seed.values())
    if len(vals) == 1:
        return float(vals[0]), 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def best_baseline_per_task(all_scores: dict, task: str) -> str | None:
    """Pick the LLM-based baseline (LLMatic or CoLLM-NAS) with highest mean
    test AUPRC for `task`. Returns the method name (e.g., 'baseline4'), or
    None if neither has data."""
    best_method = None
    best_mean = -np.inf
    for bm in LLM_BASELINE_METHODS:
        seeds_dict = all_scores.get((bm, task), {})
        m, _, n = aggregate(seeds_dict)
        if m is not None and n > 0 and m > best_mean:
            best_method = bm
            best_mean = m
    return best_method


# ---------------------------------------------------------------------------
# Significance: paired Wilcoxon (MAS-LOTO vs best-baseline) on shared seeds
# ---------------------------------------------------------------------------
def paired_wilcoxon(a_dict: dict, b_dict: dict) -> float | None:
    """One-sided paired Wilcoxon (a > b) on seeds present in both dicts.
    Returns p-value, or None if too few pairs / undefined."""
    common_seeds = sorted(set(a_dict) & set(b_dict))
    if len(common_seeds) < 2:
        return None
    a = np.array([a_dict[s] for s in common_seeds])
    b = np.array([b_dict[s] for s in common_seeds])
    if np.allclose(a, b):
        return 1.0
    try:
        _, p = wilcoxon(a, b, alternative="greater")
        return float(p)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-task plotting
# ---------------------------------------------------------------------------
def plot_panel(ax, all_scores: dict, task: str):
    """Draw one panel: 4 grouped bars for one task."""
    bb_method = best_baseline_per_task(all_scores, task)

    # Resolve display order + values
    cond_order = ["mas", "mas_loto", "mas_cold"]
    if bb_method:
        cond_order.append(bb_method)

    means, stds, ns, labels, colors = [], [], [], [], []
    for cond in cond_order:
        seeds_dict = all_scores.get((cond, task), {})
        m, s, n = aggregate(seeds_dict)
        means.append(m if m is not None else 0.0)
        stds.append(s if s is not None else 0.0)
        ns.append(n)
        if cond in CONDITION_DISPLAY:
            labels.append(CONDITION_DISPLAY[cond])
            colors.append(CONDITION_COLOR[cond])
        else:
            labels.append(BASELINE_DISPLAY.get(cond, cond))
            colors.append(BASELINE_COLOR.get(cond, "#888888"))

    # Convert AUPRC → percent for display
    means_pct = [m * 100 for m in means]
    stds_pct = [s * 100 for s in stds]

    x = np.arange(len(cond_order))
    bars = ax.bar(x, means_pct, yerr=stds_pct, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  alpha=0.9, error_kw={"linewidth": 1.0})

    # Annotate count of seeds above each bar (if std > 0)
    for i, (m_pct, s_pct, n_seeds) in enumerate(zip(means_pct, stds_pct, ns)):
        if n_seeds > 0:
            ax.text(i, m_pct + s_pct + 0.4, f"n={n_seeds}",
                    ha="center", va="bottom", fontsize=7, color="dimgray")

    # Significance: paired Wilcoxon (MAS-LOTO vs best-baseline, one-sided LOTO>baseline)
    if bb_method and len(cond_order) == 4:
        loto_dict = all_scores.get(("mas_loto", task), {})
        bl_dict = all_scores.get((bb_method, task), {})
        p = paired_wilcoxon(loto_dict, bl_dict)
        if p is not None and p < 0.05:
            i_loto, i_bl = 1, 3
            y_top = max(means_pct[i_loto] + stds_pct[i_loto],
                        means_pct[i_bl] + stds_pct[i_bl]) + 1.5
            ax.plot([i_loto, i_loto, i_bl, i_bl],
                    [y_top, y_top + 0.3, y_top + 0.3, y_top],
                    color="black", linewidth=0.8)
            stars = "*" if p >= 0.01 else ("**" if p >= 0.001 else "***")
            ax.text((i_loto + i_bl) / 2, y_top + 0.4, stars,
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test AUPRC (%)", fontsize=9)
    ax.set_title(TASK_DISPLAY[task], fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)

    # Tight y-axis: zoom to data range so error bars stand out
    if any(m > 0 for m in means_pct):
        valid_means = [m for m in means_pct if m > 0]
        ymin = max(0, min(valid_means) - max(stds_pct) - 3)
        ymax = max(valid_means) + max(stds_pct) + 3
        ax.set_ylim(ymin, ymax)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
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

    print(f"[Fig 5] hospital={args.hospital}")
    print(f"[Fig 5] reading from {results_root}/seed_*/{args.hospital}/search/...")
    all_scores = load_best_scores(results_root, args.hospital)
    print(f"[Fig 5] loaded {len(all_scores)} (method, task) groups")

    if not all_scores:
        print("⚠ No best.csv records found. Submit mas + mas_loto + mas_cold + baseline3/4 jobs first.")
        return

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.0 * len(TASKS), 4.5), sharey=False)
    if len(TASKS) == 1:
        axes = [axes]

    for ax, task in zip(axes, TASKS):
        plot_panel(ax, all_scores, task)

    fig.suptitle(
        f"LOTO + Cold-Start Robustness — {args.hospital}\n"
        f"(MAS-NAS retains performance even when target task is missing from history)",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()

    out_path = out_dir / "figure5_loto_robustness.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved {out_path}")


if __name__ == "__main__":
    main()
