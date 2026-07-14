#!/usr/bin/env python
"""
Cross-budget headline: ATHENA (MAS-NAS) lead over the best baseline as a
function of the parameter budget.

For each budget level, hospital, and task:
    lead(pp) = mean_seeds AUPRC[mas]  -  max_over_baselines mean_seeds AUPRC[baseline]
in percentage points. It then plots lead vs. budget (one line per task, one
panel per hospital) and writes the underlying numbers to CSV.

The constrained-NAS hypothesis: the lead GROWS as the budget tightens (1M)
relative to the near-unconstrained regime (3M).

Reuses aggregate_results.collect_results so the AUPRC numbers match Table 1
exactly (same MAIN_METHODS, same best-per-seed selection; baseline3/LLMatic is
already excluded from MAIN_METHODS). Cross-budget artifact → written to the
OUTER date root, NOT inside 1M/ or 3M/.

Run from the repo root:
    python analyze/plot_lead_vs_budget.py \
        --results_project /blue/mei.liu/lideyi/MAS-NAS/results \
        --budgets 1000000:1M 3000000:3M \
        --hospitals source_15 MIMIC-IV \
        --out_dir /blue/mei.liu/lideyi/MAS-NAS/analyze/<date>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact aggregation logic (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_results import (  # noqa: E402
    collect_results, MAIN_METHODS, METHOD_DISPLAY, TASK_DISPLAY, TASKS, get_seeds,
)

# Baselines that MAS is compared against for the "best baseline" reference.
# = MAIN_METHODS minus mas (baseline3/LLMatic already excluded from MAIN_METHODS).
BASELINE_METHODS = [m for m in MAIN_METHODS if m != "mas"]


def _mean_auprc(records, method, task):
    """Mean test AUPRC over available seeds for (method, task); (None, 0) if absent."""
    seeds = get_seeds(records, method, task)
    scores = [records[(method, task, s)]["best"].get("test_auprc")
              for s in seeds if (method, task, s) in records]
    scores = [x for x in scores if x is not None and not pd.isna(x)]
    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)


def parse_budgets(items):
    """['1000000:1M', '3000000:3M'] -> [(1000000,'1M'), ...] sorted ascending."""
    out = []
    for it in items:
        val, _, lab = it.partition(":")
        out.append((int(val), lab or val))
    return sorted(out, key=lambda x: x[0])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_project", required=True,
                    help="Project results root CONTAINING budget_<MAX_PARAMS>/ dirs "
                         "(e.g. /blue/mei.liu/lideyi/MAS-NAS/results).")
    ap.add_argument("--budgets", nargs="+", default=["1000000:1M", "3000000:3M"],
                    help="Budget levels as <max_params>:<label> "
                         "(e.g. 1000000:1M 3000000:3M).")
    ap.add_argument("--hospitals", nargs="+", default=["source_15", "MIMIC-IV"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)
    project = Path(args.results_project)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for max_params, label in budgets:
        results_root = project / f"budget_{max_params}"
        if not results_root.exists():
            print(f"⚠ skip budget {label}: {results_root} not found")
            continue
        for hosp in args.hospitals:
            records = collect_results(results_root, hosp)
            if not records:
                print(f"⚠ no records for {hosp} @ {label} ({results_root})")
                continue
            for task in TASKS:
                mas_mean, mas_n = _mean_auprc(records, "mas", task)
                best_b, best_b_mean = None, None
                for b in BASELINE_METHODS:
                    bm, _ = _mean_auprc(records, b, task)
                    if bm is None:
                        continue
                    if best_b_mean is None or bm > best_b_mean:
                        best_b, best_b_mean = b, bm
                lead = (None if (mas_mean is None or best_b_mean is None)
                        else round((mas_mean - best_b_mean) * 100, 2))
                rows.append({
                    "max_params": max_params, "budget": label,
                    "hospital": hosp, "task": task,
                    "task_display": TASK_DISPLAY.get(task, task),
                    "mas_auprc_pct": None if mas_mean is None else round(mas_mean * 100, 2),
                    "mas_n_seeds": mas_n,
                    "best_baseline": None if best_b is None else METHOD_DISPLAY.get(best_b, best_b),
                    "best_baseline_auprc_pct": None if best_b_mean is None
                    else round(best_b_mean * 100, 2),
                    "lead_pp": lead,
                })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "lead_vs_budget.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ wrote {csv_path}  ({len(df)} rows)")

    if df.empty or df["lead_pp"].notna().sum() == 0:
        print("⚠ no lead values to plot (need mas + >=1 baseline in >=1 budget); "
              "figure skipped.")
        return

    # ── One panel per hospital, x = budget (ascending), one line per task ──
    hosps = [h for h in args.hospitals if h in set(df.hospital)]
    n = max(len(hosps), 1)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    xlabels = [lab for _, lab in budgets]
    xmap = {lab: i for i, (_, lab) in enumerate(budgets)}

    for ax, hosp in zip(axes[0], hosps):
        sub = df[df.hospital == hosp]
        for task in TASKS:
            t = sub[(sub.task == task) & (sub.lead_pp.notna())]
            if t.empty:
                continue
            xs = np.array([xmap[b] for b in t.budget])
            ys = np.array(list(t.lead_pp), dtype=float)
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker="o",
                    label=TASK_DISPLAY.get(task, task))
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks(list(xmap.values()))
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Parameter budget (tighter → left)")
        ax.set_ylabel("ATHENA − best baseline AUPRC (pp)")
        ax.set_title(hosp)
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=8, title="Task", loc="best")
    fig.suptitle("ATHENA lead over best baseline vs. parameter budget", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = out_dir / "figure7_lead_vs_budget.png"
    fig.savefig(png_path, dpi=150)
    print(f"✅ wrote {png_path}")


if __name__ == "__main__":
    main()
