#!/usr/bin/env python
"""
Post-hoc budget simulation — ATHENA's lead vs. the parameter budget, WITHOUT re-running search.

Takes the ORIGINAL 256-grid search traces
    <results_root>/seed_*/<hospital>/search/<method>/<task>/<method>_search.csv
and re-applies progressively tighter parameter caps *after the fact*: for each cap,
drop every evaluation with `num_params > cap`, then take the best remaining
`val_auprc` as "what that method would have returned under this cap". The reported
quantity is

    lead(pp) = mean_seeds best[mas]  -  max_over_baselines mean_seeds best[baseline]

in percentage points, per (hospital, cap, task).

WHY POST-HOC INSTEAD OF RE-RUNNING WITH --max_params
----------------------------------------------------
The old 256-grid supernet was OVERWRITTEN when the grid was widened to 1008 and the
supernet retrained. Any NEW 256-grid run could therefore only use the 1008 supernet,
which spreads its capacity over ~4x more configurations — so a fresh `256 + tighter
cap` run would NOT be comparable to the existing 256 results (different supernet).
Post-hoc filtering reuses the SAME traces from the SAME supernet across all caps:
zero supernet/prior/seed confound, 5 seeds, zero compute cost. It is scientifically
*cleaner* than paying to re-run, not merely cheaper.

CAVEAT — THIS IS A LOWER BOUND (state this in the paper)
--------------------------------------------------------
Post-hoc filtering is NOT equivalent to true constrained search. A method that knew
the cap up front would spend all of its evaluation budget inside the feasible region;
here the infeasible evaluations (e.g. ~29% of the space at a 2M cap) are simply
discarded, so each method effectively searches with fewer usable evaluations.
Post-hoc therefore UNDERSTATES what true constrained search would achieve. The
inference it licenses is one-directional: if the lead does not improve even here,
true constrained search is unlikely to rescue it. (It is independently corroborated
by the real constrained run on the 1008 grid at a binding 1M cap.)

METRIC
------
`val_auprc` by default: *_search.csv records validation metrics for every evaluated
architecture; test metrics exist only for the finally-selected architecture in
*_best.csv, so test cannot be recomputed post-hoc without re-finetuning. Label the
figure axis as validation AUPRC accordingly.

Reuses aggregate_results' method/task constants so the method set matches Table 1
(MAIN_METHODS = baseline0/1/2/4 + mas; baseline3/LLMatic excluded).

Run from the repo root:
    python analyze/plot_lead_vs_budget_posthoc.py \
        --results_root /blue/mei.liu/lideyi/MAS-NAS/results \
        --hospitals source_15 MIMIC-IV \
        --caps 4000000 3000000 2000000 1000000 \
        --out_dir /blue/mei.liu/lideyi/MAS-NAS/analyze/<date>

NOTE: --results_root must point at the root holding the ORIGINAL 256-grid `seed_*/`
dirs (i.e. `.../results`), NOT at `.../results/budget_<X>/` (those are the newer
1008-grid constrained runs).
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

# Reuse the exact method/task definitions used by Table 1 (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_results import (  # noqa: E402
    MAIN_METHODS, METHOD_DISPLAY, TASK_DISPLAY, TASKS,
)

# Baselines ATHENA is compared against = MAIN_METHODS minus mas.
BASELINE_METHODS = [m for m in MAIN_METHODS if m != "mas"]


def cap_label(cap: int) -> str:
    """4000000 -> '4M'."""
    return f"{cap / 1e6:g}M"


def discover_seeds(results_root: Path) -> list:
    """All seed_<N> dirs directly under results_root (the 256-grid runs)."""
    seeds = []
    for p in sorted(results_root.glob("seed_*")):
        try:
            seeds.append(int(p.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(seeds)


def best_under_cap(results_root: Path, hosp: str, method: str, task: str,
                   seed: int, cap: int, metric: str):
    """Best `metric` among this run's evaluations with num_params <= cap.

    Returns None if the trace is missing/unreadable or no evaluation is feasible
    under the cap (the latter is itself informative — that method never proposed a
    feasible architecture at this budget)."""
    f = results_root / f"seed_{seed}" / hosp / "search" / method / task / f"{method}_search.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"  ⚠ cannot read {f}: {e}")
        return None
    if "num_params" not in df.columns or metric not in df.columns:
        print(f"  ⚠ {f} lacks num_params/{metric}")
        return None
    d = df[df["num_params"] <= cap]
    d = d[d[metric].notna()]
    if len(d) == 0:
        return None
    return float(d[metric].max())


def mean_over_seeds(results_root, hosp, method, task, seeds, cap, metric):
    """(mean*100, n_seeds) over seeds that have a feasible evaluation; (None, 0) if none."""
    vals = [best_under_cap(results_root, hosp, method, task, s, cap, metric) for s in seeds]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return float(np.mean(vals)) * 100, len(vals)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", required=True,
                    help="Root holding the ORIGINAL 256-grid seed_*/ dirs "
                         "(e.g. /blue/mei.liu/lideyi/MAS-NAS/results). NOT results/budget_<X>/.")
    ap.add_argument("--hospitals", nargs="+", default=["source_15", "MIMIC-IV"])
    ap.add_argument("--caps", type=int, nargs="+",
                    default=[4000000, 3000000, 2000000, 1000000],
                    help="Parameter caps to simulate, in params (default: 4M 3M 2M 1M).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Default: auto-discover seed_* under --results_root.")
    ap.add_argument("--metric", default="val_auprc",
                    help="Column in *_search.csv to maximize (default: val_auprc; "
                         "search traces hold validation metrics only).")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or discover_seeds(results_root)
    if not seeds:
        raise SystemExit(f"No seed_* dirs under {results_root} — wrong --results_root?")
    # Loosest -> tightest, so the x-axis reads "looser → tighter" left to right.
    caps = sorted(args.caps, reverse=True)

    print(f"[post-hoc budget sim] results_root={results_root}")
    print(f"  seeds={seeds}  caps={[cap_label(c) for c in caps]}  metric={args.metric}")
    print(f"  methods: mas vs best of {BASELINE_METHODS}")

    rows = []
    for hosp in args.hospitals:
        for cap in caps:
            for task in TASKS:
                mas_mean, mas_n = mean_over_seeds(
                    results_root, hosp, "mas", task, seeds, cap, args.metric)
                best_b, best_b_mean, best_b_n = None, None, 0
                for b in BASELINE_METHODS:
                    bm, bn = mean_over_seeds(
                        results_root, hosp, b, task, seeds, cap, args.metric)
                    if bm is None:
                        continue
                    if best_b_mean is None or bm > best_b_mean:
                        best_b, best_b_mean, best_b_n = b, bm, bn
                lead = (None if (mas_mean is None or best_b_mean is None)
                        else round(mas_mean - best_b_mean, 3))
                rows.append({
                    "hospital": hosp,
                    "cap": cap,
                    "cap_label": cap_label(cap),
                    "task": task,
                    "task_display": TASK_DISPLAY.get(task, task),
                    "mas_pct": None if mas_mean is None else round(mas_mean, 3),
                    "mas_n_seeds": mas_n,
                    "best_baseline": None if best_b is None else METHOD_DISPLAY.get(best_b, best_b),
                    "best_baseline_pct": None if best_b_mean is None else round(best_b_mean, 3),
                    "best_baseline_n_seeds": best_b_n,
                    "lead_pp": lead,
                })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "lead_vs_budget_posthoc.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ wrote {csv_path}  ({len(df)} rows)")

    # ── Console summary: the headline numbers ────────────────────────────────
    print("\n" + "=" * 76)
    print("Post-hoc lead (pp) vs parameter cap — ATHENA minus best baseline "
          f"({args.metric})")
    print("=" * 76)
    for hosp in args.hospitals:
        sub = df[df.hospital == hosp]
        if sub.lead_pp.notna().sum() == 0:
            continue
        print(f"\n{hosp}")
        for cap in caps:
            s = sub[(sub.cap == cap) & sub.lead_pp.notna()]
            if s.empty:
                continue
            per_task = "  ".join(f"{r.task_display}:{r.lead_pp:+.2f}"
                                 for r in s.itertuples())
            print(f"  {cap_label(cap):>3s} cap → mean lead {s.lead_pp.mean():+.2f}   ({per_task})")

    if df.lead_pp.notna().sum() == 0:
        print("\n⚠ no lead values computed — check --results_root / --metric; figure skipped.")
        return

    # ── Figure: one panel per hospital, x = cap (looser → tighter) ───────────
    hosps = [h for h in args.hospitals
             if h in set(df[df.lead_pp.notna()].hospital)]
    fig, axes = plt.subplots(1, max(len(hosps), 1),
                             figsize=(6 * max(len(hosps), 1), 4.6), squeeze=False)
    x = list(range(len(caps)))
    xlabels = [cap_label(c) for c in caps]

    for ax, hosp in zip(axes[0], hosps):
        sub = df[df.hospital == hosp]
        for task in TASKS:
            t = sub[sub.task == task]
            if t.lead_pp.notna().sum() == 0:
                continue
            ys = []
            for c in caps:
                v = t[t.cap == c].lead_pp
                ys.append(v.iloc[0] if len(v) and pd.notna(v.iloc[0]) else np.nan)
            ax.plot(x, ys, marker="o", lw=1.2, alpha=0.75,
                    label=TASK_DISPLAY.get(task, task))
        # Mean across tasks — the headline trend.
        means = [sub[(sub.cap == c) & sub.lead_pp.notna()].lead_pp.mean() for c in caps]
        ax.plot(x, means, marker="s", lw=2.8, color="black", zorder=5, label="Mean")
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Parameter budget (looser → tighter)")
        ax.set_ylabel(f"ATHENA − best baseline ({args.metric}, pp)")
        ax.set_title(hosp)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle("Post-hoc budget simulation: ATHENA's lead does not grow as the budget tightens",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = out_dir / "figure4_lead_vs_budget_posthoc.png"
    fig.savefig(png_path, dpi=150)
    print(f"\n✅ wrote {png_path}")


if __name__ == "__main__":
    main()
