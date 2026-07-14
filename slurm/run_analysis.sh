#!/bin/bash
# Run full Stage E analysis on HiperGator, PER PARAMETER BUDGET.
# Execute from login node: bash slurm/run_analysis.sh
#
# Layout (matches the constrained-NAS two-budget design):
#   analyze/YYYY-MM-DD/
#     ├── 1M/   ← all per-budget tables+figures for the tight  1e6 budget
#     ├── 3M/   ← all per-budget tables+figures for the loose  3e6 budget
#     ├── figure3_regression_combined.png   (Stage A, budget-independent)
#     └── figure7_lead_vs_budget*.{png,csv} (cross-budget headline comparison)
#
# Per-budget results are read from results/budget_<MAX_PARAMS>/seed_*/ (namespaced
# by submit_stage_b.sbatch). The supernet checkpoint + Stage-A regression are
# budget-independent and live under results/ directly.

set -euo pipefail

REPO=/home/lideyi/MAS-NAS             # git repo (code lives here)
PROJECT=/blue/mei.liu/lideyi/MAS-NAS  # data lives here
DATE=$(date +%Y-%m-%d)
OUTROOT=$PROJECT/analyze/$DATE
HOSPITALS="source_15 MIMIC-IV"

# Budget level → subdir label. Keep in sync with submit_stage_b.sbatch MAX_PARAMS.
BUDGETS=(1000000 3000000)
declare -A LABEL=( [1000000]=1M [3000000]=3M )

mkdir -p "$OUTROOT"
cd "$REPO"

echo "============================================"
echo " MAS-NAS Stage E Analysis (per-budget)"
echo " Output root : $OUTROOT"
echo " Hospitals   : $HOSPITALS"
echo " Budgets     : ${BUDGETS[*]}"
echo "============================================"

# ── Per-budget analysis (E.1–E.5) ────────────────────────────────────────────
for B in "${BUDGETS[@]}"; do
  L=${LABEL[$B]}
  RESULTS=$PROJECT/results/budget_$B
  OUT=$OUTROOT/$L

  if [[ ! -d "$RESULTS" ]]; then
    echo ""; echo "⚠️  skip budget $B ($L): $RESULTS not found (run Stage B for this budget first)"
    continue
  fi
  mkdir -p "$OUT"
  echo ""; echo "########## BUDGET $B → $L ##########"
  echo " Results: $RESULTS"
  echo " Output : $OUT"

  for H in $HOSPITALS; do
    echo ""; echo "[E.1] aggregate_results  budget=$L hospital=$H"
    python analyze/aggregate_results.py \
      --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT" \
      --mas_budget 20 --baseline_budget 30
  done

  for H in $HOSPITALS; do
    echo ""; echo "[E.2] plot_search_trajectory  budget=$L hospital=$H"
    python analyze/plot_search_trajectory.py \
      --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
  done

  for H in $HOSPITALS; do
    echo ""; echo "[E.3] plot_pareto  budget=$L hospital=$H"
    python analyze/plot_pareto.py \
      --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
  done

  for H in $HOSPITALS; do
    echo ""; echo "[E.4] plot_loto_ablation  budget=$L hospital=$H"
    python analyze/plot_loto_ablation.py \
      --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
  done

  echo ""; echo "[E.5] plot_source_hospital_choice  budget=$L"
  python analyze/plot_source_hospital_choice.py \
    --results_root "$RESULTS" --target_hospitals source_15 MIMIC-IV --out_dir "$OUT"
done

# ── Cross-budget / budget-independent artifacts (date root, OUTSIDE 1M/3M) ────
# E.6 Regression (Stage A data, budget-independent) → outer date root
echo ""; echo "[E.6] plot_regression_combined (budget-independent)"
python analyze/plot_regression_combined.py \
  --results_dir "$PROJECT/results" \
  --output      "$OUTROOT/figure3_regression_combined.png" \
  --summary_csv "$OUTROOT/figure3_spearman_summary.csv"

# E.7 Lead-vs-budget headline (reads the per-budget main_table_*.csv above) → root
echo ""; echo "[E.7] plot_lead_vs_budget (cross-budget headline)"
python analyze/plot_lead_vs_budget.py \
  --results_project "$PROJECT/results" \
  --budgets 1000000:1M 3000000:3M \
  --hospitals source_15 MIMIC-IV \
  --out_dir "$OUTROOT"

echo ""
echo "============================================"
echo "✅ Per-budget outputs → $OUTROOT/{1M,3M}/"
echo "✅ Cross-budget       → $OUTROOT/ (regression, lead-vs-budget)"
echo "============================================"
ls -R "$OUTROOT"
