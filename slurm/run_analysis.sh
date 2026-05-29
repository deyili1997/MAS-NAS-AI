#!/bin/bash
# Run full Stage E analysis on HiperGator.
# Execute from login node: bash slurm/run_analysis.sh
# Outputs → /blue/mei.liu/lideyi/MAS-NAS/analyze/YYYY-MM-DD/

set -euo pipefail

PROJECT=/blue/mei.liu/lideyi/MAS-NAS
RESULTS=$PROJECT/results
OUT=$PROJECT/analyze/$(date +%Y-%m-%d)
HOSPITALS="source_15 MIMIC-IV"

mkdir -p "$OUT"

cd "$PROJECT"

echo "============================================"
echo " MAS-NAS Stage E Analysis"
echo " Results  : $RESULTS"
echo " Output   : $OUT"
echo " Hospitals: $HOSPITALS"
echo "============================================"

# ── E.1  Aggregate tables (CSVs) ──────────────────────────────────────────
for H in $HOSPITALS; do
  echo ""
  echo "[E.1] aggregate_results  hospital=$H"
  python analyze/aggregate_results.py \
    --results_root "$RESULTS" \
    --hospital     "$H" \
    --out_dir      "$OUT" \
    --mas_budget   30 \
    --baseline_budget 50
done

# ── E.2  Fig 1: Search trajectory ─────────────────────────────────────────
for H in $HOSPITALS; do
  echo ""
  echo "[E.2] plot_search_trajectory  hospital=$H"
  python analyze/plot_search_trajectory.py \
    --results_root "$RESULTS" \
    --hospital     "$H" \
    --out_dir      "$OUT"
done

# ── E.3  Fig 2: Pareto front ──────────────────────────────────────────────
for H in $HOSPITALS; do
  echo ""
  echo "[E.3] plot_pareto  hospital=$H"
  python analyze/plot_pareto.py \
    --results_root "$RESULTS" \
    --hospital     "$H" \
    --out_dir      "$OUT"
done

# ── E.4  Fig 5: LOTO ablation ─────────────────────────────────────────────
for H in $HOSPITALS; do
  echo ""
  echo "[E.4] plot_loto_ablation  hospital=$H"
  python analyze/plot_loto_ablation.py \
    --results_root "$RESULTS" \
    --hospitals    "$H" \
    --out_dir      "$OUT"
done

# ── E.5  Fig 6: Source hospital choice (cross-hospital, one call) ─────────
echo ""
echo "[E.5] plot_source_hospital_choice"
python analyze/plot_source_hospital_choice.py \
  --results_root    "$RESULTS" \
  --target_hospitals source_15 MIMIC-IV \
  --out_dir         "$OUT"

# ── E.6  Fig 3: Regression combined (Stage A data) ────────────────────────
echo ""
echo "[E.6] plot_regression_combined"
python analyze/plot_regression_combined.py \
  --results_dir "$RESULTS" \
  --output      "$OUT/figure3_regression_combined.png"

echo ""
echo "============================================"
echo "✅ All outputs written to $OUT"
echo "============================================"
ls -lh "$OUT"
