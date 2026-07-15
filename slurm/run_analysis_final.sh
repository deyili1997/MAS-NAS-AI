#!/bin/bash
# =============================================================================
# ONE-SHOT FINAL ANALYSIS — 256-grid MAIN experiment → analyze/<date>_final/
# =============================================================================
# Regenerates EVERY table and figure of the main (256-grid, 4M-cap, 5-seed)
# experiment into ONE dated folder, so the paper no longer has to stitch results
# from two archived runs (2026-06-26 tables/figures + 2026-06-29 anytime).
#
# Reads the ORIGINAL 256-grid results that still live at:
#     <PROJECT>/results/seed_{111,123,456,789,999}/<hospital>/search/<method>/<task>/
# (untouched by the newer 1008-grid runs, which are namespaced under
#  <PROJECT>/results/budget_<MAX_PARAMS>/ — do NOT point this script there.)
#
# Usage (login node is fine — everything here is CPU-only and takes ~minutes):
#     module load conda && conda activate Autoformer
#     bash slurm/run_analysis_final.sh
#
# Two things this script deliberately does NOT do — see the notes at the bottom
# of the run for what to do about them:
#   1. Fig 3 (regression) — the 256-grid regression raw data was OVERWRITTEN by
#      the 1008-grid retrain, so a regenerated Fig 3 would no longer match the
#      256 main table's supernet. Carry the archived PNG over instead.
#   2. Anytime (Table II) — needs the pre-existing test re-test outputs. If they
#      are gone, the table is skipped (regenerating it needs GPU re-finetuning,
#      not just CPU).
# =============================================================================

set -uo pipefail   # NOT -e: keep going so one failure doesn't kill the whole run

REPO=/home/lideyi/MAS-NAS
PROJECT=/blue/mei.liu/lideyi/MAS-NAS
RESULTS=$PROJECT/results                              # 256-grid seed_* live HERE
OUT=$PROJECT/analyze/$(date +%Y-%m-%d)_final
ANYTIME=${ANYTIME_DIR:-$PROJECT/analyze/anytime}      # old, non-budget-namespaced
HOSPITALS="source_15 MIMIC-IV"
ARCHIVE_HINT="Results/256_version/2026-06-26/figure3_regression_combined.png"

mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "============================================================"
echo " FINAL ANALYSIS — 256-grid main experiment"
echo " Results  : $RESULTS   (expects seed_*/ directly under it)"
echo " Output   : $OUT"
echo " Hospitals: $HOSPITALS"
echo "============================================================"

# Sanity: make sure we're pointed at the 256 results, not the 1008 budget runs.
if ! compgen -G "$RESULTS/seed_*" > /dev/null; then
    echo "❌ No seed_* under $RESULTS — wrong root? (the 1008 runs live in $RESULTS/budget_*/)"
    exit 1
fi
echo "Seeds found: $(ls -d "$RESULTS"/seed_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

# ── Table I + supplementary/audit tables + significance + LOTO ───────────────
for H in $HOSPITALS; do
  echo ""; echo "[1/6] aggregate_results  hospital=$H  (Table I, supp, arch, cost, significance, LOTO)"
  python analyze/aggregate_results.py \
    --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT" \
    --mas_budget 20 --baseline_budget 30
done

# ── Fig 1 / Fig 2 / Fig 5 (per hospital) ────────────────────────────────────
for H in $HOSPITALS; do
  echo ""; echo "[2/6] plot_search_trajectory  hospital=$H  (Fig 1)"
  python analyze/plot_search_trajectory.py \
    --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"

  echo ""; echo "[3/6] plot_pareto  hospital=$H  (Fig 2)"
  python analyze/plot_pareto.py \
    --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"

  echo ""; echo "[4/6] plot_loto_ablation  hospital=$H  (Fig 5)"
  python analyze/plot_loto_ablation.py \
    --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
done

# ── Fig 6 (cross-hospital, one call) ────────────────────────────────────────
echo ""; echo "[5/6] plot_source_hospital_choice  (Fig 6)"
python analyze/plot_source_hospital_choice.py \
  --results_root "$RESULTS" --target_hospitals $HOSPITALS --out_dir "$OUT"

# ── Fig 4 (NEW): post-hoc budget simulation — the boundary evidence ──────────
# Re-applies tighter caps to the SAME 256 traces (same supernet, 5 seeds, zero
# confound, zero compute). Shows the lead does NOT grow as the budget tightens.
echo ""; echo "[6/6] plot_lead_vs_budget_posthoc  (Fig 4 — boundary analysis)"
python analyze/plot_lead_vs_budget_posthoc.py \
  --results_root "$RESULTS" --hospitals $HOSPITALS \
  --caps 4000000 3000000 2000000 1000000 --out_dir "$OUT"

# ── Table II: anytime 5/10/20/30 (needs the existing re-test outputs) ────────
echo ""
if [[ -d "$ANYTIME/retest" ]]; then
  echo "[anytime] found $ANYTIME/retest → rebuilding Table II + its significance"
  python analyze/build_anytime_table.py \
    --results_root "$RESULTS" --anytime_dir "$ANYTIME" \
    --hospitals $HOSPITALS --out_dir "$OUT"
  python analyze/build_anytime_significance.py \
    --results_root "$RESULTS" --anytime_dir "$ANYTIME" \
    --hospitals $HOSPITALS --out_dir "$OUT"
else
  echo "⚠️  [anytime] $ANYTIME/retest NOT found → Table II SKIPPED."
  echo "    The anytime table needs the test re-test outputs (GPU re-finetuning),"
  echo "    it cannot be rebuilt from search traces alone. Either:"
  echo "      (a) point ANYTIME_DIR=<dir> at the old anytime outputs and re-run, or"
  echo "      (b) carry over the archived Results/256_version/2026-06-29/main_table_anytime_*.csv"
  echo "          and anytime_significance_*.csv."
fi

# ── Fig 3: intentionally NOT regenerated ────────────────────────────────────
echo ""
echo "⚠️  [Fig 3] regression figure intentionally NOT regenerated."
echo "    The 256-grid regression raw data under results/<H>/regression/ was"
echo "    OVERWRITTEN by the 1008-grid retrain, so regenerating it now would give"
echo "    a 1008-supernet figure that does NOT match this 256 main table."
echo "    → Copy the archived 256 version into $OUT :"
echo "      $ARCHIVE_HINT"

echo ""
echo "============================================================"
echo "✅ Final analysis written to: $OUT"
echo "============================================================"
ls -lh "$OUT"
