#!/bin/bash
# =============================================================================
# ONE-SHOT FINAL ANALYSIS — 256-grid MAIN experiment → analyze/<date>_final/
# =============================================================================
# Regenerates the paper's tables and figures into one dated folder from the
# 256-grid results at <PROJECT>/results/seed_*/<hospital>/search/<method>/<task>/.
# CPU-only; login node is fine.
#     module load conda && conda activate Autoformer
#     bash slurm/run_analysis_final.sh
#
# NOT regenerated here (see the notes at the end):
#   - Fig 3 (regression): 256 raw data was overwritten by the 1008 retrain; carry
#     the archived PNG over instead.
#   - Table II (anytime): needs the GPU re-test outputs; skipped if absent.
# =============================================================================
set -uo pipefail   # NOT -e: one failure shouldn't kill the whole run

PROJECT=/blue/mei.liu/lideyi/MAS-NAS
RESULTS=$PROJECT/results                             # 256-grid seed_* live HERE
OUT=$PROJECT/analyze/$(date +%Y-%m-%d)_final
ANYTIME=${ANYTIME_DIR:-$PROJECT/analyze/anytime}
HOSPITALS="source_15 MIMIC-IV"
# Layer-2 prior forest plots: 5 tasks from the 256 backup + med_rec (skipped if absent).
PRIOR_ROOTS="$PROJECT/results_orig_256grid/meta_regression $PROJECT/results_medrec/meta_regression"

mkdir -p "$OUT"
cd /home/lideyi/MAS-NAS || exit 1

echo "FINAL ANALYSIS → $OUT   (hospitals: $HOSPITALS)"
if ! compgen -G "$RESULTS/seed_*" > /dev/null; then
    echo "❌ No seed_* under $RESULTS (the 1008 runs live in $RESULTS/budget_*/)"; exit 1
fi
echo "Seeds: $(ls -d "$RESULTS"/seed_* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

# ── Tables (Table I, supp, arch, cost, significance, LOTO) + per-hospital figures
for H in $HOSPITALS; do
  echo "── $H: tables + Fig 1/2/5"
  python analyze/aggregate_results.py   --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT" --mas_budget 20 --baseline_budget 30
  python analyze/plot_search_trajectory.py --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
  python analyze/plot_pareto.py            --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
  python analyze/plot_loto_ablation.py     --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT"
done

# ── Fig 4: post-hoc budget simulation (cross-hospital, one call)
echo "── Fig 4: lead vs budget"
python analyze/plot_lead_vs_budget_posthoc.py --results_root "$RESULTS" --hospitals $HOSPITALS \
  --caps 4000000 3000000 2000000 1000000 --out_dir "$OUT"

# ── Prior-knowledge forest grid (Layer-2 prior; replaces the old retrieval figure)
echo "── Prior knowledge: combined grid"
python analyze/plot_prior_knowledge.py --combined --prior_root $PRIOR_ROOTS --out_dir "$OUT/prior_knowledge"

# ── Table II: anytime 5/10/20/30 (needs the pre-existing GPU re-test outputs)
if [[ -d "$ANYTIME/retest" ]]; then
  echo "── Table II: anytime (from $ANYTIME/retest)"
  python analyze/build_anytime_table.py        --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT"
  python analyze/build_anytime_significance.py  --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT"
else
  echo "⚠️  Table II SKIPPED — $ANYTIME/retest not found (needs GPU re-test; or carry over the archived main_table_anytime_*.csv)."
fi

# ── Fig 3: carry over the archived 256 PNG (raw data overwritten by the 1008 retrain)
echo "⚠️  Fig 3 NOT regenerated — copy the archived 256 version into $OUT:"
echo "     Results/256_version/2026-06-26/figure3_regression_combined.png"
echo "     (its readmission panel still reads '90d'; resolve as '3 months (90 days)' in the caption)."

echo "✅ Done → $OUT"
ls -lh "$OUT"
