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

# ── Prior-knowledge forest plots (Layer-2 prior; replaces the old retrieval figure)
# Combined grid + one per-task figure. Tasks with no level_effects.csv in any
# PRIOR_ROOT are skipped, so this runs clean before med_rec exists and fills in
# the med_rec row/figure automatically once it does. `|| true` keeps a missing
# task from tripping the run.
echo "── Prior knowledge: combined grid + per-task"
python analyze/plot_prior_knowledge.py --combined --prior_root $PRIOR_ROOTS --out_dir "$OUT/prior_knowledge" || true
for T in death stay readmission next_diag_6m_pheno next_diag_12m_pheno med_rec; do
  python analyze/plot_prior_knowledge.py --task "$T" --prior_root $PRIOR_ROOTS --out_dir "$OUT/prior_knowledge" || true
done

# ── Table II: anytime 5/10/20/30 (needs the pre-existing GPU re-test outputs)
if [[ -d "$ANYTIME/retest" ]]; then
  echo "── Table II: anytime (from $ANYTIME/retest)"
  python analyze/build_anytime_table.py        --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT"
  python analyze/build_anytime_significance.py  --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT"
else
  echo "⚠️  Table II SKIPPED — $ANYTIME/retest not found (needs GPU re-test; or carry over the archived main_table_anytime_*.csv)."
fi

# ── med_rec (drug recommendation) — isolated in results_medrec, run through the
# SAME scripts with --tasks med_rec, output to OUT/med_rec so it sits parallel to
# the 5-task results. Skipped cleanly if the med_rec search hasn't run yet.
MEDREC=$PROJECT/results_medrec
if compgen -G "$MEDREC/seed_*" > /dev/null; then
  echo "── med_rec: full analysis (parallel to the 5 tasks) → $OUT/med_rec"
  mkdir -p "$OUT/med_rec"
  for H in $HOSPITALS; do
    python analyze/aggregate_results.py     --results_root "$MEDREC" --hospitals "$H" --tasks med_rec --out_dir "$OUT/med_rec" --mas_budget 20 --baseline_budget 30
    python analyze/plot_search_trajectory.py --results_root "$MEDREC" --hospitals "$H" --tasks med_rec --out_dir "$OUT/med_rec"
    python analyze/plot_pareto.py            --results_root "$MEDREC" --hospitals "$H" --tasks med_rec --out_dir "$OUT/med_rec"
    python analyze/plot_loto_ablation.py     --results_root "$MEDREC" --hospitals "$H" --tasks med_rec --out_dir "$OUT/med_rec"
  done
  if [[ -f "$MEDREC/anytime/anytime_selection_map.csv" ]]; then
    python analyze/build_anytime_table.py        --results_root "$MEDREC" --anytime_dir "$MEDREC/anytime" --hospitals $HOSPITALS --tasks med_rec --out_dir "$OUT/med_rec"
    python analyze/build_anytime_significance.py  --results_root "$MEDREC" --anytime_dir "$MEDREC/anytime" --hospitals $HOSPITALS --tasks med_rec --out_dir "$OUT/med_rec"
  else
    echo "  (med_rec anytime skipped — run extract_anytime_jobs + submit_retest_medrec + build_anytime_table first)"
  fi
else
  echo "── med_rec: no search results under $MEDREC — skipped"
fi

# ── Fig 3: carry over the archived 256 PNG (raw data overwritten by the 1008 retrain)
echo "⚠️  Fig 3 NOT regenerated — copy the archived 256 version into $OUT:"
echo "     Results/256_version/2026-06-26/figure3_regression_combined.png"
echo "     (its readmission panel still reads '90d'; resolve as '3 months (90 days)' in the caption)."

echo "✅ Done → $OUT"
ls -lh "$OUT"
