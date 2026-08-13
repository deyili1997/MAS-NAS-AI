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
RESULTS=$PROJECT/results                             # 256-grid seed_* live HERE (5 tasks)
MEDREC=$PROJECT/results_medrec                       # isolated drug-recommendation search
# All six tasks, in panel order. The figures read BOTH roots so drug rec appears
# as a 6th panel alongside the 5 tasks.
ALLTASKS="death stay readmission next_diag_6m_pheno next_diag_12m_pheno med_rec"
# Competitor methods included in ALL figures + tables. baseline3 (LLMatic) is
# EXCLUDED by default; the one knob to include it everywhere in a single run is:
#   METHODS="baseline0 baseline1 baseline2 baseline3 baseline4 mas" bash slurm/run_analysis_final.sh
# Keep canonical order and keep "mas". (The anytime tables are data-driven and
# pick up baseline3 automatically once its anytime jobs exist.)
METHODS=${METHODS:-"baseline0 baseline1 baseline2 baseline4 mas"}
# Output folder date tag. Defaults to today; override to consolidate into an
# existing final folder, e.g. FINAL_DATE=2026-07-15 bash slurm/run_analysis_final.sh
OUT=$PROJECT/analyze/${FINAL_DATE:-$(date +%Y-%m-%d)}_final
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

# ── Tables (5-task, from results/) + 6-task figures (results/ + results_medrec/)
# Figures read BOTH roots and all six tasks so drug rec is a 6th panel. The
# med_rec-only tables come from the dedicated med_rec pass below.
MEDREC_ROOT_ARG=""; compgen -G "$MEDREC/seed_*" > /dev/null && MEDREC_ROOT_ARG="$MEDREC"
for H in $HOSPITALS; do
  echo "── $H: tables (5-task) + Fig 1/2/5 (6-task incl. drug rec)   methods=[$METHODS]"
  python analyze/aggregate_results.py   --results_root "$RESULTS" --hospitals "$H" --out_dir "$OUT" --mas_budget 20 --baseline_budget 30 --methods $METHODS
  python analyze/plot_search_trajectory.py --results_root "$RESULTS" $MEDREC_ROOT_ARG --hospitals "$H" --tasks $ALLTASKS --out_dir "$OUT" --methods $METHODS
  python analyze/plot_pareto.py            --results_root "$RESULTS" $MEDREC_ROOT_ARG --hospitals "$H" --tasks $ALLTASKS --out_dir "$OUT" --methods $METHODS
  python analyze/plot_loto_ablation.py     --results_root "$RESULTS" $MEDREC_ROOT_ARG --hospitals "$H" --tasks $ALLTASKS --out_dir "$OUT" --methods $METHODS
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
  python analyze/build_anytime_table.py        --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT" --methods $METHODS
  python analyze/build_anytime_significance.py  --results_root "$RESULTS" --anytime_dir "$ANYTIME" --hospitals $HOSPITALS --out_dir "$OUT" --methods $METHODS
else
  echo "⚠️  Table II SKIPPED — $ANYTIME/retest not found (needs GPU re-test; or carry over the archived main_table_anytime_*.csv)."
fi

# ── med_rec (drug recommendation) — isolated in results_medrec, run through the
# SAME aggregate scripts with --tasks med_rec, output to OUT/med_rec — the
# med_rec-only TABLES (main/supp/arch/cost/sig/loto + anytime). Figures are NOT
# regenerated here: the main Fig 1/2/5 above already include drug rec as a 6th
# panel (they read both roots). Skipped cleanly if the med_rec search is absent.
if compgen -G "$MEDREC/seed_*" > /dev/null; then
  echo "── med_rec: tables → $OUT/med_rec"
  mkdir -p "$OUT/med_rec"
  for H in $HOSPITALS; do
    python analyze/aggregate_results.py     --results_root "$MEDREC" --hospitals "$H" --tasks med_rec --out_dir "$OUT/med_rec" --mas_budget 20 --baseline_budget 30 --methods $METHODS
  done
  if [[ -f "$MEDREC/anytime/anytime_selection_map.csv" ]]; then
    python analyze/build_anytime_table.py        --results_root "$MEDREC" --anytime_dir "$MEDREC/anytime" --hospitals $HOSPITALS --tasks med_rec --out_dir "$OUT/med_rec" --methods $METHODS
    python analyze/build_anytime_significance.py  --results_root "$MEDREC" --anytime_dir "$MEDREC/anytime" --hospitals $HOSPITALS --tasks med_rec --out_dir "$OUT/med_rec" --methods $METHODS
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
