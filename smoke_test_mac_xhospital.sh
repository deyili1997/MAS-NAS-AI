#!/bin/bash
# ============================================================================
# Cross-Hospital Production-Like Smoke Test (Mac local)
# ============================================================================
# Target = MIMIC-III   (test hospital)
# Source = MIMIC-IV    (prior — already prepared from prior smoke runs)
#
# Validates the full Phase-2-style pipeline end-to-end in a single seed at
# smoke scale, including all figures + tables expected for the paper.
#
# Pipeline stages (no skipping):
#   [Stage 0] run_pipeline.py for MIMIC-III  (5 archs × 5 tasks × 3 ep)
#   [Stage A] shap_analysis + run_meta_regression on MIMIC-IV (the prior source)
#   [Stage B] mas_search.py × 4 modes × 5 tasks on MIMIC-III (target)
#               modes: mas, mas_layer1_only, mas_loto, mas_cold (4-cond factorial)
#   [Stage C] baselines 0-4 × 5 tasks on MIMIC-III
#   [Stage D] run_regression.py × 5 tasks on MIMIC-III (Fig 3 dual-pipeline)
#   [Stage E] aggregate + plot all figures + tables
#               outputs to analyze/ : main_table, efficiency_table, supp_table,
#               significance, arch_table, cost_table, loto_ablation_table,
#               figure1_search_trajectory.png, figure2_pareto.png,
#               figure5_loto_robustness.png
#
# Estimated time on Mac MPS (M-series):
#   Stage 0: ~85 min   (pretrain 3 ep + 5 tasks × 5 archs × 3 ep finetune)
#   Stage A: ~1 min
#   Stage B: ~100 min  (4 modes × 5 tasks × budget=2)
#   Stage C: ~75 min   (5 baselines × 5 tasks × budget=2)
#   Stage D: ~50 min   (5 tasks × k=3 dual-pipeline)
#   Stage E: ~2 min
#   TOTAL:   ~5 hours  (overnight-feasible)
#
# Self-match exclusion (mas_search.py:185) ensures cross-hospital semantics:
# even though MIMIC-III's metadata.csv is in results/, the cosine matcher
# excludes target hospital from candidates → falls back to MIMIC-IV.
#
# Usage:
#   bash smoke_test_mac_xhospital.sh                  # FULL ~5 hours
#   bash smoke_test_mac_xhospital.sh --no-pipeline    # Skip Stage 0
#   bash smoke_test_mac_xhospital.sh --pipeline-only  # Only Stage 0
#   bash smoke_test_mac_xhospital.sh --mas-only       # Only A+B
#   bash smoke_test_mac_xhospital.sh --baselines-only # Only Stage C
#   bash smoke_test_mac_xhospital.sh --regression-only # Only Stage D
#   bash smoke_test_mac_xhospital.sh --analyze-only   # Only Stage E (assumes B/C/D ran)
#   bash smoke_test_mac_xhospital.sh --clean          # Remove all xhospital artifacts
#
# What --clean removes:
#   results/MIMIC-III/  (full)
#   results/shap_analysis/, results/meta_regression/  (regenerated)
#   results/seed_${SMOKE_SEED}/  (mas + baseline runs)
#   analyze/*.csv, analyze/figure*.png  (Stage E outputs)
# What --clean PRESERVES:
#   results/MIMIC-IV/  (prior source, must stay warm)
# ============================================================================

set -e

PROJECT_ROOT="/Users/deyili/Documents_Local/BMI_PhD@UF/Projects/AgentCare/MAS-NAS-Working/MAS-NAS"
PY="/Users/deyili/anaconda3/envs/agent/bin/python"

# ─── Configuration ───────────────────────────────────────────────────────
TARGET_HOSPITAL="MIMIC-III"
SOURCE_HOSPITAL="MIMIC-IV"
SMOKE_SEED=999
SMOKE_TASKS=("death" "stay" "readmission" "next_diag_6m_pheno" "next_diag_12m_pheno")

# Smoke-scale params (3 ep, 1 patience for both pretrain + finetune)
SMOKE_PRETRAIN_EPOCHS=3
SMOKE_PRETRAIN_PATIENCE=1
SMOKE_FINETUNE_EPOCHS=3
SMOKE_FINETUNE_PATIENCE=1
SMOKE_NUM_ARCHS=5         # Stage 0 K
SMOKE_BUDGET=2            # Stage B/C budget
SMOKE_REGRESSION_K=3      # Stage D k

# Paths
RESULTS_ROOT="${PROJECT_ROOT}/results"
SEED_OUTPUT="${RESULTS_ROOT}/seed_${SMOKE_SEED}"
ANALYZE_DIR="${PROJECT_ROOT}/analyze"

# ─── CLI parsing ─────────────────────────────────────────────────────────
RUN_PIPELINE=true
RUN_STAGE_A=true
RUN_MAS=true
RUN_BASELINES=true
RUN_REGRESSION=true
RUN_ANALYZE=true
CLEAN_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --no-pipeline)      RUN_PIPELINE=false ;;
        --pipeline-only)    RUN_STAGE_A=false; RUN_MAS=false; RUN_BASELINES=false; RUN_REGRESSION=false; RUN_ANALYZE=false ;;
        --mas-only)         RUN_PIPELINE=false; RUN_BASELINES=false; RUN_REGRESSION=false; RUN_ANALYZE=false ;;
        --baselines-only)   RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_REGRESSION=false; RUN_ANALYZE=false ;;
        --regression-only)  RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_BASELINES=false; RUN_ANALYZE=false ;;
        --analyze-only)     RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_BASELINES=false; RUN_REGRESSION=false ;;
        --clean)            CLEAN_ONLY=true ;;
        --help|-h)          grep '^#' "$0" | head -55; exit 0 ;;
    esac
done

# ─── Colors / helpers ────────────────────────────────────────────────────
BOLD=$'\e[1m'; BLUE=$'\e[0;34m'; YELLOW=$'\e[1;33m'
GREEN=$'\e[0;32m'; RED=$'\e[0;31m'; NC=$'\e[0m'
section() { echo -e "\n${BOLD}${BLUE}===================================================================="
            echo -e "$1"
            echo -e "====================================================================${NC}"; }
step()    { echo -e "${BOLD}${YELLOW}\n→ $1${NC}"; }
ok()      { echo -e "${GREEN}  ✓ $1${NC}"; }
fail()    { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
warn()    { echo -e "${YELLOW}  ⚠ $1${NC}"; }
check_file() { [ -f "$1" ] && ok "exists: $1" || fail "missing: $1"; }

cd "${PROJECT_ROOT}" || fail "cannot cd to ${PROJECT_ROOT}"

# ─── --clean ─────────────────────────────────────────────────────────────
if [ "$CLEAN_ONLY" = true ]; then
    section "CLEAN xhospital smoke artifacts (preserves MIMIC-IV prior)"
    rm -rf "${RESULTS_ROOT}/${TARGET_HOSPITAL}"
    ok "removed results/${TARGET_HOSPITAL}/"
    rm -rf "${RESULTS_ROOT}/shap_analysis" "${RESULTS_ROOT}/meta_regression"
    ok "removed results/shap_analysis/, results/meta_regression/"
    rm -rf "${SEED_OUTPUT}"
    ok "removed results/seed_${SMOKE_SEED}/"
    # Stage E artifacts (only the per-hospital ones for this target)
    rm -f "${ANALYZE_DIR}"/{main_table,efficiency_table,supp_table,significance,arch_table,cost_table,loto_ablation_table}.csv
    rm -f "${ANALYZE_DIR}"/figure{1_search_trajectory,2_pareto,5_loto_robustness}.png
    ok "removed analyze/*.csv, analyze/figure*.png"
    ok "preserved results/${SOURCE_HOSPITAL}/ (prior source)"
    exit 0
fi

# ─── Prerequisite checks ─────────────────────────────────────────────────
section "PREREQUISITES"

step "Python interpreter"
[ -x "$PY" ] && ok "$PY" || fail "missing: $PY"
$PY --version

step "Critical deps"
$PY -c "
import sys
need = ['xgboost', 'shap', 'matplotlib', 'statsmodels', 'pandas', 'numpy', 'sklearn',
        'torch', 'anthropic', 'dotenv', 'scipy']
missing = [p for p in need if __import__('importlib').util.find_spec(p) is None]
if missing:
    print(f'  MISSING: {missing}'); sys.exit(1)
print(f'  ✓ all {len(need)} deps present')
import torch
print(f'  ✓ torch backend: cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}')
"

step "Source data — MIMIC-III (target) + MIMIC-IV (source)"
for HOSP in "${TARGET_HOSPITAL}" "${SOURCE_HOSPITAL}"; do
    for f in mimic.pkl mimic_pretrain.pkl mimic_downstream.pkl mimic_nextdiag_6m.pkl mimic_nextdiag_12m.pkl; do
        check_file "data_process/${HOSP}/${HOSP}-processed/${f}"
    done
done

step ".env file"
check_file ".env"
grep -q "ANTHROPIC_API_KEY" .env && ok ".env contains ANTHROPIC_API_KEY" || fail "ANTHROPIC_API_KEY missing"

step "Source prior (${SOURCE_HOSPITAL}) must already be warm"
check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv"
check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/dataset_summary.csv"
check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/checkpoint_mlm/mlm_model.pt"
n_rows=$($PY -c "import pandas as pd; print(len(pd.read_csv('${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv')))")
n_tasks=$($PY -c "import pandas as pd; print(pd.read_csv('${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv')['task'].nunique())")
ok "${SOURCE_HOSPITAL} metadata.csv: ${n_rows} rows × ${n_tasks} tasks"

# ============================================================================
# STAGE 0 — run_pipeline.py for MIMIC-III (target)
# ============================================================================
if [ "$RUN_PIPELINE" = true ]; then
    section "STAGE 0 — run_pipeline.py for ${TARGET_HOSPITAL} (~85 min)"

    # Stage 0 has TWO scripts (per plan A1 + A2 design):
    #   0.0  dataset_summary.py  → <hospital>/dataset_summary.csv  (used by Layer 1 cosine)
    #   0.1  run_pipeline.py     → <hospital>/{metadata.csv, checkpoint_mlm/mlm_model.pt}
    step "0.0  dataset_summary.py --hospital ${TARGET_HOSPITAL}  (~1 min CPU)"
    $PY dataset_summary.py --hospital ${TARGET_HOSPITAL} \
        --output_dir "${RESULTS_ROOT}"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/dataset_summary.csv"

    step "0.1  pretrain + finetune ${SMOKE_NUM_ARCHS} archs × ${#SMOKE_TASKS[@]} tasks"
    $PY run_pipeline.py \
        --hospital ${TARGET_HOSPITAL} \
        --num_archs ${SMOKE_NUM_ARCHS} \
        --tasks ${SMOKE_TASKS[@]} \
        --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
        --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
        --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
        --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
        --num_workers 0 \
        --output_dir "${RESULTS_ROOT}"

    step "0.2  verify outputs"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/metadata.csv"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/dataset_summary.csv"
    n_rows=$($PY -c "import pandas as pd; print(len(pd.read_csv('${RESULTS_ROOT}/${TARGET_HOSPITAL}/metadata.csv')))")
    expected=$((SMOKE_NUM_ARCHS * ${#SMOKE_TASKS[@]}))
    [ "$n_rows" = "$expected" ] && ok "metadata: $n_rows rows (expected $expected = ${SMOKE_NUM_ARCHS} archs × ${#SMOKE_TASKS[@]} tasks)" \
        || warn "metadata rows: $n_rows (expected $expected)"

    ok "Stage 0 complete"
fi

# ============================================================================
# STAGE A — shap_analysis + run_meta_regression on SOURCE prior (MIMIC-IV)
# ============================================================================
if [ "$RUN_STAGE_A" = true ]; then
    section "STAGE A — Layer 2 prior generation from ${SOURCE_HOSPITAL} (~1 min)"

    step "A.1  shap_analysis.py --hospitals ${SOURCE_HOSPITAL}"
    $PY shap_analysis.py --hospitals ${SOURCE_HOSPITAL} \
        --results_dir "${RESULTS_ROOT}" \
        --interaction_main embed_dim depth num_heads mlp_ratio

    for task in "${SMOKE_TASKS[@]}"; do
        check_file "${RESULTS_ROOT}/shap_analysis/${task}/shap_summary.png"
        check_file "${RESULTS_ROOT}/shap_analysis/${task}/shap_values.csv"
    done

    step "A.2  run_meta_regression.py (Layer 2 architecture_prior.json)"
    $PY run_meta_regression.py \
        --shap_dir "${RESULTS_ROOT}/shap_analysis" \
        --output_dir "${RESULTS_ROOT}/meta_regression"

    for task in "${SMOKE_TASKS[@]}"; do
        check_file "${RESULTS_ROOT}/meta_regression/${task}/architecture_prior.json"
    done

    ok "Stage A complete (Layer 2 prior built from ${SOURCE_HOSPITAL})"
fi

# ============================================================================
# STAGE B — mas_search × 4 modes × 5 tasks on ${TARGET_HOSPITAL}
# ============================================================================
if [ "$RUN_MAS" = true ]; then
    section "STAGE B — mas_search × 4 modes × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~100 min)"

    # Modes and their flag triple: name | extra_flag(s)
    declare -a MODES=("mas:" "mas_layer1_only:--no_meta_regression" \
                      "mas_loto:--exclude_exact_task_from_history" "mas_cold:--no_history")

    total_runs=$((4 * ${#SMOKE_TASKS[@]}))
    counter=0
    for mode_spec in "${MODES[@]}"; do
        mode="${mode_spec%%:*}"
        flag="${mode_spec#*:}"
        for task in "${SMOKE_TASKS[@]}"; do
            counter=$((counter + 1))
            step "B.${counter}/${total_runs}  mas_search ${mode} on ${task}"
            $PY mas_search.py \
                --hospital ${TARGET_HOSPITAL} --task ${task} --max_params 2000000 \
                --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers 0 \
                --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
                --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
                --top_k_epochs 1 \
                ${flag} \
                --results_dir "${SEED_OUTPUT}" \
                --history_root "${RESULTS_ROOT}"

            DIR="${SEED_OUTPUT}/${TARGET_HOSPITAL}/search/${mode}/${task}"
            check_file "${DIR}/${mode}_search.csv"
            check_file "${DIR}/${mode}_best.csv"
            check_file "${DIR}/agent_io_log.txt"
            METHOD=$($PY -c "import json; print(json.load(open('${DIR}/search_meta.json'))['method'])")
            [ "$METHOD" = "$mode" ] && ok "method=${mode}" || fail "method mismatch: $METHOD"
        done
    done

    ok "Stage B complete (4 modes × ${#SMOKE_TASKS[@]} tasks = $total_runs runs)"
fi

# ============================================================================
# STAGE C — Baselines 0-4 × 5 tasks on ${TARGET_HOSPITAL}
# ============================================================================
if [ "$RUN_BASELINES" = true ]; then
    section "STAGE C — Baselines 0-4 × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~75 min)"

    CKPT_PATH="${RESULTS_ROOT}/${TARGET_HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    [ -f "${CKPT_PATH}" ] || fail "Pretrain ckpt missing: ${CKPT_PATH}. Run Stage 0 first."
    ok "Reusing warm ckpt: ${CKPT_PATH}"

    BASELINE_NAMES=("Random" "EA" "LLM-1shot" "LLMatic" "CoLLM-NAS")
    total_runs=$((5 * ${#SMOKE_TASKS[@]}))
    counter=0
    for i in 0 1 2 3 4; do
        bname=${BASELINE_NAMES[$i]}
        for task in "${SMOKE_TASKS[@]}"; do
            counter=$((counter + 1))
            step "C.${counter}/${total_runs}  baseline${i} (${bname}) on ${task}"
            $PY baselines/baseline${i}.py \
                --hospital ${TARGET_HOSPITAL} --task ${task} --max_params 2000000 \
                --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers 0 \
                --ckpt_path "${CKPT_PATH}" \
                --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
                --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
                --top_k_epochs 1 \
                --results_dir "${SEED_OUTPUT}"

            DIR="${SEED_OUTPUT}/${TARGET_HOSPITAL}/search/baseline${i}/${task}"
            check_file "${DIR}/baseline${i}_search.csv"
            check_file "${DIR}/baseline${i}_best.csv"
        done
    done

    ok "Stage C complete (5 baselines × ${#SMOKE_TASKS[@]} tasks = $total_runs runs)"
fi

# ============================================================================
# STAGE D — run_regression × 5 tasks on ${TARGET_HOSPITAL}
# ============================================================================
if [ "$RUN_REGRESSION" = true ]; then
    section "STAGE D — run_regression × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~50 min)"

    for task in "${SMOKE_TASKS[@]}"; do
        step "D.  run_regression --k ${SMOKE_REGRESSION_K} --task ${task}"
        $PY run_regression.py \
            --hospital ${TARGET_HOSPITAL} --task ${task} --k ${SMOKE_REGRESSION_K} \
            --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
            --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --top_k_epochs 1 \
            --seed ${SMOKE_SEED} --num_workers 0 \
            --results_dir "${RESULTS_ROOT}"

        check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/regression/${task}/regression_panel.png"
        ok "Fig 3 panel for ${task} generated"
    done

    ok "Stage D complete (${#SMOKE_TASKS[@]} regression panels = Fig 3 inputs)"
fi

# ============================================================================
# STAGE E — Aggregate + plot (Fig 1, 2, 5 + all CSV tables)
# ============================================================================
if [ "$RUN_ANALYZE" = true ]; then
    section "STAGE E — Aggregate results + generate figures/tables (~2 min)"

    step "E.1  aggregate_results.py — main_table, efficiency, supp, sig, arch, cost, loto_ablation"
    $PY analyze/aggregate_results.py \
        --results_root "${RESULTS_ROOT}" \
        --hospital ${TARGET_HOSPITAL} \
        --out_dir "${ANALYZE_DIR}" \
        --mas_budget ${SMOKE_BUDGET} \
        --baseline_budget ${SMOKE_BUDGET}

    for csv in main_table efficiency_table supp_table significance arch_table cost_table loto_ablation_table; do
        check_file "${ANALYZE_DIR}/${csv}.csv"
    done

    step "E.2  plot_search_trajectory.py — Fig 1"
    $PY analyze/plot_search_trajectory.py \
        --results_root "${RESULTS_ROOT}" \
        --hospital ${TARGET_HOSPITAL} \
        --out_dir "${ANALYZE_DIR}"
    check_file "${ANALYZE_DIR}/figure1_search_trajectory.png"

    step "E.3  plot_pareto.py — Fig 2"
    $PY analyze/plot_pareto.py \
        --results_root "${RESULTS_ROOT}" \
        --hospital ${TARGET_HOSPITAL} \
        --out_dir "${ANALYZE_DIR}"
    check_file "${ANALYZE_DIR}/figure2_pareto.png"

    step "E.4  plot_loto_ablation.py — Fig 5 (4-condition factorial)"
    $PY analyze/plot_loto_ablation.py \
        --results_root "${RESULTS_ROOT}" \
        --hospital ${TARGET_HOSPITAL} \
        --out_dir "${ANALYZE_DIR}"
    check_file "${ANALYZE_DIR}/figure5_loto_robustness.png"

    ok "Stage E complete — all figures + tables generated"
fi

# ============================================================================
# FINAL REPORT
# ============================================================================
section "XHOSPITAL SMOKE TEST PASSED ✓"
echo ""
echo "Target = ${TARGET_HOSPITAL}, Source = ${SOURCE_HOSPITAL}, Seed = ${SMOKE_SEED}"
echo ""
echo "─── Stage A artifacts ───"
[ -d "${RESULTS_ROOT}/shap_analysis" ] && \
    echo "  SHAP PNGs:                $(ls ${RESULTS_ROOT}/shap_analysis/*/*.png 2>/dev/null | wc -l | xargs)"
[ -d "${RESULTS_ROOT}/meta_regression" ] && \
    echo "  architecture_prior.json:  $(ls ${RESULTS_ROOT}/meta_regression/*/architecture_prior.json 2>/dev/null | wc -l | xargs)"

echo ""
echo "─── Stage B/C runs (seed_${SMOKE_SEED}) ───"
if [ -d "${SEED_OUTPUT}/${TARGET_HOSPITAL}/search" ]; then
    for method in mas mas_layer1_only mas_loto mas_cold baseline0 baseline1 baseline2 baseline3 baseline4; do
        n=$(ls -d "${SEED_OUTPUT}/${TARGET_HOSPITAL}/search/${method}"/* 2>/dev/null | wc -l | xargs)
        [ "$n" -gt 0 ] && printf "  %-20s %s tasks\n" "${method}:" "${n}"
    done
fi

echo ""
echo "─── Stage D regression panels (Fig 3) ───"
ls -1 "${RESULTS_ROOT}/${TARGET_HOSPITAL}/regression/"*/regression_panel.png 2>/dev/null | \
    while read p; do echo "  $p"; done

echo ""
echo "─── Stage E figures + tables (paper outputs) ───"
ls -1 "${ANALYZE_DIR}"/*.csv "${ANALYZE_DIR}"/figure*.png 2>/dev/null | \
    while read p; do echo "  $p"; done

echo ""
echo "Layer 2 SHAP figures (Fig 4) are at:"
echo "  ${RESULTS_ROOT}/shap_analysis/<task>/shap_*.png"
echo ""
echo "To clean (preserves ${SOURCE_HOSPITAL}):  bash $0 --clean"
