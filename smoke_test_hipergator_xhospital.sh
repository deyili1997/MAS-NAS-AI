#!/bin/bash
# ============================================================================
# Cross-Hospital Near-Production Smoke (HiPerGator)
# ============================================================================
# Target = MIMIC-III   (test hospital)
# Source = MIMIC-IV    (prior — must be pre-warmed via Mac smoke or HPC pipeline)
#
# Near-production scale (budget=10, 10 ep, 10 archs) — bigger than Mac smoke
# (budget=2, 3 ep, 5 archs) to validate behavior closer to real Phase 2 jobs.
#
# Pipeline stages (no skipping):
#   [Stage 0] dataset_summary + run_pipeline for MIMIC-III
#   [Stage A] shap_analysis + run_meta_regression on MIMIC-IV (source prior)
#   [Stage B] mas_search × 4 modes × 5 tasks on MIMIC-III (target)
#               modes: mas, mas_layer1_only, mas_loto, mas_cold
#   [Stage C] baselines 0-4 × 5 tasks on MIMIC-III
#   [Stage D] run_regression × 5 tasks on MIMIC-III (Fig 3)
#   [Stage E] aggregate + plot all figures + tables
#
# Estimated wall-time on HiPerGator A100 (single GPU):
#   Stage 0: ~70 min   (pretrain 10 ep + 10 archs × 5 tasks × 10 ep finetune)
#   Stage A: ~2 min
#   Stage B: ~3-4 h    (4 modes × 5 tasks × budget=10 + 10 ep finetune)
#   Stage C: ~2-3 h    (5 baselines × 5 tasks × budget=10)
#   Stage D: ~75 min   (5 tasks × k=3 dual-pipeline)
#   Stage E: ~3 min
#   TOTAL:   ~8-10 hours
#
# Submission:
#   sbatch smoke_test_hipergator_xhospital.sbatch
#   # OR run interactively on srun --pty session:
#   bash smoke_test_hipergator_xhospital.sh
#
# CLI flags (same as Mac version):
#   --no-pipeline / --pipeline-only / --mas-only / --baselines-only
#   --regression-only / --analyze-only / --clean
# ============================================================================

set -e

# ─── Configuration ───────────────────────────────────────────────────────
PROJECT_ROOT="/home/lideyi/MAS-NAS"
PY=$(which python)   # Resolved AFTER conda env activation in sbatch wrapper
# If running this .sh directly (not via sbatch), activate manually first:
#   module load conda && conda activate Autoformer

TARGET_HOSPITAL="MIMIC-III"
SOURCE_HOSPITAL="MIMIC-IV"
SMOKE_SEED=999
SMOKE_TASKS=("death" "stay" "readmission" "next_diag_6m_pheno" "next_diag_12m_pheno")

# Near-production scale (vs Mac smoke at 3/3/5/2/3)
SMOKE_PRETRAIN_EPOCHS=10
SMOKE_PRETRAIN_PATIENCE=2
SMOKE_FINETUNE_EPOCHS=10
SMOKE_FINETUNE_PATIENCE=2
SMOKE_NUM_ARCHS=10        # Stage 0 K (vs production K=100)
SMOKE_BUDGET=10           # Stage B/C budget (vs production budget=100)
SMOKE_REGRESSION_K=3      # Stage D k (regression is expensive — keep modest)

# Paths — outputs go to /blue (lab shared, large quota), reads from project /home
RESULTS_ROOT="/blue/mei.liu/lideyi/MAS-NAS/results"
SEED_OUTPUT="${RESULTS_ROOT}/seed_${SMOKE_SEED}"
ANALYZE_DIR="${PROJECT_ROOT}/analyze"   # CSVs/PNGs stay small → /home is fine

# CPU workers: HPC A100 node has plenty; bump from Mac's 0
NUM_WORKERS=4

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
        --help|-h)          grep '^#' "$0" | head -50; exit 0 ;;
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
    section "CLEAN xhospital smoke artifacts (preserves ${SOURCE_HOSPITAL} prior)"
    rm -rf "${RESULTS_ROOT}/${TARGET_HOSPITAL}"
    ok "removed ${RESULTS_ROOT}/${TARGET_HOSPITAL}/"
    rm -rf "${RESULTS_ROOT}/shap_analysis" "${RESULTS_ROOT}/meta_regression"
    ok "removed shap_analysis/, meta_regression/"
    rm -rf "${SEED_OUTPUT}"
    ok "removed seed_${SMOKE_SEED}/"
    rm -f "${ANALYZE_DIR}"/{main_table,efficiency_table,supp_table,significance,arch_table,cost_table,loto_ablation_table}.csv
    rm -f "${ANALYZE_DIR}"/figure1_search_trajectory_*.png \
          "${ANALYZE_DIR}"/figure2_pareto.png \
          "${ANALYZE_DIR}"/figure5_loto_robustness.png
    ok "removed analyze/*.csv, analyze/figure*.png"
    ok "preserved ${RESULTS_ROOT}/${SOURCE_HOSPITAL}/ (prior source)"
    exit 0
fi

# ─── Prerequisite checks ─────────────────────────────────────────────────
section "PREREQUISITES (HiPerGator)"

step "Python interpreter"
[ -x "$PY" ] && ok "$PY" || fail "missing python. Did you 'conda activate Autoformer' first?"
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
print(f'  ✓ torch backend: cuda={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available! Smoke needs GPU.'
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

step "Source prior (${SOURCE_HOSPITAL}) must already be warm at ${RESULTS_ROOT}"
if [ ! -f "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv" ] || \
   [ ! -f "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/dataset_summary.csv" ]; then
    warn "Source prior missing at ${RESULTS_ROOT}/${SOURCE_HOSPITAL}/"
    warn "Will pre-generate MIMIC-IV prior first (Stage 0 will run for BOTH hospitals)."
    NEED_SOURCE_PIPELINE=true
else
    n_rows=$($PY -c "import pandas as pd; print(len(pd.read_csv('${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv')))")
    n_tasks=$($PY -c "import pandas as pd; print(pd.read_csv('${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv')['task'].nunique())")
    ok "${SOURCE_HOSPITAL} metadata.csv: ${n_rows} rows × ${n_tasks} tasks"
    NEED_SOURCE_PIPELINE=false
fi

mkdir -p "${RESULTS_ROOT}"

# ============================================================================
# STAGE 0 — run_pipeline.py for target (and source if missing)
# ============================================================================
if [ "$RUN_PIPELINE" = true ]; then
    section "STAGE 0 — run_pipeline.py (~70 min target, +~70 min source if needed)"

    # If source prior is missing, run it FIRST (Layer 1 candidate)
    if [ "$NEED_SOURCE_PIPELINE" = true ]; then
        step "0.0a  dataset_summary.py --hospital ${SOURCE_HOSPITAL}  (~1 min CPU)"
        $PY dataset_summary.py --hospital ${SOURCE_HOSPITAL} \
            --output_dir "${RESULTS_ROOT}"
        check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/dataset_summary.csv"

        step "0.0b  run_pipeline for ${SOURCE_HOSPITAL} (pretrain + ${SMOKE_NUM_ARCHS} archs × ${#SMOKE_TASKS[@]} tasks)"
        $PY run_pipeline.py \
            --hospital ${SOURCE_HOSPITAL} \
            --num_archs ${SMOKE_NUM_ARCHS} \
            --tasks ${SMOKE_TASKS[@]} \
            --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
            --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --num_workers ${NUM_WORKERS} \
            --output_dir "${RESULTS_ROOT}"
        check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv"
        check_file "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    fi

    step "0.1  dataset_summary.py --hospital ${TARGET_HOSPITAL}  (~1 min CPU)"
    $PY dataset_summary.py --hospital ${TARGET_HOSPITAL} \
        --output_dir "${RESULTS_ROOT}"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/dataset_summary.csv"

    step "0.2  run_pipeline for ${TARGET_HOSPITAL} (pretrain + ${SMOKE_NUM_ARCHS} archs × ${#SMOKE_TASKS[@]} tasks)"
    $PY run_pipeline.py \
        --hospital ${TARGET_HOSPITAL} \
        --num_archs ${SMOKE_NUM_ARCHS} \
        --tasks ${SMOKE_TASKS[@]} \
        --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
        --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
        --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
        --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
        --num_workers ${NUM_WORKERS} \
        --output_dir "${RESULTS_ROOT}"

    step "0.3  verify outputs"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/metadata.csv"
    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/dataset_summary.csv"
    n_rows=$($PY -c "import pandas as pd; print(len(pd.read_csv('${RESULTS_ROOT}/${TARGET_HOSPITAL}/metadata.csv')))")
    expected=$((SMOKE_NUM_ARCHS * ${#SMOKE_TASKS[@]}))
    [ "$n_rows" = "$expected" ] && ok "metadata: $n_rows rows (expected $expected)" \
        || warn "metadata rows: $n_rows (expected $expected)"

    ok "Stage 0 complete"
fi

# ============================================================================
# STAGE A — shap_analysis + run_meta_regression on SOURCE prior
# ============================================================================
if [ "$RUN_STAGE_A" = true ]; then
    section "STAGE A — Layer 2 prior generation from ${SOURCE_HOSPITAL} (~2 min)"

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
# STAGE B — mas_search × 4 modes × 5 tasks on TARGET
# ============================================================================
if [ "$RUN_MAS" = true ]; then
    section "STAGE B — mas_search × 4 modes × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~3-4 h)"

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
                --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
                --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
                --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
                --top_k_epochs 2 \
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
# STAGE C — Baselines 0-4 × 5 tasks on TARGET
# ============================================================================
if [ "$RUN_BASELINES" = true ]; then
    section "STAGE C — Baselines 0-4 × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~2-3 h)"

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
                --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
                --ckpt_path "${CKPT_PATH}" \
                --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
                --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
                --top_k_epochs 2 \
                --results_dir "${SEED_OUTPUT}"

            DIR="${SEED_OUTPUT}/${TARGET_HOSPITAL}/search/baseline${i}/${task}"
            check_file "${DIR}/baseline${i}_search.csv"
            check_file "${DIR}/baseline${i}_best.csv"
        done
    done

    ok "Stage C complete (5 baselines × ${#SMOKE_TASKS[@]} tasks = $total_runs runs)"
fi

# ============================================================================
# STAGE D — run_regression × 5 tasks on TARGET
# ============================================================================
if [ "$RUN_REGRESSION" = true ]; then
    section "STAGE D — run_regression × ${#SMOKE_TASKS[@]} tasks on ${TARGET_HOSPITAL} (~75 min)"

    for task in "${SMOKE_TASKS[@]}"; do
        step "D.  run_regression --k ${SMOKE_REGRESSION_K} --task ${task}"
        $PY run_regression.py \
            --hospital ${TARGET_HOSPITAL} --task ${task} --k ${SMOKE_REGRESSION_K} \
            --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
            --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --top_k_epochs 2 \
            --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
            --results_dir "${RESULTS_ROOT}"

        check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/regression/${task}/regression_panel.png"
        ok "Fig 3 panel for ${task} generated"
    done

    ok "Stage D complete (${#SMOKE_TASKS[@]} regression panels = Fig 3 inputs)"
fi

# ============================================================================
# STAGE E — Aggregate + plot
# ============================================================================
if [ "$RUN_ANALYZE" = true ]; then
    section "STAGE E — Aggregate results + generate figures/tables (~3 min)"

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

    step "E.2  plot_search_trajectory.py — Fig 1 (default metric val_auprc)"
    $PY analyze/plot_search_trajectory.py \
        --results_root "${RESULTS_ROOT}" \
        --hospital ${TARGET_HOSPITAL} \
        --out_dir "${ANALYZE_DIR}"
    check_file "${ANALYZE_DIR}/figure1_search_trajectory_val_auprc.png"

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
section "HPC XHOSPITAL SMOKE PASSED ✓"
echo ""
echo "Target = ${TARGET_HOSPITAL}, Source = ${SOURCE_HOSPITAL}, Seed = ${SMOKE_SEED}"
echo "Scale  = budget ${SMOKE_BUDGET}, pretrain ${SMOKE_PRETRAIN_EPOCHS} ep, finetune ${SMOKE_FINETUNE_EPOCHS} ep, num_archs ${SMOKE_NUM_ARCHS}"
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
echo "Layer 2 SHAP figures (Fig 4):  ${RESULTS_ROOT}/shap_analysis/<task>/shap_*.png"
echo ""
