#!/bin/bash
#SBATCH --job-name=xh-smoke
#SBATCH --account=mei.liu
#SBATCH --qos=mei.liu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err

# ============================================================================
# Cross-Hospital Near-Production Smoke (HiPerGator, single self-contained sbatch)
# ============================================================================
# Target = MIMIC-III   (test hospital)
# Source = MIMIC-IV    (prior — auto-bootstrapped if missing)
#
# Near-production scale (budget=10, 10 ep, 10 archs) — bigger than Mac smoke
# (budget=2, 3 ep, 5 archs) to validate behavior closer to real Phase 2 jobs.
#
# Pipeline stages (no skipping):
#   [Stage 0] dataset_summary + run_pipeline for MIMIC-III (+MIMIC-IV if missing)
#   [Stage A] shap_analysis + run_meta_regression on MIMIC-IV (source prior)
#   [Stage B] mas_search × 4 modes × 5 tasks on MIMIC-III (target)
#               modes: mas, mas_layer1_only, mas_loto, mas_cold
#   [Stage C] baselines 0-4 × 5 tasks on MIMIC-III
#   [Stage D] run_regression × 5 tasks on MIMIC-III (Fig 3)
#   [Stage E] aggregate + plot all figures + tables
#
# Estimated wall-time on A100 (single GPU): ~8-10 hours
# Wall request: 24:00:00 (double the estimate; plenty of margin for retries)
#
# Submit:    sbatch smoke_test_hipergator_xhospital.sbatch
# Monitor:   squeue -u $USER  +  tail -f slurm-<jobid>-xhospital-smoke.out
# Cancel:    scancel <jobid>
# ============================================================================

set -e

# Ensure slurm_logs/ exists on /blue (--output/--error paths use this)
mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Started at:    $(date)"
echo "Working dir:   $(pwd)"
echo "=========================================="

# Activate conda env
module load conda
conda activate Autoformer

echo "Python:        $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible:  $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -25
echo "=========================================="

# ─── Configuration ───────────────────────────────────────────────────────
PROJECT_ROOT="/home/lideyi/MAS-NAS"
PY=$(which python)

TARGET_HOSPITAL="MIMIC-III"
SOURCE_HOSPITAL="MIMIC-IV"
SMOKE_SEED=999
SMOKE_TASKS=("death" "stay" "readmission" "next_diag_6m_pheno" "next_diag_12m_pheno")

# Near-production scale
SMOKE_PRETRAIN_EPOCHS=10
SMOKE_PRETRAIN_PATIENCE=2
SMOKE_FINETUNE_EPOCHS=10
SMOKE_FINETUNE_PATIENCE=2
SMOKE_NUM_ARCHS=10
SMOKE_BUDGET=10
SMOKE_REGRESSION_K=3
SMOKE_BATCH_SIZE=128         # match production sbatch (run_baseline/run_pipeline)

# Cheap model for LLM calls (mas_search agents + LLM-based baselines).
# Haiku is ~3.7x cheaper than Sonnet ($0.80/$4 vs $3/$15 per M tokens) → at
# SMOKE_BUDGET=10 this brings total LLM cost from ~$10-15 to ~$3-5. Smoke
# only needs code-path validation, not best-in-class agent reasoning;
# production paper runs should switch back to sonnet/opus by overriding here.
LLM_MODEL="claude-haiku-4-5"

# Paths — ALL outputs go to /blue (lab shared, large quota); /home is small.
# Only source code stays in PROJECT_ROOT (cd'd into so relative imports work).
RESULTS_ROOT="/blue/mei.liu/lideyi/MAS-NAS/results"
SEED_OUTPUT="${RESULTS_ROOT}/seed_${SMOKE_SEED}"
ANALYZE_DIR="/blue/mei.liu/lideyi/MAS-NAS/analyze"
mkdir -p "${ANALYZE_DIR}"

NUM_WORKERS=4

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

# ─── Prerequisite checks ─────────────────────────────────────────────────
section "PREREQUISITES"

step "Python interpreter"
[ -x "$PY" ] && ok "$PY" || fail "missing python. Did conda activation succeed?"
$PY --version

step "Critical deps"
$PY -c "
import sys, importlib.util
need = ['xgboost', 'shap', 'matplotlib', 'statsmodels', 'pandas', 'numpy', 'sklearn',
        'torch', 'anthropic', 'dotenv', 'scipy']
missing = [p for p in need if importlib.util.find_spec(p) is None]
if missing:
    print(f'  MISSING: {missing}'); sys.exit(1)
print(f'  ✓ all {len(need)} deps present')
import torch
print(f'  ✓ torch backend: cuda={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available! Smoke needs GPU.'
"

step "Source data — MIMIC-III (target) + MIMIC-IV (source)"
# utils.paths.get_processed_root() routes data to /blue on HPC (it detects HPC
# by /blue/mei.liu/lideyi/MAS-NAS existing). The relative path "data_process/..."
# would resolve via cwd /home/lideyi/MAS-NAS, missing the actual data location.
for HOSP in "${TARGET_HOSPITAL}" "${SOURCE_HOSPITAL}"; do
    DATA_ROOT=$($PY -c "from utils.paths import get_processed_root; print(get_processed_root('${HOSP}'))")
    for f in mimic.pkl mimic_pretrain.pkl mimic_downstream.pkl mimic_nextdiag_6m.pkl mimic_nextdiag_12m.pkl; do
        check_file "${DATA_ROOT}/${f}"
    done
done

step ".env file"
check_file ".env"
grep -q "ANTHROPIC_API_KEY" .env && ok ".env contains ANTHROPIC_API_KEY" || fail "ANTHROPIC_API_KEY missing"

step "Source prior (${SOURCE_HOSPITAL}) check at ${RESULTS_ROOT}"
if [ ! -f "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/metadata.csv" ] || \
   [ ! -f "${RESULTS_ROOT}/${SOURCE_HOSPITAL}/dataset_summary.csv" ]; then
    warn "Source prior missing — Stage 0 will bootstrap MIMIC-IV first (+~70 min)"
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
section "STAGE 0 — run_pipeline.py (~70 min target, +~70 min source if needed)"

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
        --batch_size ${SMOKE_BATCH_SIZE} \
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
    --batch_size ${SMOKE_BATCH_SIZE} \
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

# ============================================================================
# STAGE A — shap_analysis + run_meta_regression on SOURCE prior
# ============================================================================
section "STAGE A — Layer 2 prior generation from ${SOURCE_HOSPITAL} (~2 min)"

step "A.1  shap_analysis.py --hospitals ${SOURCE_HOSPITAL}"
# IMPORTANT: pass --output_dir explicitly — default is "./results/shap_analysis"
# which on HPC resolves to /home (cwd is /home/lideyi/MAS-NAS), bypassing /blue.
$PY shap_analysis.py --hospitals ${SOURCE_HOSPITAL} \
    --results_dir "${RESULTS_ROOT}" \
    --output_dir "${RESULTS_ROOT}/shap_analysis" \
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

# ============================================================================
# STAGE B — mas_search × 4 modes × 5 tasks on TARGET
# ============================================================================
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
        step "B.${counter}/${total_runs}  mas_search ${mode} on ${task} (model=${LLM_MODEL})"
        $PY mas_search.py \
            --hospital ${TARGET_HOSPITAL} --task ${task} --max_params 2000000 \
            --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
            --batch_size ${SMOKE_BATCH_SIZE} \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --top_k_epochs 2 \
            --model ${LLM_MODEL} \
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

# ============================================================================
# STAGE C — Baselines 0-4 × 5 tasks on TARGET
# ============================================================================
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
        # baseline0 (Random) + baseline1 (EA) don't call LLM; baseline 2/3/4 do.
        # Pass --model uniformly — baseline0/1 ignore it harmlessly.
        MODEL_ARG=""
        if [ "$i" -ge 2 ]; then
            MODEL_ARG="--model ${LLM_MODEL}"
        fi
        step "C.${counter}/${total_runs}  baseline${i} (${bname}) on ${task} ${MODEL_ARG}"
        $PY baselines/baseline${i}.py \
            --hospital ${TARGET_HOSPITAL} --task ${task} --max_params 2000000 \
            --budget ${SMOKE_BUDGET} --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
            --ckpt_path "${CKPT_PATH}" \
            --batch_size ${SMOKE_BATCH_SIZE} \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --top_k_epochs 2 \
            ${MODEL_ARG} \
            --results_dir "${SEED_OUTPUT}"

        DIR="${SEED_OUTPUT}/${TARGET_HOSPITAL}/search/baseline${i}/${task}"
        check_file "${DIR}/baseline${i}_search.csv"
        check_file "${DIR}/baseline${i}_best.csv"
    done
done

ok "Stage C complete (5 baselines × ${#SMOKE_TASKS[@]} tasks = $total_runs runs)"

# ============================================================================
# STAGE D — run_regression × 5 tasks on TARGET
# ============================================================================
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
        --batch_size ${SMOKE_BATCH_SIZE} \
        --seed ${SMOKE_SEED} --num_workers ${NUM_WORKERS} \
        --results_dir "${RESULTS_ROOT}"

    check_file "${RESULTS_ROOT}/${TARGET_HOSPITAL}/regression/${task}/regression_panel.png"
    ok "Fig 3 panel for ${task} generated"
done

ok "Stage D complete (${#SMOKE_TASKS[@]} regression panels = Fig 3 inputs)"

# ============================================================================
# STAGE E — Aggregate + plot
# ============================================================================
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

# ============================================================================
# FINAL REPORT
# ============================================================================
section "HPC XHOSPITAL SMOKE PASSED ✓"
echo ""
echo "Target = ${TARGET_HOSPITAL}, Source = ${SOURCE_HOSPITAL}, Seed = ${SMOKE_SEED}"
echo "Scale  = budget ${SMOKE_BUDGET}, pretrain ${SMOKE_PRETRAIN_EPOCHS} ep, finetune ${SMOKE_FINETUNE_EPOCHS} ep, num_archs ${SMOKE_NUM_ARCHS}"
echo "LLM    = ${LLM_MODEL}"
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
echo "=========================================="
echo "Finished at:   $(date)"
echo "=========================================="
