#!/bin/bash
# ============================================================================
# Full systematic Mac smoke test — covers entire pipeline end-to-end.
#
# Pipeline stages:
#   [Stage 0] dataset_summary.py + run_pipeline.py  (reduced scale: pretrain 3 ep, finetune 3 ep, 5 archs)
#   [Stage A] shap_analysis.py    (per-task pooled SHAP + Fig 4 PNGs)
#   [Stage A] run_meta_regression (Layer 2 architecture_prior.json)
#   [Stage B] mas_search.py × 4 modes (mas / mas_layer1_only / mas_loto / mas_cold, budget=2 each)
#   [Stage C] baselines 0-4 × budget=2 each (5 baselines: Random, EA, LLM-1shot, LLMatic, CoLLM-NAS)
#   [Stage D] run_regression.py   (k=3 archs, traditional vs AutoFormer pipeline comparison)
#
# Time on Mac MPS (M-series chip):
#   - Stage 0 (run_pipeline reduced):  ~90 min
#   - Stage A:                         ~1 min
#   - Stage B (4 modes × budget=2):    ~40-60 min
#   - Stage C (5 baselines × budget=2):~25-35 min
#   - Stage D (regression k=3):        ~15-20 min
#   - Total full smoke:                ~3-3.5 hours
#
# Backup/restore:
#   Stage 0 overwrites results/MIMIC-IV/{metadata.csv, checkpoint_mlm/mlm_model.pt}.
#   Originals are auto-backed up to .smoke_backup/ before Stage 0.
#   Use --restore to restore originals after smoke.
#
# Usage:
#   bash smoke_test_mac.sh                  # FULL (Stage 0+A+B+C+D), ~3-3.5 hours
#   bash smoke_test_mac.sh --no-pipeline    # Skip Stage 0 (uses existing artifacts), ~80-115 min
#   bash smoke_test_mac.sh --stage-a-only   # Only Stage A (shap + meta_regression), ~1 min
#   bash smoke_test_mac.sh --wiring-only    # Only Layer 2 wiring check (no model runs), ~5 sec
#   bash smoke_test_mac.sh --mas-only       # Stages A+B (mas_search 4 modes, no baselines/regression)
#   bash smoke_test_mac.sh --baselines-only # Stage C only (assumes Stage 0 ckpt already exists)
#   bash smoke_test_mac.sh --regression-only # Stage D only
#   bash smoke_test_mac.sh --restore        # Restore originals from .smoke_backup/
#   bash smoke_test_mac.sh --clean          # Remove smoke artifacts + restore originals
# ============================================================================

set -e

# ─── Constants ───────────────────────────────────────────────────────────
SMOKE_SEED=999
SMOKE_TASK=death
HOSPITAL=MIMIC-IV
PROJECT_ROOT="/Users/deyili/Documents_Local/BMI_PhD@UF/Projects/AgentCare/MAS-NAS-Working/MAS-NAS"
RESULTS_ROOT="${PROJECT_ROOT}/results"
SEED_OUTPUT="${RESULTS_ROOT}/seed_${SMOKE_SEED}"
BACKUP_DIR="${PROJECT_ROOT}/.smoke_backup"
PY="/Users/deyili/anaconda3/envs/agent/bin/python"

# Reduced run_pipeline params for smoke
SMOKE_PRETRAIN_EPOCHS=3
SMOKE_PRETRAIN_PATIENCE=1
SMOKE_FINETUNE_EPOCHS=3
SMOKE_FINETUNE_PATIENCE=1
SMOKE_NUM_ARCHS=5
SMOKE_TASKS="death stay readmission next_diag_6m_pheno next_diag_12m_pheno"

# Color output
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

# ─── CLI parsing ─────────────────────────────────────────────────────────
RUN_PIPELINE=true
RUN_STAGE_A=true
RUN_MAS=true
RUN_BASELINES=true
RUN_REGRESSION=true
WIRING_ONLY=false
CLEAN_ONLY=false
RESTORE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --no-pipeline)      RUN_PIPELINE=false ;;
        --stage-a-only)     RUN_PIPELINE=false; RUN_MAS=false; RUN_BASELINES=false; RUN_REGRESSION=false ;;
        --wiring-only)      RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_BASELINES=false; RUN_REGRESSION=false; WIRING_ONLY=true ;;
        --mas-only)         RUN_PIPELINE=false; RUN_BASELINES=false; RUN_REGRESSION=false ;;
        --baselines-only)   RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_REGRESSION=false ;;
        --regression-only)  RUN_PIPELINE=false; RUN_STAGE_A=false; RUN_MAS=false; RUN_BASELINES=false ;;
        --clean)            CLEAN_ONLY=true ;;
        --restore)          RESTORE_ONLY=true ;;
        --help|-h)          grep '^#' "$0" | head -50; exit 0 ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────
section() { echo -e "\n${BOLD}${BLUE}===================================================================="
            echo -e "$1"
            echo -e "====================================================================${NC}"; }
step()    { echo -e "${BOLD}${YELLOW}\n→ $1${NC}"; }
ok()      { echo -e "${GREEN}  ✓ $1${NC}"; }
fail()    { echo -e "${RED}  ✗ $1${NC}"; exit 1; }
warn()    { echo -e "${YELLOW}  ⚠ $1${NC}"; }
check_file() { [ -f "$1" ] && ok "exists: $1" || fail "missing: $1"; }
check_dir()  { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] && ok "non-empty: $1" || fail "empty/missing: $1"; }
check_count() {
    local pattern="$1"; local expected="$2"; local label="$3"
    local actual=$(ls -1 $pattern 2>/dev/null | wc -l | xargs)
    [ "$actual" -eq "$expected" ] && ok "$label: $actual files (expected $expected)" \
        || fail "$label: $actual (expected $expected)  — pattern: $pattern"
}

cd "${PROJECT_ROOT}" || fail "cannot cd to ${PROJECT_ROOT}"

# ─── --restore mode ──────────────────────────────────────────────────────
if [ "$RESTORE_ONLY" = true ]; then
    section "RESTORE originals from .smoke_backup/"
    if [ ! -d "${BACKUP_DIR}" ]; then
        warn "no .smoke_backup/ directory; nothing to restore"
        exit 0
    fi
    [ -f "${BACKUP_DIR}/mlm_model.pt" ] && cp -f "${BACKUP_DIR}/mlm_model.pt" \
        "${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt" && ok "restored mlm_model.pt" \
        || warn "no mlm_model.pt in backup"
    [ -f "${BACKUP_DIR}/metadata.csv" ] && cp -f "${BACKUP_DIR}/metadata.csv" \
        "${RESULTS_ROOT}/${HOSPITAL}/metadata.csv" && ok "restored metadata.csv" \
        || warn "no metadata.csv in backup"
    exit 0
fi

# ─── --clean mode ────────────────────────────────────────────────────────
if [ "$CLEAN_ONLY" = true ]; then
    section "CLEAN smoke artifacts + RESTORE originals"
    rm -rf "${RESULTS_ROOT}/shap_analysis" "${RESULTS_ROOT}/meta_regression" "${SEED_OUTPUT}"
    ok "removed: results/shap_analysis, results/meta_regression, results/seed_${SMOKE_SEED}"
    if [ -d "${BACKUP_DIR}" ]; then
        [ -f "${BACKUP_DIR}/mlm_model.pt" ] && cp -f "${BACKUP_DIR}/mlm_model.pt" \
            "${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt" && ok "restored mlm_model.pt"
        [ -f "${BACKUP_DIR}/metadata.csv" ] && cp -f "${BACKUP_DIR}/metadata.csv" \
            "${RESULTS_ROOT}/${HOSPITAL}/metadata.csv" && ok "restored metadata.csv"
        # Restore Stage D regression artifacts (smoke version overwrites them in place)
        if [ -d "${BACKUP_DIR}/regression_${SMOKE_TASK}" ]; then
            rm -rf "${RESULTS_ROOT}/${HOSPITAL}/regression/${SMOKE_TASK}"
            cp -R "${BACKUP_DIR}/regression_${SMOKE_TASK}" \
                "${RESULTS_ROOT}/${HOSPITAL}/regression/${SMOKE_TASK}" \
                && ok "restored regression/${SMOKE_TASK}/"
        fi
        rm -rf "${BACKUP_DIR}" && ok "removed .smoke_backup/"
    fi
    exit 0
fi

# ─── Prerequisite checks ─────────────────────────────────────────────────
section "PREREQUISITES (Mac local)"

step "Python interpreter (Mac local agent env)"
[ -x "$PY" ] && ok "$PY" || fail "missing: $PY"
$PY --version

step "Critical deps"
$PY -c "
import sys
need = ['xgboost', 'shap', 'matplotlib', 'statsmodels', 'pandas', 'numpy', 'sklearn',
        'torch', 'anthropic', 'dotenv']
missing = [p for p in need if __import__('importlib').util.find_spec(p) is None]
if missing:
    print(f'  MISSING: {missing}')
    print(f'  Install:  $PY -m pip install ' + ' '.join(missing))
    sys.exit(1)
print(f'  ✓ all 10 deps present')
import torch
print(f'  ✓ torch backend: cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()}')
" || fail "missing deps; install with the command shown"

step ".env file (mas_search auto-loads ANTHROPIC_API_KEY)"
check_file ".env"
grep -q "ANTHROPIC_API_KEY" .env && ok ".env contains ANTHROPIC_API_KEY" || fail ".env missing key"

step "Source data (raw pkls)"
for pkl in mimic.pkl mimic_pretrain.pkl mimic_downstream.pkl mimic_nextdiag_6m.pkl mimic_nextdiag_12m.pkl; do
    check_file "data_process/${HOSPITAL}/${HOSPITAL}-processed/${pkl}"
done

# ============================================================================
# STAGE 0 — run_pipeline.py (reduced scale)
# ============================================================================
if [ "$RUN_PIPELINE" = true ]; then
    section "STAGE 0 — run_pipeline.py (reduced scale, ~90 min on Mac MPS)"

    step "Pre-Stage 0: backup existing artifacts"
    mkdir -p "${BACKUP_DIR}"
    if [ -f "${RESULTS_ROOT}/${HOSPITAL}/metadata.csv" ]; then
        cp -f "${RESULTS_ROOT}/${HOSPITAL}/metadata.csv" "${BACKUP_DIR}/metadata.csv"
        ok "backed up metadata.csv → ${BACKUP_DIR}/"
    fi
    if [ -f "${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt" ]; then
        cp -f "${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt" "${BACKUP_DIR}/mlm_model.pt"
        ok "backed up mlm_model.pt → ${BACKUP_DIR}/"
    fi

    # Stage 0 has TWO scripts (per plan A1 + A2 design):
    #   0.0  dataset_summary.py  → <hospital>/dataset_summary.csv  (Layer 1 cosine input)
    #   0.1  run_pipeline.py     → <hospital>/{metadata.csv, checkpoint_mlm/mlm_model.pt}
    # If dataset_summary.csv is missing, mas_search silently degrades to cold-start
    # (Layer 1 returns "No historical data found"), invalidating the smoke check.
    step "0.0  dataset_summary.py --hospital ${HOSPITAL}  (~1 min CPU)"
    $PY dataset_summary.py --hospital ${HOSPITAL} \
        --output_dir "${RESULTS_ROOT}"
    check_file "${RESULTS_ROOT}/${HOSPITAL}/dataset_summary.csv"

    step "0.1  run_pipeline.py with smoke-scale params"
    echo "  --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS}    (vs production 50)"
    echo "  --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE}  (vs production 5)"
    echo "  --finetune_epochs ${SMOKE_FINETUNE_EPOCHS}    (vs production 20)"
    echo "  --finetune_patience ${SMOKE_FINETUNE_PATIENCE}  (vs production 5)"
    echo "  --num_archs ${SMOKE_NUM_ARCHS}        (vs production 10-100)"
    echo "  --tasks ${SMOKE_TASKS}"
    echo ""

    # Run the reduced pipeline. num_workers=0 (Mac MPS friendly), --max_params 2M (production budget)
    $PY run_pipeline.py \
        --hospital ${HOSPITAL} \
        --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
        --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
        --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
        --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
        --num_archs ${SMOKE_NUM_ARCHS} \
        --tasks ${SMOKE_TASKS} \
        --num_workers 0

    step "0.2  Stage 0 verify outputs"
    check_file "${RESULTS_ROOT}/${HOSPITAL}/metadata.csv"
    check_file "${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    check_file "${RESULTS_ROOT}/${HOSPITAL}/dataset_summary.csv"

    # Verify metadata.csv has all 5 tasks + new columns (label_entropy, positive_ratio)
    $PY <<EOF
import pandas as pd, sys
df = pd.read_csv("${RESULTS_ROOT}/${HOSPITAL}/metadata.csv")
n_tasks = df['task'].nunique()
required_cols = {'hospital', 'task', 'embed_dim', 'depth', 'mlp_ratio', 'num_heads',
                 'num_params', 'flops', 'accuracy', 'f1', 'auroc', 'auprc',
                 'label_entropy', 'positive_ratio'}
missing = required_cols - set(df.columns)
if missing:
    print(f"  ✗ metadata.csv missing cols: {missing}"); sys.exit(1)
if n_tasks < 5:
    print(f"  ✗ metadata has {n_tasks} tasks (expected 5)"); sys.exit(1)
print(f"  ✓ metadata.csv: {len(df)} rows, {n_tasks} tasks, all required cols present")
print(f"    tasks: {sorted(df['task'].unique())}")
EOF

    ok "Stage 0 complete (~90 min)"
fi

# ============================================================================
# STAGE A — shap_analysis + run_meta_regression (~1 min)
# ============================================================================
if [ "$RUN_STAGE_A" = true ]; then
    section "STAGE A — Layer 1+2 prior generation (~1 min)"

    step "A.1  shap_analysis.py  (per-task pooled SHAP)"
    rm -rf "${RESULTS_ROOT}/shap_analysis"
    $PY shap_analysis.py \
        --hospitals ${HOSPITAL} \
        --results_dir "${RESULTS_ROOT}" \
        --output_dir "${RESULTS_ROOT}/shap_analysis" \
        --interaction_main embed_dim depth num_heads mlp_ratio

    step "A.1 verify outputs"
    EXPECTED_TASKS=("death" "stay" "readmission" "next_diag_6m_pheno" "next_diag_12m_pheno")
    for task in "${EXPECTED_TASKS[@]}"; do
        check_dir "${RESULTS_ROOT}/shap_analysis/${task}"
        check_file "${RESULTS_ROOT}/shap_analysis/${task}/shap_summary.png"
        check_file "${RESULTS_ROOT}/shap_analysis/${task}/shap_values.csv"
        check_count "${RESULTS_ROOT}/shap_analysis/${task}/shap_interaction_*.png" 4 "[${task}] interaction PNGs"
    done

    step "A.1 schema check on shap_values.csv"
    $PY <<EOF
import pandas as pd, sys
df = pd.read_csv("${RESULTS_ROOT}/shap_analysis/death/shap_values.csv")
required = {"hospital", "embed_dim", "depth", "mlp_ratio", "num_heads",
            "shap_embed_dim", "shap_depth", "shap_mlp_ratio", "shap_num_heads"}
missing = required - set(df.columns)
if missing: print(f"  ✗ missing: {missing}"); sys.exit(1)
print(f"  ✓ schema OK (n_rows={len(df)}, hospitals={df['hospital'].nunique()})")
EOF

    step "A.2  run_meta_regression.py  (Layer 2 architecture_prior.json)"
    rm -rf "${RESULTS_ROOT}/meta_regression"
    $PY run_meta_regression.py \
        --shap_dir "${RESULTS_ROOT}/shap_analysis" \
        --output_dir "${RESULTS_ROOT}/meta_regression"

    step "A.2 verify outputs"
    for task in "${EXPECTED_TASKS[@]}"; do
        check_file "${RESULTS_ROOT}/meta_regression/${task}/architecture_prior.json"
    done

    step "A.2 schema check on architecture_prior.json"
    $PY <<EOF
import json, sys
required = {"task", "feature_importance_order", "preferred_levels",
            "discouraged_levels", "interaction_rules", "confidence", "_caveat"}
for task in ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]:
    with open(f"${RESULTS_ROOT}/meta_regression/{task}/architecture_prior.json") as f:
        prior = json.load(f)
    missing = required - set(prior.keys())
    if missing: print(f"  ✗ {task} missing: {missing}"); sys.exit(1)
    print(f"  ✓ {task}: order={prior['feature_importance_order'][:2]}..., preferred={prior['preferred_levels']}")
EOF

    ok "Stage A complete (~1 min)"
fi

# ============================================================================
# STAGE B — mas_search (wiring check + 4 modes)
# ============================================================================
# Always do quick wiring check first (5 sec, no GPU)
section "STAGE B — mas_search Layer 1+2 wiring + 4-mode runs"

step "B.0  Layer 1+2 wiring quick check (no GPU, ~5 sec)"
$PY <<EOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
from mas_search import gather_historical_context

print("  Testing 4 modes' context dicts (Layer 1, Layer 2):")

# Mode 1: mas — Layer 1 ON (exact), Layer 2 ON
ctx_full = gather_historical_context(
    "${HOSPITAL}", "death", 2_000_000, "${RESULTS_ROOT}", top_k=5,
    exclude_exact_task_from_history=False, no_history=False, no_meta_regression=False,
)
mp_full = ctx_full["meta_regression_prior"]
assert mp_full and "preferred_levels" in mp_full, f"Layer 2 prior empty in mas mode: {mp_full}"
assert ctx_full["top_k_archs"], "Layer 1 top_k_archs should be non-empty in mas mode"
print(f"  ✓ mas              (L1 ON, L2 ON):  L1 top_k={len(ctx_full['top_k_archs'])} | L2 keys={sorted(mp_full.keys())[:3]}...")

# Mode 2: mas_layer1_only — Layer 1 ON (exact), Layer 2 OFF
ctx_l1only = gather_historical_context(
    "${HOSPITAL}", "death", 2_000_000, "${RESULTS_ROOT}", top_k=5,
    exclude_exact_task_from_history=False, no_history=False, no_meta_regression=True,
)
mp_l1only = ctx_l1only["meta_regression_prior"]
assert mp_l1only == {}, f"Layer 2 must be empty in mas_layer1_only, got {mp_l1only}"
assert ctx_l1only["top_k_archs"], "Layer 1 top_k_archs should be non-empty in mas_layer1_only"
print(f"  ✓ mas_layer1_only  (L1 ON, L2 OFF): L1 top_k={len(ctx_l1only['top_k_archs'])} | L2 = {{}}")

# Mode 3: mas_loto — Layer 1 fallback, Layer 2 ON
ctx_loto = gather_historical_context(
    "${HOSPITAL}", "death", 2_000_000, "${RESULTS_ROOT}", top_k=5,
    exclude_exact_task_from_history=True, no_history=False, no_meta_regression=False,
)
mp_loto = ctx_loto["meta_regression_prior"]
assert mp_loto, f"Layer 2 should stay ON in mas_loto, got {mp_loto}"
print(f"  ✓ mas_loto         (L1 fallback, L2 ON): Layer 2 still active (correct)")

# Mode 4: mas_cold — BOTH off
ctx_cold = gather_historical_context(
    "${HOSPITAL}", "death", 2_000_000, "${RESULTS_ROOT}", top_k=5,
    exclude_exact_task_from_history=False, no_history=True, no_meta_regression=False,
)
mp_cold = ctx_cold["meta_regression_prior"]
assert mp_cold == {}, f"Layer 2 must be empty in mas_cold, got {mp_cold}"
assert ctx_cold["top_k_archs"] == [], f"Layer 1 must be empty in mas_cold, got {ctx_cold['top_k_archs']}"
print(f"  ✓ mas_cold         (L1 OFF, L2 OFF): both layers empty (correct)")

print("\n  ✓ All 4 modes correctly toggle Layer 1+2 priors")
EOF

if [ "$WIRING_ONLY" = true ]; then
    ok "Wiring-only smoke complete (~5 sec)"
    exit 0
fi

if [ "$RUN_MAS" = true ]; then
    rm -rf "${SEED_OUTPUT}"

    step "B.1  mas (full Layer 1 + Layer 2)"
    $PY mas_search.py \
        --hospital ${HOSPITAL} --task ${SMOKE_TASK} --max_params 2000000 \
        --budget 2 --seed ${SMOKE_SEED} --num_workers 0 \
        --results_dir "${SEED_OUTPUT}" \
        --history_root "${RESULTS_ROOT}"

    DIR_B1="${SEED_OUTPUT}/${HOSPITAL}/search/mas/${SMOKE_TASK}"
    check_file "${DIR_B1}/mas_search.csv"
    check_file "${DIR_B1}/mas_best.csv"
    check_file "${DIR_B1}/agent_io_log.txt"
    # Use "## Architecture Prior" (markdown header) as the discriminating marker —
    # it's rendered conditionally on `arch_prior` being non-empty.
    # (The bare phrase "Architecture Prior" also appears in the Critic's static
    # rubric text "(from Architecture Prior)" regardless of mode.)
    grep -q "## Architecture Prior" "${DIR_B1}/agent_io_log.txt" \
        && ok "agent_io_log.txt contains '## Architecture Prior' header (Layer 2 active)" \
        || fail "Layer 2 not in agent_io_log.txt"
    METHOD=$($PY -c "import json; print(json.load(open('${DIR_B1}/search_meta.json'))['method'])")
    [ "$METHOD" = "mas" ] && ok "search_meta.method = mas" || fail "method = $METHOD"

    step "B.2  mas_layer1_only (Layer 1 exact ON + Layer 2 OFF — isolates Layer 2 contribution)"
    $PY mas_search.py \
        --hospital ${HOSPITAL} --task ${SMOKE_TASK} --max_params 2000000 \
        --budget 2 --seed ${SMOKE_SEED} --num_workers 0 \
        --no_meta_regression \
        --results_dir "${SEED_OUTPUT}" \
        --history_root "${RESULTS_ROOT}"

    DIR_B2="${SEED_OUTPUT}/${HOSPITAL}/search/mas_layer1_only/${SMOKE_TASK}"
    check_file "${DIR_B2}/mas_layer1_only_search.csv"
    check_file "${DIR_B2}/mas_layer1_only_best.csv"
    check_file "${DIR_B2}/agent_io_log.txt"
    METHOD=$($PY -c "import json; print(json.load(open('${DIR_B2}/search_meta.json'))['method'])")
    [ "$METHOD" = "mas_layer1_only" ] && ok "search_meta.method = mas_layer1_only" || fail "method = $METHOD"
    # Layer 2 OFF → no '## Architecture Prior' header rendered
    grep -q "## Architecture Prior" "${DIR_B2}/agent_io_log.txt" \
        && fail "mas_layer1_only should NOT have '## Architecture Prior' (Layer 2 should be OFF)" \
        || ok "Layer 2 correctly OFF in mas_layer1_only (no '## Architecture Prior' header)"
    # Layer 1 ON → tracer.log_kv("Similar hospital", "<name>  (similarity = X)")
    # When Layer 1 is OFF (mas_cold), the value is "N/A" with no "similarity = " text.
    grep -qE "Similar hospital +: .*similarity = " "${DIR_B2}/agent_io_log.txt" \
        && ok "Layer 1 still active in mas_layer1_only (similar-hospital match logged)" \
        || fail "Layer 1 missing in mas_layer1_only (no 'similarity = ' in agent_io_log)"

    step "B.3  mas_loto (Layer 1 task fallback + Layer 2 unchanged)"
    $PY mas_search.py \
        --hospital ${HOSPITAL} --task ${SMOKE_TASK} --max_params 2000000 \
        --budget 2 --seed ${SMOKE_SEED} --num_workers 0 \
        --exclude_exact_task_from_history \
        --results_dir "${SEED_OUTPUT}" \
        --history_root "${RESULTS_ROOT}"

    DIR_B3="${SEED_OUTPUT}/${HOSPITAL}/search/mas_loto/${SMOKE_TASK}"
    check_file "${DIR_B3}/mas_loto_search.csv"
    check_file "${DIR_B3}/agent_io_log.txt"
    METHOD=$($PY -c "import json; print(json.load(open('${DIR_B3}/search_meta.json'))['method'])")
    [ "$METHOD" = "mas_loto" ] && ok "search_meta.method = mas_loto" || fail "method = $METHOD"
    grep -q "## Architecture Prior" "${DIR_B3}/agent_io_log.txt" \
        && ok "Layer 2 still active in mas_loto (correct)" \
        || fail "Layer 2 missing in mas_loto"

    step "B.4  mas_cold (Layer 1 OFF + Layer 2 OFF)"
    $PY mas_search.py \
        --hospital ${HOSPITAL} --task ${SMOKE_TASK} --max_params 2000000 \
        --budget 2 --seed ${SMOKE_SEED} --num_workers 0 \
        --no_history \
        --results_dir "${SEED_OUTPUT}" \
        --history_root "${RESULTS_ROOT}"

    DIR_B4="${SEED_OUTPUT}/${HOSPITAL}/search/mas_cold/${SMOKE_TASK}"
    check_file "${DIR_B4}/mas_cold_search.csv"
    check_file "${DIR_B4}/agent_io_log.txt"
    METHOD=$($PY -c "import json; print(json.load(open('${DIR_B4}/search_meta.json'))['method'])")
    [ "$METHOD" = "mas_cold" ] && ok "search_meta.method = mas_cold" || fail "method = $METHOD"
    grep -q "## Architecture Prior" "${DIR_B4}/agent_io_log.txt" \
        && fail "mas_cold should NOT have '## Architecture Prior' header (Layer 2 should be OFF)" \
        || ok "Layer 2 correctly disabled in mas_cold (no '## Architecture Prior' header rendered)"

    ok "Stage B complete (4/4 modes)"
fi

# ============================================================================
# STAGE C — Baselines 0-4 (Random / EA / LLM-1shot / LLMatic / CoLLM-NAS)
# ============================================================================
# All 5 baselines share the warm pretrain ckpt at <history_root>/<hospital>/checkpoint_mlm/.
# We don't pass --ckpt_path explicitly; baselines auto-detect via --results_dir layout.
# The seeded output goes to ${SEED_OUTPUT}/<hospital>/search/baseline<N>/<task>/.
if [ "$RUN_BASELINES" = true ]; then
    section "STAGE C — Baselines 0-4 (5 methods × budget=2 on ${SMOKE_TASK})"

    CKPT_PATH="${RESULTS_ROOT}/${HOSPITAL}/checkpoint_mlm/mlm_model.pt"
    if [ ! -f "${CKPT_PATH}" ]; then
        fail "Pretrain checkpoint missing: ${CKPT_PATH}. Run Stage 0 (full smoke) first."
    fi
    ok "Reusing warm checkpoint: ${CKPT_PATH}"

    BASELINE_NAMES=("Random" "EA" "LLM-1shot" "LLMatic" "CoLLM-NAS")
    for i in 0 1 2 3 4; do
        bname=${BASELINE_NAMES[$i]}
        step "C.$((i+1))  baseline${i} (${bname})"
        $PY baselines/baseline${i}.py \
            --hospital ${HOSPITAL} --task ${SMOKE_TASK} --max_params 2000000 \
            --budget 2 --seed ${SMOKE_SEED} --num_workers 0 \
            --ckpt_path "${CKPT_PATH}" \
            --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
            --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
            --top_k_epochs 1 \
            --results_dir "${SEED_OUTPUT}"

        DIR_C="${SEED_OUTPUT}/${HOSPITAL}/search/baseline${i}/${SMOKE_TASK}"
        # Output naming: baselines emit <baseline>_search.csv + <baseline>_best.csv
        check_file "${DIR_C}/baseline${i}_search.csv"
        check_file "${DIR_C}/baseline${i}_best.csv"
        ok "baseline${i} (${bname}) complete"
    done

    ok "Stage C complete (5/5 baselines)"
fi

# ============================================================================
# STAGE D — run_regression.py (Traditional vs AutoFormer pipeline comparison, Fig 3)
# ============================================================================
# Uses --k=3 random architectures (smoke), reduced epochs. Outputs to
# results/<hospital>/regression/<task>/{regression_panel.png, ...csv}.
# This validates the Fig 3 dual-pipeline scatter plot path.
if [ "$RUN_REGRESSION" = true ]; then
    section "STAGE D — run_regression.py (k=3 archs, ${SMOKE_TASK} only)"

    # Backup existing regression artifacts (full-scale outputs from prior runs)
    REG_DIR="${RESULTS_ROOT}/${HOSPITAL}/regression/${SMOKE_TASK}"
    REG_BACKUP="${BACKUP_DIR}/regression_${SMOKE_TASK}"
    if [ -d "${REG_DIR}" ]; then
        mkdir -p "${BACKUP_DIR}"
        rm -rf "${REG_BACKUP}"
        cp -R "${REG_DIR}" "${REG_BACKUP}" && ok "backed up regression/${SMOKE_TASK} → ${REG_BACKUP}"
    fi

    step "D.1  run_regression.py --k 3 --task ${SMOKE_TASK}"
    $PY run_regression.py \
        --hospital ${HOSPITAL} --task ${SMOKE_TASK} --k 3 \
        --pretrain_epochs ${SMOKE_PRETRAIN_EPOCHS} \
        --pretrain_patience ${SMOKE_PRETRAIN_PATIENCE} \
        --finetune_epochs ${SMOKE_FINETUNE_EPOCHS} \
        --finetune_patience ${SMOKE_FINETUNE_PATIENCE} \
        --top_k_epochs 1 \
        --seed ${SMOKE_SEED} --num_workers 0 \
        --results_dir "${RESULTS_ROOT}"

    check_file "${REG_DIR}/regression_panel.png"
    ok "regression_panel.png generated (Fig 3 path verified)"

    ok "Stage D complete"
fi

# ============================================================================
# FINAL REPORT
# ============================================================================
section "SMOKE TEST PASSED ✓"

echo "Outputs summary:"
[ -d "${RESULTS_ROOT}/shap_analysis" ] && \
    echo "  Stage A SHAP:             $(ls ${RESULTS_ROOT}/shap_analysis/*/*.png 2>/dev/null | wc -l | xargs) PNGs"
[ -d "${RESULTS_ROOT}/meta_regression" ] && \
    echo "  Stage A architecture_prior.json:  $(ls ${RESULTS_ROOT}/meta_regression/*/architecture_prior.json 2>/dev/null | wc -l | xargs) files"
if [ -d "${SEED_OUTPUT}" ]; then
    echo "  Stage B (mas_search) outputs (seed_${SMOKE_SEED}):"
    for mode in mas mas_layer1_only mas_loto mas_cold; do
        n=$(ls "${SEED_OUTPUT}/${HOSPITAL}/search/${mode}/${SMOKE_TASK}/" 2>/dev/null | wc -l | xargs)
        [ "$n" -gt 0 ] && echo "    ${mode}: ${n} files"
    done
    echo "  Stage C (baselines) outputs (seed_${SMOKE_SEED}):"
    for i in 0 1 2 3 4; do
        n=$(ls "${SEED_OUTPUT}/${HOSPITAL}/search/baseline${i}/${SMOKE_TASK}/" 2>/dev/null | wc -l | xargs)
        [ "$n" -gt 0 ] && echo "    baseline${i}: ${n} files"
    done
fi
if [ -f "${RESULTS_ROOT}/${HOSPITAL}/regression/${SMOKE_TASK}/regression_panel.png" ]; then
    echo "  Stage D (regression):"
    echo "    regression_panel.png: ${RESULTS_ROOT}/${HOSPITAL}/regression/${SMOKE_TASK}/regression_panel.png"
fi

echo ""
if [ -d "${BACKUP_DIR}" ]; then
    echo "⚠ Smoke modified results/${HOSPITAL}/{metadata.csv, checkpoint_mlm/mlm_model.pt, regression/${SMOKE_TASK}/}."
    echo "  Originals are backed up at ${BACKUP_DIR}/"
    echo "  To restore: bash $0 --restore"
fi
echo ""
echo "To clean (remove smoke artifacts + restore):  bash $0 --clean"
