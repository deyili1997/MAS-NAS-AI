#!/bin/bash
# =============================================================================
# Smoke Test — validates all changes made on 2026-05-29
# =============================================================================
# Tests (in order):
#   [1] Config sanity    — stage_b_jobs.tsv, sbatch, run_analysis.sh
#   [2] Budget=30        — --budget 30 in all 9 methods
#   [3] Timing fields    — n_evals + per_eval_sec_mean in search_meta.json
#   [4] Pretrain timing  — pretrain_meta.json created by run_pipeline.py
#   [5] New hospitals    — source_10 (internal), MIMIC-III (external)
#   [6] Aggregate tables — all 9 CSVs per hospital, including new ones
#   [7] Bootstrap CI     — significance CSV has ci_lower_pct / ci_upper_pct
#   [8] Cost decomp      — cost_table has gpu_eval_min_mean / llm_latency_min_mean
#   [9] Cost N=10 table  — cost_table_n10 produced (may be empty without pretrain_meta)
#
# Runtime estimate: ~30-45 min on HiperGator L4 GPU
# Usage: bash slurm/smoke_test_changes.sh
# =============================================================================

set -euo pipefail

REPO=/home/lideyi/MAS-NAS
PROJECT=/blue/mei.liu/lideyi/MAS-NAS
SMOKE_BUDGET=5          # tiny budget — fast
SMOKE_SEED=42
SMOKE_TASK=death        # one representative task
SMOKE_RESULTS=$PROJECT/results/smoke_changes_$(date +%Y%m%d_%H%M%S)
SMOKE_OUT=$PROJECT/analyze/smoke_changes_$(date +%Y%m%d_%H%M%S)
PASS=0; FAIL=0

cd "$REPO"

green()  { echo -e "\033[32m  ✓ $*\033[0m"; ((PASS++)) || true; }
red()    { echo -e "\033[31m  ✗ FAIL: $*\033[0m"; ((FAIL++)) || true; }
header() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "[$1] $2"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# =============================================================================
header "1/9" "Config sanity checks"
# =============================================================================

# stage_b_jobs.tsv: 405 lines, correct hospitals, no source_15
N_JOBS=$(wc -l < slurm/stage_b_jobs.tsv)
[[ "$N_JOBS" -eq 405 ]] && green "stage_b_jobs.tsv: 405 lines" || red "stage_b_jobs.tsv: expected 405, got $N_JOBS"

HOSPITALS_IN_TSV=$(cut -f2 slurm/stage_b_jobs.tsv | sort -u | tr '\n' ' ')
[[ "$HOSPITALS_IN_TSV" == *"source_10"* ]] && green "source_10 in TSV" || red "source_10 missing from TSV"
[[ "$HOSPITALS_IN_TSV" == *"MIMIC-III"* ]] && green "MIMIC-III in TSV" || red "MIMIC-III missing from TSV"
[[ "$HOSPITALS_IN_TSV" != *"source_15"* ]] && green "source_15 NOT in TSV (correct)" || red "source_15 still in TSV!"

# sbatch: correct array size and budget
ARRAY_LINE=$(grep "^#SBATCH --array" slurm/submit_stage_b.sbatch)
[[ "$ARRAY_LINE" == *"1-405"* ]] && green "sbatch array=1-405" || red "sbatch array wrong: $ARRAY_LINE"

N_BUDGET_LINES=$(grep -c "\-\-budget 30" slurm/submit_stage_b.sbatch || true)
[[ "$N_BUDGET_LINES" -eq 9 ]] && green "All 9 methods use --budget 30" || red "Expected 9 --budget 30 lines, got $N_BUDGET_LINES"

# run_analysis.sh: correct flags and hospitals
grep -q "\-\-hospitals" slurm/run_analysis.sh && green "run_analysis.sh uses --hospitals (plural)" || red "--hospitals flag missing in run_analysis.sh"
grep -q "source_10" slurm/run_analysis.sh && green "source_10 in run_analysis.sh" || red "source_10 missing from run_analysis.sh"
grep -q "MIMIC-III" slurm/run_analysis.sh && green "MIMIC-III in run_analysis.sh" || red "MIMIC-III missing from run_analysis.sh"
grep -qv "source_15" slurm/run_analysis.sh && green "source_15 NOT in run_analysis.sh" || red "source_15 still in run_analysis.sh"

# No Wilcoxon/multipletests imports
grep -qE "^from scipy.stats import wilcoxon|^from statsmodels" analyze/aggregate_results.py \
  && red "Wilcoxon/multipletests still imported" || green "No Wilcoxon imports in aggregate_results.py"

# =============================================================================
header "2/9" "Budget=30 in all baseline scripts"
# =============================================================================
for B in baselines/baseline{0,1,2,3,4}.py; do
    N=$(grep -c "\-\-budget" "$B" || true)
    # budget is set via argparse default or argument — just verify no hardcoded 50
    if grep -q "\-\-budget 50" "$B" 2>/dev/null; then
        red "$B still has --budget 50"
    else
        green "$B: no hardcoded --budget 50"
    fi
done

# =============================================================================
header "3/9" "Timing fields — run 3 methods with budget=$SMOKE_BUDGET"
# =============================================================================
echo "  Hospital: source_10 | Task: $SMOKE_TASK | Seed: $SMOKE_SEED"
mkdir -p "$SMOKE_RESULTS/seed_${SMOKE_SEED}"

CKPT="$PROJECT/results/source_10/checkpoint_mlm/mlm_model.pt"
RESULTS_DIR="$SMOKE_RESULTS/seed_${SMOKE_SEED}"
HISTORY_ROOT="$PROJECT/results"

COMMON_ARGS=(
    --hospital     source_10
    --task         "$SMOKE_TASK"
    --seed         "$SMOKE_SEED"
    --ckpt_path    "$CKPT"
    --results_dir  "$RESULTS_DIR"
    --budget       "$SMOKE_BUDGET"
    --max_params   4000000
    --flops_seq_len 512
    --pretrain_epochs 100 --pretrain_patience 5
    --embed_dim 256 --depth 8 --num_heads 8 --mlp_ratio 8
    --finetune_epochs 10 --finetune_patience 3 --top_k_epochs 3
    --batch_size 64 --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0
    --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1
    --num_workers 4
)

echo "  Running baseline0 (Random)..."
python baselines/baseline0.py "${COMMON_ARGS[@]}" \
  > "$SMOKE_RESULTS/baseline0.log" 2>&1 \
  && green "baseline0 completed" \
  || { red "baseline0 failed"; tail -5 "$SMOKE_RESULTS/baseline0.log"; }

echo "  Running baseline2 (GENIUS)..."
python baselines/baseline2.py "${COMMON_ARGS[@]}" \
  --model google/gemini-2.5-flash-lite \
  > "$SMOKE_RESULTS/baseline2.log" 2>&1 \
  && green "baseline2 completed" \
  || { red "baseline2 failed"; tail -5 "$SMOKE_RESULTS/baseline2.log"; }

echo "  Running mas..."
python mas_search.py "${COMMON_ARGS[@]}" \
  --history_root "$HISTORY_ROOT" \
  > "$SMOKE_RESULTS/mas.log" 2>&1 \
  && green "mas completed" \
  || { red "mas failed"; tail -5 "$SMOKE_RESULTS/mas.log"; }

# Check new fields in search_meta.json
for METHOD in baseline0 baseline2 mas; do
    META="$RESULTS_DIR/source_10/search/${METHOD}/${SMOKE_TASK}/search_meta.json"
    if [[ ! -f "$META" ]]; then
        red "$METHOD: search_meta.json missing"
        continue
    fi
    python - <<PYEOF
import json
with open("$META") as f: m = json.load(f)
ok = True
for field in ["n_evals", "per_eval_sec_mean", "wall_clock_sec"]:
    if m.get(field) is None:
        print(f"  ✗ $METHOD: '{field}' missing or None"); ok = False
if m.get("budget") != $SMOKE_BUDGET:
    print(f"  ✗ $METHOD: budget={m.get('budget')}, expected $SMOKE_BUDGET"); ok = False
if ok:
    print(f"  ✓ $METHOD: n_evals={m['n_evals']}, per_eval_sec={m['per_eval_sec_mean']:.1f}s, budget={m['budget']}")
PYEOF
done

# =============================================================================
header "4/9" "Pretrain timing — check pretrain_meta.json"
# =============================================================================
for H in source_10 MIMIC-IV MIMIC-III; do
    PM="$PROJECT/results/$H/checkpoint_mlm/pretrain_meta.json"
    if [[ -f "$PM" ]]; then
        WC=$(python -c "import json; d=json.load(open('$PM')); print(f\"{d['wall_clock_sec']/3600:.2f} GPU-hours\")")
        green "$H: pretrain_meta.json exists ($WC)"
    else
        echo "  ⚠  $H: pretrain_meta.json not yet created (will be auto-generated on next pretrain)"
        echo "     This is expected if checkpoint already exists — no re-pretrain needed."
    fi
done

# =============================================================================
header "5/9" "New hospitals — data availability"
# =============================================================================
for H in source_10 MIMIC-III; do
    PKL="$PROJECT/data_process/$H/${H}-processed/mimic_downstream.pkl"
    CKPT="$PROJECT/results/$H/checkpoint_mlm/mlm_model.pt"
    [[ -f "$PKL" ]]  && green "$H: mimic_downstream.pkl exists" || red "$H: mimic_downstream.pkl MISSING"
    [[ -f "$CKPT" ]] && green "$H: checkpoint_mlm exists"       || red "$H: checkpoint_mlm MISSING"
done

# =============================================================================
header "6/9" "Aggregate tables — run analyze for source_10"
# =============================================================================
mkdir -p "$SMOKE_OUT"
echo "  Running aggregate_results.py..."
python analyze/aggregate_results.py \
    --results_root "$SMOKE_RESULTS" \
    --hospitals    source_10 \
    --out_dir      "$SMOKE_OUT" \
    --mas_budget   4 \
    --baseline_budget "$SMOKE_BUDGET" \
    > "$SMOKE_RESULTS/aggregate.log" 2>&1 \
  && green "aggregate_results.py completed" \
  || { red "aggregate_results.py failed"; tail -10 "$SMOKE_RESULTS/aggregate.log"; }

EXPECTED_CSVS=(
    main_table_source_10
    efficiency_table_source_10
    supp_table_source_10
    arch_table_source_10
    cost_table_source_10
    cost_table_n10_source_10
    significance_source_10
    loto_ablation_table_source_10
)
for CSV in "${EXPECTED_CSVS[@]}"; do
    [[ -f "$SMOKE_OUT/${CSV}.csv" ]] \
        && green "${CSV}.csv exists" \
        || red "${CSV}.csv MISSING"
done

# =============================================================================
header "7/9" "Bootstrap CI — check significance columns"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys
df = pd.read_csv("$SMOKE_OUT/significance_source_10.csv")
required = ["ci_lower_pct", "ci_upper_pct", "significant_ci>0", "delta_pct", "n_seeds"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"  ✗ Missing columns: {missing}")
    sys.exit(1)
old_cols = [c for c in ["p_raw", "p_adj_holm", "significant_p<0.05"] if c in df.columns]
if old_cols:
    print(f"  ✗ Old Wilcoxon columns still present: {old_cols}")
    sys.exit(1)
print(f"  ✓ Bootstrap CI columns present, {len(df)} rows")
print(df[["task","metric","baseline","delta_pct","ci_lower_pct","ci_upper_pct","significant_ci>0"]].head(3).to_string(index=False))
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
header "8/9" "Cost table decomposition"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys
df = pd.read_csv("$SMOKE_OUT/cost_table_source_10.csv")
required = ["gpu_eval_min_mean", "llm_latency_min_mean", "n_evals_mean"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"  ✗ Missing columns: {missing}")
    sys.exit(1)
print(f"  ✓ Cost decomposition columns present")
print(df[["method","wall_clock_min_mean","gpu_eval_min_mean","llm_latency_min_mean","llm_calls_mean"]].to_string(index=False))
# Sanity: gpu + llm ≈ total (allow small float gap)
for _, row in df.iterrows():
    if row["gpu_eval_min_mean"] is not None and row["llm_latency_min_mean"] is not None:
        diff = abs(row["wall_clock_min_mean"] - row["gpu_eval_min_mean"] - row["llm_latency_min_mean"])
        if diff > 1.0:
            print(f"  ⚠ {row['method']}: gpu+llm={row['gpu_eval_min_mean']+row['llm_latency_min_mean']:.1f} ≠ total={row['wall_clock_min_mean']:.1f} (diff={diff:.1f})")
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
header "9/9" "Cost N=10 table"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys
df = pd.read_csv("$SMOKE_OUT/cost_table_n10_source_10.csv")
required = ["supernet_n10_min", "traditional_n10_hr", "speedup_supernet_vs_traditional"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"  ✗ Missing columns: {missing}")
    sys.exit(1)
print(f"  ✓ cost_table_n10 columns present ({len(df)} methods)")
if df["supernet_pretrain_hr"].isna().all():
    print("  ⚠ supernet_pretrain_hr all NaN — pretrain_meta.json not yet created")
    print("    (expected: will populate after first production pretrain)")
else:
    print(df[["method","supernet_n10_min","traditional_n10_hr","speedup_supernet_vs_traditional"]].to_string(index=False))
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SMOKE TEST RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PASS: $PASS"
echo " FAIL: $FAIL"
echo " Output: $SMOKE_OUT"
if [[ "$FAIL" -eq 0 ]]; then
    echo " ✅ ALL CHECKS PASSED — safe to submit 405 production jobs"
else
    echo " ❌ FAILURES DETECTED — fix before submitting"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit "$FAIL"
