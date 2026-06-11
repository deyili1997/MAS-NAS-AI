#!/bin/bash
# =============================================================================
# Smoke test — validates 2026-05-29 changes on a real GPU node
# =============================================================================
#SBATCH --job-name=smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/smoke_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/smoke_%j.err
# =============================================================================

set -euo pipefail

module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

PROJECT=/blue/mei.liu/lideyi/MAS-NAS
SMOKE_BUDGET=5
SMOKE_SEED=42
SMOKE_TASK=death
SMOKE_RESULTS=$PROJECT/results/smoke_$SLURM_JOB_ID
SMOKE_OUT=$PROJECT/analyze/smoke_$SLURM_JOB_ID
PASS=0; FAIL=0

mkdir -p "$SMOKE_RESULTS/seed_${SMOKE_SEED}" "$SMOKE_OUT" \
         "$PROJECT/slurm_logs"

green()  { echo "  ✓ $*"; ((PASS++)) || true; }
red()    { echo "  ✗ FAIL: $*"; ((FAIL++)) || true; }
header() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
           echo "[$1] $2"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# =============================================================================
header "1/9" "Config sanity checks"
# =============================================================================
N_JOBS=$(wc -l < slurm/stage_b_jobs.tsv)
[[ "$N_JOBS" -eq 405 ]] && green "stage_b_jobs.tsv: 405 lines" || red "stage_b_jobs.tsv: got $N_JOBS"

cut -f2 slurm/stage_b_jobs.tsv | sort -u | grep -q "source_10"  && green "source_10 in TSV"   || red "source_10 missing"
cut -f2 slurm/stage_b_jobs.tsv | sort -u | grep -q "MIMIC-III"  && green "MIMIC-III in TSV"   || red "MIMIC-III missing"
cut -f2 slurm/stage_b_jobs.tsv | sort -u | grep -qv "source_15" && green "source_15 NOT in TSV" || red "source_15 still in TSV"

grep -q "^#SBATCH --array=1-405" slurm/submit_stage_b.sbatch && green "sbatch array=1-405" || red "sbatch array wrong"
N30=$(grep -c "\-\-budget 30" slurm/submit_stage_b.sbatch || true)
[[ "$N30" -eq 9 ]] && green "All 9 methods --budget 30" || red "Expected 9, got $N30"

grep -q "\-\-hospitals" slurm/run_analysis.sh    && green "--hospitals (plural) in run_analysis.sh" || red "--hospital singular found"
grep -q "source_10"    slurm/run_analysis.sh    && green "source_10 in run_analysis.sh"            || red "source_10 missing"
grep -q "MIMIC-III"    slurm/run_analysis.sh    && green "MIMIC-III in run_analysis.sh"            || red "MIMIC-III missing"
grep -qv "source_15"   slurm/run_analysis.sh    && green "source_15 NOT in run_analysis.sh"        || red "source_15 still present"

grep -qE "^from scipy.stats import wilcoxon|^from statsmodels" analyze/aggregate_results.py \
  && red "Wilcoxon/multipletests still imported" || green "No Wilcoxon imports"

# =============================================================================
header "2/9" "Budget=30 — no hardcoded --budget 50 in baselines"
# =============================================================================
for B in baselines/baseline{0,1,2,3,4}.py; do
    grep -q "\-\-budget 50" "$B" 2>/dev/null \
      && red "$B still has --budget 50" \
      || green "$B: clean"
done

# =============================================================================
header "3/9" "Timing fields — run 3 methods (budget=$SMOKE_BUDGET, GPU required)"
# =============================================================================
CKPT="$PROJECT/results/source_10/checkpoint_mlm/mlm_model.pt"
RESULTS_DIR="$SMOKE_RESULTS/seed_${SMOKE_SEED}"

COMMON_ARGS=(
    --hospital      source_10
    --task          "$SMOKE_TASK"
    --seed          "$SMOKE_SEED"
    --ckpt_path     "$CKPT"
    --results_dir   "$RESULTS_DIR"
    --budget        "$SMOKE_BUDGET"
    --max_params    4000000
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
  || { red "baseline0 failed"; tail -8 "$SMOKE_RESULTS/baseline0.log"; }

echo "  Running baseline2 (GENIUS)..."
python baselines/baseline2.py "${COMMON_ARGS[@]}" \
    --model google/gemini-2.5-flash-lite \
    > "$SMOKE_RESULTS/baseline2.log" 2>&1 \
  && green "baseline2 completed" \
  || { red "baseline2 failed"; tail -8 "$SMOKE_RESULTS/baseline2.log"; }

echo "  Running mas..."
python mas_search.py "${COMMON_ARGS[@]}" \
    --history_root "$PROJECT/results" \
    > "$SMOKE_RESULTS/mas.log" 2>&1 \
  && green "mas completed" \
  || { red "mas failed"; tail -8 "$SMOKE_RESULTS/mas.log"; }

# Verify new timing fields in search_meta.json
for METHOD in baseline0 baseline2 mas; do
    META="$RESULTS_DIR/source_10/search/${METHOD}/${SMOKE_TASK}/search_meta.json"
    if [[ ! -f "$META" ]]; then
        red "$METHOD: search_meta.json missing"; continue
    fi
    python - <<PYEOF
import json
with open("$META") as f: m = json.load(f)
ok = True
for field in ["n_evals", "per_eval_sec_mean", "wall_clock_sec"]:
    if m.get(field) is None:
        print(f"  ✗ $METHOD: '{field}' missing"); ok = False
if m.get("budget") != $SMOKE_BUDGET:
    print(f"  ✗ $METHOD: budget={m.get('budget')}, expected $SMOKE_BUDGET"); ok = False
if ok:
    print(f"  ✓ $METHOD: n_evals={m['n_evals']}, per_eval_sec={m['per_eval_sec_mean']:.1f}s, budget={m['budget']}")
PYEOF
done

# =============================================================================
header "4/9" "Pretrain timing — pretrain_meta.json"
# =============================================================================
for H in source_10 MIMIC-IV MIMIC-III; do
    PM="$PROJECT/results/$H/checkpoint_mlm/pretrain_meta.json"
    if [[ -f "$PM" ]]; then
        WC=$(python -c "import json; d=json.load(open('$PM')); print(f\"{d['wall_clock_sec']/3600:.2f} GPU-hrs\")")
        green "$H: pretrain_meta.json exists ($WC)"
    else
        echo "  ⚠  $H: pretrain_meta.json absent — will generate on next pretrain run (expected)"
    fi
done

# =============================================================================
header "5/9" "New hospitals — data + checkpoint availability"
# =============================================================================
for H in source_10 MIMIC-III; do
    [[ -f "$PROJECT/data_process/$H/${H}-processed/mimic_downstream.pkl" ]] \
      && green "$H: mimic_downstream.pkl" || red "$H: mimic_downstream.pkl MISSING"
    [[ -f "$PROJECT/results/$H/checkpoint_mlm/mlm_model.pt" ]] \
      && green "$H: checkpoint_mlm"       || red "$H: checkpoint_mlm MISSING"
done

# =============================================================================
header "6/9" "Aggregate tables — all 8 CSVs produced"
# =============================================================================
python analyze/aggregate_results.py \
    --results_root "$SMOKE_RESULTS" \
    --hospitals    source_10 \
    --out_dir      "$SMOKE_OUT" \
    --mas_budget   4 \
    --baseline_budget "$SMOKE_BUDGET" \
    > "$SMOKE_RESULTS/aggregate.log" 2>&1 \
  && green "aggregate_results.py completed" \
  || { red "aggregate_results.py failed"; tail -10 "$SMOKE_RESULTS/aggregate.log"; }

for CSV in main_table efficiency_table supp_table arch_table \
           cost_table cost_table_n10 significance loto_ablation_table; do
    [[ -f "$SMOKE_OUT/${CSV}_source_10.csv" ]] \
      && green "${CSV}_source_10.csv" \
      || red "${CSV}_source_10.csv MISSING"
done

# =============================================================================
header "7/9" "Bootstrap CI — significance columns"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys, os
f = "$SMOKE_OUT/significance_source_10.csv"
if not os.path.exists(f):
    print("  ✗ significance CSV not found"); sys.exit(1)
# Empty is expected with 1 seed — check columns via header only
with open(f) as fh: header = fh.readline().strip()
if not header:
    print("  ✓ significance CSV exists but empty (expected: 1 seed, n<2 → no CI computed)")
    sys.exit(0)
df = pd.read_csv(f)
if len(df) == 0:
    print("  ✓ significance CSV has header, 0 rows (expected with 1 seed)")
    required = ["ci_lower_pct","ci_upper_pct","significant_ci>0","delta_pct"]
    missing = [c for c in required if c not in df.columns]
    old = [c for c in ["p_raw","p_adj_holm"] if c in df.columns]
    if missing: print(f"  ✗ Missing columns: {missing}"); sys.exit(1)
    if old:     print(f"  ✗ Old Wilcoxon columns: {old}"); sys.exit(1)
    print("  ✓ Correct columns present")
    sys.exit(0)
missing = [c for c in ["ci_lower_pct","ci_upper_pct","significant_ci>0","delta_pct"] if c not in df.columns]
old    = [c for c in ["p_raw","p_adj_holm","significant_p<0.05"] if c in df.columns]
if missing: print(f"  ✗ Missing: {missing}"); sys.exit(1)
if old:     print(f"  ✗ Old columns: {old}"); sys.exit(1)
print(f"  ✓ Bootstrap CI columns OK ({len(df)} rows)")
print(df[["task","metric","baseline","delta_pct","ci_lower_pct","ci_upper_pct"]].head(3).to_string(index=False))
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
header "8/9" "Cost table decomposition"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys
try:
    df = pd.read_csv("$SMOKE_OUT/cost_table_source_10.csv")
except FileNotFoundError:
    print("  ✗ cost_table CSV not found"); sys.exit(1)
missing = [c for c in ["gpu_eval_min_mean","llm_latency_min_mean","n_evals_mean"] if c not in df.columns]
if missing: print(f"  ✗ Missing columns: {missing}"); sys.exit(1)
print("  ✓ Decomposition columns present")
print(df[["method","wall_clock_min_mean","gpu_eval_min_mean","llm_latency_min_mean","llm_calls_mean"]].to_string(index=False))
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
header "9/9" "Cost N=10 table"
# =============================================================================
python - <<PYEOF
import pandas as pd, sys
try:
    df = pd.read_csv("$SMOKE_OUT/cost_table_n10_source_10.csv")
except FileNotFoundError:
    print("  ✗ cost_table_n10 CSV not found"); sys.exit(1)
missing = [c for c in ["supernet_n10_min","traditional_n10_hr","speedup_supernet_vs_traditional"] if c not in df.columns]
if missing: print(f"  ✗ Missing columns: {missing}"); sys.exit(1)
print(f"  ✓ cost_table_n10 OK ({len(df)} methods)")
if df["supernet_pretrain_hr"].isna().all():
    print("  ⚠ supernet_pretrain_hr all NaN — pretrain_meta.json not yet created (expected)")
else:
    print(df[["method","supernet_n10_min","traditional_n10_hr","speedup_supernet_vs_traditional"]].to_string(index=False))
PYEOF
[[ $? -eq 0 ]] && ((PASS++)) || ((FAIL++))

# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " SMOKE TEST COMPLETE"
echo " PASS: $PASS  |  FAIL: $FAIL"
echo " Logs : $SMOKE_RESULTS/"
echo " Tables: $SMOKE_OUT/"
if [[ "$FAIL" -eq 0 ]]; then
    echo " ✅ ALL CHECKS PASSED — safe to submit 405 production jobs"
    echo ""
    echo " Next step:"
    echo "   sbatch slurm/submit_stage_b.sbatch"
else
    echo " ❌ $FAIL FAILURE(S) — fix before submitting production jobs"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit "$FAIL"
