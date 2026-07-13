#!/bin/bash
# =============================================================================
# Stage B smoke test — run ALL 9 methods once on ONE interactive node with a
# tiny budget, BEFORE launching the full 450x2 array. Verifies every method
# runs end-to-end under the new 1008 grid + new prior + param budget, and that
# MAS loads the new Layer-2 prior and excludes the contaminated hospitals.
#
# Uses the EXACT production args from submit_stage_b.sbatch (COMMON_ARGS /
# MAS_EXTRA) — only budget is tiny and output goes to a throwaway dir, so a
# green run here means the real array will dispatch identically.
#
# Usage (grab an interactive GPU node first):
#   srun --partition=hpg-turin --gres=gpu:l4:1 --cpus-per-task=4 \
#        --mem-per-cpu=10gb --time=4:00:00 --pty bash
#   module load conda; conda activate Autoformer
#   cd /home/lideyi/MAS-NAS
#   bash slurm/smoke_stage_b.sh                 # defaults: source_15/death, budget 5, 1M
#   HOSPITAL=MIMIC-IV bash slurm/smoke_stage_b.sh   # or the other test hospital
# =============================================================================
set -uo pipefail   # NOT -e: keep going past a failing method to test them all

HOSPITAL=${HOSPITAL:-source_15}
TASK=${TASK:-death}
SEED=${SEED:-123}
BUDGET=${BUDGET:-5}                 # tiny — just proves the loop runs, not quality
MAX_PARAMS=${MAX_PARAMS:-1000000}   # test the BINDING budget (hardest path)

CKPT=/blue/mei.liu/lideyi/MAS-NAS/results/${HOSPITAL}/checkpoint_mlm/mlm_model.pt
HISTORY_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results
SMOKE_DIR=/blue/mei.liu/lideyi/MAS-NAS/results/_smoke_stage_b/${HOSPITAL}/seed_${SEED}
mkdir -p "$SMOKE_DIR"

echo "Smoke: HOSPITAL=$HOSPITAL TASK=$TASK SEED=$SEED BUDGET=$BUDGET MAX_PARAMS=$MAX_PARAMS"
echo "Output → $SMOKE_DIR"
[[ -f "$CKPT" ]] || { echo "❌ supernet ckpt missing: $CKPT"; exit 1; }

# EXACT production args (mirror submit_stage_b.sbatch)
COMMON_ARGS=(
  --hospital "$HOSPITAL" --task "$TASK" --max_params "$MAX_PARAMS"
  --flops_seq_len 512 --ckpt_path "$CKPT" --results_dir "$SMOKE_DIR"
  --pretrain_epochs 100 --pretrain_patience 5
  --embed_dim 256 --depth 8 --num_heads 8 --mlp_ratio 8
  --finetune_epochs 30 --finetune_patience 5 --top_k_epochs 3
  --batch_size 64 --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0
  --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1
  --seed "$SEED" --num_workers 4 --cudnn_benchmark
)
MAS_EXTRA=(
  --history_root "$HISTORY_ROOT"
  --exclude_from_layer1 source_15 MIMIC-IV source_3 source_10 source_11
  --model anthropic/claude-haiku-4.5
)

declare -a RESULTS=()
run() {   # $1 = label, rest = command
  local name=$1; shift
  echo ""; echo "───────── $name ─────────"
  local t0=$SECONDS
  if "$@" > "$SMOKE_DIR/${name}.log" 2>&1; then
    RESULTS+=("✅ $name  ($((SECONDS - t0))s)")
    echo "✅ $name OK ($((SECONDS - t0))s)"
  else
    RESULTS+=("❌ $name  ($((SECONDS - t0))s)")
    echo "❌ $name FAILED — last 20 lines:"; tail -20 "$SMOKE_DIR/${name}.log"
  fi
}

run baseline0 python baselines/baseline0.py "${COMMON_ARGS[@]}" --budget "$BUDGET"
run baseline1 python baselines/baseline1.py "${COMMON_ARGS[@]}" --budget "$BUDGET" \
    --population_num 8 --select_num 4 --mutation_num 4 --crossover_num 4 --m_prob 0.3
run baseline2 python baselines/baseline2.py "${COMMON_ARGS[@]}" --budget "$BUDGET" \
    --model anthropic/claude-haiku-4.5
run baseline3 python baselines/baseline3.py "${COMMON_ARGS[@]}" --budget "$BUDGET" \
    --model anthropic/claude-haiku-4.5 --temperature 0.7 \
    --n_niches 16 --random_init_frac 0.3 --mutation_prob 0.85 --temp_jitter 0.1 --top_n_select 3
run baseline4 python baselines/baseline4.py "${COMMON_ARGS[@]}" --budget "$BUDGET" \
    --model anthropic/claude-haiku-4.5 \
    --navigator_temperature 0.7 --generator_temperature 0.7 --candidates_per_iter 3
run mas             python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget "$BUDGET"
run mas_loto        python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget "$BUDGET" --exclude_exact_task_from_history
run mas_cold        python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget "$BUDGET" --no_history
run mas_layer1_only python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget "$BUDGET" --no_meta_regression

echo ""; echo "══════════ SMOKE SUMMARY ══════════"
printf '%s\n' "${RESULTS[@]}"
echo "logs + search.csv under: $SMOKE_DIR"
echo "(remove with: rm -rf /blue/mei.liu/lideyi/MAS-NAS/results/_smoke_stage_b )"
