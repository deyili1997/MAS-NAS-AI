#!/bin/bash
#SBATCH --job-name=medrec
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
# 168h covers MODE=regression (k=150 from-scratch pretrains); MODE=search
# finishes in a few hours and just ends early under the same ceiling.
#SBATCH --time=168:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err
# =============================================================================
# ISOLATED drug-recommendation (med_rec) launcher.
#
# WHY THIS FILE EXISTS: every med_rec output is written under a SEPARATE root
# (results_medrec/), so a med_rec run can NEVER overwrite the finalized 5-task
# results under results/. The task is hard-locked to med_rec and the write root
# is hard-locked to results_medrec here — you cannot pass a flag that redirects
# a write into results/. The ONLY reference back to results/ is the supernet
# checkpoint, which is opened read-only.
#
# It mirrors run_baseline.sbatch's per-method args EXACTLY (same LLM backbone
# google/gemini-2.5-flash-lite, same EA archive variant, same budget/caps) so
# the med_rec column is comparable to the main table. If you retune the main
# table, mirror the change here.
#
# Required env vars (via sbatch --export):
#   MODE ∈ {search, regression}          (default: search)
#
#   MODE=search  — one (method, seed) search cell:
#     METHOD ∈ {0,1,2,3,4,mas,mas_loto,mas_cold,mas_layer1_only}
#     SEED   ∈ {111,123,456,789,999}
#     HOSPITAL (default MIMIC-IV; use source_15 for the internal target)
#   MODE=regression — one (hospital) supernet-vs-scratch panel (k=150, seed 123):
#     HOSPITAL (default MIMIC-IV)
#
# PREREQUISITES for MODE=search with the `mas*` methods (the cross-hospital
# prior): the med_rec metadata + Layer-2 prior must already exist UNDER
# results_medrec/ (built by the source sweep + shap_analysis + run_meta_regression,
# all pointed at results_medrec). Baselines 0–4 need no prior and run immediately.
#
# Examples:
#   # regression panel (the target-only Fig-3 analogue) — safe to run now:
#   sbatch --export=ALL,MODE=regression,HOSPITAL=MIMIC-IV run_medrec.sbatch
#   # one main-table search cell:
#   sbatch --export=ALL,MODE=search,METHOD=mas,SEED=123,HOSPITAL=MIMIC-IV run_medrec.sbatch
# =============================================================================
set -euo pipefail

# ---- hard-locked roots ------------------------------------------------------
CANON_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results            # read-only (supernet ckpt)
MEDREC_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results_medrec    # every write lands here
TASK=med_rec                                               # not overridable — the whole point

if [[ "$MEDREC_ROOT" == "$CANON_ROOT" ]]; then
    echo "FATAL: write root equals canonical root — refusing to run."; exit 1
fi

MODE=${MODE:-search}
HOSPITAL=${HOSPITAL:-MIMIC-IV}
SEED=${SEED:-123}

date; hostname; pwd
echo "MODE=$MODE  TASK=$TASK  HOSPITAL=$HOSPITAL  WRITE_ROOT=$MEDREC_ROOT"

mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs "$MEDREC_ROOT"
module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

# Supernet is task-agnostic (MLM) and lives under the canonical root; read-only.
CKPT=$CANON_ROOT/${HOSPITAL}/checkpoint_mlm/mlm_model.pt
[[ -f "$CKPT" ]] || { echo "FATAL: supernet ckpt missing: $CKPT"; exit 1; }

# -----------------------------------------------------------------------------
if [[ "$MODE" == "regression" ]]; then
    # Target-only supernet-vs-scratch panel; writes results_medrec/<H>/regression/med_rec/
    python run_regression.py \
        --hospital "$HOSPITAL" --task "$TASK" \
        --k 150 --seed 123 \
        --batch_size 128 \
        --pretrain_epochs 100 --pretrain_patience 5 \
        --finetune_epochs 30 --finetune_patience 5 --top_k_epochs 3 \
        --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0 \
        --embed_dim 256 --depth 8 --num_heads 8 --mlp_ratio 8 \
        --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1 \
        --flops_seq_len 512 \
        --results_dir "$MEDREC_ROOT" \
        --num_workers 4 --cudnn_benchmark
    date; echo "DONE regression: hospital=$HOSPITAL task=$TASK -> $MEDREC_ROOT"
    exit 0
fi

# -----------------------------------------------------------------------------
# MODE=search — mirror run_baseline.sbatch, but TASK and roots are locked above.
METHOD=${METHOD:?set METHOD for MODE=search}
BUDGET=${BUDGET:-30}
RESULTS_DIR=$MEDREC_ROOT/seed_${SEED}
HISTORY_ROOT=$MEDREC_ROOT     # med_rec metadata + Layer-2 prior live under the isolated root

echo "METHOD=$METHOD  SEED=$SEED  BUDGET=$BUDGET  RESULTS_DIR=$RESULTS_DIR"

COMMON_ARGS=(
    --hospital $HOSPITAL
    --task $TASK
    --max_params 4000000
    --flops_seq_len 512
    --ckpt_path $CKPT
    --results_dir $RESULTS_DIR
    --pretrain_epochs 100
    --pretrain_patience 5
    --embed_dim 256
    --depth 8
    --num_heads 8
    --mlp_ratio 8
    --finetune_epochs 30
    --finetune_patience 5
    --top_k_epochs 3
    --batch_size 64
    --lr 2e-4
    --weight_decay 1e-2
    --max_grad_norm 1.0
    --drop_rate 0.1
    --attn_drop_rate 0.1
    --drop_path_rate 0.1
    --seed $SEED
    --num_workers 4
    --cudnn_benchmark
)
MAS_EXTRA=(--history_root $HISTORY_ROOT)

case $METHOD in
  0) python baselines/baseline0.py "${COMMON_ARGS[@]}" --budget $BUDGET ;;
  1) python baselines/baseline1.py "${COMMON_ARGS[@]}" --budget $BUDGET \
        --ea_variant ${EA_VARIANT:-archive} \
        --population_num ${EA_POP:-8} --select_num ${EA_SEL:-4} \
        --mutation_num ${EA_MUT:-4} --crossover_num ${EA_CROSS:-4} \
        --m_prob ${EA_MPROB:-0.3} ;;
  2) python baselines/baseline2.py "${COMMON_ARGS[@]}" --budget $BUDGET \
        --model google/gemini-2.5-flash-lite ;;
  3) python baselines/baseline3.py "${COMMON_ARGS[@]}" --budget $BUDGET \
        --model google/gemini-2.5-flash-lite --temperature 0.7 \
        --n_niches 16 --random_init_frac 0.3 --mutation_prob 0.85 --temp_jitter 0.1 --top_n_select 3 ;;
  4) python baselines/baseline4.py "${COMMON_ARGS[@]}" --budget $BUDGET \
        --model google/gemini-2.5-flash-lite \
        --navigator_temperature 0.7 --generator_temperature 0.7 --candidates_per_iter 3 ;;
  mas) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET ;;
  mas_loto) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --exclude_exact_task_from_history ;;
  mas_cold) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --no_history ;;
  mas_layer1_only) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --no_meta_regression ;;
  *) echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

date
echo "DONE search: method=$METHOD seed=$SEED hospital=$HOSPITAL task=$TASK -> $RESULTS_DIR"
