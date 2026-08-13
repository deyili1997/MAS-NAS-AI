#!/bin/bash
#SBATCH --job-name=bl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=06:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err

# Required env vars (via sbatch --export):
#   BASELINE  ∈ {0,1,2,3,4,5,6}   (5=ENAS/RL, 6=TPE/BO)
#   TASK      ∈ {death, stay, readmission, next_diag_6m_pheno, next_diag_12m_pheno}
#   SEED      ∈ {123, 456, 789, 1000, 1234}
# Optional (default MIMIC-IV):
#   HOSPITAL
HOSPITAL=${HOSPITAL:-MIMIC-IV}
BL=${BASELINE}
TASK=${TASK}
SEED=${SEED}
# Search budget (evaluations). Default 30 to MATCH the reported 256 main table
# (baseline0 has 30 rows; mas fewer due to Critic dedup). The old hard-coded 50
# did NOT generate the main results — do not use it, or a new task's column
# would get more budget than the rest and be unfairly advantaged.
BUDGET=${BUDGET:-30}

date; hostname; pwd
echo "HOSPITAL=$HOSPITAL  BASELINE=$BL  TASK=$TASK  SEED=$SEED"

mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs

module load conda
conda activate Autoformer

cd /home/lideyi/MAS-NAS

# Hospital-aware ckpt path (matches utils.paths.get_checkpoint_dir output)
CKPT=/blue/mei.liu/lideyi/MAS-NAS/results/${HOSPITAL}/checkpoint_mlm/mlm_model.pt

# Per-seed results dir keeps each seed's CSVs isolated; aggregate_results.py
# globs across results/seed_*/<hospital>/search/<baseline>/<task>/.
# RESULTS_ROOT can be overridden (e.g. an isolated dir for a baseline re-tune)
# so a rerun does NOT overwrite the reported main-table results under results/.
RESULTS_ROOT=${RESULTS_ROOT:-/blue/mei.liu/lideyi/MAS-NAS/results}
RESULTS_DIR=${RESULTS_ROOT}/seed_${SEED}
# hospital-level INPUT root for mas (Layer 1 metadata.csv + Layer 2 prior). MUST
# match the grid of this run. The live results/ tree was OVERWRITTEN by the 1008
# retrain (metadata.csv now carries embed_dim 48/96/192), so a 256-grid run MUST
# point history at the untouched 256 backup:
#   sbatch --export=ALL,...,HISTORY_ROOT=/blue/.../results_orig_256grid run_baseline.sbatch
HISTORY_ROOT=${HISTORY_ROOT:-/blue/mei.liu/lideyi/MAS-NAS/results_orig_256grid}

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

# --history_root is consumed only by mas_search.py (Layer 1 / Layer 2 prior
# loading paths). Baselines 0-4 don't define this argparse flag.
# --include_layer1 hard-locks Layer 1 retrieval to the paper's four source sites
# {1,4,14,16}, regardless of what else has metadata under HISTORY_ROOT (the
# target self-excludes). Layer 2 is already fixed to the same four sites at
# prior-BUILD time (shap_analysis --hospitals 1/4/14/16), so with HISTORY_ROOT
# pointed at the 256 backup both layers use exactly {1,4,14,16}.
MAS_EXTRA=(--history_root $HISTORY_ROOT \
           --include_layer1 source_1 source_4 source_14 source_16)

case $BL in
  0) python baselines/baseline0.py "${COMMON_ARGS[@]}" --budget $BUDGET ;;
  # EA defaults to the 'archive' variant (SPOS/AutoFormer-style top-k archive
  # search) — this is what the REPORTED results use. Set EA_VARIANT=aging for
  # Aging/Regularized Evolution (Real et al. 2019); under 'aging', EA_MUT/EA_CROSS
  # are ignored (mutation-only, one child per step). Never mix the two variants'
  # numbers in one table.
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
  5) python baselines/baseline5.py "${COMMON_ARGS[@]}" --budget $BUDGET ;;   # ENAS (RL controller)
  6) python baselines/baseline6.py "${COMMON_ARGS[@]}" --budget $BUDGET ;;   # TPE (Bayesian optimization)
  mas) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET ;;
  mas_loto) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --exclude_exact_task_from_history ;;
  mas_cold) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --no_history ;;
  mas_layer1_only) python mas_search.py "${COMMON_ARGS[@]}" "${MAS_EXTRA[@]}" --budget $BUDGET --no_meta_regression ;;
  *) echo "Unknown baseline: $BL"; exit 1 ;;
esac

date
