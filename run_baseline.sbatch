#!/bin/bash
#SBATCH --job-name=bl
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err

# Required env vars (via sbatch --export):
#   BASELINE  ∈ {0,1,2,3,4}
#   TASK      ∈ {death, stay, readmission, next_diag_6m_pheno, next_diag_12m_pheno}
#   SEED      ∈ {123, 456, 789, 1000, 1234}
# Optional (default MIMIC-IV):
#   HOSPITAL
HOSPITAL=${HOSPITAL:-MIMIC-IV}
BL=${BASELINE}
TASK=${TASK}
SEED=${SEED}

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
RESULTS_DIR=/blue/mei.liu/lideyi/MAS-NAS/results/seed_${SEED}
HISTORY_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results    # hospital-level INPUT root (metadata.csv, shap_analysis/, checkpoint_mlm/) — shared across seeds

COMMON_ARGS=(
    --hospital $HOSPITAL
    --task $TASK
    --max_params 2000000
    --flops_seq_len 512
    --ckpt_path $CKPT
    --results_dir $RESULTS_DIR
    --history_root $HISTORY_ROOT
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

case $BL in
  0) python baselines/baseline0.py "${COMMON_ARGS[@]}" --budget 100 ;;
  1) python baselines/baseline1.py "${COMMON_ARGS[@]}" --budget 100 \
        --population_num 8 --select_num 4 --mutation_num 4 --crossover_num 4 --m_prob 0.3 ;;
  2) python baselines/baseline2.py "${COMMON_ARGS[@]}" --budget 100 \
        --model claude-sonnet-4-5 ;;
  3) python baselines/baseline3.py "${COMMON_ARGS[@]}" --budget 100 \
        --model claude-sonnet-4-5 --temperature 0.7 \
        --n_niches 16 --random_init_frac 0.3 --mutation_prob 0.85 --temp_jitter 0.1 --top_n_select 3 ;;
  4) python baselines/baseline4.py "${COMMON_ARGS[@]}" --budget 100 \
        --model claude-sonnet-4-5 \
        --navigator_temperature 0.7 --generator_temperature 0.7 --candidates_per_iter 3 ;;
  mas) python mas_search.py "${COMMON_ARGS[@]}" --budget 100 ;;
  mas_loto) python mas_search.py "${COMMON_ARGS[@]}" --budget 100 --exclude_exact_task_from_history ;;
  mas_cold) python mas_search.py "${COMMON_ARGS[@]}" --budget 100 --no_history ;;
  mas_layer1_only) python mas_search.py "${COMMON_ARGS[@]}" --budget 100 --no_meta_regression ;;
  *) echo "Unknown baseline: $BL"; exit 1 ;;
esac

date
