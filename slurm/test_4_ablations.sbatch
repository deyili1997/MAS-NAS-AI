#!/bin/bash
# Quick test: 4 MAS ablation variants on source_15/death, seed=123, budget=50
#SBATCH --job-name=mas_test
#SBATCH --array=1-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=12:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/mas_test_%A_%a.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/mas_test_%A_%a.err

set -euo pipefail
mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs

module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

HOSPITAL=source_15
TASK=death
SEED=123
CKPT=/blue/mei.liu/lideyi/MAS-NAS/results/${HOSPITAL}/checkpoint_mlm/mlm_model.pt
RESULTS_DIR=/blue/mei.liu/lideyi/MAS-NAS/results/seed_${SEED}
HISTORY_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results

COMMON_ARGS=(
    --hospital $HOSPITAL
    --task $TASK
    --max_params 4000000
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
    --budget 50
)

case "$SLURM_ARRAY_TASK_ID" in
  1) METHOD=mas;              EXTRA_ARGS=() ;;
  2) METHOD=mas_layer1_only;  EXTRA_ARGS=(--no_meta_regression) ;;
  3) METHOD=mas_loto;         EXTRA_ARGS=(--exclude_exact_task_from_history) ;;
  4) METHOD=mas_cold;         EXTRA_ARGS=(--no_history) ;;
esac

date
hostname
echo "METHOD=$METHOD  HOSPITAL=$HOSPITAL  TASK=$TASK  SEED=$SEED"

python mas_search.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

date
echo "DONE: $METHOD"
