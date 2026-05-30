#!/bin/bash
# =============================================================================
# Run run_pipeline.py for a single (hospital, task) — parallelises Stage 0.
# Submit 5 jobs in parallel (one per task), then merge with merge_metadata.py.
#
# Usage:
#   sbatch --export=ALL,HOSPITAL=source_15,TASK=death       slurm/submit_pipeline_task.sbatch
#   sbatch --export=ALL,HOSPITAL=source_15,TASK=stay        slurm/submit_pipeline_task.sbatch
#   sbatch --export=ALL,HOSPITAL=source_15,TASK=readmission slurm/submit_pipeline_task.sbatch
#   sbatch --export=ALL,HOSPITAL=source_15,TASK=next_diag_6m_pheno  slurm/submit_pipeline_task.sbatch
#   sbatch --export=ALL,HOSPITAL=source_15,TASK=next_diag_12m_pheno slurm/submit_pipeline_task.sbatch
# =============================================================================
#SBATCH --job-name=pipe-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/pipe-%x-%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/pipe-%x-%j.err
# =============================================================================

set -e

HOSPITAL=${HOSPITAL:-source_15}
TASK=${TASK:?ERROR: TASK env var required}

# Write to a task-specific temp directory to avoid conflicts
TASK_TMP=/blue/mei.liu/lideyi/MAS-NAS/results/pipeline_tmp/${HOSPITAL}_${TASK}
mkdir -p "$TASK_TMP"

date; hostname
echo "HOSPITAL=$HOSPITAL  TASK=$TASK  OUTPUT=$TASK_TMP"

module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

# dataset_summary only needs to run once — skip if already exists
if [ ! -f "/blue/mei.liu/lideyi/MAS-NAS/results/$HOSPITAL/dataset_summary.csv" ]; then
    python dataset_summary.py --hospital $HOSPITAL \
        --output_dir /blue/mei.liu/lideyi/MAS-NAS/results
fi

python run_pipeline.py \
    --hospital $HOSPITAL \
    --tasks $TASK \
    --seed 123 \
    --batch_size 64 \
    --pretrain_epochs 100 \
    --pretrain_patience 5 \
    --finetune_epochs 30 \
    --finetune_patience 5 \
    --top_k 3 \
    --lr 2e-4 \
    --weight_decay 1e-2 \
    --max_grad_norm 1.0 \
    --embed_dim 256 \
    --depth 8 \
    --num_heads 8 \
    --mlp_ratio 8 \
    --drop_rate 0.1 \
    --attn_drop_rate 0.1 \
    --drop_path_rate 0.1 \
    --num_archs 100 \
    --flops_seq_len 512 \
    --output_dir $TASK_TMP \
    --num_workers 4 \
    --cudnn_benchmark

echo "Done: $TASK_TMP/$HOSPITAL/metadata.csv"
date
