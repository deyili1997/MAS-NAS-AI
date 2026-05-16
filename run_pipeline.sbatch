#!/bin/bash
#SBATCH --job-name=pipe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err

# Hospital is passed via env var. Falls back to MIMIC-IV if not set.
#   sbatch --export=ALL,HOSPITAL=MIMIC-IV --job-name=pipe-MIMIC-IV run_pipeline.sbatch
#   sbatch --export=ALL,HOSPITAL=BIDMC    --job-name=pipe-BIDMC    run_pipeline.sbatch
HOSPITAL=${HOSPITAL:-MIMIC-IV}

date; hostname; pwd; echo "HOSPITAL=$HOSPITAL"
mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs

module load conda
conda activate Autoformer

cd /home/lideyi/MAS-NAS

# Layer 1 cosine input — must run BEFORE run_pipeline so mas_search later
# can find <hospital>/dataset_summary.csv (otherwise mas falls back to cold).
python dataset_summary.py --hospital $HOSPITAL \
    --output_dir /blue/mei.liu/lideyi/MAS-NAS/results

python run_pipeline.py \
    --hospital $HOSPITAL \
    --tasks death stay readmission next_diag_6m_pheno next_diag_12m_pheno \
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
    --output_dir /blue/mei.liu/lideyi/MAS-NAS/results \
    --num_workers 4 \
    --cudnn_benchmark

date
