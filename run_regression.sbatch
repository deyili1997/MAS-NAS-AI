#!/bin/bash
#SBATCH --job-name=reg
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# 10gb x 4 = 40GB per job. Regression peaks at ~25GB; the old 16gb (64GB/job)
# was heavily over-requested and, with many jobs concurrent, exhausted the QOS
# group memory quota -> jobs killed mid-run with Reason=QOSGrpMemLimit.
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=168:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu

set -e

# Required env vars (via sbatch --export):
#   TASK      ∈ {death, stay, readmission, next_diag_6m_pheno, next_diag_12m_pheno}
# Optional (with default):
#   HOSPITAL  default MIMIC-IV
#
# regression is run ONCE per (hospital, task) — no seed sweep.
# Output: results/<hospital>/regression/<task>/{regression_panel.png, *.csv}
HOSPITAL=${HOSPITAL:-MIMIC-IV}
TASK=${TASK}

date; hostname; pwd
echo "HOSPITAL=$HOSPITAL  TASK=$TASK"

export TQDM_DISABLE=1

export PYTHONUNBUFFERED=1
module load conda
conda activate Autoformer

cd /home/lideyi/MAS-NAS

python run_regression.py \
    --hospital $HOSPITAL \
    --task $TASK \
    --k 150 \
    --seed 123 \
    --batch_size 128 \
    --pretrain_epochs 100 \
    --pretrain_patience 5 \
    --finetune_epochs 30 \
    --finetune_patience 5 \
    --top_k_epochs 3 \
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
    --flops_seq_len 512 \
    --results_dir /blue/mei.liu/lideyi/MAS-NAS/results \
    --num_workers 4 \
    --cudnn_benchmark

date
