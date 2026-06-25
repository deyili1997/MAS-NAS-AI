#!/bin/bash
# =============================================================================
# Anytime test re-evaluation — one array task per line of anytime_rerun_jobs.tsv
# =============================================================================
# Workflow:
#   1. python analyze/extract_anytime_jobs.py ...   → anytime_rerun_jobs.tsv
#   2. sbatch --array=1-<N>%20 slurm/submit_retest.sbatch
#   3. python analyze/build_anytime_table.py ...     → 3-panel main table
# Header line of the TSV is row 0; array indices are 1-based DATA rows, so we
# read line ($SLURM_ARRAY_TASK_ID + 1).
# =============================================================================
#SBATCH --job-name=retest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/retest_%A_%a.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/retest_%A_%a.err
# =============================================================================

set -euo pipefail

JOBS_TSV=/blue/mei.liu/lideyi/MAS-NAS/analyze/anytime/anytime_rerun_jobs.tsv
OUT_DIR=/blue/mei.liu/lideyi/MAS-NAS/analyze/anytime/retest
CKPT_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results

# Data row = array index + 1 (skip header)
LINE_NO=$(( SLURM_ARRAY_TASK_ID + 1 ))
LINE=$(sed -n "${LINE_NO}p" "$JOBS_TSV")
if [[ -z "$LINE" ]]; then
    echo "ERROR: empty line $LINE_NO in $JOBS_TSV"; exit 1
fi

HOSPITAL=$(echo "$LINE" | cut -f1)
TASK=$(echo "$LINE" | cut -f2)
SEED=$(echo "$LINE" | cut -f3)
EMBED=$(echo "$LINE" | cut -f4)
DEPTH=$(echo "$LINE" | cut -f5)
MLP=$(echo "$LINE" | cut -f6)
HEADS=$(echo "$LINE" | cut -f7)

date; hostname
echo "Retest: $HOSPITAL/$TASK seed=$SEED arch=($EMBED,$DEPTH,$MLP,$HEADS)"

module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

python retest_arch.py \
    --hospital "$HOSPITAL" --task "$TASK" --seed "$SEED" \
    --embed_dim "$EMBED" --depth "$DEPTH" --mlp_ratio "$MLP" --num_heads "$HEADS" \
    --ckpt_path "$CKPT_ROOT/$HOSPITAL/checkpoint_mlm/mlm_model.pt" \
    --out_dir "$OUT_DIR" \
    --finetune_epochs 30 --finetune_patience 5 --top_k_epochs 3 \
    --batch_size 64 --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0 \
    --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1 \
    --num_workers 4 --cudnn_benchmark

date
