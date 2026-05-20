#!/bin/bash
#SBATCH --job-name=smoke-onefl
#SBATCH --account=mei.liu
#SBATCH --qos=mei.liu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=01:30:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err

# ============================================================================
# OneFL+ smoke pipeline — verifies one hospital's pkls + new MAX_SEQ_LEN=256
# truncation + patient-level LOINC dedup work end-to-end at bs=128.
#
# Submit one job per hospital:
#   sbatch --export=ALL,HOSPITAL=source_10 --job-name=smoke-source_10 run_smoke_onefl.sbatch
#
# Or batch loop (see deploy commands in conversation).
#
# Scale: 5 archs × 5 tasks × 5 epochs ≈ 15-30 min per hospital (varies with
# pretrain dataset size; source_10 ~28K patients → ~15min,
# source_14 ~112K patients → ~30min).
#
# Output: /blue/.../results/<hospital>/metadata.csv (26 rows = 25 + header)
# ============================================================================

set -e
mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Hospital:      ${HOSPITAL:-<UNSET>}"
echo "Node:          $SLURMD_NODENAME"
echo "Started at:    $(date)"
echo "=========================================="

[ -z "$HOSPITAL" ] && { echo "ERROR: HOSPITAL env var required"; exit 1; }

module load conda
conda activate Autoformer

echo "Python:        $(which python)"
nvidia-smi | head -25
echo "=========================================="

cd /home/lideyi/MAS-NAS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python run_pipeline.py \
    --hospital ${HOSPITAL} \
    --num_archs 5 \
    --tasks death stay readmission next_diag_6m_pheno next_diag_12m_pheno \
    --pretrain_epochs 5 \
    --pretrain_patience 2 \
    --finetune_epochs 5 \
    --finetune_patience 2 \
    --batch_size 128 \
    --num_workers 4 \
    --output_dir /blue/mei.liu/lideyi/MAS-NAS/results

# Verify
META=/blue/mei.liu/lideyi/MAS-NAS/results/${HOSPITAL}/metadata.csv
if [ -f "$META" ]; then
    N=$(wc -l < "$META")
    echo "✓ $HOSPITAL metadata: $N rows (expect 26 = 25 data + 1 header)"
else
    echo "✗ $HOSPITAL metadata MISSING"
    exit 1
fi

echo "Finished at: $(date)"
