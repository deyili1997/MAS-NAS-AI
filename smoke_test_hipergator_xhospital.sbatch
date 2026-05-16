#!/bin/bash
#SBATCH --job-name=mas_nas_xhospital_smoke
#SBATCH --account=mei.liu
#SBATCH --qos=mei.liu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j-xhospital-smoke.out
#SBATCH --error=slurm-%j-xhospital-smoke.err

# ============================================================================
# HiPerGator sbatch wrapper for cross-hospital near-production smoke
#
# Target = MIMIC-III, Source = MIMIC-IV (prior)
# Scale  = budget=10, 10 ep, 10 archs   (vs Mac smoke 2/3/5)
# Estimated wall-time ≈ 8-10 hours on a single A100 → request 12h to be safe
#
# Submit:
#     sbatch smoke_test_hipergator_xhospital.sbatch
#
# Monitor:
#     squeue -u $USER
#     tail -f slurm-<jobid>-xhospital-smoke.out
#
# Cancel:
#     scancel <jobid>
# ============================================================================

set -e

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "Started at:    $(date)"
echo "Working dir:   $(pwd)"
echo "=========================================="

# Activate conda env
module purge
module load conda
conda activate Autoformer

echo "Python:        $(which python)"
echo "Python version: $(python --version)"
echo "CUDA visible:  $CUDA_VISIBLE_DEVICES"
nvidia-smi | head -25
echo "=========================================="

# cd to project root
cd /home/lideyi/MAS-NAS

# Run the smoke (logic lives in the .sh file for interactive testability)
bash smoke_test_hipergator_xhospital.sh

echo "=========================================="
echo "Finished at:   $(date)"
echo "=========================================="
