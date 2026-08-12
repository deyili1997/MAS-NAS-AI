#!/bin/bash
#SBATCH --job-name=rtmed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# 10gb x 4 = 40GB/job. Single-arch finetune, but med_rec splits are larger
# (~150-200k rows), so keep a little more headroom than the 5-task retest (32G).
#SBATCH --mem-per-cpu=10gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/rtmed_%A_%a.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/rtmed_%A_%a.err
# =============================================================================
# med_rec anytime test re-evaluation — ISOLATED to results_medrec.
# One array task per line of results_medrec/anytime/anytime_rerun_jobs.tsv
# (the archs each method WOULD have selected at cutoffs 5/10/20 but whose test
# value isn't in best.csv). Re-finetunes the arch on FULL med_rec data (same as
# the budget-30 search — NOT subsampled) and evaluates on test.
#
# Sequence:
#   python analyze/extract_anytime_jobs.py \
#       --results_root /blue/mei.liu/lideyi/MAS-NAS/results_medrec \
#       --hospitals source_15 MIMIC-IV \
#       --out_dir /blue/mei.liu/lideyi/MAS-NAS/results_medrec/anytime
#   N=$(($(wc -l < /blue/mei.liu/lideyi/MAS-NAS/results_medrec/anytime/anytime_rerun_jobs.tsv)-1))
#   sbatch --array=1-$N%20 slurm/submit_retest_medrec.sbatch
#   python analyze/build_anytime_table.py \
#       --results_root /blue/mei.liu/lideyi/MAS-NAS/results_medrec \
#       --anytime_dir  /blue/mei.liu/lideyi/MAS-NAS/results_medrec/anytime \
#       --hospitals source_15 MIMIC-IV \
#       --out_dir      /blue/mei.liu/lideyi/MAS-NAS/results_medrec/anytime
#
# Every write lands under results_medrec/anytime/; the supernet ckpt is read
# read-only from the canonical results/ tree (task-agnostic MLM).
# =============================================================================
set -euo pipefail

MEDREC_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results_medrec
CANON_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results     # read-only (supernet ckpt)
JOBS_TSV=$MEDREC_ROOT/anytime/anytime_rerun_jobs.tsv
OUT_DIR=$MEDREC_ROOT/anytime/retest
mkdir -p "$OUT_DIR"

# TSV has a header line, so array index i maps to file line i+1.
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

CKPT=$CANON_ROOT/$HOSPITAL/checkpoint_mlm/mlm_model.pt
[[ -f "$CKPT" ]] || { echo "FATAL: supernet ckpt missing: $CKPT"; exit 1; }

date; hostname
echo "Retest med_rec: $HOSPITAL/$TASK seed=$SEED arch=($EMBED,$DEPTH,$MLP,$HEADS)"

module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

python retest_arch.py \
    --hospital "$HOSPITAL" --task "$TASK" --seed "$SEED" \
    --embed_dim "$EMBED" --depth "$DEPTH" --mlp_ratio "$MLP" --num_heads "$HEADS" \
    --ckpt_path "$CKPT" \
    --out_dir "$OUT_DIR" \
    --finetune_epochs 30 --finetune_patience 5 --top_k_epochs 3 \
    --batch_size 64 --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0 \
    --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1 \
    --num_workers 4 --cudnn_benchmark

date
