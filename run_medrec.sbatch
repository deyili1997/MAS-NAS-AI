#!/bin/bash
#SBATCH --job-name=medrec
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lideyi@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# 12gb x 4 = 48GB/job. med_rec splits are larger (~150-200k rows) and MODE=
# regression trains from scratch, so keep more headroom than run_baseline (32G).
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
# 168h covers MODE=regression (k=150 from-scratch pretrains); MODE=search
# finishes in a few hours and just ends early under the same ceiling.
#SBATCH --time=168:00:00
#SBATCH --output=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.out
#SBATCH --error=/blue/mei.liu/lideyi/MAS-NAS/slurm_logs/%x_%j.err
# =============================================================================
# ISOLATED drug-recommendation (med_rec) launcher.
#
# WHY THIS FILE EXISTS: every med_rec output is written under a SEPARATE root
# (results_medrec/), so a med_rec run can NEVER overwrite the finalized 5-task
# results under results/. The task is hard-locked to med_rec and the write root
# is hard-locked to results_medrec here — you cannot pass a flag that redirects
# a write into results/. The ONLY reference back to results/ is the supernet
# checkpoint, which is opened read-only.
#
# It mirrors run_baseline.sbatch's per-method args EXACTLY (same LLM backbone
# google/gemini-2.5-flash-lite, same EA archive variant, same budget/caps) so
# the med_rec column is comparable to the main table. If you retune the main
# table, mirror the change here.
#
# Required env vars (via sbatch --export):
#   MODE ∈ {sweep, prior, search, regression}     (default: search)
#
# The med_rec pipeline runs in this ORDER; every stage writes ONLY to
# results_medrec/, and the supernet is reused read-only from results/:
#
#   MODE=sweep — build the med_rec NAS metadata for ONE hospital (GPU). Reuses
#     that hospital's existing supernet (asserted present, never re-pretrained)
#     and finetunes num_archs sub-paths, writing results_medrec/<H>/metadata.csv
#     and results_medrec/<H>/dataset_summary.csv. Run once per hospital that
#     should feed the prior pool, plus each target.
#       HOSPITAL (default MIMIC-IV)
#
#   MODE=prior — pool the swept metadata into the Layer-2 prior (CPU, once).
#     Runs shap_analysis + run_meta_regression over the SOURCE pool, writing
#     results_medrec/{shap_analysis,meta_regression}/med_rec/. Confirm HOSPITALS
#     matches the pool used for the 5-task prior.
#       HOSPITALS (default: the 7 source sites, targets excluded)
#
#   MODE=search — one (method, seed) search cell (GPU):
#     METHOD ∈ {0,1,2,3,4,5,6,mas,mas_loto,mas_cold,mas_layer1_only}  (5=ENAS, 6=TPE)
#     SEED   ∈ {111,123,456,789,999}
#     HOSPITAL (default MIMIC-IV; use source_15 for the internal target)
#     mas* additionally need MODE=prior to have completed. Baselines 0–4 do not.
#
#   MODE=regression — one (hospital) supernet-vs-scratch panel (k=150, seed 123):
#     HOSPITAL (default MIMIC-IV)
#
# Examples:
#   sbatch --export=ALL,MODE=regression,HOSPITAL=MIMIC-IV run_medrec.sbatch
#   sbatch --export=ALL,MODE=sweep,HOSPITAL=source_1 run_medrec.sbatch
#   sbatch --export=ALL,MODE=prior run_medrec.sbatch
#   sbatch --export=ALL,MODE=search,METHOD=mas,SEED=123,HOSPITAL=MIMIC-IV run_medrec.sbatch
# =============================================================================
set -euo pipefail

# ---- hard-locked roots ------------------------------------------------------
CANON_ROOT=/blue/mei.liu/lideyi/MAS-NAS/results            # read-only (supernet ckpt)
# Default write root is the canonical med_rec store. MEDREC_ROOT may be
# overridden via --export ONLY for isolated side analyses (e.g. the low-budget
# scratch-equivalence proxy) so a rerun can NEVER overwrite the reported
# results_medrec/ numbers. Guards below refuse results/ and require any
# override to live under an explicitly separate directory.
MEDREC_ROOT=${MEDREC_ROOT:-/blue/mei.liu/lideyi/MAS-NAS/results_medrec}
TASK=med_rec                                               # not overridable — the whole point

if [[ "$MEDREC_ROOT" == "$CANON_ROOT" ]]; then
    echo "FATAL: write root equals canonical root — refusing to run."; exit 1
fi
case "$MEDREC_ROOT" in
  /blue/mei.liu/lideyi/MAS-NAS/results/*)
    echo "FATAL: MEDREC_ROOT override points inside results/ — refusing to run."; exit 1 ;;
esac

MODE=${MODE:-search}
HOSPITAL=${HOSPITAL:-MIMIC-IV}
SEED=${SEED:-123}

date; hostname; pwd
echo "MODE=$MODE  TASK=$TASK  HOSPITAL=$HOSPITAL  WRITE_ROOT=$MEDREC_ROOT"

mkdir -p /blue/mei.liu/lideyi/MAS-NAS/slurm_logs "$MEDREC_ROOT"
module load conda
conda activate Autoformer
cd /home/lideyi/MAS-NAS

# Supernet is task-agnostic (MLM) and lives under the canonical root; read-only.
CKPT=$CANON_ROOT/${HOSPITAL}/checkpoint_mlm/mlm_model.pt

# -----------------------------------------------------------------------------
if [[ "$MODE" == "sweep" ]]; then
    # Build med_rec NAS metadata for ONE hospital. run_pipeline resolves the
    # supernet via get_checkpoint_dir() = results/<H>/checkpoint_mlm/ and REUSES
    # it if present (returns before pretraining). We assert it exists so a
    # missing ckpt can never trigger a fresh pretrain that would WRITE into
    # results/<H>/checkpoint_mlm/. Metadata + dataset_summary go to results_medrec.
    [[ -f "$CKPT" ]] || { echo "FATAL: supernet ckpt missing ($CKPT) — refusing to run sweep (would pretrain into results/)."; exit 1; }

    python dataset_summary.py --hospital "$HOSPITAL" --output_dir "$MEDREC_ROOT"

    # med_rec's eligible-visit splits are ~150-200k rows (vs ~20k for the other
    # tasks); each epoch evaluates val AND test, so a full 100-arch sweep blows
    # the wall clock. Cap to FT_SUBSAMPLE (default 20k, ~the other tasks' scale)
    # with a fixed seed → every arch is ranked on the same rows, prior unchanged.
    python run_pipeline.py \
        --hospital "$HOSPITAL" --tasks "$TASK" \
        --seed 123 --batch_size 64 \
        --pretrain_epochs 100 --pretrain_patience 5 \
        --finetune_epochs 30 --finetune_patience 5 --top_k 3 \
        --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0 \
        --embed_dim 256 --depth 8 --num_heads 8 --mlp_ratio 8 \
        --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1 \
        --num_archs 100 --flops_seq_len 512 \
        --finetune_subsample ${FT_SUBSAMPLE:-20000} \
        --output_dir "$MEDREC_ROOT" \
        --num_workers 4 --cudnn_benchmark
    date; echo "DONE sweep: hospital=$HOSPITAL task=$TASK -> $MEDREC_ROOT/$HOSPITAL/metadata.csv"
    exit 0
fi

# -----------------------------------------------------------------------------
if [[ "$MODE" == "prior" ]]; then
    # Pool the swept med_rec metadata into the Layer-2 prior. CPU-only, run once.
    # Default pool = source_1/4/14/16. CONFIRMED: this is exactly the hospital
    # set in every results/shap_analysis/<task>/shap_values.csv, i.e. the SHAP
    # --hospitals used to build the 5-task prior. It is deliberately NOT the same
    # as the Layer-1 metadata pool (all 8 OneFlorida sites + MIMIC-IV, from which
    # retrieval matched {1,3,10,11,14,16}); the two are different mechanisms, so
    # do not "reconcile" them by widening this.
    HOSPITALS=${HOSPITALS:-source_1 source_4 source_14 source_16}
    echo "prior pool HOSPITALS=$HOSPITALS"

    python shap_analysis.py \
        --results_dir "$MEDREC_ROOT" \
        --hospitals $HOSPITALS

    python run_meta_regression.py \
        --results_dir "$MEDREC_ROOT" \
        --tasks "$TASK"
    date; echo "DONE prior: pool=[$HOSPITALS] -> $MEDREC_ROOT/meta_regression/$TASK/architecture_prior.json"
    exit 0
fi

# -----------------------------------------------------------------------------
if [[ "$MODE" == "regression" ]]; then
    [[ -f "$CKPT" ]] || { echo "FATAL: supernet ckpt missing: $CKPT"; exit 1; }
    # Target-only supernet-vs-scratch panel; writes results_medrec/<H>/regression/med_rec/
    python run_regression.py \
        --hospital "$HOSPITAL" --task "$TASK" \
        --k 150 --seed 123 \
        --batch_size 128 \
        --pretrain_epochs 100 --pretrain_patience 5 \
        --finetune_epochs 30 --finetune_patience 5 --top_k_epochs 3 \
        --lr 2e-4 --weight_decay 1e-2 --max_grad_norm 1.0 \
        --embed_dim 256 --depth 8 --num_heads 8 --mlp_ratio 8 \
        --drop_rate 0.1 --attn_drop_rate 0.1 --drop_path_rate 0.1 \
        --flops_seq_len 512 \
        --results_dir "$MEDREC_ROOT" \
        --num_workers 4 --cudnn_benchmark
    date; echo "DONE regression: hospital=$HOSPITAL task=$TASK -> $MEDREC_ROOT"
    exit 0
fi

# -----------------------------------------------------------------------------
# MODE=search — mirror run_baseline.sbatch, but TASK and roots are locked above.
[[ -f "$CKPT" ]] || { echo "FATAL: supernet ckpt missing: $CKPT"; exit 1; }
METHOD=${METHOD:?set METHOD for MODE=search}
BUDGET=${BUDGET:-30}
RESULTS_DIR=$MEDREC_ROOT/seed_${SEED}
HISTORY_ROOT=$MEDREC_ROOT     # med_rec metadata + Layer-2 prior live under the isolated root

echo "METHOD=$METHOD  SEED=$SEED  BUDGET=$BUDGET  RESULTS_DIR=$RESULTS_DIR"

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
# Hard-lock Layer 1 retrieval to the fixed source pool {1,4,14,16}, independent
# of whatever has metadata under results_medrec (the target self-excludes).
MAS_EXTRA=(--history_root $HISTORY_ROOT \
           --include_layer1 source_1 source_4 source_14 source_16)

case $METHOD in
  0) python baselines/baseline0.py "${COMMON_ARGS[@]}" --budget $BUDGET ;;
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
  *) echo "Unknown METHOD: $METHOD"; exit 1 ;;
esac

date
echo "DONE search: method=$METHOD seed=$SEED hospital=$HOSPITAL task=$TASK -> $RESULTS_DIR"
