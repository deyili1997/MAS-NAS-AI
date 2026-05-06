"""
Multi-Agent Neural Architecture Search (MAS)
=============================================
3-agent system for NAS on longitudinal EHR Transformer models:
  Agent 1 (Proposal):   LLM-based architecture proposal (guided by search strategy)
  Agent 2 (Critic):     LLM-based proposal review and revision
  Agent 3 (Experiment): Finetune accepted architectures + LLM-based strategy decision

Historical context (dataset similarity, top-k archs, SHAP) is gathered
directly in the orchestration layer before the search loop begins.

Usage:
    # With pretrain integrated (no --ckpt_path):
    python mas_search.py \
        --hospital MIMIC-IV --task death --max_params 2000000 --budget 10

    # With existing checkpoint:
    python mas_search.py \
        --hospital MIMIC-IV --task death --max_params 2000000 --budget 10 \
        --ckpt_path /blue/mei.liu/lideyi/MAS-NAS/results/MIMIC-IV/checkpoint_mlm/mlm_model.pt
"""

import argparse
import glob
import json
import os
import pickle
import time

from dotenv import load_dotenv
load_dotenv(override=True)

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from utils.seed import set_random_seed
from utils.dataset import FineTuneEHRDataset, batcher
from utils.engine import evaluate
from utils.device_helpers import dataloader_kwargs, pick_device
from utils.task_registry import task_info, task_time_horizon, ALL_TASKS
from utils.paths import get_processed_root
from utils.llm_counter import increment as _llm_increment, reset as _llm_reset, get as _llm_get
from run_pipeline import (
    build_tokenizer, pretrain,
    _label_entropy_for_task, _label_positive_ratio_for_task,
)
from model.supernet_transformer import TransformerSuper
from dataset_summary import summarize_dataset

from agents import proposal_agent, critic_agent, experiment_agent
from utils.tracer import Tracer, set_global_tracer


# ---------------------------------------------------------------------------
# Historical context helpers (dataset similarity, top-k archs, SHAP)
# ---------------------------------------------------------------------------

METRIC_COLS = ["accuracy", "f1", "auroc", "auprc"]

SUMMARY_FEATURE_COLS = [
    "Pretrain_num_samples", "Pretrain_diag_num", "Pretrain_med_num",
    "Pretrain_lab_num", "Pretrain_pro_num", "Pretrain_avg_num_encounters",
    "Pretrain_avg_diag_per_patient", "Pretrain_avg_med_per_patient",
    "Pretrain_avg_lab_per_patient", "Pretrain_avg_pro_per_patient",
    "Finetune_num_samples", "Finetune_diag_num", "Finetune_med_num",
    "Finetune_lab_num", "Finetune_pro_num", "Finetune_avg_num_encounters",
    "Finetune_avg_diag_per_patient", "Finetune_avg_med_per_patient",
    "Finetune_avg_lab_per_patient", "Finetune_avg_pro_per_patient",
]


# ---------------------------------------------------------------------------
# Task-similarity features (used when target task is missing from source hospital)
# ---------------------------------------------------------------------------

# Normalization constants — keep all dims in roughly comparable range so cosine
# similarity reflects feature alignment instead of being dominated by one axis.
_TASK_NUMCLASSES_LOG_NORM = np.log(20.0)   # log(20) ≈ 3.0; max num_classes ~18
_TASK_HORIZON_NORM = 365.0                  # 1 year as upper anchor

# Order of components in task_feature_vector(); used for tracer / logs.
TASK_FEATURE_COLS = [
    "is_binary", "is_multilabel",
    "num_classes_log_norm",
    "label_entropy",
    "positive_ratio",
    "time_horizon_norm",
]


def task_feature_vector(task: str, label_entropy: float, positive_ratio: float) -> np.ndarray:
    """6-D feature vector describing a task's prediction setup + label statistics.

    Components (matching `TASK_FEATURE_COLS` order):
      0 is_binary            ∈ {0, 1}
      1 is_multilabel        ∈ {0, 1}
      2 num_classes_log_norm log(num_classes)/log(20), ~0.23 (binary) … ~0.96 (18-pheno)
      3 label_entropy        passed in (binary entropy or mean per-label binary entropy)
      4 positive_ratio       passed in (binary P(y=1) or mean per-(sample,class) P=1)
      5 time_horizon_norm    days/365 (0 in-stay, ~0.25 readmission-3M, 0.5/1 pheno)

    Static dims (0,1,2,5) come from `task_info(task)`; dynamic dims (3,4) come
    from data and are passed in by caller.
    """
    info = task_info(task)
    is_binary = (info["type"] == "binary")
    return np.array([
        1.0 if is_binary else 0.0,
        0.0 if is_binary else 1.0,
        float(np.log(max(info["num_classes"], 2)) / _TASK_NUMCLASSES_LOG_NORM),
        float(label_entropy),
        float(positive_ratio),
        float(task_time_horizon(task)) / _TASK_HORIZON_NORM,
    ], dtype=float)


def _compute_target_task_label_stats(hospital: str, task: str) -> tuple:
    """Load the train split for (hospital, task) and compute (label_entropy,
    positive_ratio). Mirrors run_pipeline._load_task_split + the two label
    helpers, but standalone (no `args` needed)."""
    info = task_info(task)
    pkl_path = get_processed_root(hospital) / info["data_pkl"]
    with open(pkl_path, "rb") as f:
        train_data, _, _ = pickle.load(f)
    return (
        _label_entropy_for_task(train_data, task),
        _label_positive_ratio_for_task(train_data, task),
    )


def _source_task_feature_vector(source_task: str, hosp_df: pd.DataFrame) -> np.ndarray | None:
    """Build feature vector for a source task using metadata aggregates.

    Pulls `label_entropy` and (if present) `positive_ratio` from `hosp_df`
    rows for `source_task`. Falls back gracefully when older metadata.csv
    files lack `positive_ratio` (assumes 0.5 — neutral imbalance prior).
    Returns None if the source task is not registered (unknown semantics)."""
    if source_task not in ALL_TASKS:
        # Unregistered task — we can't trust task_type / num_classes / horizon.
        return None
    sub = hosp_df[hosp_df["task"] == source_task]
    if sub.empty:
        return None
    label_entropy = float(sub["label_entropy"].median())
    if "positive_ratio" in sub.columns and sub["positive_ratio"].notna().any():
        positive_ratio = float(sub["positive_ratio"].median())
    else:
        positive_ratio = 0.5   # backward-compat: no info → neutral
    return task_feature_vector(source_task, label_entropy, positive_ratio)


def _compute_target_summary(hospital):
    """Compute dataset summary features for the target hospital."""
    data_root = get_processed_root(hospital)
    pretrain_path = data_root / "mimic_pretrain.pkl"
    downstream_path = data_root / "mimic_downstream.pkl"

    pretrain_df = pickle.load(open(pretrain_path, "rb"))
    pretrain_stats = summarize_dataset(pretrain_df, "Pretrain")

    train_df, _, _ = pickle.load(open(downstream_path, "rb"))
    finetune_stats = summarize_dataset(train_df, "Finetune")

    return {"hospital": hospital, **pretrain_stats, **finetune_stats}


def _load_historical_summaries(results_dir):
    """Load all dataset_summary.csv files from results/*/."""
    pattern = os.path.join(results_dir, "*", "dataset_summary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _find_most_similar_hospital(target_summary, historical_df):
    """Find the most similar hospital by cosine similarity on summary features."""
    target_hospital = target_summary["hospital"]

    candidates = historical_df[historical_df["hospital"] != target_hospital].copy()
    if candidates.empty:
        candidates = historical_df.copy()

    target_vec = np.array([[target_summary.get(c, 0) for c in SUMMARY_FEATURE_COLS]])
    candidate_vecs = candidates[SUMMARY_FEATURE_COLS].fillna(0).values

    sims = cosine_similarity(target_vec, candidate_vecs)[0]
    best_idx = np.argmax(sims)
    best_hospital = candidates.iloc[best_idx]["hospital"]
    best_score = sims[best_idx]

    return best_hospital, float(best_score)


def _context_compute_avg_rank(group):
    """Rank architectures by each metric (higher=better), return avg rank."""
    ranks = pd.DataFrame()
    for col in METRIC_COLS:
        ranks[col] = group[col].rank(ascending=False, method="average")
    return ranks.mean(axis=1).values


def _get_top_k_archs(metadata_df, hospital, task, max_params, top_k,
                     target_task_features=None):
    """Get top-k architectures from the most similar hospital.

    Match logic:
      1. If `task` has metadata rows in `hospital`           → use exact task.
      2. Else fallback via cosine similarity on `task_feature_vector` between
         target task and each source task available in `hospital`. Picks the
         source task with highest similarity. Requires `target_task_features`
         to be passed in (computed once in gather_historical_context).
      3. If `target_task_features` is None (e.g. caller skipped it), degrade
         to the legacy "first available task" fallback so we never crash.
    """
    hosp_df = metadata_df[metadata_df["hospital"] == hospital].copy()
    if hosp_df.empty:
        return [], task, None

    task_df = hosp_df[hosp_df["task"] == task]
    matched_task = task
    matched_similarity: float | None = None   # None for exact match

    if task_df.empty:
        available_tasks = list(hosp_df["task"].unique())
        if len(available_tasks) == 0:
            return [], task, None

        # ---- Multi-feature task similarity (Plan A) ----
        if target_task_features is not None:
            sims = []
            for src_task in available_tasks:
                src_vec = _source_task_feature_vector(src_task, hosp_df)
                if src_vec is None:
                    continue
                cs = float(cosine_similarity(
                    target_task_features.reshape(1, -1),
                    src_vec.reshape(1, -1),
                )[0, 0])
                sims.append((src_task, cs))

            if sims:
                sims.sort(key=lambda x: x[1], reverse=True)
                matched_task, matched_similarity = sims[0]
                # Pretty-print all candidates for transparency
                ranked_str = ", ".join(f"{t}={s:.3f}" for t, s in sims)
                print(f"  [Context] Task '{task}' not found for {hospital}; "
                      f"fallback by feature-cosine → '{matched_task}' (sim={matched_similarity:.3f}). "
                      f"Candidates: {ranked_str}")
            else:
                # No registered candidate task to compare → first available
                matched_task = available_tasks[0]
                print(f"  [Context] Task '{task}' not found for {hospital}; "
                      f"no registered fallback candidates, using '{matched_task}'")
        else:
            matched_task = available_tasks[0]
            print(f"  [Context] Task '{task}' not found for {hospital}; "
                  f"no target-task features supplied, using '{matched_task}'")

        task_df = hosp_df[hosp_df["task"] == matched_task]

    if "num_params" in task_df.columns:
        task_df = task_df[task_df["num_params"] <= max_params]

    if task_df.empty:
        return [], matched_task, matched_similarity

    task_df = task_df.reset_index(drop=True)
    task_df["avg_rank"] = _context_compute_avg_rank(task_df)
    task_df = task_df.sort_values("avg_rank").head(top_k)

    archs = []
    for _, row in task_df.iterrows():
        arch = {
            "embed_dim": int(row["embed_dim"]),
            "depth": int(row["depth"]),
            "mlp_ratio": int(row["mlp_ratio"]),
            "num_heads": int(row["num_heads"]),
            "num_params": int(row["num_params"]) if "num_params" in row else None,
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
        }
        archs.append(arch)

    return archs, matched_task, matched_similarity


def _load_meta_regression_prior(history_root, task) -> dict:
    """Load Layer 2 architecture_prior.json for the matched task.

    Per-task pooled prior produced by `run_meta_regression.py` (mixed-effect
    models on signed SHAP across all source hospitals). Contains preferred /
    discouraged / interaction rules + confidence labels — for LLM agents'
    soft directional guidance.

    Path: <history_root>/meta_regression/<task>/architecture_prior.json

    Returns empty dict if file doesn't exist (Phase 1, or before Stage A.4 has
    been run). The agent prompts handle empty meta_regression_prior gracefully
    by skipping the architecture_prior section."""
    prior_path = Path(history_root) / "meta_regression" / task / "architecture_prior.json"
    if not prior_path.exists():
        print(f"  [Context] No meta-regression prior at {prior_path}")
        return {}

    import json
    with open(prior_path) as f:
        prior = json.load(f)
    print(f"  Layer 2 architecture_prior loaded ({task}):")
    print(f"    feature importance order: {prior.get('feature_importance_order', [])}")
    print(f"    preferred levels:         {prior.get('preferred_levels', {})}")
    print(f"    discouraged levels:       {prior.get('discouraged_levels', {})}")
    print(f"    confidence:               {prior.get('confidence', {})}")
    return prior


def gather_historical_context(hospital, task, max_params, history_root, top_k,
                              exclude_exact_task_from_history=False,
                              no_history=False,
                              no_meta_regression=False):
    """
    Gather historical context for the target hospital and task.

    Two-layer prior design:
      Layer 1 (dataset similarity, in this function):
        - <history_root>/<*>/dataset_summary.csv   → cosine to pick matched source hospital
        - <history_root>/<*>/metadata.csv          → top-k concrete architecture examples
        - Plan A 6-D task feature cosine fallback when exact target task missing
      Layer 2 (architecture-effect, NEW via run_meta_regression.py):
        - <history_root>/meta_regression/<task>/architecture_prior.json
          → preferred / discouraged levels + interaction rules + confidence labels
          → SOFT directional priors for LLM agents (NOT causal rules; see _caveat).

    Returns dict with:
      target_summary, similar_hospital, similarity_score,
      matched_task, matched_task_similarity, top_k_archs,
      meta_regression_prior  (Layer 2 architecture_prior.json content; {} if absent)

    Note: legacy `shap_importance` field has been removed — Layer 2 supersedes it
    with finer-grained per-level effects + interaction rules.

    Robustness-ablation knobs (Fig 5):
      - `exclude_exact_task_from_history` (bool): drop rows where the target
        task already exists in the matched source hospital's metadata, forcing
        the cosine-fallback path even when an exact match exists. Used by the
        `mas_loto` ablation method. Layer 2 stays ON (architecture_prior still loaded).
      - `no_meta_regression` (bool): keep Layer 1 retrieval ON but disable Layer 2
        (skip loading `architecture_prior.json`; return `meta_regression_prior={}`).
        Used by the `mas_layer1_only` ablation, which isolates the incremental
        contribution of Layer 2 beyond exact-task retrieval alone.
      - `no_history` (bool): bypass everything (Layer 1 AND Layer 2) and return an
        empty context dict (top_k_archs=[], meta_regression_prior={}). Used by
        `mas_cold` — both prior layers are off; pure multi-agent reasoning only.
    """
    print("\n[Historical Context]")

    if no_history:
        print("  [COLD] --no_history set; skipping historical context entirely.")
        return {
            "target_summary": {"hospital": hospital},
            "similar_hospital": None,
            "similarity_score": 0.0,
            "matched_task": task,
            "matched_task_similarity": None,
            "top_k_archs": [],
            "meta_regression_prior": {},
        }

    print(f"  Computing dataset summary for {hospital}...")
    target_summary = _compute_target_summary(hospital)
    print(f"  Target summary: {target_summary}")

    # Compute target task feature vector (Plan A: multi-feature task similarity)
    try:
        tgt_entropy, tgt_pos_ratio = _compute_target_task_label_stats(hospital, task)
        target_task_features = task_feature_vector(task, tgt_entropy, tgt_pos_ratio)
        print(f"  Target task features [{','.join(TASK_FEATURE_COLS)}] = "
              f"{np.round(target_task_features, 4).tolist()}")
    except Exception as e:
        print(f"  ⚠ Could not compute target task features ({e}); "
              f"task-similarity fallback will degrade.")
        target_task_features = None

    historical_df = _load_historical_summaries(history_root)

    if historical_df.empty:
        print("  No historical data found — starting cold (no prior experiments)")
        return {
            "target_summary": target_summary,
            "similar_hospital": None,
            "similarity_score": 0.0,
            "matched_task": task,
            "matched_task_similarity": None,
            "top_k_archs": [],
            "meta_regression_prior": {},
        }

    similar_hospital, sim_score = _find_most_similar_hospital(target_summary, historical_df)
    print(f"  Most similar hospital: {similar_hospital} (cosine sim = {sim_score:.4f})")

    if similar_hospital is None:
        return {
            "target_summary": target_summary,
            "similar_hospital": None,
            "similarity_score": 0.0,
            "matched_task": task,
            "matched_task_similarity": None,
            "top_k_archs": [],
            "meta_regression_prior": {},
        }

    metadata_pattern = os.path.join(history_root, "*", "metadata.csv")
    metadata_files = sorted(glob.glob(metadata_pattern))
    if metadata_files:
        metadata_df = pd.concat([pd.read_csv(f) for f in metadata_files], ignore_index=True)
    else:
        metadata_df = pd.DataFrame()

    # LOTO ablation: drop the (similar_hospital, target_task) rows so the
    # exact-match branch in _get_top_k_archs misses, forcing the cosine
    # task-feature fallback. This validates Plan A's robustness when no
    # historical record exists for the target task.
    if exclude_exact_task_from_history and not metadata_df.empty:
        n_before = len(metadata_df)
        metadata_df = metadata_df[~(
            (metadata_df["hospital"] == similar_hospital) &
            (metadata_df["task"] == task)
        )].copy()
        n_dropped = n_before - len(metadata_df)
        print(f"  [LOTO] Dropped {n_dropped} rows for ({similar_hospital}, {task}) "
              f"— forcing cosine fallback path")

    top_k_archs, matched_task, matched_task_similarity = _get_top_k_archs(
        metadata_df, similar_hospital, task, max_params, top_k,
        target_task_features=target_task_features,
    )
    if matched_task_similarity is None:
        match_str = f"{similar_hospital}/{matched_task} (exact match)"
    else:
        match_str = f"{similar_hospital}/{matched_task} (task-feature sim={matched_task_similarity:.4f})"
    print(f"  Retrieved {len(top_k_archs)} top architectures from {match_str}")

    # Layer 2: load architecture_prior.json (per-task pooled meta-regression)
    # Note: keyed by matched_task (not similar_hospital) — Layer 2 is hospital-agnostic
    # by design (pooled across all source hospitals when generated).
    # The `--no_meta_regression` knob (mas_layer1_only ablation) skips this load
    # so the agent prompts never see Layer 2 — Layer 1 retrieval stays untouched.
    if no_meta_regression:
        print("  [LAYER1-ONLY] --no_meta_regression set; skipping Layer 2 architecture_prior load.")
        meta_regression_prior = {}
    else:
        meta_regression_prior = _load_meta_regression_prior(history_root, matched_task)

    return {
        "target_summary": target_summary,
        "similar_hospital": similar_hospital,
        "similarity_score": sim_score,
        "matched_task": matched_task,
        "matched_task_similarity": matched_task_similarity,
        "top_k_archs": top_k_archs,
        "meta_regression_prior": meta_regression_prior,
    }


# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Multi-Agent NAS for EHR Transformers")

    # Target
    p.add_argument("--hospital", type=str, required=True, help="Target hospital name")
    p.add_argument("--task", type=str, required=True,
                   choices=ALL_TASKS,
                   help=("Downstream task. Binary: death/stay/readmission. "
                         "Multilabel (18 phenotypes): next_diag_6m_pheno, next_diag_12m_pheno."))

    # Constraints
    p.add_argument("--max_params", type=int, required=True,
                   help="Maximum subnet parameter count")
    p.add_argument("--max_flops", type=int, default=None,
                   help="Maximum subnet FLOPs (computed at --flops_seq_len reference length)")
    p.add_argument("--flops_seq_len", type=int, default=512,
                   help="Reference sequence length for FLOPs estimation")
    p.add_argument("--budget", type=int, default=20,
                   help="Total number of architectures to try")

    # Historical context
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of historical best architectures to retrieve")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="OUTPUT root for this run (search/<method>/<task>/ outputs + "
                        "<hospital>/checkpoint_mlm/ for pretrain). For seeded runs, "
                        "set this to ./results/seed_<N> so seeds don't collide.")
    p.add_argument("--history_root", type=str, default=None,
                   help="Hospital-level INPUT root for past experiments — where "
                        "<hospital>/metadata.csv, <hospital>/dataset_summary.csv, "
                        "shap_analysis/ live (always seed-independent). If unset, "
                        "defaults to --results_dir (backward-compatible single-seed mode).")

    # Robustness ablation flags (Fig 5)
    p.add_argument("--exclude_exact_task_from_history", action="store_true",
                   help="LOTO ablation: drop rows where task==target from metadata "
                        "before _get_top_k_archs, forcing the cosine-similarity "
                        "fallback path. Used by mas_loto.")
    p.add_argument("--no_meta_regression", action="store_true",
                   help="Layer-1-only ablation: keep Layer 1 retrieval (top-k archs + "
                        "dataset similarity) but disable Layer 2 (skip loading "
                        "architecture_prior.json; meta_regression_prior={}). Used by "
                        "mas_layer1_only — isolates the incremental contribution of "
                        "Layer 2 beyond exact-task retrieval alone.")
    p.add_argument("--no_history", action="store_true",
                   help="Cold-start ablation: skip historical context entirely "
                        "(empty top_k_archs, no SHAP, no similar-hospital). "
                        "Used by mas_cold.")

    # Pretrained model (optional — if omitted, pretrain runs automatically)
    p.add_argument("--ckpt_path", type=str, default=None,
                   help="Path to pretrained supernet checkpoint (if omitted, pretrain runs first)")

    # Pretrain hyperparams (used only when --ckpt_path is not provided)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_patience", type=int, default=5,
                   help="Pretrain early stopping patience")
    # Supernet max dimensions
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=8)

    # LLM settings
    p.add_argument("--model", type=str, default="claude-sonnet-4-6",
                   help="Claude model ID for LLM agents")

    # Finetune hyperparams
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--finetune_patience", type=int, default=5)
    p.add_argument("--top_k_epochs", type=int, default=3,
                   help="Average test metrics from top-k validation epochs")
    # NOTE: --batch_size is SHARED by both phases:
    #   (a) Supernet MLM pretrain — only when --ckpt_path is omitted; routed
    #       through run_pipeline.pretrain(), which reads args.batch_size.
    #   (b) Finetune (cls) — every accepted subnet's train/val/test DataLoader.
    # If reusing an existing --ckpt_path, only finetune actually consumes this.
    p.add_argument("--batch_size", type=int, default=32,
                   help="DataLoader batch size shared by pretrain (when "
                        "--ckpt_path is absent) and finetune.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=123)

    # GPU throughput knobs
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers (forced to 0 off-CUDA)")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="Enable cuDNN benchmark (faster, nondeterministic)")

    return p.parse_args()


def _compute_avg_rank(df):
    """Compute composite average rank (lower = better)."""
    ranks = pd.DataFrame()
    for col in ["accuracy", "f1", "auroc", "auprc"]:
        ranks[col] = df[col].rank(ascending=False, method="average")
    return ranks.mean(axis=1)


def _method_name(args) -> str:
    """Resolve method name (and output subdir) from ablation flags.

    Default = `mas`; ablation variants get distinct names so their outputs
    don't clobber the regular MAS run. Precedence (most-restrictive first):
      --no_history                       → `mas_cold`        (Layer 1 OFF, Layer 2 OFF)
      --exclude_exact_task_from_history  → `mas_loto`        (Layer 1 task fallback, Layer 2 ON)
      --no_meta_regression               → `mas_layer1_only` (Layer 1 ON exact, Layer 2 OFF)
      (none)                             → `mas`             (Layer 1 ON exact, Layer 2 ON)
    """
    if getattr(args, "no_history", False):
        return "mas_cold"
    if getattr(args, "exclude_exact_task_from_history", False):
        return "mas_loto"
    if getattr(args, "no_meta_regression", False):
        return "mas_layer1_only"
    return "mas"


def main():
    args = parse_args()
    # Default --history_root to --results_dir if not specified (single-seed legacy mode).
    # In production multi-seed runs, sbatch should pass --results_dir ./results/seed_<N>
    # AND --history_root ./results so per-seed outputs don't collide while hospital-level
    # inputs (metadata.csv, shap_analysis/, checkpoint_mlm/) stay shared.
    if args.history_root is None:
        args.history_root = args.results_dir
    t_start = time.perf_counter()
    _llm_reset()
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    device = pick_device()
    method_name = _method_name(args)
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}  (method={method_name})")
    print(f"Budget: {args.budget} architectures, max_params={args.max_params:,}")
    print(f"Output root: {args.results_dir}   |   History root: {args.history_root}")

    # --- Initialize I/O tracer ---
    # Output layout: results/<hospital>/search/<method>/<task>/{<method>_search.csv, <method>_best.csv, agent_io_log.txt}
    output_dir = Path(args.results_dir) / args.hospital / "search" / method_name / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(str(output_dir / "agent_io_log.txt"))
    set_global_tracer(tracer)
    tracer.log_section("RUN CONFIGURATION")
    tracer.log_kv("hospital", args.hospital)
    tracer.log_kv("task", args.task)
    tracer.log_kv("budget", args.budget)
    tracer.log_kv("max_params", args.max_params)
    tracer.log_kv("max_flops", args.max_flops)
    tracer.log_kv("model", args.model)
    tracer.log_kv("seed", args.seed)
    tracer.log_kv("finetune_epochs", args.finetune_epochs)
    tracer.log_kv("batch_size", args.batch_size)
    tracer.log_kv("lr", args.lr)

    # --- Load data ---
    data_root = get_processed_root(args.hospital)
    full_data_path = data_root / "mimic.pkl"
    pretrain_data_path = data_root / "mimic_pretrain.pkl"
    # Per-task split — binary tasks share mimic_downstream.pkl; multilabel
    # tasks each have their own pkl produced by MIMIC-IV.ipynb.
    finetune_data_path = data_root / task_info(args.task)["data_pkl"]

    special_tokens = ["[PAD]", "[CLS]", "[MASK]"]
    full_data = pickle.load(open(full_data_path, "rb"))
    tokenizer = build_tokenizer(full_data, special_tokens)
    max_adm = full_data.groupby("SUBJECT_ID")["HADM_ID"].nunique().max()
    vocab_size = len(tokenizer.vocab.id2word)
    print(f"Vocab size: {vocab_size}, max admissions: {max_adm}")

    # --- Pretrain or load checkpoint ---
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        print(f"Using existing checkpoint: {ckpt_path}")
    else:
        print("\nNo checkpoint provided — running pretrain phase first...")
        pretrain_data = pickle.load(open(pretrain_data_path, "rb"))
        # pretrain() writes <output_dir>/<hospital>/checkpoint_mlm/mlm_model.pt.
        # Use history_root (hospital-level) NOT results_dir (per-seed) so the
        # mlm checkpoint is shared across seeds — otherwise every seed re-runs
        # 50 epochs of pretrain (~100+ wasted GPU-hours across 25 mas seeds).
        args.output_dir = args.history_root
        ckpt_path = pretrain(args, tokenizer, pretrain_data, max_adm, device)

    # weights_only=True: see run_pipeline.py:~314 for ckpt schema (all metadata
    # cast to python types so the safe loader accepts them).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- Prepare finetune data ---
    train_data, val_data, test_data = pickle.load(open(finetune_data_path, "rb"))

    token_type = ["diag", "med", "lab", "pro"]
    train_dataset = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, args.task)
    val_dataset = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, args.task)
    test_dataset = FineTuneEHRDataset(test_data, tokenizer, token_type, max_adm, args.task)

    dl_kwargs = dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=batcher(tokenizer, mode="finetune"),
                              shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=batcher(tokenizer, mode="finetune"),
                            shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=batcher(tokenizer, mode="finetune"),
                             shuffle=False, **dl_kwargs)

    # --- Initialize Anthropic client ---
    import anthropic
    client = anthropic.Anthropic()

    # =============================================
    # Historical Context (runs once)
    # =============================================
    context = gather_historical_context(
        hospital=args.hospital,
        task=args.task,
        max_params=args.max_params,
        history_root=args.history_root,
        top_k=args.top_k,
        exclude_exact_task_from_history=args.exclude_exact_task_from_history,
        no_history=args.no_history,
        no_meta_regression=args.no_meta_regression,
    )

    tracer.log_section("PHASE 0 — HISTORICAL CONTEXT")
    sim_hosp = context.get("similar_hospital", "N/A")
    sim_score = context.get("similarity_score")
    sim_str = f"{sim_hosp}  (similarity = {sim_score:.4f})" if sim_score else sim_hosp
    tracer.log_kv("Similar hospital", sim_str)
    tracer.log_kv("Matched task", context.get("matched_task", args.task))
    tracer.log_arch_table(context.get("top_k_archs", []), label="Top-k historical architectures")

    # Layer 2 architecture_prior summary (replaces old Layer 1 shap_importance log)
    arch_prior = context.get("meta_regression_prior") or {}
    if arch_prior:
        tracer.log_subsection("Layer 2: Architecture Prior (meta-regression)")
        tracer.log_kv("feature_importance_order", arch_prior.get("feature_importance_order", []))
        tracer.log_kv("preferred_levels", arch_prior.get("preferred_levels", {}))
        tracer.log_kv("discouraged_levels", arch_prior.get("discouraged_levels", {}))
        tracer.log_kv("confidence", arch_prior.get("confidence", {}))
    else:
        tracer.log_kv("Layer 2 architecture_prior", "(not available — Phase 1 / no meta_regression run)")

    # =============================================
    # Search loop: Agent 1 (Proposal) -> Agent 2 (Critic) -> Agent 3 (Experiment + Strategy)
    # =============================================
    search_state = {
        "completed_experiments": [],
        "budget_remaining": args.budget,
    }

    # First round always starts with exploration
    strategy = {"strategy": "exploration", "rationale": "initial round"}

    round_num = 0
    max_consecutive_failures = 3
    consecutive_failures = 0

    while search_state["budget_remaining"] > 0:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"Round {round_num} — Budget remaining: {search_state['budget_remaining']} "
              f"— Strategy: {strategy['strategy']}")
        print(f"{'='*60}")

        # Agent 1: Propose architectures
        tracer.log_section(f"ROUND {round_num} — AGENT 1: PROPOSAL")
        tracer.log_subsection("Inputs")
        tracer.log_kv("Strategy", f"{strategy['strategy']} (\"{strategy['rationale']}\")")
        tracer.log_kv("Budget remaining", search_state["budget_remaining"])
        tracer.log_kv("Completed exps", len(search_state["completed_experiments"]))
        flops_str = Tracer._fmt_int(args.max_flops) if args.max_flops else "None"
        tracer.log_kv("Constraints", f"max_params={Tracer._fmt_int(args.max_params)}  max_flops={flops_str}")

        proposals = proposal_agent.propose(
            context=context,
            search_state=search_state,
            max_params=args.max_params,
            client=client,
            model=args.model,
            strategy=strategy,
            max_flops=args.max_flops,
        )

        tracer.log_subsection(f"Output: {len(proposals)} Proposals")
        tracer.log_archs("Proposals", proposals)

        if not proposals:
            consecutive_failures += 1
            print(f"No valid proposals generated. "
                  f"({consecutive_failures}/{max_consecutive_failures} consecutive failures)")
            if consecutive_failures >= max_consecutive_failures:
                print("Too many consecutive failures, stopping search.")
                break
            continue

        # Agent 1 ↔ Agent 2: Propose-Critique-Revise loop
        max_revision_rounds = 3
        all_accepted = []

        for revision_round in range(max_revision_rounds):
            # Agent 2: Critique
            tracer.log_section(f"ROUND {round_num} — AGENT 2: CRITIC (pass {revision_round+1}/{max_revision_rounds})")
            tracer.log_subsection("Inputs")
            tracer.log_kv("Reviewing", f"{len(proposals)} proposals")
            tracer.log_kv("Strategy", strategy["strategy"])

            accepted, rejected = critic_agent.critique(
                context=context,
                search_state=search_state,
                proposals=proposals,
                max_params=args.max_params,
                client=client,
                vocab_size=vocab_size,
                max_adm=max_adm,
                model=args.model,
                strategy=strategy,
                max_flops=args.max_flops,
                num_classes=task_info(args.task)["num_classes"],
                flops_seq_len=args.flops_seq_len,
            )

            tracer.log_subsection("Output")
            tracer.log_archs("Accepted", accepted)
            tracer.log_rejected(rejected)

            all_accepted.extend(accepted)

            if not rejected:
                print(f"  All proposals accepted (revision round {revision_round+1})")
                break

            if revision_round < max_revision_rounds - 1:
                # Agent 1: Revise rejected proposals
                tracer.log_section(f"ROUND {round_num} — AGENT 1: REVISE (pass {revision_round+1})")
                tracer.log_subsection("Inputs")
                tracer.log_rejected(rejected, label="Critiques to address")

                proposals = proposal_agent.revise(
                    context=context,
                    search_state=search_state,
                    rejected_with_critiques=rejected,
                    max_params=args.max_params,
                    client=client,
                    model=args.model,
                    strategy=strategy,
                    max_flops=args.max_flops,
                )

                tracer.log_subsection("Output: Revised Proposals")
                tracer.log_archs("Revised", proposals)

                if not proposals:
                    print(f"  Revision produced no valid proposals, stopping revision loop")
                    break
            else:
                print(f"  Max revision rounds ({max_revision_rounds}) reached, "
                      f"dropping {len(rejected)} still-rejected proposals")

        reviewed = all_accepted

        if not reviewed:
            consecutive_failures += 1
            print(f"No proposals accepted after revision. "
                  f"({consecutive_failures}/{max_consecutive_failures} consecutive failures)")
            if consecutive_failures >= max_consecutive_failures:
                print("Too many consecutive failures, stopping search.")
                break
            continue

        consecutive_failures = 0  # Reset on success

        # Agent 3: Run experiments (val only)
        tracer.log_section(f"ROUND {round_num} — AGENT 3a: EXPERIMENT")
        tracer.log_subsection("Inputs")
        tracer.log_kv("Architectures", f"{len(reviewed)} accepted proposals")
        tracer.log_kv("Budget remaining", search_state["budget_remaining"])

        n_before = len(search_state["completed_experiments"])
        search_state = experiment_agent.run_trials(
            reviewed_proposals=reviewed,
            search_state=search_state,
            ckpt=ckpt,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
            vocab_size=vocab_size,
            max_adm=max_adm,
        )

        new_results = search_state["completed_experiments"][n_before:]
        tracer.log_subsection("Results")
        tracer.log_arch_table(new_results, label="New experiment results")
        tracer.log_kv("Budget remaining", search_state["budget_remaining"])
        best = search_state.get("best_proposal")
        if best:
            tracer.log_kv("Current best", f"{Tracer._fmt_arch(best)} (avg_rank={best.get('avg_rank', '?')})")

        # Agent 3: Decide strategy for next round
        if search_state["budget_remaining"] > 0:
            tracer.log_section(f"ROUND {round_num} — AGENT 3b: STRATEGY")
            tracer.log_subsection("Inputs")
            tracer.log_kv("Completed exps", len(search_state["completed_experiments"]))
            tracer.log_kv("Budget remaining", search_state["budget_remaining"])

            strategy = experiment_agent.decide_strategy(
                context=context,
                search_state=search_state,
                client=client,
                model=args.model,
            )

            tracer.log_subsection("Decision")
            tracer.log_kv("Strategy", f"{strategy['strategy']}")
            tracer.log_kv("Rationale", f"\"{strategy.get('rationale', '')}\"")


    # =============================================
    # Save search results (val only)
    # =============================================
    print(f"\n{'='*60}")
    print("Search Complete")
    print(f"{'='*60}")

    val_results = search_state["completed_experiments"]
    if not val_results:
        print("No experiments completed.")
        return

    df = pd.DataFrame(val_results)
    df["iteration"] = range(1, len(df) + 1)   # chronological eval order
    df["hospital"] = args.hospital
    df["task"] = args.task

    # Rank by val metrics
    val_rank = pd.DataFrame()
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        val_rank[col] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = val_rank.mean(axis=1)
    df = df.sort_values("avg_rank").reset_index(drop=True)

    # Save all val results — same dir as the tracer (results/<hospital>/search/<method>/<task>/)
    output_dir = Path(args.results_dir) / args.hospital / "search" / method_name / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{method_name}_search.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} val results to {csv_path}")

    # Print all architectures ranked by val
    print(f"\nAll architectures (ranked by val):")
    display_cols = ["embed_dim", "depth", "num_params",
                    "val_accuracy", "val_f1", "val_auroc", "val_auprc", "avg_rank"]
    print(df[display_cols].to_string(index=False))

    # =============================================
    # Final test evaluation: best architecture by composite val rank
    # =============================================
    # Find the best arch by composite rank (already computed above)
    best_idx = df["avg_rank"].idxmin()
    best_row_info = df.iloc[best_idx]

    # Retrieve saved model state dict for this architecture
    best_model_sd = search_state.get("best_model_sd")
    best_config = search_state.get("best_config")

    if best_model_sd is None or best_config is None:
        print("\nNo best model found, skipping test evaluation.")
        return

    print(f"\n{'='*60}")
    print("Final Test Evaluation — Best Architecture by Composite Val Rank")
    print(f"{'='*60}")
    print(f"  embed_dim={best_row_info['embed_dim']}, "
          f"depth={best_row_info['depth']}, "
          f"params={best_row_info['num_params']:,}")

    # Build model and load best weights — num_classes routed via task registry
    info = task_info(args.task)
    model = TransformerSuper(
        num_classes=info["num_classes"],
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"],
        mlp_ratio=ckpt["mlp_ratio"],
        depth=ckpt["depth"],
        num_heads=ckpt["num_heads"],
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        pre_norm=True,
        max_adm_num=ckpt["max_adm_num"],
    ).to(device)
    model.load_state_dict(best_model_sd)

    test_metrics = evaluate(test_loader, model, device, retrain_config=best_config,
                            task_type=info["type"])

    print(f"\n  Test Results:")
    print(f"    Accuracy = {test_metrics['accuracy']:.4f}")
    print(f"    F1       = {test_metrics['f1']:.4f}")
    print(f"    AUROC    = {test_metrics['auroc']:.4f}")
    print(f"    AUPRC    = {test_metrics['auprc']:.4f}")

    tracer.log_section("FINAL TEST EVALUATION")
    best_arch_dict = {
        "embed_dim": int(best_row_info["embed_dim"]),
        "depth": int(best_row_info["depth"]),
        "mlp_ratio": int(best_row_info.get("mlp_ratio", 0)),
        "num_heads": int(best_row_info.get("num_heads", 0)),
    }
    tracer.log_kv("Best architecture", f"{Tracer._fmt_arch(best_arch_dict)}  (params={Tracer._fmt_int(int(best_row_info['num_params']))})")
    tracer.log_subsection("Test Results")
    tracer.log_kv("Accuracy", f"{test_metrics['accuracy']:.4f}")
    tracer.log_kv("F1", f"{test_metrics['f1']:.4f}")
    tracer.log_kv("AUROC", f"{test_metrics['auroc']:.4f}")
    tracer.log_kv("AUPRC", f"{test_metrics['auprc']:.4f}")

    # Append test results to CSV
    best_row = df.iloc[0].to_dict()
    best_row["test_accuracy"] = test_metrics["accuracy"]
    best_row["test_f1"] = test_metrics["f1"]
    best_row["test_auroc"] = test_metrics["auroc"]
    best_row["test_auprc"] = test_metrics["auprc"]
    test_csv_path = output_dir / f"{method_name}_best.csv"
    pd.DataFrame([best_row]).to_csv(test_csv_path, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv_path}")

    # Run-level metadata for cost / fairness audit
    meta = {
        "method": method_name,        # mas / mas_loto / mas_cold (from _method_name(args))
        "hospital": args.hospital,
        "task": args.task,
        "seed": int(args.seed),
        "budget": int(args.budget),
        "max_params": int(args.max_params),
        "ckpt_used": str(ckpt_path),
        "wall_clock_sec": time.perf_counter() - t_start,
        "llm_calls": _llm_get(),  # sum across proposal + experiment + critic agents
    }
    with open(output_dir / "search_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved to {output_dir / 'search_meta.json'}  (LLM calls: {meta['llm_calls']})")

    tracer.close()
    print(f"  Agent I/O log saved to {tracer.path}")


if __name__ == "__main__":
    main()
