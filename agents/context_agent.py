"""
Agent 1: Historical Context (Algorithmic)
==========================================
Gathers historical experiment data to guide architecture search:
1. Compute target hospital's dataset summary
2. Find most similar hospital via cosine similarity
3. Retrieve top-k architectures from that hospital
4. Load SHAP feature importance
"""

import ast
import glob
import os
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Reuse summarize_dataset from dataset_summary.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_summary import summarize_dataset


METRIC_COLS = ["accuracy", "f1", "auroc", "auprc"]

# The 20 numeric feature columns from dataset_summary
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


def _compute_target_summary(hospital):
    """Compute dataset summary features for the target hospital."""
    data_root = Path(f"./data_process/{hospital}/{hospital}-processed")
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

    # Prefer a different hospital; if only the target exists, use it (self-match)
    candidates = historical_df[historical_df["hospital"] != target_hospital].copy()
    if candidates.empty:
        candidates = historical_df.copy()

    # Build feature vectors
    target_vec = np.array([[target_summary.get(c, 0) for c in SUMMARY_FEATURE_COLS]])
    candidate_vecs = candidates[SUMMARY_FEATURE_COLS].fillna(0).values

    # Cosine similarity
    sims = cosine_similarity(target_vec, candidate_vecs)[0]
    best_idx = np.argmax(sims)
    best_hospital = candidates.iloc[best_idx]["hospital"]
    best_score = sims[best_idx]

    return best_hospital, float(best_score)


def _compute_avg_rank(group):
    """Rank architectures by each metric (higher=better, rank 1=best), return avg rank."""
    ranks = pd.DataFrame()
    for col in METRIC_COLS:
        ranks[col] = group[col].rank(ascending=False, method="average")
    return ranks.mean(axis=1).values


def _get_top_k_archs(metadata_df, hospital, task, max_params, top_k):
    """
    Get top-k architectures from the most similar hospital.
    Falls back to closest task by label_entropy if exact task not found.
    """
    hosp_df = metadata_df[metadata_df["hospital"] == hospital].copy()
    if hosp_df.empty:
        return [], task

    # Try exact task match first
    task_df = hosp_df[hosp_df["task"] == task]
    matched_task = task

    if task_df.empty:
        # Fallback: find task with closest label_entropy
        available_tasks = hosp_df["task"].unique()
        if len(available_tasks) == 0:
            return [], task

        # Use median label_entropy per task as proxy
        task_entropies = hosp_df.groupby("task")["label_entropy"].median()

        # Get target task's typical entropy (use 0.5 as default if unknown)
        target_entropy = 0.5
        if "label_entropy" in hosp_df.columns:
            # Pick the task with closest entropy
            diffs = (task_entropies - target_entropy).abs()
            matched_task = diffs.idxmin()
        else:
            matched_task = available_tasks[0]

        task_df = hosp_df[hosp_df["task"] == matched_task]
        print(f"  [Context] Task '{task}' not found for {hospital}, "
              f"falling back to '{matched_task}'")

    # Filter by max_params
    if "num_params" in task_df.columns:
        task_df = task_df[task_df["num_params"] <= max_params]

    if task_df.empty:
        return [], matched_task

    # Compute composite rank and sort
    task_df = task_df.reset_index(drop=True)
    task_df["avg_rank"] = _compute_avg_rank(task_df)
    task_df = task_df.sort_values("avg_rank").head(top_k)

    # Extract config + metrics from historical experiments
    archs = []
    for _, row in task_df.iterrows():
        mlp_ratio = row["mlp_ratio"]
        num_heads = row["num_heads"]
        # CSV stores lists as strings — parse them back
        if isinstance(mlp_ratio, str):
            mlp_ratio = ast.literal_eval(mlp_ratio)
        if isinstance(num_heads, str):
            num_heads = ast.literal_eval(num_heads)

        arch = {
            "embed_dim": int(row["embed_dim"]),
            "depth": int(row["depth"]),
            "mlp_ratio": mlp_ratio,
            "num_heads": num_heads,
            "num_params": int(row["num_params"]) if "num_params" in row else None,
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
            "auroc": float(row["auroc"]),
            "auprc": float(row["auprc"]),
        }
        archs.append(arch)

    return archs, matched_task


def _load_shap_importance(results_dir, hospital, task):
    """Load mean |SHAP| values for the given hospital+task."""
    shap_path = Path(results_dir) / "shap_analysis" / hospital / task / "mean_abs_shap.csv"
    if not shap_path.exists():
        print(f"  [Context] No SHAP data at {shap_path}")
        return {}

    df = pd.read_csv(shap_path, index_col=0, header=None)
    return {str(k): float(v) for k, v in zip(df.index, df.iloc[:, 0])}


def run(hospital, task, max_params, results_dir, top_k):
    """
    Gather historical context for the target hospital and task.

    Returns:
        dict with target_summary, similar_hospital, similarity_score,
        matched_task, top_k_archs, shap_importance
    """
    print("\n[Agent 1: Historical Context]")

    # Step 1: Compute target hospital dataset features
    print(f"  Computing dataset summary for {hospital}...")
    target_summary = _compute_target_summary(hospital)
    print(f"  Target summary: {target_summary}")

    # Step 2: Find most similar hospital
    historical_df = _load_historical_summaries(results_dir)

    if historical_df.empty:
        print("  No historical data found — starting cold (no prior experiments)")
        return {
            "target_summary": target_summary,
            "similar_hospital": None,
            "similarity_score": 0.0,
            "matched_task": task,
            "top_k_archs": [],
            "shap_importance": {},
        }

    similar_hospital, sim_score = _find_most_similar_hospital(target_summary, historical_df)
    print(f"  Most similar hospital: {similar_hospital} (cosine sim = {sim_score:.4f})")

    if similar_hospital is None:
        return {
            "target_summary": target_summary,
            "similar_hospital": None,
            "similarity_score": 0.0,
            "matched_task": task,
            "top_k_archs": [],
            "shap_importance": {},
        }

    # Step 3: Load metadata and get top-k architectures
    metadata_pattern = os.path.join(results_dir, "*", "metadata.csv")
    metadata_files = sorted(glob.glob(metadata_pattern))
    if metadata_files:
        metadata_df = pd.concat([pd.read_csv(f) for f in metadata_files], ignore_index=True)
    else:
        metadata_df = pd.DataFrame()

    top_k_archs, matched_task = _get_top_k_archs(
        metadata_df, similar_hospital, task, max_params, top_k
    )
    print(f"  Retrieved {len(top_k_archs)} top architectures from {similar_hospital}/{matched_task}")

    # Step 4: Load SHAP importance
    shap_importance = _load_shap_importance(results_dir, similar_hospital, matched_task)
    if shap_importance:
        print(f"  SHAP importance: {shap_importance}")
    else:
        print("  No SHAP data available")

    return {
        "target_summary": target_summary,
        "similar_hospital": similar_hospital,
        "similarity_score": sim_score,
        "matched_task": matched_task,
        "top_k_archs": top_k_archs,
        "shap_importance": shap_importance,
    }
