"""
Random Search Baseline for EHR Transformer NAS
================================================
The simplest possible NAS baseline. For each of `--budget` iterations:
  1. Uniformly sample a scalar config (embed_dim, depth, mlp_ratio, num_heads)
     from CHOICES, rejecting duplicates and configs that violate the
     `embed_dim % num_heads == 0` constraint or the param/FLOPs budget.
  2. Finetune the sampled architecture on the SAME pretrained EHR Transformer
     supernet using `_finetune_one_arch` (val-only, top-k epoch averaging).
  3. Record validation metrics.

After the search, the architecture with the best composite val rank is
re-built and evaluated on the held-out test set.

This baseline removes ALL search intelligence — no LLM, no evolution, no
quality-diversity, no historical context. It establishes a floor: any
"smart" baseline must beat random sampling under the same compute budget
to claim that its search procedure is contributing value.

It uses:
  - The SAME pretrained EHR Transformer supernet checkpoint
  - The SAME scalar search space (CHOICES from run_pipeline)
  - The SAME finetune-based evaluation (`_finetune_one_arch`)
  - The SAME composite val-rank selection rule and final test evaluation
  - The SAME --budget semantics (total architectures finetuned)

Usage:
    # Reuse an existing MAS-produced checkpoint
    python baselines/baseline0.py \\
        --hospital MIMIC-IV --task death \\
        --max_params 1000000 --budget 10 \\
        --ckpt_path results/MIMIC-IV/checkpoint_mlm/mlm_model.pt

    # Or pretrain automatically
    python baselines/baseline0.py \\
        --hospital MIMIC-IV --task death --max_params 1000000 --budget 10
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make MAS-NAS importable (baselines/ lives inside MAS-NAS/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_NAS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _MAS_NAS_DIR)

from agents.experiment_agent import (  # noqa: E402
    _finetune_one_arch,
    _to_internal_config,
    _update_best_by_composite_rank,
)
from run_pipeline import (  # noqa: E402
    build_tokenizer, pretrain, count_subnet_params, count_subnet_flops, CHOICES,
)
from utils.dataset import FineTuneEHRDataset, batcher  # noqa: E402
from utils.engine import evaluate  # noqa: E402
from utils.seed import set_random_seed  # noqa: E402
from utils.device_helpers import dataloader_kwargs  # noqa: E402
from utils.task_registry import task_info, ALL_TASKS  # noqa: E402
from model.supernet_transformer import TransformerSuper  # noqa: E402


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _cand_key(cand):
    return (cand["embed_dim"], cand["depth"], cand["mlp_ratio"], cand["num_heads"])


def _random_cand():
    """Uniform sample one scalar config from CHOICES (no constraint check yet)."""
    return {
        "embed_dim": random.choice(CHOICES["embed_dim"]),
        "depth": random.choice(CHOICES["depth"]),
        "mlp_ratio": random.choice(CHOICES["mlp_ratio"]),
        "num_heads": random.choice(CHOICES["num_heads"]),
    }


def _validate(cand, vocab_size, max_adm, max_params, num_classes,
              max_flops=None, flops_seq_len=512):
    """Constraint + param budget + FLOPs budget. Returns (ok, n_params, n_flops)."""
    if cand["embed_dim"] % cand["num_heads"] != 0:
        return False, 0, 0
    internal = _to_internal_config(cand)
    n_params = count_subnet_params(internal, vocab_size, num_classes=num_classes, max_adm=max_adm)
    if n_params > max_params:
        return False, n_params, 0
    n_flops = count_subnet_flops(internal, flops_seq_len)
    if max_flops is not None and n_flops > max_flops:
        return False, n_params, n_flops
    return True, n_params, n_flops


def _sample_unique_valid(visited, vocab_size, max_adm, max_params, num_classes,
                         max_flops=None, flops_seq_len=512, max_attempts=200):
    """
    Sample uniformly until we hit a valid, in-budget, unvisited config.
    Returns (cand, internal, n_params, n_flops) or (None, ..., ..., ...)
    if exhaustion (search space too tightly constrained).
    """
    for _ in range(max_attempts):
        cand = _random_cand()
        key = _cand_key(cand)
        if key in visited:
            continue
        ok, n_params, n_flops = _validate(
            cand, vocab_size, max_adm, max_params, num_classes,
            max_flops=max_flops, flops_seq_len=flops_seq_len,
        )
        if ok:
            internal = _to_internal_config(cand)
            return cand, internal, n_params, n_flops
    return None, None, None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Random search baseline for EHR Transformer NAS")
    # Target
    p.add_argument("--hospital", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=ALL_TASKS,
                   help="Binary or multilabel task. Multilabel: next_diag_*_pheno.")
    # Constraints
    p.add_argument("--max_params", type=int, required=True)
    p.add_argument("--max_flops", type=int, default=None,
                   help="Maximum subnet FLOPs (computed at --flops_seq_len reference length)")
    p.add_argument("--flops_seq_len", type=int, default=512,
                   help="Reference sequence length for FLOPs estimation")
    p.add_argument("--budget", type=int, default=20,
                   help="Total number of architectures to finetune")
    # Pretrained model
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="./results")
    # Pretrain hyperparams (used only if ckpt_path absent)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_patience", type=int, default=5)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=8)
    # Finetune hyperparams
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--finetune_patience", type=int, default=5)
    p.add_argument("--top_k_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    # Common
    p.add_argument("--seed", type=int, default=123)
    # GPU throughput knobs
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers (forced to 0 off-CUDA)")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="Enable cuDNN benchmark (faster, nondeterministic)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}")
    print(f"Budget: {args.budget}, max_params={args.max_params:,}")

    # Resolve ckpt_path before chdir
    if args.ckpt_path is not None and not os.path.isabs(args.ckpt_path):
        orig_abs = os.path.abspath(args.ckpt_path)
        if os.path.exists(orig_abs):
            args.ckpt_path = orig_abs
    os.chdir(_MAS_NAS_DIR)
    print(f"Working directory: {os.getcwd()}")

    # --- Data ---
    data_root = Path(f"./data_process/{args.hospital}/{args.hospital}-processed")
    full_data = pickle.load(open(data_root / "mimic.pkl", "rb"))
    tokenizer = build_tokenizer(full_data, ["[PAD]", "[CLS]", "[MASK]"])
    max_adm = full_data.groupby("SUBJECT_ID")["HADM_ID"].nunique().max()
    vocab_size = len(tokenizer.vocab.id2word)
    print(f"Vocab size: {vocab_size}, max admissions: {max_adm}")

    # --- Pretrain or load checkpoint ---
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        print(f"Using existing checkpoint: {ckpt_path}")
    else:
        print("\nNo checkpoint provided — running pretrain phase first...")
        pretrain_data = pickle.load(open(data_root / "mimic_pretrain.pkl", "rb"))
        args.output_dir = args.results_dir
        ckpt_path = pretrain(args, tokenizer, pretrain_data, max_adm, device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- Finetune loaders ---
    train_data, val_data, test_data = pickle.load(open(data_root / task_info(args.task)["data_pkl"], "rb"))
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

    # =========================================================================
    # Random-search loop
    # =========================================================================
    search_state = {
        "completed_experiments": [],
        "budget_remaining": args.budget,
    }
    visited = set()

    iteration = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    while search_state["budget_remaining"] > 0:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} — budget remaining: {search_state['budget_remaining']}")
        print(f"{'='*60}")

        cand, internal, n_params, n_flops = _sample_unique_valid(
            visited, vocab_size, max_adm, args.max_params,
            task_info(args.task)["num_classes"],
            max_flops=args.max_flops, flops_seq_len=args.flops_seq_len,
        )

        if cand is None:
            consecutive_failures += 1
            print(f"  Sampling exhausted — could not find a valid unvisited config "
                  f"({consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print("  Search space appears exhausted, stopping early.")
                break
            continue

        consecutive_failures = 0
        visited.add(_cand_key(cand))

        print(f"  Sampled: embed_dim={cand['embed_dim']}, depth={cand['depth']}, "
              f"mlp_ratio={cand['mlp_ratio']}, num_heads={cand['num_heads']}, "
              f"params={n_params:,}, FLOPs={n_flops:,}")

        # Finetune (shared with all other baselines)
        avg_val, best_model_sd = _finetune_one_arch(
            internal, ckpt, train_loader, val_loader, device, args
        )
        print(f"    Val: Acc={avg_val['accuracy']:.4f}  F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  AUPRC={avg_val['auprc']:.4f}")

        val_result = {
            "embed_dim": cand["embed_dim"],
            "depth": cand["depth"],
            "mlp_ratio": cand["mlp_ratio"],
            "num_heads": cand["num_heads"],
            "num_params": n_params,
            "flops": n_flops,
            "val_accuracy": avg_val["accuracy"],
            "val_f1": avg_val["f1"],
            "val_auroc": avg_val["auroc"],
            "val_auprc": avg_val["auprc"],
        }
        search_state["completed_experiments"].append(val_result)
        search_state.setdefault("_model_sds", []).append(best_model_sd)
        search_state.setdefault("_configs", []).append(internal)
        search_state["budget_remaining"] -= 1

        _update_best_by_composite_rank(search_state)

    # =========================================================================
    # Save search results
    # =========================================================================
    print(f"\n{'='*60}")
    print("Random Search Complete")
    print(f"{'='*60}")

    val_results = search_state["completed_experiments"]
    if not val_results:
        print("No experiments completed.")
        return

    df = pd.DataFrame(val_results)
    df["hospital"] = args.hospital
    df["task"] = args.task

    val_rank = pd.DataFrame()
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        val_rank[col] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = val_rank.mean(axis=1)
    df = df.sort_values("avg_rank").reset_index(drop=True)

    output_dir = Path(args.results_dir) / args.hospital
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"baseline0_search_{args.task}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} val results to {csv_path}")

    print(f"\nAll architectures (ranked by composite val rank):")
    display_cols = ["embed_dim", "depth", "mlp_ratio", "num_heads", "num_params",
                    "val_accuracy", "val_f1", "val_auroc", "val_auprc", "avg_rank"]
    print(df[display_cols].to_string(index=False))

    # =========================================================================
    # Final test evaluation: best by composite val rank
    # =========================================================================
    best_row_info = df.iloc[0]
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
          f"mlp_ratio={best_row_info['mlp_ratio']}, "
          f"num_heads={best_row_info['num_heads']}, "
          f"params={best_row_info['num_params']:,}")

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

    best_row = df.iloc[0].to_dict()
    best_row["test_accuracy"] = test_metrics["accuracy"]
    best_row["test_f1"] = test_metrics["f1"]
    best_row["test_auroc"] = test_metrics["auroc"]
    best_row["test_auprc"] = test_metrics["auprc"]
    test_csv_path = output_dir / f"baseline0_best_{args.task}.csv"
    pd.DataFrame([best_row]).to_csv(test_csv_path, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv_path}")


if __name__ == "__main__":
    main()
