"""
TPE (Bayesian Optimization) Baseline for EHR Transformer NAS
============================================================
The canonical sample-efficient black-box optimizer: a Tree-structured Parzen
Estimator (Bergstra et al., 2011) over the scalar search space. TPE models the
config distribution of the best-so-far architectures l(x) = p(x | y is good) and
the rest g(x) = p(x | y is bad), then proposes the config maximizing the
Expected-Improvement proxy l(x) / g(x). BO shines in the low-evaluation-budget
regime, making it a strong, sample-efficient competitor at our ~30-eval budget.

Adapted to our scalar setting:
  - The search space is 4 categorical dimensions (embed_dim, depth, mlp_ratio,
    num_heads) drawn from CHOICES. TPE is implemented directly for categoricals:
    per-dimension "good"/"bad" histograms with Laplace smoothing, treated as
    independent (the standard TPE factorization).
  - `y` = validation AUPRC (the primary metric); "good" = the top-gamma quantile
    of observed configs.
  - The first `--tpe_warmup` architectures are random valid samples (TPE needs an
    initial observation set); thereafter each proposal maximizes l(x)/g(x) among
    candidates sampled from l, restricted to valid, in-budget, unvisited configs.
  - Only architectures that are actually finetuned consume --budget, exactly like
    every other baseline.

It reuses (for fair, apples-to-apples comparison):
  - The SAME pretrained EHR Transformer supernet checkpoint
  - The SAME scalar search space (CHOICES from run_pipeline)
  - The SAME finetune-based evaluation (`_finetune_one_arch`)
  - The SAME composite val-rank selection rule and final test evaluation
  - The SAME --budget semantics (total architectures finetuned)

Usage:
    python baselines/baseline6.py \\
        --hospital MIMIC-IV --task death \\
        --max_params 4000000 --budget 30 \\
        --ckpt_path /blue/mei.liu/lideyi/MAS-NAS/results/MIMIC-IV/checkpoint_mlm/mlm_model.pt
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
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
from utils.device_helpers import dataloader_kwargs, pick_device, empty_cache  # noqa: E402
from utils.task_registry import task_info, ALL_TASKS  # noqa: E402
from utils.paths import get_processed_root  # noqa: E402
from model.supernet_transformer import TransformerSuper  # noqa: E402


# The dimensions TPE models (fixed, part of the setup).
DIMS = ["embed_dim", "depth", "mlp_ratio", "num_heads"]


# ---------------------------------------------------------------------------
# Constraint / validity helpers (identical to baseline0 for comparability)
# ---------------------------------------------------------------------------

def _cand_key(cand):
    return (cand["embed_dim"], cand["depth"], cand["mlp_ratio"], cand["num_heads"])


def _random_cand():
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
    """Random valid sampler (same as baseline0). Returns (cand, internal,
    n_params, n_flops) or (None, ...) on exhaustion."""
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
# TPE acquisition — categorical, per-dimension l(x)/g(x)
# ---------------------------------------------------------------------------

def _cat_dist(configs, dim, smoothing=1.0):
    """Smoothed categorical distribution over CHOICES[dim] from a list of configs."""
    cs = CHOICES[dim]
    counts = {c: float(smoothing) for c in cs}
    for cfg in configs:
        counts[cfg[dim]] += 1.0
    total = sum(counts.values())
    return {c: counts[c] / total for c in cs}


def _tpe_suggest(observations, visited, vocab_size, max_adm, max_params, num_classes,
                 gamma=0.25, n_candidates=100, smoothing=1.0,
                 max_flops=None, flops_seq_len=512):
    """Propose the next config by maximizing the TPE EI proxy l(x)/g(x).

    observations: list of (cand_dict, reward). Splits into good (top-gamma by
    reward) and bad; builds per-dim l (good) and g (bad); samples candidates from
    l and returns the valid, unvisited one with the highest sum_d log l/g.
    Returns (cand, n_params, n_flops) or (None, None, None) if none found."""
    obs = sorted(observations, key=lambda o: -o[1])          # reward descending
    n = len(obs)
    n_good = max(1, int(np.ceil(gamma * n)))
    n_good = min(n_good, n - 1) if n > 1 else 1              # keep bad non-empty
    good = [o[0] for o in obs[:n_good]]
    bad = [o[0] for o in obs[n_good:]] or [obs[-1][0]]

    L = {d: _cat_dist(good, d, smoothing) for d in DIMS}
    G = {d: _cat_dist(bad, d, smoothing) for d in DIMS}

    best_cand, best_score = None, -np.inf
    best_np, best_nf = None, None
    for _ in range(n_candidates):
        cand = {}
        for d in DIMS:
            cs = CHOICES[d]
            cand[d] = int(np.random.choice(cs, p=[L[d][c] for c in cs]))
        if _cand_key(cand) in visited:
            continue
        ok, p_, f_ = _validate(
            cand, vocab_size, max_adm, max_params, num_classes,
            max_flops=max_flops, flops_seq_len=flops_seq_len,
        )
        if not ok:
            continue
        score = float(sum(np.log(L[d][cand[d]]) - np.log(G[d][cand[d]]) for d in DIMS))
        if score > best_score:
            best_cand, best_score, best_np, best_nf = cand, score, p_, f_
    return best_cand, best_np, best_nf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TPE Bayesian-optimization baseline for EHR Transformer NAS")
    # Target
    p.add_argument("--hospital", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=ALL_TASKS,
                   help="Binary or multilabel task. Multilabel: next_diag_*_pheno.")
    # Constraints
    p.add_argument("--max_params", type=int, required=True)
    p.add_argument("--max_flops", type=int, default=None,
                   help="Maximum subnet FLOPs (computed at --flops_seq_len reference length)")
    p.add_argument("--flops_seq_len", type=int, default=512)
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
    # TPE hyperparams
    p.add_argument("--tpe_warmup", type=int, default=8,
                   help="Random valid samples before TPE kicks in (initial design)")
    p.add_argument("--tpe_gamma", type=float, default=0.25,
                   help="Top-quantile treated as 'good' when splitting observations")
    p.add_argument("--tpe_candidates", type=int, default=100,
                   help="Candidates sampled from l(x) per proposal (EI maximization)")
    p.add_argument("--tpe_smoothing", type=float, default=1.0,
                   help="Laplace smoothing added to each category's count")
    # Common
    p.add_argument("--seed", type=int, default=123)
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
    t_start = time.perf_counter()
    eval_times_sec = []
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device()
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
    data_root = get_processed_root(args.hospital)
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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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

    num_classes = task_info(args.task)["num_classes"]

    # =========================================================================
    # TPE search loop — random warmup, then EI-guided proposals
    # =========================================================================
    search_state = {
        "completed_experiments": [],
        "budget_remaining": args.budget,
    }
    visited = set()
    observations = []   # list of (cand_dict, reward) for the TPE surrogate

    iteration = 0
    consecutive_failures = 0
    max_consecutive_failures = 3

    while search_state["budget_remaining"] > 0:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} — budget remaining: {search_state['budget_remaining']}")
        print(f"{'='*60}")

        # --- Propose next config: random warmup, then TPE EI ---
        if len(observations) < args.tpe_warmup:
            cand, _internal, n_params, n_flops = _sample_unique_valid(
                visited, vocab_size, max_adm, args.max_params, num_classes,
                max_flops=args.max_flops, flops_seq_len=args.flops_seq_len,
            )
            source = "warmup"
        else:
            cand, n_params, n_flops = _tpe_suggest(
                observations, visited, vocab_size, max_adm, args.max_params, num_classes,
                gamma=args.tpe_gamma, n_candidates=args.tpe_candidates,
                smoothing=args.tpe_smoothing,
                max_flops=args.max_flops, flops_seq_len=args.flops_seq_len,
            )
            source = "tpe"
            if cand is None:   # TPE found nothing valid/unvisited — random fallback
                cand, _internal, n_params, n_flops = _sample_unique_valid(
                    visited, vocab_size, max_adm, args.max_params, num_classes,
                    max_flops=args.max_flops, flops_seq_len=args.flops_seq_len,
                )
                source = "tpe->random"

        if cand is None:
            consecutive_failures += 1
            print(f"  Sampling exhausted ({consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print("  Search space appears exhausted, stopping early.")
                break
            continue

        consecutive_failures = 0
        visited.add(_cand_key(cand))
        internal = _to_internal_config(cand)

        print(f"  [{source}] embed_dim={cand['embed_dim']}, depth={cand['depth']}, "
              f"mlp_ratio={cand['mlp_ratio']}, num_heads={cand['num_heads']}, "
              f"params={n_params:,}, FLOPs={n_flops:,}")

        # --- Evaluate on the shared supernet (same as every baseline) ---
        _t_eval = time.perf_counter()
        avg_val, best_model_sd = _finetune_one_arch(
            internal, ckpt, train_loader, val_loader, device, args
        )
        eval_times_sec.append(time.perf_counter() - _t_eval)
        print(f"    Val: Acc={avg_val['accuracy']:.4f}  F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  AUPRC={avg_val['auprc']:.4f}")

        reward = float(avg_val["auprc"])   # primary metric drives the surrogate
        observations.append((cand, reward))

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
    # Save search results (identical layout/columns to every baseline)
    # =========================================================================
    print(f"\n{'='*60}")
    print("TPE Search Complete")
    print(f"{'='*60}")

    val_results = search_state["completed_experiments"]
    if not val_results:
        print("No experiments completed.")
        return

    df = pd.DataFrame(val_results)
    df["iteration"] = range(1, len(df) + 1)   # chronological eval order
    df["hospital"] = args.hospital
    df["task"] = args.task

    val_rank = pd.DataFrame()
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        val_rank[col] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = val_rank.mean(axis=1)
    df = df.sort_values("avg_rank").reset_index(drop=True)

    output_dir = Path(args.results_dir) / args.hospital / "search" / "baseline6" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline6_search.csv"
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
    print(f"  embed_dim={best_row_info['embed_dim']}, depth={best_row_info['depth']}, "
          f"mlp_ratio={best_row_info['mlp_ratio']}, num_heads={best_row_info['num_heads']}, "
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
    test_csv_path = output_dir / "baseline6_best.csv"
    pd.DataFrame([best_row]).to_csv(test_csv_path, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv_path}")

    # Run-level metadata for cost / fairness audit
    meta = {
        "method": "baseline6",
        "hospital": args.hospital,
        "task": args.task,
        "seed": int(args.seed),
        "budget": int(args.budget),
        "max_params": int(args.max_params),
        "ckpt_used": str(ckpt_path),
        "wall_clock_sec": time.perf_counter() - t_start,
        "llm_calls": 0,    # Bayesian optimization — no LLM
        "n_evals": len(search_state["completed_experiments"]),
        "tpe_warmup": int(args.tpe_warmup),
        "per_eval_sec_mean": float(np.mean(eval_times_sec)) if eval_times_sec else None,
    }
    with open(output_dir / "search_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved to {output_dir / 'search_meta.json'}")


if __name__ == "__main__":
    main()
