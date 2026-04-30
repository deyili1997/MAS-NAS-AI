"""
Evolution Search Baseline for EHR Transformer NAS
===================================================
A fair-comparison baseline for MAS-NAS. It uses:
  - The SAME pretrained EHR Transformer supernet checkpoint
  - The SAME scalar search space (embed_dim, depth, mlp_ratio, num_heads)
  - The SAME finetune-based evaluation (`_finetune_one_arch` from MAS)
  - The SAME composite val-rank selection rule
  - The SAME --budget semantics (total architectures finetuned)

It removes all historical context (no SHAP, no top-k archs from similar
hospitals, no dataset similarity, no LLM agents). The searcher sees only
the val metrics of architectures it has itself evaluated.

Usage:
    # Reuse an existing MAS-produced checkpoint
    python baselines/baseline1.py \
        --hospital MIMIC-IV --task death \
        --max_params 2000000 --budget 10 \
        --ckpt_path /blue/mei.liu/lideyi/MAS-NAS/results/MIMIC-IV/checkpoint_mlm/mlm_model.pt

    # Or pretrain automatically
    python baselines/baseline1.py \
        --hospital MIMIC-IV --task death --max_params 2000000 --budget 10
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

# Make MAS-NAS importable (baselines/ now lives inside MAS-NAS/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_NAS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _MAS_NAS_DIR)

from agents.experiment_agent import (  # noqa: E402
    _finetune_one_arch,
    _to_internal_config,
    _update_best_by_composite_rank,
)
from run_pipeline import build_tokenizer, pretrain, count_subnet_params, count_subnet_flops, CHOICES  # noqa: E402
from utils.dataset import FineTuneEHRDataset, batcher  # noqa: E402
from utils.engine import evaluate  # noqa: E402
from utils.seed import set_random_seed  # noqa: E402
from utils.device_helpers import dataloader_kwargs, pick_device, empty_cache  # noqa: E402
from utils.task_registry import task_info, ALL_TASKS  # noqa: E402
from utils.paths import get_processed_root  # noqa: E402
from model.supernet_transformer import TransformerSuper  # noqa: E402


# ---------------------------------------------------------------------------
# Candidate helpers
# ---------------------------------------------------------------------------

def _cand_key(cand):
    return (cand["embed_dim"], cand["depth"], cand["mlp_ratio"], cand["num_heads"])


def _valid(cand, vocab_size, max_adm, max_params, num_classes,
           max_flops=None, flops_seq_len=512):
    """Check constraint + parameter budget + FLOPs budget."""
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


def _random_cand():
    return {
        "embed_dim": random.choice(CHOICES["embed_dim"]),
        "depth": random.choice(CHOICES["depth"]),
        "mlp_ratio": random.choice(CHOICES["mlp_ratio"]),
        "num_heads": random.choice(CHOICES["num_heads"]),
    }


def _mutate(parent, m_prob):
    child = dict(parent)
    for k in ("embed_dim", "depth", "mlp_ratio", "num_heads"):
        if random.random() < m_prob:
            child[k] = random.choice(CHOICES[k])
    return child


def _crossover(p1, p2):
    child = {}
    for k in ("embed_dim", "depth", "mlp_ratio", "num_heads"):
        child[k] = random.choice([p1[k], p2[k]])
    return child


def _sample_random_valid(visited, vocab_size, max_adm, max_params, num_classes,
                         max_flops=None, flops_seq_len=512, max_tries=500):
    for _ in range(max_tries):
        cand = _random_cand()
        if _cand_key(cand) in visited:
            continue
        ok, _, _ = _valid(cand, vocab_size, max_adm, max_params, num_classes,
                          max_flops, flops_seq_len)
        if ok:
            return cand
    return None


def _generate_children(parents, n_mut, n_cross, m_prob, visited,
                       vocab_size, max_adm, max_params, num_classes,
                       max_flops=None, flops_seq_len=512, max_tries_each=100):
    """Generate up to n_mut mutation + n_cross crossover children, all unique & valid."""
    children = []

    # Mutation
    tries = 0
    while len(children) < n_mut and tries < n_mut * max_tries_each:
        tries += 1
        parent = random.choice(parents)
        child = _mutate(parent, m_prob)
        key = _cand_key(child)
        if key in visited:
            continue
        ok, _, _ = _valid(child, vocab_size, max_adm, max_params, num_classes,
                          max_flops, flops_seq_len)
        if not ok:
            continue
        visited.add(key)
        children.append(child)

    # Crossover
    tries = 0
    while len(children) < n_mut + n_cross and tries < n_cross * max_tries_each:
        tries += 1
        if len(parents) < 2:
            break
        p1, p2 = random.sample(parents, 2)
        child = _crossover(p1, p2)
        key = _cand_key(child)
        if key in visited:
            continue
        ok, _, _ = _valid(child, vocab_size, max_adm, max_params, num_classes,
                          max_flops, flops_seq_len)
        if not ok:
            continue
        visited.add(key)
        children.append(child)

    return children


# ---------------------------------------------------------------------------
# Evaluation (reuse MAS _finetune_one_arch)
# ---------------------------------------------------------------------------

def _evaluate_candidates(cands, search_state, ckpt, train_loader, val_loader,
                         device, args, vocab_size, max_adm, max_params, num_classes, tag,
                         max_flops=None, flops_seq_len=512):
    """Finetune each candidate, append results to search_state. Respects budget."""
    for i, cand in enumerate(cands):
        if search_state["budget_remaining"] <= 0:
            print(f"  Budget exhausted, stopping {tag} evaluation.")
            break

        ok, n_params, n_flops = _valid(cand, vocab_size, max_adm, max_params, num_classes,
                                       max_flops, flops_seq_len)
        if not ok:
            print(f"  [{tag} {i+1}/{len(cands)}] INVALID, skipping: {cand}")
            continue

        internal = _to_internal_config(cand)
        print(f"\n  [{tag} {i+1}/{len(cands)}] "
              f"(budget remaining: {search_state['budget_remaining']}) "
              f"embed_dim={cand['embed_dim']}, depth={cand['depth']}, "
              f"mlp_ratio={cand['mlp_ratio']}, num_heads={cand['num_heads']}, "
              f"params={n_params:,}, FLOPs={n_flops:,}")

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


def _top_parents(search_state, k):
    """Pick top-k parent configs by composite val rank."""
    experiments = search_state["completed_experiments"]
    if not experiments:
        return []
    df = pd.DataFrame(experiments)
    ranks = pd.DataFrame()
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        ranks[col] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = ranks.mean(axis=1)
    df = df.sort_values("avg_rank").head(k)

    parents = []
    for _, row in df.iterrows():
        parents.append({
            "embed_dim": int(row["embed_dim"]),
            "depth": int(row["depth"]),
            "mlp_ratio": int(row["mlp_ratio"]),
            "num_heads": int(row["num_heads"]),
        })
    return parents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evolution baseline for EHR Transformer NAS")

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

    # Pretrained model (optional — if omitted, pretrain runs automatically)
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="./results")

    # Pretrain hyperparams (used when --ckpt_path not provided)
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

    # Evolution-specific
    p.add_argument("--population_num", type=int, default=8,
                   help="Initial random population size")
    p.add_argument("--select_num", type=int, default=4,
                   help="Number of parents kept each generation")
    p.add_argument("--mutation_num", type=int, default=4,
                   help="Number of mutation children per generation")
    p.add_argument("--crossover_num", type=int, default=4,
                   help="Number of crossover children per generation")
    p.add_argument("--m_prob", type=float, default=0.3,
                   help="Per-scalar mutation probability")

    p.add_argument("--seed", type=int, default=123)

    # GPU throughput knobs
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers (forced to 0 off-CUDA)")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="Enable cuDNN benchmark (faster, nondeterministic)")

    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.perf_counter()
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}")
    print(f"Budget: {args.budget} architectures, max_params={args.max_params:,}")
    print(f"Evolution: population={args.population_num}, select={args.select_num}, "
          f"mutation={args.mutation_num}, crossover={args.crossover_num}, "
          f"m_prob={args.m_prob}")

    # --- Load data ---
    # Data and results paths in MAS-NAS are relative to the MAS-NAS directory,
    # so chdir into it for consistency with `mas_search.py`. For a
    # user-provided `--ckpt_path`, prefer: (1) the original cwd location if it
    # exists, else (2) the MAS-NAS-relative location (matching mas_search.py).
    if args.ckpt_path is not None and not os.path.isabs(args.ckpt_path):
        orig_abs = os.path.abspath(args.ckpt_path)
        if os.path.exists(orig_abs):
            args.ckpt_path = orig_abs
        # else: leave relative so it resolves under MAS-NAS after chdir
    os.chdir(_MAS_NAS_DIR)
    print(f"Working directory: {os.getcwd()}")

    data_root = get_processed_root(args.hospital)
    full_data_path = data_root / "mimic.pkl"
    pretrain_data_path = data_root / "mimic_pretrain.pkl"
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
        args.output_dir = args.results_dir
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

    # =========================================================================
    # Evolution search loop
    # =========================================================================
    num_classes = task_info(args.task)["num_classes"]
    search_state = {
        "completed_experiments": [],
        "budget_remaining": args.budget,
    }
    visited = set()

    # ----- Generation 0: random initial population -----
    print(f"\n{'='*60}")
    print("Generation 0: random initial population")
    print(f"{'='*60}")

    init_pop = []
    init_size = min(args.population_num, args.budget)
    for _ in range(init_size):
        cand = _sample_random_valid(visited, vocab_size, max_adm, args.max_params,
                                     num_classes,
                                     args.max_flops, args.flops_seq_len)
        if cand is None:
            print("  Could not sample a valid random candidate; stopping init.")
            break
        visited.add(_cand_key(cand))
        init_pop.append(cand)

    _evaluate_candidates(init_pop, search_state, ckpt, train_loader, val_loader,
                         device, args, vocab_size, max_adm, args.max_params, num_classes,
                         tag="random", max_flops=args.max_flops, flops_seq_len=args.flops_seq_len)

    # ----- Subsequent generations: mutation + crossover -----
    generation = 0
    while search_state["budget_remaining"] > 0:
        generation += 1
        print(f"\n{'='*60}")
        print(f"Generation {generation}: mutation + crossover "
              f"(budget remaining: {search_state['budget_remaining']})")
        print(f"{'='*60}")

        parents = _top_parents(search_state, args.select_num)
        if not parents:
            print("  No parents available, stopping.")
            break
        print(f"  Parents (top-{len(parents)} by composite val rank):")
        for p in parents:
            print(f"    {p}")

        budget = search_state["budget_remaining"]
        # Split budget proportionally between mutation and crossover
        total = args.mutation_num + args.crossover_num
        if total == 0:
            print("  mutation_num + crossover_num == 0, stopping.")
            break
        n_mut = min(args.mutation_num, budget)
        n_cross = min(args.crossover_num, budget - n_mut)

        children = _generate_children(
            parents, n_mut, n_cross, args.m_prob, visited,
            vocab_size, max_adm, args.max_params, num_classes,
            args.max_flops, args.flops_seq_len,
        )

        if not children:
            print("  No new children could be generated; stopping.")
            break

        _evaluate_candidates(children, search_state, ckpt, train_loader, val_loader,
                             device, args, vocab_size, max_adm, args.max_params, num_classes,
                             tag=f"gen{generation}", max_flops=args.max_flops,
                             flops_seq_len=args.flops_seq_len)

    # =========================================================================
    # Save search results (val only)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Evolution Search Complete")
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

    # Output layout: results/<hospital>/search/baseline1/<task>/{baseline1_search,baseline1_best}.csv
    output_dir = Path(args.results_dir) / args.hospital / "search" / "baseline1" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline1_search.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} val results to {csv_path}")

    print(f"\nAll architectures (ranked by val):")
    display_cols = ["embed_dim", "depth", "mlp_ratio", "num_heads", "num_params",
                    "val_accuracy", "val_f1", "val_auroc", "val_auprc", "avg_rank"]
    print(df[display_cols].to_string(index=False))

    # =========================================================================
    # Final test evaluation: best architecture by composite val rank
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
    test_csv_path = output_dir / "baseline1_best.csv"
    pd.DataFrame([best_row]).to_csv(test_csv_path, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv_path}")

    # Run-level metadata for cost / fairness audit
    meta = {
        "method": "baseline1",
        "hospital": args.hospital,
        "task": args.task,
        "seed": int(args.seed),
        "budget": int(args.budget),
        "max_params": int(args.max_params),
        "ckpt_used": str(ckpt_path),
        "wall_clock_sec": time.perf_counter() - t_start,
        "llm_calls": 0,    # evolutionary search — no LLM
    }
    with open(output_dir / "search_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved to {output_dir / 'search_meta.json'}")


if __name__ == "__main__":
    main()
