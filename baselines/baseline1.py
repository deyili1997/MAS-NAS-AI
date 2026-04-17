"""
Single-LLM Baseline for EHR Transformer NAS
=============================================
A fair-comparison baseline for MAS-NAS. Uses a single LLM call per iteration
(no multi-agent propose/critique/revise). Each iteration:
  1. Build a prompt containing ALL previously evaluated architectures + their
     validation metrics.
  2. Ask the LLM to propose exactly ONE new architecture.
  3. Finetune the architecture on the SAME pretrained supernet and evaluate it.
  4. Append the result and repeat until `--budget` is exhausted.

It uses:
  - The SAME pretrained EHR Transformer supernet checkpoint
  - The SAME scalar search space (embed_dim, depth, mlp_ratio, num_heads)
  - The SAME finetune-based evaluation (`_finetune_one_arch` from MAS)
  - The SAME composite val-rank selection rule and final test evaluation
  - The SAME --budget semantics (total architectures finetuned)

It removes all historical context (no SHAP, no top-k archs from similar
hospitals, no dataset similarity, no multi-agent debate). The LLM sees only
the architectures it has itself evaluated in this run.

Usage:
    python baselines/baseline1.py \
        --hospital MIMIC-IV --task death \
        --max_params 1000000 --budget 10 \
        --ckpt_path results/MIMIC-IV/checkpoint_mlm/mlm_model.pt
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Make MAS-NAS importable (baselines/ now lives inside MAS-NAS/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_NAS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _MAS_NAS_DIR)

# Load .env from several likely locations (current cwd, MAS-NAS/, baselines/)
# so ANTHROPIC_API_KEY is found regardless of where the user runs from.
for _env_dir in (os.getcwd(), _MAS_NAS_DIR, _THIS_DIR):
    _env_path = os.path.join(_env_dir, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
        print(f"Loaded .env from {_env_path}")
        break

from agents.experiment_agent import (  # noqa: E402
    _finetune_one_arch,
    _to_internal_config,
    _update_best_by_composite_rank,
)
from run_pipeline import build_tokenizer, pretrain, count_subnet_params, count_subnet_flops, CHOICES  # noqa: E402
from utils.dataset import FineTuneEHRDataset, batcher  # noqa: E402
from utils.engine import evaluate  # noqa: E402
from utils.seed import set_random_seed  # noqa: E402
from model.supernet_transformer import TransformerSuper  # noqa: E402


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _build_prompt(search_state, max_params, hospital, task, max_flops=None):
    """Build the single-LLM prompt. No historical context — only past experiments in this run."""
    parts = []

    parts.append(
        "You are an expert neural architecture search agent for Transformer models "
        "applied to longitudinal Electronic Health Record (EHR) data. Your job is to "
        "propose ONE architecture that will perform well on the target task.\n"
    )

    # Target task
    parts.append(f"\n## Target\n")
    parts.append(f"Hospital: {hospital}\n")
    parts.append(f"Task: {task} (binary classification)\n")

    # Search space
    parts.append("\n## Search Space\n")
    parts.append(f"Available choices: {json.dumps(CHOICES)}\n")
    parts.append(
        "Each architecture is defined by four scalar hyperparameters:\n"
        "- embed_dim: one value from [64, 128, 256]\n"
        "- depth: number of transformer layers, from [2, 4, 8]\n"
        "- mlp_ratio: one value from [2, 4, 8], constant across all layers\n"
        "- num_heads: one value from [2, 4, 8], constant across all layers\n\n"
        "CONSTRAINT: embed_dim must be divisible by num_heads.\n"
    )

    # Parameter budget
    parts.append(f"\n## Parameter Budget\n")
    parts.append(f"Maximum allowed parameters: {max_params:,}\n")
    if max_flops is not None:
        parts.append(f"Maximum allowed FLOPs: {max_flops:,}\n")

    # Already tried architectures (val metrics only)
    completed = search_state.get("completed_experiments", [])
    if completed:
        parts.append(f"\n## Previously Evaluated Architectures ({len(completed)})\n")
        parts.append("All metrics below are on the VALIDATION set.\n\n")
        for i, exp in enumerate(completed):
            parts.append(f"  [{i+1}] {json.dumps(exp)}\n")
        parts.append(
            "\nLearn from these results. Avoid proposing an architecture that is "
            "identical to any above. Use the trend in val metrics to guide your next "
            "proposal — balance exploration (covering new regions of the space) and "
            "exploitation (refining around the best performers).\n"
        )
    else:
        parts.append(
            "\n## Previously Evaluated Architectures\n"
            "None yet — this is the first iteration. Propose a reasonable starting "
            "architecture to begin exploration.\n"
        )

    # Budget remaining
    budget_remaining = search_state.get("budget_remaining", 0)
    parts.append(f"\n## Budget Remaining: {budget_remaining} architectures\n")

    # Output format
    parts.append(
        "\n## Output Format\n"
        "Return a JSON object for a SINGLE architecture:\n"
        "```json\n"
        "{\n"
        '  "embed_dim": 128,\n'
        '  "depth": 4,\n'
        '  "mlp_ratio": 4,\n'
        '  "num_heads": 4,\n'
        '  "rationale": "Brief explanation of why this architecture"\n'
        "}\n"
        "```\n"
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )

    return "".join(parts)


def _parse_proposal(response_text):
    """Parse LLM response into a single proposal dict."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { / last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
        else:
            raise

    # If LLM returned a list, take the first element
    if isinstance(obj, list):
        if not obj:
            raise ValueError("LLM returned empty list")
        obj = obj[0]
    return obj


def _validate_proposal(proposal):
    embed_dim = proposal.get("embed_dim")
    depth = proposal.get("depth")
    mlp_ratio = proposal.get("mlp_ratio")
    num_heads = proposal.get("num_heads")

    if embed_dim not in CHOICES["embed_dim"]:
        return False, f"embed_dim {embed_dim} not in {CHOICES['embed_dim']}"
    if depth not in CHOICES["depth"]:
        return False, f"depth {depth} not in {CHOICES['depth']}"
    if mlp_ratio not in CHOICES["mlp_ratio"]:
        return False, f"mlp_ratio {mlp_ratio} not in {CHOICES['mlp_ratio']}"
    if num_heads not in CHOICES["num_heads"]:
        return False, f"num_heads {num_heads} not in {CHOICES['num_heads']}"
    if embed_dim % num_heads != 0:
        return False, f"embed_dim {embed_dim} not divisible by num_heads={num_heads}"
    return True, "ok"


def _call_llm(prompt, client, model, max_retries=5):
    """Call Claude with exponential backoff on API errors."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _cand_key(cand):
    return (cand["embed_dim"], cand["depth"], cand["mlp_ratio"], cand["num_heads"])


def _propose_one(search_state, max_params, client, model, hospital, task,
                 vocab_size, max_adm, visited, max_flops=None, flops_seq_len=512,
                 max_attempts=5):
    """
    Ask the LLM for one valid, unique, in-budget architecture. Retries up to
    max_attempts with extra feedback on invalid / duplicate / over-budget.
    """
    base_prompt = _build_prompt(search_state, max_params, hospital, task, max_flops=max_flops)
    extra_feedback = ""

    for attempt in range(max_attempts):
        prompt = base_prompt + extra_feedback
        print(f"\n  [LLM call] attempt {attempt+1}/{max_attempts}")
        response_text = _call_llm(prompt, client, model)

        try:
            prop = _parse_proposal(response_text)
        except Exception as e:
            print(f"  Parse error: {e}. Raw: {response_text[:300]}")
            extra_feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Your previous response could not be parsed as JSON. "
                f"Return ONLY a single JSON object, no preamble.\n"
            )
            continue

        ok, msg = _validate_proposal(prop)
        if not ok:
            print(f"  INVALID: {msg}")
            extra_feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Your previous proposal {json.dumps(prop)} was invalid: {msg}. "
                f"Please fix this issue.\n"
            )
            continue

        key = _cand_key(prop)
        if key in visited:
            print(f"  DUPLICATE of an already-evaluated architecture")
            extra_feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Your previous proposal {json.dumps(prop)} is identical to an "
                f"already-evaluated architecture. Propose a DIFFERENT architecture.\n"
            )
            continue

        internal = _to_internal_config(prop)
        n_params = count_subnet_params(internal, vocab_size, max_adm=max_adm)
        if n_params > max_params:
            print(f"  OVER BUDGET (params): {n_params:,} > {max_params:,}")
            extra_feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Your previous proposal {json.dumps(prop)} has {n_params:,} "
                f"parameters, which exceeds the budget of {max_params:,}. "
                f"Propose a smaller architecture.\n"
            )
            continue

        n_flops = count_subnet_flops(internal, flops_seq_len)
        if max_flops is not None and n_flops > max_flops:
            print(f"  OVER BUDGET (FLOPs): {n_flops:,} > {max_flops:,}")
            extra_feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Your previous proposal {json.dumps(prop)} has {n_flops:,} "
                f"FLOPs, which exceeds the budget of {max_flops:,}. "
                f"Propose a smaller architecture.\n"
            )
            continue

        # Valid!
        print(f"  ACCEPTED: embed_dim={prop['embed_dim']}, depth={prop['depth']}, "
              f"mlp_ratio={prop['mlp_ratio']}, num_heads={prop['num_heads']}, "
              f"params={n_params:,}, FLOPs={n_flops:,}")
        if prop.get("rationale"):
            print(f"    Rationale: {prop['rationale']}")
        return prop, internal, n_params, n_flops

    print(f"  Failed to get a valid proposal after {max_attempts} attempts.")
    return None, None, None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-LLM baseline for EHR Transformer NAS")

    # Target
    p.add_argument("--hospital", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=["death", "stay", "readmission"])

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

    # LLM
    p.add_argument("--model", type=str, default="claude-sonnet-4-6",
                   help="Claude model ID")

    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}")
    print(f"Budget: {args.budget} architectures, max_params={args.max_params:,}")
    print(f"LLM: {args.model}")

    # Resolve ckpt_path against original cwd if it exists there; else leave
    # relative so it resolves under MAS-NAS after chdir.
    if args.ckpt_path is not None and not os.path.isabs(args.ckpt_path):
        orig_abs = os.path.abspath(args.ckpt_path)
        if os.path.exists(orig_abs):
            args.ckpt_path = orig_abs
    os.chdir(_MAS_NAS_DIR)
    print(f"Working directory: {os.getcwd()}")

    # --- Load data ---
    data_root = Path(f"./data_process/{args.hospital}/{args.hospital}-processed")
    full_data_path = data_root / "mimic.pkl"
    pretrain_data_path = data_root / "mimic_pretrain.pkl"
    finetune_data_path = data_root / "mimic_downstream.pkl"

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

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- Prepare finetune data ---
    train_data, val_data, test_data = pickle.load(open(finetune_data_path, "rb"))

    token_type = ["diag", "med", "lab", "pro"]
    train_dataset = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, args.task)
    val_dataset = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, args.task)
    test_dataset = FineTuneEHRDataset(test_data, tokenizer, token_type, max_adm, args.task)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=batcher(tokenizer, mode="finetune"), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=batcher(tokenizer, mode="finetune"), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=batcher(tokenizer, mode="finetune"), shuffle=False)

    # --- Anthropic client ---
    import anthropic
    client = anthropic.Anthropic()

    # =========================================================================
    # Single-LLM search loop
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

        prop, internal, n_params, n_flops = _propose_one(
            search_state, args.max_params, client, args.model,
            args.hospital, args.task, vocab_size, max_adm, visited,
            max_flops=args.max_flops, flops_seq_len=args.flops_seq_len,
        )

        if prop is None:
            consecutive_failures += 1
            print(f"  LLM failed to produce a valid architecture "
                  f"({consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print("  Too many consecutive failures, stopping.")
                break
            continue

        consecutive_failures = 0
        visited.add(_cand_key(prop))

        # Finetune (shared with MAS)
        print(f"\n  [Finetune]")
        avg_val, best_model_sd = _finetune_one_arch(
            internal, ckpt, train_loader, val_loader, device, args
        )
        print(f"    Val: Acc={avg_val['accuracy']:.4f}  F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  AUPRC={avg_val['auprc']:.4f}")

        val_result = {
            "embed_dim": prop["embed_dim"],
            "depth": prop["depth"],
            "mlp_ratio": prop["mlp_ratio"],
            "num_heads": prop["num_heads"],
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
    print("Single-LLM Search Complete")
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
    csv_path = output_dir / f"baseline1_search_{args.task}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} val results to {csv_path}")

    print(f"\nAll architectures (ranked by val):")
    display_cols = ["embed_dim", "depth", "mlp_ratio", "num_heads", "num_params",
                    "val_accuracy", "val_f1", "val_auroc", "val_auprc", "avg_rank"]
    print(df[display_cols].to_string(index=False))

    # =========================================================================
    # Final test evaluation — best by composite val rank
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

    model = TransformerSuper(
        num_classes=2,
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

    test_metrics = evaluate(test_loader, model, device, retrain_config=best_config)
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
    test_csv_path = output_dir / f"baseline1_best_{args.task}.csv"
    pd.DataFrame([best_row]).to_csv(test_csv_path, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv_path}")


if __name__ == "__main__":
    main()
