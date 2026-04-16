"""
Agent 3: Experiment Manager + Strategy Decision
=================================================
Dual role:
  1. (Algorithmic) Run finetune experiments for accepted proposals, record val results,
     and track the globally best model for final test evaluation.
  2. (LLM-based) After each round of experiments, analyze results and decide the
     search strategy ("exploration" or "exploitation") for the next round.
"""

import copy
import json
import time

import numpy as np
import torch
from torch.optim import AdamW

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.engine import train_one_epoch, evaluate
from run_pipeline import count_subnet_params, count_subnet_flops
from model.supernet_transformer import TransformerSuper
from utils.tracer import get_tracer


def _to_internal_config(proposal):
    """Convert proposal dict (scalars) to the internal config format (per-layer lists)."""
    depth = proposal["depth"]
    return {
        "embed_dim": [proposal["embed_dim"]] * depth,
        "layer_num": depth,
        "mlp_ratio": [proposal["mlp_ratio"]] * depth,
        "num_heads": [proposal["num_heads"]] * depth,
    }


def _finetune_one_arch(config, ckpt, train_loader, val_loader, device, args):
    """
    Finetune a single architecture with early stopping + top-k averaging on val only.

    Returns:
        (avg_val, best_model_sd) — averaged val metrics and the model state dict
        from the best val epoch (for potential later test evaluation).
    """
    vocab_size = ckpt["vocab_size"]
    max_adm = ckpt["max_adm_num"]

    model = TransformerSuper(
        num_classes=2,
        vocab_size=vocab_size,
        embed_dim=ckpt["embed_dim"],
        mlp_ratio=ckpt["mlp_ratio"],
        depth=ckpt["depth"],
        num_heads=ckpt["num_heads"],
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        pre_norm=True,
        max_adm_num=max_adm,
    ).to(device)

    # Load pretrained encoder weights (skip cls head)
    pretrained_sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    filtered = {k: v for k, v in pretrained_sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model_sd.update(filtered)
    model.load_state_dict(model_sd)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_records = []  # (val_auprc, val_metrics)
    best_val_auprc = -1.0
    best_model_sd = None
    epochs_without_improve = 0

    for epoch in range(args.finetune_epochs):
        train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch,
            mode="retrain", retrain_config=config, task="cls",
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate(val_loader, model, device, retrain_config=config)

        val_auprc = val_metrics["auprc"]
        epoch_records.append((val_auprc, val_metrics))

        print(f"      Epoch {epoch+1}: val_AUPRC={val_auprc:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_sd = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.finetune_patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    # Top-k averaging on val
    epoch_records.sort(key=lambda x: x[0], reverse=True)
    top_k = min(args.top_k_epochs, len(epoch_records))
    top_k_val = [rec[1] for rec in epoch_records[:top_k]]

    avg_val = {}
    for key in ["accuracy", "f1", "auroc", "auprc"]:
        avg_val[key] = sum(m[key] for m in top_k_val) / top_k

    return avg_val, best_model_sd


def run_trials(reviewed_proposals, search_state, ckpt, train_loader, val_loader,
               device, args, vocab_size, max_adm):
    """
    Run finetune experiments for accepted proposals (val only, no test).

    search_state["completed_experiments"] stores VAL metrics (visible to agents).
    search_state["best_model"] tracks the globally best model for final test eval.

    Returns:
        updated search_state
    """
    print("\n[Agent 4: Experiment Manager]")

    if not reviewed_proposals:
        print("  No proposals to run.")
        return search_state

    # Trim to fit remaining budget
    budget = search_state["budget_remaining"]
    proposals_to_run = reviewed_proposals[:budget]
    print(f"  Running {len(proposals_to_run)} experiments (budget remaining: {budget})")

    for i, proposal in enumerate(proposals_to_run):
        config = _to_internal_config(proposal)

        # Validate param count + compute FLOPs
        n_params = count_subnet_params(config, vocab_size, max_adm=max_adm)
        flops_seq_len = getattr(args, "flops_seq_len", 512)
        n_flops = count_subnet_flops(config, flops_seq_len)
        print(f"\n    Experiment {i+1}/{len(proposals_to_run)}: "
              f"embed_dim={proposal['embed_dim']}, depth={proposal['depth']}, "
              f"params={n_params:,}, FLOPs={n_flops:,}")

        # Run finetune (val only)
        avg_val, best_model_sd = _finetune_one_arch(
            config, ckpt, train_loader, val_loader, device, args
        )

        print(f"    Val: Acc={avg_val['accuracy']:.4f}  F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  AUPRC={avg_val['auprc']:.4f}")

        arch_info = {
            "embed_dim": proposal["embed_dim"],
            "depth": proposal["depth"],
            "mlp_ratio": proposal["mlp_ratio"],
            "num_heads": proposal["num_heads"],
            "num_params": n_params,
            "flops": n_flops,
        }

        # Val metrics — visible to LLM agents in subsequent rounds
        val_result = {
            **arch_info,
            "val_accuracy": avg_val["accuracy"],
            "val_f1": avg_val["f1"],
            "val_auroc": avg_val["auroc"],
            "val_auprc": avg_val["auprc"],
        }
        search_state["completed_experiments"].append(val_result)

        # Save model state dict temporarily for this experiment
        search_state.setdefault("_model_sds", []).append(best_model_sd)
        search_state.setdefault("_configs", []).append(config)

        search_state["budget_remaining"] -= 1

    # Recompute composite rank across ALL completed experiments to find global best
    _update_best_by_composite_rank(search_state)

    print(f"\n  Budget remaining: {search_state['budget_remaining']}")
    return search_state


def _update_best_by_composite_rank(search_state):
    """Recompute composite rank across all experiments, update best model."""
    import pandas as pd

    experiments = search_state["completed_experiments"]
    if not experiments:
        return

    df = pd.DataFrame(experiments)
    val_cols = ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]
    ranks = pd.DataFrame()
    for col in val_cols:
        ranks[col] = df[col].rank(ascending=False, method="average")
    avg_rank = ranks.mean(axis=1)

    best_idx = avg_rank.idxmin()

    model_sds = search_state.get("_model_sds", [])
    configs = search_state.get("_configs", [])

    if best_idx < len(model_sds):
        search_state["best_model_sd"] = model_sds[best_idx]
        search_state["best_config"] = configs[best_idx]
        search_state["best_proposal"] = experiments[best_idx]
        print(f"\n  Best architecture (composite rank): idx={best_idx}, "
              f"embed_dim={experiments[best_idx]['embed_dim']}, "
              f"depth={experiments[best_idx]['depth']}")


# ---------------------------------------------------------------------------
# Strategy Decision (LLM-based)
# ---------------------------------------------------------------------------

def _build_strategy_prompt(context, search_state):
    """Build prompt for the strategy decision."""
    parts = []

    parts.append(
        "You are a search strategy agent for Neural Architecture Search on EHR Transformer models. "
        "Your job is to decide whether the next round of architecture proposals should focus on "
        "EXPLORATION (diverse, covering unexplored regions) or EXPLOITATION (refining around the "
        "best-performing architectures found so far).\n"
    )

    # Historical top-k architectures
    top_k = context.get("top_k_archs", [])
    if top_k:
        parts.append("\n## Historical Best Architectures (from similar hospital)\n")
        for i, arch in enumerate(top_k):
            parts.append(f"  Top-{i+1}: {json.dumps(arch)}\n")

    # SHAP importance
    shap = context.get("shap_importance", {})
    if shap:
        parts.append("\n## SHAP Feature Importance\n")
        parts.append("These show which architecture features matter most for performance:\n")
        for feat, val in sorted(shap.items(), key=lambda x: -x[1]):
            parts.append(f"  {feat}: {val:.4f}\n")

    # Completed experiments
    completed = search_state.get("completed_experiments", [])
    if completed:
        parts.append(f"\n## Completed Experiments ({len(completed)} architectures)\n")
        for exp in completed:
            parts.append(f"  {json.dumps(exp)}\n")

    # Budget
    budget = search_state.get("budget_remaining", 0)
    parts.append(f"\n## Budget Remaining: {budget} architectures\n")

    # Decision criteria
    parts.append(
        "\n## Decision Guidelines\n"
        "Choose EXPLORATION when:\n"
        "- Few experiments completed so far (search space not well covered)\n"
        "- Results are similar across tried architectures (no clear winner yet)\n"
        "- Important SHAP features have not been sufficiently varied\n"
        "- Budget is still large relative to search space\n\n"
        "Choose EXPLOITATION when:\n"
        "- A clear best-performing region has emerged\n"
        "- Budget is running low — focus on refining the best candidates\n"
        "- High-SHAP features have a clear optimal range from the data\n\n"
    )

    # Output format
    parts.append(
        "## Output Format\n"
        "Return a JSON object:\n"
        '{"strategy": "exploration" or "exploitation", "rationale": "brief explanation"}\n\n'
        "Return ONLY valid JSON, no extra text.\n"
    )

    return "".join(parts)


def decide_strategy(context, search_state, client, model="claude-sonnet-4-6"):
    """
    Use Claude to decide the search strategy for the next round.

    Returns:
        dict with "strategy" ("exploration" or "exploitation") and "rationale"
    """
    print("\n[Agent 3: Strategy Decision]")

    prompt = _build_strategy_prompt(context, search_state)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    text = response.content[0].text.strip()

    # Trace LLM prompt/response
    tracer = get_tracer()
    if tracer:
        tracer.log_subsection("LLM Call")
        tracer.log_prompt(prompt)
        tracer.log_response(text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Extract JSON object from response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            result = json.loads(text[start:end + 1])
        else:
            print(f"  Failed to parse strategy response, defaulting to exploration")
            return {"strategy": "exploration", "rationale": "parse failure fallback"}

    strategy = result.get("strategy", "exploration")
    if strategy not in ("exploration", "exploitation"):
        strategy = "exploration"
    rationale = result.get("rationale", "")

    print(f"  Strategy: {strategy}")
    print(f"  Rationale: {rationale}")

    return {"strategy": strategy, "rationale": rationale}
