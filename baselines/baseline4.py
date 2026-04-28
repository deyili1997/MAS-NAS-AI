"""
CoLLM-NAS Baseline for EHR Transformer NAS
============================================
Adapted from Wang et al., "CoLLM-NAS: Exploring High-Performing Architectures
via Collaborative LLMs", §3.2 and Algorithm 1.

The framework decomposes the LLM-driven NAS workflow into three components:

  • Navigator LLM (stateful):  Receives the full history H = [(strategy, [eval
    results]), ...] and outputs a free-text *search strategy* describing which
    region of the space the next iteration should target. It does NOT propose
    architectures directly. Starts in exploration mode and progressively
    transitions to exploitation as the history grows.

  • Generator LLM (stateless): Receives ONLY the current strategy from the
    Navigator and the search-space rules. It outputs N concrete candidate
    architectures as JSON. It has no memory across iterations and never sees
    H or the visited set — this prevents "prior-knowledge contamination" of
    the Navigator's strategic guidance (paper §3.2).

  • Coordinator (this script's main loop): Verifies legality, enforces the
    resource constraint Λ, deduplicates against the visited set V, evaluates
    each unique legal candidate via the SAME pretrained supernet + finetune
    pipeline used by every other baseline, and tracks the best architecture.

Same supernet ckpt + `_finetune_one_arch` + `--budget` semantics as baselines
0–3, for fair head-to-head comparison.

Usage:
    python baselines/baseline4.py \\
        --hospital MIMIC-IV --task death \\
        --max_params 1000000 --budget 20 \\
        --candidates_per_iter 3 \\
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

# Make MAS-NAS importable (baselines/ lives inside MAS-NAS/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_NAS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _MAS_NAS_DIR)

# Load .env
for _env_dir in (os.getcwd(), _MAS_NAS_DIR, _THIS_DIR):
    _env_path = os.path.join(_env_dir, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
        print(f"Loaded .env from {_env_path}")
        break

from agents.experiment_agent import (  # noqa: E402
    _finetune_one_arch,
    _to_internal_config,
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


# ===========================================================================
# Navigator LLM — stateful, outputs free-text strategy
# ===========================================================================

NAVIGATOR_SYSTEM_PROMPT = (
    "You are the Navigator agent in a CoLLM-NAS pipeline. Your role is to "
    "design and refine search strategies for Neural Architecture Search. You "
    "do NOT propose specific architectures yourself — that is the Generator's "
    "job. Your output is a free-text strategy directive (1-2 short paragraphs) "
    "that describes which regions of the search space to target next, what "
    "trade-offs to make, and (after history accumulates) what patterns suggest "
    "high-performing regions. Be specific about the dimensions to vary."
)


def _format_search_space():
    return (
        f"Search space:\n"
        f"  embed_dim ∈ {CHOICES['embed_dim']}\n"
        f"  depth ∈ {CHOICES['depth']}\n"
        f"  mlp_ratio ∈ {CHOICES['mlp_ratio']}\n"
        f"  num_heads ∈ {CHOICES['num_heads']}\n"
        f"Constraint: embed_dim must be divisible by num_heads.\n"
    )


def _format_constraint(max_params, max_flops):
    s = f"Resource constraint Λ: param budget ≤ {max_params:,}"
    if max_flops:
        s += f"; FLOPs budget ≤ {max_flops:,}"
    return s + ".\n"


def _format_target(target_auprc):
    if target_auprc is None:
        return "Performance target: maximize val AUPRC (no fixed target — keep improving).\n"
    return f"Performance target: val AUPRC ≥ {target_auprc:.4f}.\n"


def _build_navigator_init_prompt(target_auprc, max_params, max_flops, hospital, task):
    """Stage-1 (line 2 of Algorithm 1): define an INITIAL EXPLORATION strategy."""
    info = task_info(task)
    if info["type"] == "multilabel":
        task_descr = f"multilabel ({info['num_classes']}-class phenotype prediction)"
    else:
        task_descr = "binary classification"
    return (
        f"Target task: hospital={hospital}, task={task} ({task_descr} "
        f"on EHR Transformer subnets).\n\n"
        f"{_format_search_space()}\n"
        f"{_format_constraint(max_params, max_flops)}"
        f"{_format_target(target_auprc)}\n"
        "This is iteration 0 — no architectures have been evaluated yet.\n\n"
        "Task: Define an INITIAL search strategy that emphasizes EXPLORATION. "
        "Cover diverse regions of the space so the next iteration's "
        "candidates produce useful signal. Suggest which dimensions to vary "
        "first and why.\n\n"
        "Output: Free-text strategy directive (1-2 short paragraphs). "
        "Do NOT propose specific {embed_dim, depth, mlp_ratio, num_heads} "
        "tuples — that is the Generator's job."
    )


def _summarize_results_for_history(results):
    """Compact one-arch-per-line summary used inside the Navigator history."""
    if not results:
        return "  (no legal unvisited architectures evaluated this iteration)\n"
    lines = []
    for r in results:
        cfg = r["config"]
        lines.append(
            f"  - embed={cfg['embed_dim']:<3} depth={cfg['depth']:<2} "
            f"mlp={cfg['mlp_ratio']:<2} heads={cfg['num_heads']:<2}  "
            f"val_auprc={r['perf']:.4f}  params={r['cost']:,}"
        )
    return "\n".join(lines) + "\n"


def _build_navigator_refine_prompt(history, best_arch, best_perf,
                                   target_auprc, max_params, max_flops,
                                   hospital, task):
    """Stage-N (line 15 of Algorithm 1): refine strategy from full history."""
    parts = [
        f"Target task: hospital={hospital}, task={task}.\n\n",
        _format_search_space(),
        _format_constraint(max_params, max_flops),
        _format_target(target_auprc),
        f"\n## Best so far\n",
    ]
    if best_arch is not None:
        parts.append(
            f"  embed_dim={best_arch['embed_dim']}, depth={best_arch['depth']}, "
            f"mlp_ratio={best_arch['mlp_ratio']}, num_heads={best_arch['num_heads']}  "
            f"with val AUPRC = {best_perf:.4f}\n"
        )
    else:
        parts.append("  (no legal architectures evaluated yet)\n")

    parts.append(f"\n## Full search history H ({len(history)} prior iteration(s))\n")
    for h in history:
        # Truncate strategy to keep prompt size bounded
        strat = h["strategy"]
        strat_show = strat if len(strat) <= 700 else strat[:700] + " …(truncated)"
        parts.append(
            f"\n### Iteration {h['iteration']}\n"
            f"Strategy used:\n  {strat_show}\n"
            f"Architectures evaluated:\n"
            f"{_summarize_results_for_history(h['results'])}"
        )

    parts.append(
        "\n## Task\n"
        "Refine the search strategy for the NEXT iteration. Use the patterns "
        "in the history above to identify high-performing regions, and bias "
        "the next search toward them while keeping useful diversity. As the "
        "history grows and a clear winner emerges, transition from broad "
        "EXPLORATION toward targeted EXPLOITATION of the high-performing "
        "region.\n\n"
        "Output: Free-text strategy directive (1-2 short paragraphs). "
        "Do NOT propose specific {embed_dim, depth, mlp_ratio, num_heads} "
        "tuples — that is the Generator's job."
    )
    return "".join(parts)


def _navigator_call(client, model, prompt, temperature, max_retries=5):
    """Stateful in semantics (we re-send full H each time); stateless at API
    level. Returns the strategy as plain text."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=float(temperature),
                system=NAVIGATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Navigator] API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ===========================================================================
# Generator LLM — stateless, outputs candidate architectures
# ===========================================================================

GENERATOR_SYSTEM_PROMPT = (
    "You are the Generator agent in a CoLLM-NAS pipeline. Your role is to "
    "translate the search strategy provided to you into concrete candidate "
    "neural-network architectures. You have NO memory of prior iterations and "
    "see no history — only the current strategy directive. Output strict JSON "
    "matching the requested format. Adhere strictly to the search-space rules "
    "and the resource constraint."
)


def _build_generator_prompt(strategy, n_candidates, max_params, max_flops):
    return (
        f"{_format_search_space()}\n"
        f"{_format_constraint(max_params, max_flops)}\n"
        f"## Search strategy (from the Navigator)\n{strategy}\n\n"
        f"## Task\n"
        f"Propose exactly {n_candidates} candidate architectures that follow "
        f"the strategy above. Make them DIVERSE within the strategy's intent.\n\n"
        f"## Output format\n"
        f"Return a JSON array of exactly {n_candidates} architectures, each "
        f"shaped:\n"
        '  {"embed_dim": <int>, "depth": <int>, "mlp_ratio": <int>, "num_heads": <int>}\n\n'
        f"Return ONLY valid JSON, no markdown code fences, no extra text."
    )


def _parse_candidates(text):
    """Parse a JSON array of candidates from the Generator's response."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e > s:
            obj = json.loads(text[s:e + 1])
        else:
            # Fall back: maybe a single object
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e > s:
                obj = json.loads(text[s:e + 1])
            else:
                raise
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array, got {type(obj)}")
    return obj


def _validate_proposal(prop):
    e, d, m, h = (prop.get("embed_dim"), prop.get("depth"),
                  prop.get("mlp_ratio"), prop.get("num_heads"))
    if e not in CHOICES["embed_dim"]:
        return False, f"embed_dim {e} not in {CHOICES['embed_dim']}"
    if d not in CHOICES["depth"]:
        return False, f"depth {d} not in {CHOICES['depth']}"
    if m not in CHOICES["mlp_ratio"]:
        return False, f"mlp_ratio {m} not in {CHOICES['mlp_ratio']}"
    if h not in CHOICES["num_heads"]:
        return False, f"num_heads {h} not in {CHOICES['num_heads']}"
    if e % h != 0:
        return False, f"embed_dim {e} not divisible by num_heads {h}"
    return True, "ok"


def _generator_call(client, model, prompt, temperature, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=float(temperature),
                system=GENERATOR_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [Generator] API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _generate_candidates(client, model, strategy, n_candidates,
                          max_params, max_flops, temperature):
    """One Generator call → parse → return list of valid arch dicts (may
    contain fewer than n_candidates if the Generator returned invalid ones)."""
    prompt = _build_generator_prompt(strategy, n_candidates, max_params, max_flops)
    text = _generator_call(client, model, prompt, temperature)
    try:
        raw = _parse_candidates(text)
    except Exception as ex:
        print(f"  [Generator] parse error: {ex}. Raw[:200]: {text[:200]}")
        return []
    valid = []
    for prop in raw:
        ok, msg = _validate_proposal(prop)
        if ok:
            valid.append({
                "embed_dim": prop["embed_dim"], "depth": prop["depth"],
                "mlp_ratio": prop["mlp_ratio"], "num_heads": prop["num_heads"],
            })
        else:
            print(f"  [Generator] dropped invalid candidate {prop}: {msg}")
    return valid


def _cand_key(c):
    return (c["embed_dim"], c["depth"], c["mlp_ratio"], c["num_heads"])


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="CoLLM-NAS baseline for EHR Transformer NAS"
    )
    # Target
    p.add_argument("--hospital", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=ALL_TASKS,
                   help="Binary or multilabel task. Multilabel: next_diag_*_pheno.")
    # Constraints
    p.add_argument("--max_params", type=int, required=True)
    p.add_argument("--max_flops", type=int, default=None)
    p.add_argument("--flops_seq_len", type=int, default=512)
    p.add_argument("--budget", type=int, default=20,
                   help="Total architectures to finetune")
    # Pretrained model
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="./results")
    # Pretrain hyperparams (only if no ckpt_path)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_patience", type=int, default=5)
    p.add_argument("--embed_dim", type=int, default=128)
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
                   help="Default Claude model used for both Navigator and Generator "
                        "if not overridden separately.")
    p.add_argument("--navigator_model", type=str, default=None,
                   help="Navigator-specific model ID (defaults to --model).")
    p.add_argument("--generator_model", type=str, default=None,
                   help="Generator-specific model ID (defaults to --model).")
    p.add_argument("--navigator_temperature", type=float, default=0.7,
                   help="Sampling temperature for the Navigator (paper used 0.6)")
    p.add_argument("--generator_temperature", type=float, default=0.7,
                   help="Sampling temperature for the Generator")
    # CoLLM-NAS specifics
    p.add_argument("--candidates_per_iter", type=int, default=3,
                   help="|C_t|: number of candidates the Generator emits per iteration")
    p.add_argument("--max_iterations", type=int, default=None,
                   help="Iteration limit T (default: 2 * budget)")
    p.add_argument("--target_auprc", type=float, default=None,
                   help="Optional P_target for early stop on val AUPRC; "
                        "default None = no early stop")
    # Common
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers (forced to 0 off-CUDA)")
    p.add_argument("--cudnn_benchmark", action="store_true",
                   help="Enable cuDNN benchmark (faster, nondeterministic)")
    return p.parse_args()


# ===========================================================================
# Main (Coordinator)
# ===========================================================================
def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    np.random.seed(args.seed)

    # Default Navigator/Generator to --model if not separately set
    nav_model = args.navigator_model or args.model
    gen_model = args.generator_model or args.model
    if args.max_iterations is None:
        args.max_iterations = 2 * args.budget

    device = pick_device()
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}")
    print(f"Budget: {args.budget}, max_iterations={args.max_iterations}, "
          f"candidates_per_iter={args.candidates_per_iter}")
    print(f"Navigator: {nav_model}  (T={args.navigator_temperature})")
    print(f"Generator: {gen_model}  (T={args.generator_temperature})")
    print(f"Resource Λ: max_params={args.max_params:,}"
          + (f", max_flops={args.max_flops:,}" if args.max_flops else ""))
    if args.target_auprc is not None:
        print(f"Target P: val AUPRC ≥ {args.target_auprc}")

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
    print(f"Vocab: {vocab_size}, max_adm: {max_adm}")

    # --- Pretrain or load supernet ckpt ---
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
    train_ds = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, args.task)
    val_ds = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, args.task)
    test_ds = FineTuneEHRDataset(test_data, tokenizer, token_type, max_adm, args.task)
    dl_kw = dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=batcher(tokenizer, mode="finetune"),
                              shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=batcher(tokenizer, mode="finetune"),
                            shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             collate_fn=batcher(tokenizer, mode="finetune"),
                             shuffle=False, **dl_kw)

    # --- LLM client ---
    import anthropic
    client = anthropic.Anthropic()

    # --- Coordinator state (Algorithm 1, line 1) ---
    visited = set()             # V
    history = []                # H = [{iteration, strategy, results}]
    completed = []              # all evaluated archs (for final ranking + CSV)
    model_sds = []              # parallel list: best CPU snapshots
    internals = []              # parallel list: internal configs
    best_arch = None            # α*
    best_perf = -1.0            # p* (val AUPRC; range [0,1])
    consecutive_empty = 0       # iterations w/ 0 new evaluations
    max_consecutive_empty = 3

    # --- Initial strategy (line 2) ---
    print(f"\n{'='*60}")
    print("Navigator → Initial strategy (line 2)")
    print(f"{'='*60}")
    init_prompt = _build_navigator_init_prompt(
        args.target_auprc, args.max_params, args.max_flops,
        args.hospital, args.task,
    )
    strategy = _navigator_call(client, nav_model, init_prompt,
                                args.navigator_temperature)
    print(f"\n[Strategy 0]\n{strategy}\n")

    strategies_log = [{"iteration": 0, "stage": "init", "strategy": strategy}]

    # --- Search loop (lines 3-16) ---
    iteration = 0
    while len(completed) < args.budget and iteration < args.max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.max_iterations}  "
              f"evaluated={len(completed)}/{args.budget}  "
              f"|V|={len(visited)}  best_auprc={best_perf:.4f}")
        print(f"{'='*60}")

        # --- Generator (line 4) ---
        candidates = _generate_candidates(
            client, gen_model, strategy, args.candidates_per_iter,
            args.max_params, args.max_flops, args.generator_temperature,
        )
        print(f"  [Generator] returned {len(candidates)} valid candidates")

        # --- Evaluate each legal unvisited candidate (lines 5-11) ---
        results_this_iter = []
        for cand in candidates:
            if len(completed) >= args.budget:
                break
            key = _cand_key(cand)
            if key in visited:
                print(f"  ↳ {cand} already visited, skip")
                continue

            internal = _to_internal_config(cand)
            n_params = count_subnet_params(internal, vocab_size,
                                           num_classes=task_info(args.task)["num_classes"],
                                           max_adm=max_adm)
            n_flops = count_subnet_flops(internal, args.flops_seq_len)

            # Resource constraint check (paper's c_i ≤ Λ — we pre-filter here
            # to save finetune compute; the paper's algorithm allows wasting
            # compute on over-budget configs but skipping is equivalent up to
            # the best-update rule on line 9)
            if n_params > args.max_params or (
                args.max_flops and n_flops > args.max_flops
            ):
                print(f"  ↳ {cand} OVER BUDGET (params={n_params:,}, "
                      f"flops={n_flops:,}); marking visited and skipping")
                visited.add(key)
                continue

            visited.add(key)
            print(f"  ↳ Evaluating {cand}  params={n_params:,}  flops={n_flops:,}")

            avg_val, best_sd = _finetune_one_arch(
                internal, ckpt, train_loader, val_loader, device, args,
            )
            print(f"    val: Acc={avg_val['accuracy']:.4f}  "
                  f"F1={avg_val['f1']:.4f}  "
                  f"AUROC={avg_val['auroc']:.4f}  "
                  f"AUPRC={avg_val['auprc']:.4f}")

            val_result = {
                "embed_dim": cand["embed_dim"], "depth": cand["depth"],
                "mlp_ratio": cand["mlp_ratio"], "num_heads": cand["num_heads"],
                "num_params": n_params, "flops": n_flops,
                "val_accuracy": avg_val["accuracy"],
                "val_f1": avg_val["f1"],
                "val_auroc": avg_val["auroc"],
                "val_auprc": avg_val["auprc"],
                "iteration": iteration,
            }
            completed.append(val_result)
            model_sds.append(best_sd)
            internals.append(internal)
            results_this_iter.append({
                "config": cand, "perf": avg_val["auprc"], "cost": n_params,
            })

            # Best update (line 9) — c_i ≤ Λ already satisfied above
            if avg_val["auprc"] > best_perf:
                best_perf = avg_val["auprc"]
                best_arch = cand
                print(f"    *** new best: AUPRC={best_perf:.4f}")

        # --- Termination: target reached (line 12) ---
        if args.target_auprc is not None and best_perf >= args.target_auprc:
            print(f"\nTarget P_target={args.target_auprc} reached "
                  f"(best={best_perf:.4f}). Stopping.")
            history.append({"iteration": iteration, "strategy": strategy,
                            "results": results_this_iter})
            break

        # Track empty iterations to stop if Generator/duplicates exhaust
        if not results_this_iter:
            consecutive_empty += 1
            print(f"  No new evaluations this iteration "
                  f"({consecutive_empty}/{max_consecutive_empty})")
            if consecutive_empty >= max_consecutive_empty:
                print("  Too many empty iterations, stopping.")
                history.append({"iteration": iteration, "strategy": strategy,
                                "results": results_this_iter})
                break
        else:
            consecutive_empty = 0

        # --- Update history (line 14) ---
        history.append({"iteration": iteration, "strategy": strategy,
                        "results": results_this_iter})

        # --- Refine strategy (line 15) ---
        if len(completed) < args.budget and iteration < args.max_iterations:
            print(f"\n  Navigator → refining strategy (line 15)")
            refine_prompt = _build_navigator_refine_prompt(
                history, best_arch, best_perf, args.target_auprc,
                args.max_params, args.max_flops, args.hospital, args.task,
            )
            strategy = _navigator_call(client, nav_model, refine_prompt,
                                        args.navigator_temperature)
            print(f"\n[Strategy {iteration}]\n{strategy}\n")
            strategies_log.append({
                "iteration": iteration, "stage": "refine", "strategy": strategy,
            })

    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n{'='*60}")
    print("CoLLM-NAS Search Complete")
    print(f"{'='*60}")

    if not completed:
        print("No experiments completed.")
        return

    df = pd.DataFrame(completed)
    df["hospital"] = args.hospital
    df["task"] = args.task

    val_rank = pd.DataFrame()
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        val_rank[col] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = val_rank.mean(axis=1)
    df_sorted = df.sort_values("avg_rank").reset_index(drop=True)

    # Output layout: results/<hospital>/search/baseline4/<task>/{baseline4_search,baseline4_best}.csv + baseline4_strategies.json
    out_dir = Path(args.results_dir) / args.hospital / "search" / "baseline4" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline4_search.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_sorted)} val results to {csv_path}")

    # Strategies log (Navigator outputs per iteration)
    strat_json_path = out_dir / "baseline4_strategies.json"
    # Also include per-iter result summaries
    strategies_with_history = []
    for entry in strategies_log:
        h = next((h for h in history if h["iteration"] == entry["iteration"]), None)
        strategies_with_history.append({
            "iteration": entry["iteration"],
            "stage": entry["stage"],
            "strategy": entry["strategy"],
            "results_summary": (
                _summarize_results_for_history(h["results"]).strip()
                if h else None
            ),
        })
    with open(strat_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "hospital": args.hospital, "task": args.task,
            "navigator_model": nav_model, "generator_model": gen_model,
            "best_arch": best_arch, "best_val_auprc": best_perf,
            "strategies": strategies_with_history,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved Navigator strategies log to {strat_json_path}")

    print(f"\nAll architectures (ranked by composite val rank):")
    display_cols = ["embed_dim", "depth", "mlp_ratio", "num_heads",
                    "num_params", "iteration",
                    "val_accuracy", "val_f1", "val_auroc", "val_auprc",
                    "avg_rank"]
    print(df_sorted[display_cols].to_string(index=False))

    # =========================================================================
    # Final test evaluation: best by composite val rank
    # =========================================================================
    val_rank_full = pd.DataFrame()
    df_unsorted = pd.DataFrame(completed)
    for col in ["val_accuracy", "val_f1", "val_auroc", "val_auprc"]:
        val_rank_full[col] = df_unsorted[col].rank(ascending=False, method="average")
    avg_rank_full = val_rank_full.mean(axis=1)
    best_idx = int(avg_rank_full.idxmin())
    best_sd = model_sds[best_idx]
    best_internal = internals[best_idx]
    best_info = completed[best_idx]

    print(f"\n{'='*60}")
    print("Final Test Evaluation — Best Architecture by Composite Val Rank")
    print(f"{'='*60}")
    print(f"  embed_dim={best_info['embed_dim']}, depth={best_info['depth']}, "
          f"mlp_ratio={best_info['mlp_ratio']}, num_heads={best_info['num_heads']}, "
          f"params={best_info['num_params']:,}")

    info = task_info(args.task)
    model = TransformerSuper(
        num_classes=info["num_classes"],
        vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"], mlp_ratio=ckpt["mlp_ratio"],
        depth=ckpt["depth"], num_heads=ckpt["num_heads"],
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        pre_norm=True, max_adm_num=ckpt["max_adm_num"],
    ).to(device)
    model.load_state_dict(best_sd)
    test_metrics = evaluate(test_loader, model, device, retrain_config=best_internal,
                            task_type=info["type"])

    print(f"\n  Test Results:")
    print(f"    Accuracy = {test_metrics['accuracy']:.4f}")
    print(f"    F1       = {test_metrics['f1']:.4f}")
    print(f"    AUROC    = {test_metrics['auroc']:.4f}")
    print(f"    AUPRC    = {test_metrics['auprc']:.4f}")

    if best_arch is not None and (
        best_arch["embed_dim"] != best_info["embed_dim"]
        or best_arch["depth"] != best_info["depth"]
        or best_arch["mlp_ratio"] != best_info["mlp_ratio"]
        or best_arch["num_heads"] != best_info["num_heads"]
    ):
        print(f"\n  Note: paper's α* (best by val_auprc alone) differs from "
              f"composite-val-rank best. α* = {best_arch} with "
              f"val_auprc={best_perf:.4f}.")

    best_row = df_sorted.iloc[0].to_dict()
    best_row["test_accuracy"] = test_metrics["accuracy"]
    best_row["test_f1"] = test_metrics["f1"]
    best_row["test_auroc"] = test_metrics["auroc"]
    best_row["test_auprc"] = test_metrics["auprc"]
    test_csv = out_dir / "baseline4_best.csv"
    pd.DataFrame([best_row]).to_csv(test_csv, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv}")


if __name__ == "__main__":
    main()
