"""
LLMatic-Style Quality-Diversity Baseline for EHR Transformer NAS
=================================================================
Adapted from LLMatic (Nasir et al., 2023, arXiv:2306.01102).

LLMatic combines a MAP-Elites quality-diversity (QD) archive with LLM-driven
mutation/crossover. The original paper generates Python `nn.Module` source
code via LLM and trains each candidate from scratch on CIFAR-10.

This adaptation keeps LLMatic's core ideas — MAP-Elites niching, prompt
curiosity, mutation/crossover via LLM — but specializes them to our scalar
NAS setting:
  - Architectures are 4 scalars (embed_dim, depth, mlp_ratio, num_heads),
    not arbitrary Python code. The LLM emits a JSON config, not source code.
  - Evaluation reuses the SAME pretrained EHR Transformer supernet checkpoint
    and the SAME `_finetune_one_arch` finetune+early-stop+top-k-epoch averaging
    used by every other baseline in this repo, for fair comparison.
  - The 2D Behavioral Descriptor (BD) is `(log_params, log_flops)` normalized
    to [0,1]^2 (paper uses depth/width-ratio + flops; ours is closer to a
    capacity-vs-compute layout, more meaningful for transformer subnets).
  - A single KDTree of CVT-style centroids in [0,1]^2 niches both the
    `nets_archive` and the `prompt_archive`, mirroring LLMatic.
  - Mutation prompts are short natural-language directives ("increase depth",
    "make the network smaller", etc.) injected alongside the parent JSON.
  - Crossover concatenates two parents' JSON into a single LLM prompt asking
    for a hybrid.
  - Prompt curiosity is +1 when the generated child enters `nets_archive`
    (improves a niche) and -0.5 otherwise; the QD phase prefers high-curiosity
    prompts.

It uses:
  - The SAME pretrained supernet checkpoint
  - The SAME scalar search space (CHOICES)
  - The SAME `_finetune_one_arch` (val-only, top-k epoch averaging)
  - The SAME composite val-rank rule for final test selection
  - The SAME --budget semantics (total architectures finetuned)

Removed: dataset similarity, top-k archs from similar hospitals, SHAP.
The LLM only sees its own search trajectory in this run.

Usage:
    python baselines/baseline3.py \\
        --hospital MIMIC-IV --task death \\
        --max_params 1000000 --budget 20 --n_niches 16 \\
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
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Make MAS-NAS importable (baselines/ lives inside MAS-NAS/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_NAS_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _MAS_NAS_DIR)

# Load .env from likely locations
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


# ---------------------------------------------------------------------------
# LLMatic-style mutation prompts (adapted to scalar Transformer search space)
# ---------------------------------------------------------------------------
MUTATION_PROMPTS = [
    "Increase the depth (more transformer layers).",
    "Decrease the depth (fewer transformer layers).",
    "Increase the embedding dimension.",
    "Decrease the embedding dimension.",
    "Increase the MLP ratio.",
    "Decrease the MLP ratio.",
    "Increase the number of attention heads.",
    "Decrease the number of attention heads.",
    "Make the architecture larger overall (more parameters / FLOPs).",
    "Make the architecture smaller overall (fewer parameters / FLOPs).",
    "Make a small targeted change to refine performance.",
    "Make a bold change to a region not yet explored.",
]
N_PROMPTS = len(MUTATION_PROMPTS)
CROSSOVER_DIRECTIVE = (
    "Combine the two parent architectures above. Produce a third architecture "
    "that inherits characteristics from both — e.g., the depth of one and the "
    "embedding dim of the other."
)


# ---------------------------------------------------------------------------
# MAP-Elites primitives  (mirrors LLMatic.map_elites.common)
# ---------------------------------------------------------------------------
class Species:
    """Archive entry. Mirrors LLMatic.map_elites.common.Species."""
    __slots__ = (
        "config", "internal", "desc", "fitness", "centroid",
        "curiosity", "val_metrics", "n_params", "flops",
        "model_sd", "prompt_idx", "temperature",
    )

    def __init__(self, config, internal, desc, fitness, n_params, flops,
                 val_metrics, model_sd, prompt_idx=None, temperature=0.7):
        self.config = config
        self.internal = internal
        self.desc = np.asarray(desc, dtype=float)
        self.fitness = fitness
        self.centroid = None
        self.curiosity = 0.0
        self.val_metrics = val_metrics
        self.n_params = n_params
        self.flops = flops
        self.model_sd = model_sd
        self.prompt_idx = prompt_idx
        self.temperature = temperature


def _build_centroids(n_niches, dim=2, seed=123, n_samples=10000):
    """CVT-style centroids in [0,1]^dim via k-means on uniform samples.
    Equivalent to LLMatic's cvt() but inlined for self-contained baseline."""
    from sklearn.cluster import KMeans
    rs = np.random.RandomState(seed)
    samples = rs.uniform(0, 1, (n_samples, dim))
    km = KMeans(n_clusters=n_niches, random_state=seed, n_init=10).fit(samples)
    return km.cluster_centers_


def _make_hashable(arr):
    return tuple(map(float, arr))


def _bd_nets(n_params, flops, max_params_ref, max_flops_ref):
    """Behavioral descriptor for nets_archive: (log_params, log_flops) in [0,1]^2."""
    bd0 = np.log10(max(n_params, 1)) / np.log10(max(max_params_ref, 10))
    bd1 = np.log10(max(flops, 1)) / np.log10(max(max_flops_ref, 10))
    return np.array([float(np.clip(bd0, 0, 1)), float(np.clip(bd1, 0, 1))])


def _bd_prompts(prompt_idx, temperature):
    """Behavioral descriptor for prompt_archive: (prompt_idx_norm, temp) in [0,1]^2."""
    return np.array([prompt_idx / max(N_PROMPTS - 1, 1), float(temperature)])


def _add_to_archive(specie, archive, kdt, lower_is_better):
    """Add specie to archive. Returns True if niche was newly filled or improved.
    Mirrors LLMatic.__add_to_archive."""
    niche_idx = kdt.query([specie.desc], k=1)[1][0][0]
    niche = kdt.data[niche_idx]
    n = _make_hashable(niche)
    specie.centroid = n
    if n in archive:
        if lower_is_better:
            if specie.fitness < archive[n].fitness:
                archive[n] = specie
                return True
        else:
            if specie.fitness > archive[n].fitness:
                archive[n] = specie
                return True
        return False
    archive[n] = specie
    return True


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _format_search_space():
    return (
        f"Search space: embed_dim ∈ {CHOICES['embed_dim']}, "
        f"depth ∈ {CHOICES['depth']}, mlp_ratio ∈ {CHOICES['mlp_ratio']}, "
        f"num_heads ∈ {CHOICES['num_heads']}.\n"
        "CONSTRAINT: embed_dim must be divisible by num_heads.\n"
    )


def _format_budget(max_params, max_flops):
    s = f"Param budget ≤ {max_params:,}"
    if max_flops:
        s += f"; FLOPs budget ≤ {max_flops:,}"
    return s + ".\n"


def _build_init_prompt(hospital, task, max_params, max_flops):
    info = task_info(task)
    if info["type"] == "multilabel":
        task_descr = f"multilabel ({info['num_classes']}-class phenotype prediction)"
    else:
        task_descr = "binary classification"
    return (
        f"You are designing a Transformer for EHR data on hospital={hospital}, "
        f"task={task} ({task_descr}).\n\n"
        f"{_format_search_space()}{_format_budget(max_params, max_flops)}\n"
        "Propose ONE architecture as JSON of exactly this shape:\n"
        '{"embed_dim": <int>, "depth": <int>, "mlp_ratio": <int>, "num_heads": <int>}\n'
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )


def _build_mutation_prompt(parent_config, parent_metrics, directive,
                           hospital, task, max_params, max_flops):
    parts = [
        f"You are evolving Transformer architectures for EHR data "
        f"(hospital={hospital}, task={task}).\n\n",
        _format_search_space(),
        _format_budget(max_params, max_flops),
        f"\n## Parent architecture\n{json.dumps(parent_config)}\n",
    ]
    if parent_metrics:
        parts.append(
            f"Parent val metrics: "
            f"acc={parent_metrics['accuracy']:.4f}, f1={parent_metrics['f1']:.4f}, "
            f"auroc={parent_metrics['auroc']:.4f}, auprc={parent_metrics['auprc']:.4f}\n"
        )
    parts.append(f"\n## Directive\n{directive}\n")
    parts.append(
        "\nPropose ONE new architecture as JSON of exactly this shape:\n"
        '{"embed_dim": <int>, "depth": <int>, "mlp_ratio": <int>, "num_heads": <int>}\n'
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )
    return "".join(parts)


def _build_crossover_prompt(parent_a, parent_b, metrics_a, metrics_b,
                            hospital, task, max_params, max_flops):
    parts = [
        f"You are evolving Transformer architectures for EHR data "
        f"(hospital={hospital}, task={task}).\n\n",
        _format_search_space(),
        _format_budget(max_params, max_flops),
        f"\n## Parent A\n{json.dumps(parent_a)}\n",
    ]
    if metrics_a:
        parts.append(
            f"Parent A val metrics: auroc={metrics_a['auroc']:.4f}, "
            f"auprc={metrics_a['auprc']:.4f}\n"
        )
    parts.append(f"\n## Parent B\n{json.dumps(parent_b)}\n")
    if metrics_b:
        parts.append(
            f"Parent B val metrics: auroc={metrics_b['auroc']:.4f}, "
            f"auprc={metrics_b['auprc']:.4f}\n"
        )
    parts.append(f"\n## Directive\n{CROSSOVER_DIRECTIVE}\n")
    parts.append(
        "\nPropose ONE new architecture as JSON of exactly this shape:\n"
        '{"embed_dim": <int>, "depth": <int>, "mlp_ratio": <int>, "num_heads": <int>}\n'
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# LLM I/O (parse + validate; same conventions as baseline2)
# ---------------------------------------------------------------------------
def _parse_proposal(text):
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            obj = json.loads(text[s:e + 1])
        else:
            raise
    if isinstance(obj, list):
        if not obj:
            raise ValueError("LLM returned empty list")
        obj = obj[0]
    return obj


def _validate_proposal(prop):
    e = prop.get("embed_dim")
    d = prop.get("depth")
    m = prop.get("mlp_ratio")
    h = prop.get("num_heads")
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


def _call_llm(prompt, client, model, temperature=0.7, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=512, temperature=float(temperature),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def _generate_one(prompt, client, model, temperature, vocab_size, max_adm,
                  max_params, max_flops, flops_seq_len, visited, num_classes,
                  max_attempts=5):
    """Get one valid, in-budget, unvisited config from the LLM."""
    feedback = ""
    for attempt in range(max_attempts):
        text = _call_llm(prompt + feedback, client, model, temperature=temperature)
        try:
            prop = _parse_proposal(text)
        except Exception as ex:
            print(f"  Parse error: {ex}. Raw[:200]: {text[:200]}")
            feedback = (
                "\n\n## Previous Attempt Feedback\n"
                "Previous response was not valid JSON. Return ONLY a single JSON "
                "object, no preamble, no markdown.\n"
            )
            continue
        ok, msg = _validate_proposal(prop)
        if not ok:
            print(f"  INVALID: {msg}")
            feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Previous proposal {json.dumps(prop)} was invalid: {msg}. Fix it.\n"
            )
            continue
        key = (prop["embed_dim"], prop["depth"], prop["mlp_ratio"], prop["num_heads"])
        if key in visited:
            print(f"  DUPLICATE: {prop} already evaluated")
            feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Previous proposal {json.dumps(prop)} is identical to an "
                f"already-evaluated architecture. Propose a DIFFERENT one.\n"
            )
            continue
        internal = _to_internal_config(prop)
        n_params = count_subnet_params(internal, vocab_size, num_classes=num_classes, max_adm=max_adm)
        if n_params > max_params:
            print(f"  OVER PARAMS: {n_params:,} > {max_params:,}")
            feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Previous proposal {json.dumps(prop)} has {n_params:,} params, "
                f"exceeds budget {max_params:,}. Propose smaller.\n"
            )
            continue
        n_flops = count_subnet_flops(internal, flops_seq_len)
        if max_flops and n_flops > max_flops:
            print(f"  OVER FLOPs: {n_flops:,} > {max_flops:,}")
            feedback = (
                f"\n\n## Previous Attempt Feedback\n"
                f"Previous proposal {json.dumps(prop)} has {n_flops:,} FLOPs, "
                f"exceeds budget {max_flops:,}. Propose smaller.\n"
            )
            continue
        print(f"  ACCEPTED: {prop}  params={n_params:,}  FLOPs={n_flops:,}")
        return prop, internal, n_params, n_flops
    return None, None, None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="LLMatic-style Quality-Diversity baseline for EHR Transformer NAS"
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
    # Pretrain hyperparams (only used if ckpt_path absent)
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
    p.add_argument("--model", type=str, default="claude-sonnet-4-6")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Initial LLM sampling temperature")
    # Quality-Diversity specifics
    p.add_argument("--n_niches", type=int, default=16,
                   help="Number of MAP-Elites niches (CVT centroids in [0,1]^2)")
    p.add_argument("--random_init_frac", type=float, default=0.3,
                   help="Fraction of budget reserved for random-init phase")
    p.add_argument("--mutation_prob", type=float, default=0.85,
                   help="P(mutation) vs P(crossover) during QD phase")
    p.add_argument("--temp_jitter", type=float, default=0.1,
                   help="Per-step temperature mutation magnitude")
    p.add_argument("--top_n_select", type=int, default=3,
                   help="Pick parent/prompt from top-N by fitness/curiosity")
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
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}")
    print(f"Budget: {args.budget}, n_niches={args.n_niches}, "
          f"max_params={args.max_params:,}")
    print(f"LLM: {args.model}, init_temp={args.temperature}")

    # Resolve ckpt_path before chdir (mirrors baseline2)
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

    # --- MAP-Elites archives + KDTree ---
    from sklearn.neighbors import KDTree
    centroids = _build_centroids(args.n_niches, dim=2, seed=args.seed)
    kdt = KDTree(centroids, leaf_size=10, metric="euclidean")
    print(f"\nMAP-Elites: {args.n_niches} niches, CVT centroids in [0,1]^2")

    nets_archive = {}      # niche_key -> Species (lower fitness = better)
    prompt_archive = {}    # niche_key -> Species (higher curiosity = better)

    # BD normalization references
    max_flops_ref = args.max_flops if args.max_flops else 10 ** 10

    # Bookkeeping for final ranking + test eval (parallel lists)
    completed = []
    model_sds = []
    internals = []
    visited = set()

    # --- Search loop ---
    n_random_init = max(2, int(args.random_init_frac * args.budget))
    cur_temp = float(args.temperature)
    consecutive_failures = 0
    max_consecutive_failures = 3
    iteration = 0

    while len(completed) < args.budget:
        iteration += 1
        n_done = len(completed)
        in_init = n_done < n_random_init

        print(f"\n{'='*60}")
        print(f"Generation {iteration}  evaluated={n_done}/{args.budget}  "
              f"phase={'random_init' if in_init else 'qd'}  "
              f"|nets|={len(nets_archive)}  |prompts|={len(prompt_archive)}")
        print(f"{'='*60}")

        prop = internal = n_params = n_flops = None
        chosen_prompt_idx = None
        chosen_temp = cur_temp
        op_name = "init"

        if in_init:
            # Random-init phase: random prompt + (random parent if any)
            chosen_prompt_idx = random.randrange(N_PROMPTS)
            directive = MUTATION_PROMPTS[chosen_prompt_idx]
            if not nets_archive:
                base_prompt = _build_init_prompt(
                    args.hospital, args.task, args.max_params, args.max_flops,
                )
                op_name = "init/cold"
            else:
                parent = random.choice(list(nets_archive.values()))
                base_prompt = _build_mutation_prompt(
                    parent.config, parent.val_metrics, directive,
                    args.hospital, args.task, args.max_params, args.max_flops,
                )
                op_name = "init/mutation"

            prop, internal, n_params, n_flops = _generate_one(
                base_prompt, client, args.model, chosen_temp,
                vocab_size, max_adm, args.max_params, args.max_flops,
                args.flops_seq_len, visited,
                num_classes=task_info(args.task)["num_classes"],
            )

        else:
            # QD phase: mutation (mut_prob) or crossover
            do_mutation = (random.random() < args.mutation_prob) or (len(nets_archive) < 2)
            if do_mutation:
                op_name = "qd/mutation"
                # Prompt: pick from top-N by curiosity (fall back to random if empty)
                if prompt_archive:
                    sorted_p = sorted(
                        prompt_archive.values(), key=lambda s: s.curiosity, reverse=True,
                    )
                    top_p = sorted_p[: max(1, args.top_n_select)]
                    chosen = random.choice(top_p)
                    chosen_prompt_idx = (
                        chosen.prompt_idx if chosen.prompt_idx is not None
                        and 0 <= chosen.prompt_idx < N_PROMPTS
                        else random.randrange(N_PROMPTS)
                    )
                    chosen_temp = float(chosen.temperature)
                else:
                    chosen_prompt_idx = random.randrange(N_PROMPTS)
                    chosen_temp = cur_temp

                # Temperature jitter (LLMatic-style)
                chosen_temp = float(np.clip(
                    chosen_temp + random.uniform(-args.temp_jitter, args.temp_jitter),
                    0.1, 1.0,
                ))

                # Parent: pick from top-N by fitness (lower better)
                sorted_n = sorted(nets_archive.values(), key=lambda s: s.fitness)
                parent = random.choice(sorted_n[: max(1, args.top_n_select)])
                directive = MUTATION_PROMPTS[chosen_prompt_idx]
                base_prompt = _build_mutation_prompt(
                    parent.config, parent.val_metrics, directive,
                    args.hospital, args.task, args.max_params, args.max_flops,
                )
                prop, internal, n_params, n_flops = _generate_one(
                    base_prompt, client, args.model, chosen_temp,
                    vocab_size, max_adm, args.max_params, args.max_flops,
                    args.flops_seq_len, visited,
                    num_classes=task_info(args.task)["num_classes"],
                )

            else:
                op_name = "qd/crossover"
                sorted_n = sorted(nets_archive.values(), key=lambda s: s.fitness)
                parent_a = sorted_n[0]
                parent_b = random.choice(sorted_n[1:])
                base_prompt = _build_crossover_prompt(
                    parent_a.config, parent_b.config,
                    parent_a.val_metrics, parent_b.val_metrics,
                    args.hospital, args.task, args.max_params, args.max_flops,
                )
                # Crossover doesn't have its own prompt — use parent_a's as
                # bookkeeping (we won't write to prompt_archive for crossover).
                chosen_prompt_idx = parent_a.prompt_idx
                chosen_temp = float(np.clip(parent_a.temperature, 0.1, 1.0))
                prop, internal, n_params, n_flops = _generate_one(
                    base_prompt, client, args.model, chosen_temp,
                    vocab_size, max_adm, args.max_params, args.max_flops,
                    args.flops_seq_len, visited,
                    num_classes=task_info(args.task)["num_classes"],
                )

        if prop is None:
            consecutive_failures += 1
            print(f"  LLM failed ({consecutive_failures}/{max_consecutive_failures})")
            if consecutive_failures >= max_consecutive_failures:
                print("  Too many consecutive failures, stopping.")
                break
            continue
        consecutive_failures = 0
        visited.add((prop["embed_dim"], prop["depth"], prop["mlp_ratio"], prop["num_heads"]))

        # --- Finetune ---
        print(f"  [{op_name}] Finetuning ...")
        avg_val, best_sd = _finetune_one_arch(
            internal, ckpt, train_loader, val_loader, device, args,
        )
        print(f"    Val: Acc={avg_val['accuracy']:.4f}  F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  AUPRC={avg_val['auprc']:.4f}")

        val_result = {
            "embed_dim": prop["embed_dim"], "depth": prop["depth"],
            "mlp_ratio": prop["mlp_ratio"], "num_heads": prop["num_heads"],
            "num_params": n_params, "flops": n_flops,
            "val_accuracy": avg_val["accuracy"], "val_f1": avg_val["f1"],
            "val_auroc": avg_val["auroc"], "val_auprc": avg_val["auprc"],
            "operator": op_name,
            "prompt_idx": chosen_prompt_idx if chosen_prompt_idx is not None else -1,
            "temperature": float(chosen_temp),
        }
        completed.append(val_result)
        model_sds.append(best_sd)
        internals.append(internal)

        # --- Add to nets_archive ---
        bd_n = _bd_nets(n_params, n_flops, args.max_params, max_flops_ref)
        # fitness: lower is better (we use -auprc so MAP-Elites' < comparator works)
        fitness = -float(avg_val["auprc"])
        spec_net = Species(
            prop, internal, bd_n, fitness, n_params, n_flops,
            avg_val, best_sd, chosen_prompt_idx, chosen_temp,
        )
        net_added = _add_to_archive(spec_net, nets_archive, kdt, lower_is_better=True)
        print(f"    nets_archive: {'ADDED to niche' if net_added else 'rejected'}")

        # --- Update prompt_archive (only for mutation/random_init, not crossover) ---
        if op_name != "qd/crossover" and chosen_prompt_idx is not None:
            bd_p = _bd_prompts(chosen_prompt_idx, chosen_temp)
            prompt_fit = 1.0 if net_added else 0.0
            spec_prompt = Species(
                prop, internal, bd_p, prompt_fit, n_params, n_flops,
                avg_val, None, chosen_prompt_idx, chosen_temp,
            )
            spec_prompt.curiosity = 1.0 if net_added else -0.5
            _add_to_archive(spec_prompt, prompt_archive, kdt, lower_is_better=False)

    # =========================================================================
    # Save results
    # =========================================================================
    print(f"\n{'='*60}")
    print("LLMatic-style QD Search Complete")
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

    # Output layout: results/<hospital>/search/baseline3/<task>/{baseline3_search,baseline3_archive,baseline3_best}.csv
    out_dir = Path(args.results_dir) / args.hospital / "search" / "baseline3" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline3_search.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_sorted)} val results to {csv_path}")

    # MAP-Elites archive snapshot
    arch_rows = []
    for niche_key, spec in nets_archive.items():
        arch_rows.append({
            "niche_x": niche_key[0],
            "niche_y": niche_key[1],
            "embed_dim": spec.config["embed_dim"],
            "depth": spec.config["depth"],
            "mlp_ratio": spec.config["mlp_ratio"],
            "num_heads": spec.config["num_heads"],
            "num_params": spec.n_params,
            "flops": spec.flops,
            "val_accuracy": spec.val_metrics["accuracy"],
            "val_f1": spec.val_metrics["f1"],
            "val_auroc": spec.val_metrics["auroc"],
            "val_auprc": spec.val_metrics["auprc"],
            "fitness_neg_auprc": spec.fitness,
        })
    arch_df = pd.DataFrame(arch_rows)
    arch_csv = out_dir / "baseline3_archive.csv"
    arch_df.to_csv(arch_csv, index=False)
    print(f"Filled {len(nets_archive)}/{args.n_niches} niches → {arch_csv}")

    print(f"\nAll architectures (ranked by composite val rank):")
    display_cols = ["embed_dim", "depth", "mlp_ratio", "num_heads",
                    "num_params", "val_accuracy", "val_f1", "val_auroc",
                    "val_auprc", "operator", "avg_rank"]
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

    best_row = df_sorted.iloc[0].to_dict()
    best_row["test_accuracy"] = test_metrics["accuracy"]
    best_row["test_f1"] = test_metrics["f1"]
    best_row["test_auroc"] = test_metrics["auroc"]
    best_row["test_auprc"] = test_metrics["auprc"]
    test_csv = out_dir / "baseline3_best.csv"
    pd.DataFrame([best_row]).to_csv(test_csv, index=False)
    print(f"\n  Best architecture + test results saved to {test_csv}")


if __name__ == "__main__":
    main()
