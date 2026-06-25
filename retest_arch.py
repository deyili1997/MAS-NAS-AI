"""Finetune a single fully-specified architecture and evaluate it on the held-out
TEST split. Used to obtain the true test performance of the architecture a method
WOULD have selected at an intermediate search budget (anytime test curve).

The finetune + test protocol is identical to the baselines' final-selection
evaluation: load the shared supernet checkpoint, slice the subnet, finetune on
train (early stopping + top-k epoch averaging on val), then evaluate the best-val
weights once on test.

Usage:
    python retest_arch.py \
        --hospital MIMIC-IV --task death --seed 123 \
        --embed_dim 128 --depth 2 --mlp_ratio 8 --num_heads 2 \
        --ckpt_path /blue/.../results/MIMIC-IV/checkpoint_mlm/mlm_model.pt \
        --out_dir /blue/.../analyze/anytime/retest
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from agents.experiment_agent import _finetune_one_arch, _to_internal_config  # noqa: E402
from run_pipeline import build_tokenizer, count_subnet_params, count_subnet_flops  # noqa: E402
from utils.dataset import FineTuneEHRDataset, batcher  # noqa: E402
from utils.engine import evaluate  # noqa: E402
from utils.seed import set_random_seed  # noqa: E402
from utils.device_helpers import dataloader_kwargs, pick_device  # noqa: E402
from utils.task_registry import task_info, ALL_TASKS  # noqa: E402
from utils.paths import get_processed_root  # noqa: E402
from model.supernet_transformer import TransformerSuper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Finetune one arch and evaluate on test")
    p.add_argument("--hospital", required=True)
    p.add_argument("--task", required=True, choices=ALL_TASKS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--embed_dim", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--mlp_ratio", type=int, required=True)
    p.add_argument("--num_heads", type=int, required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--out_dir", required=True)
    # Finetune hyperparams — MUST match the production runs (submit_stage_b.sbatch)
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--finetune_patience", type=int, default=5)
    p.add_argument("--top_k_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cudnn_benchmark", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()
    set_random_seed(args.seed, deterministic=not args.cudnn_benchmark)
    np.random.seed(args.seed)
    device = pick_device()

    print(f"Retest: {args.hospital}/{args.task} seed={args.seed} "
          f"arch=({args.embed_dim},{args.depth},{args.mlp_ratio},{args.num_heads})")

    if not os.path.isabs(args.ckpt_path):
        args.ckpt_path = os.path.abspath(args.ckpt_path)
    os.chdir(_THIS_DIR)

    # --- Data ---
    data_root = get_processed_root(args.hospital)
    full_data = pickle.load(open(data_root / "mimic.pkl", "rb"))
    tokenizer = build_tokenizer(full_data, ["[PAD]", "[CLS]", "[MASK]"])
    max_adm = full_data.groupby("SUBJECT_ID")["HADM_ID"].nunique().max()
    vocab_size = len(tokenizer.vocab.id2word)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)

    train_data, val_data, test_data = pickle.load(
        open(data_root / task_info(args.task)["data_pkl"], "rb"))
    token_type = ["diag", "med", "lab", "pro"]
    train_ds = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, args.task)
    val_ds = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, args.task)
    test_ds = FineTuneEHRDataset(test_data, tokenizer, token_type, max_adm, args.task)

    dlk = dataloader_kwargs(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=batcher(tokenizer, mode="finetune"),
                              shuffle=True, **dlk)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=batcher(tokenizer, mode="finetune"),
                            shuffle=False, **dlk)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             collate_fn=batcher(tokenizer, mode="finetune"),
                             shuffle=False, **dlk)

    # --- Build internal config + validate ---
    proposal = {"embed_dim": args.embed_dim, "depth": args.depth,
                "mlp_ratio": args.mlp_ratio, "num_heads": args.num_heads}
    internal = _to_internal_config(proposal)
    info = task_info(args.task)
    n_params = count_subnet_params(internal, vocab_size,
                                   num_classes=info["num_classes"], max_adm=max_adm)
    n_flops = count_subnet_flops(internal, 512)

    # --- Finetune (val-only, returns best-val weights) ---
    avg_val, best_model_sd = _finetune_one_arch(
        internal, ckpt, train_loader, val_loader, device, args)

    # --- Test evaluation with best-val weights ---
    model = TransformerSuper(
        num_classes=info["num_classes"], vocab_size=ckpt["vocab_size"],
        embed_dim=ckpt["embed_dim"], mlp_ratio=ckpt["mlp_ratio"],
        depth=ckpt["depth"], num_heads=ckpt["num_heads"], qkv_bias=True,
        drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate, pre_norm=True,
        max_adm_num=ckpt["max_adm_num"],
    ).to(device)
    model.load_state_dict(best_model_sd)
    test_metrics = evaluate(test_loader, model, device,
                            retrain_config=internal, task_type=info["type"])

    result = {
        "hospital": args.hospital, "task": args.task, "seed": args.seed,
        "embed_dim": args.embed_dim, "depth": args.depth,
        "mlp_ratio": args.mlp_ratio, "num_heads": args.num_heads,
        "num_params": int(n_params), "flops": int(n_flops),
        "val_accuracy": avg_val["accuracy"], "val_f1": avg_val["f1"],
        "val_auroc": avg_val["auroc"], "val_auprc": avg_val["auprc"],
        "test_accuracy": test_metrics["accuracy"], "test_f1": test_metrics["f1"],
        "test_auroc": test_metrics["auroc"], "test_auprc": test_metrics["auprc"],
        "wall_clock_sec": time.perf_counter() - t0,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = (f"{args.hospital}__{args.task}__seed{args.seed}__"
           f"{args.embed_dim}_{args.depth}_{args.mlp_ratio}_{args.num_heads}")
    out_path = out_dir / f"{key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  test_AUPRC={test_metrics['auprc']:.4f}  →  {out_path}")


if __name__ == "__main__":
    main()
