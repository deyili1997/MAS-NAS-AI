"""
AutoFormer-style NAS Pipeline for Longitudinal EHR Data
========================================================
1. Pretrain supernet with MLM (weight entanglement)
2. For each task (death, stay, readmission), randomly sample 10 architectures
   and finetune each on the downstream task
3. Collect all metrics (Accuracy, F1, AUROC, AUPRC) into a CSV metadata file
"""

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.seed import set_random_seed
from utils.tokenizer import EHRTokenizer
from utils.dataset import PreTrainEHRDataset, FineTuneEHRDataset, batcher
from utils.engine import sample_configs, train_one_epoch, evaluate, evaluate_mlm
from model.supernet_transformer import TransformerSuper


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
CHOICES = {
    "mlp_ratio": [2, 4, 8],
    "num_heads": [2, 4, 8],
    "embed_dim": [64, 128, 256],
    "depth": [2, 4, 8],
}

TASKS = ["death", "stay", "readmission"]


def parse_args():
    p = argparse.ArgumentParser(description="EHR NAS Pipeline")
    p.add_argument("--hospital", type=str, required=True,
                   help="Hospital name (used as subfolder under data_process/)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # supernet max dimensions (must be >= max of CHOICES)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=8)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--tasks", nargs="+", default=["death", "stay", "readmission"],
                   choices=["death", "stay", "readmission"],
                   help="Downstream tasks to finetune on (default: all three)")
    p.add_argument("--num_archs", type=int, default=10,
                   help="Number of random architectures to sample for finetuning")
    p.add_argument("--pretrain_patience", type=int, default=5,
                   help="Pretrain early stopping patience (epochs without val MLM loss improvement)")
    p.add_argument("--finetune_patience", type=int, default=5,
                   help="Finetune early stopping patience (epochs without val AUPRC improvement)")
    p.add_argument("--top_k", type=int, default=3,
                   help="Average test performance from top-k validation epochs")
    p.add_argument("--output_dir", type=str, default="./results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: count subnet parameters analytically
# ---------------------------------------------------------------------------
def count_subnet_params(config, vocab_size, num_classes=2, type_vocab_size=7, max_adm=8):
    d = config["embed_dim"][0]
    depth = config["layer_num"]

    # Embeddings
    params = vocab_size * d          # code_embed
    params += type_vocab_size * d    # type_embed
    params += (max_adm + 2) * d     # adm_embed

    # Transformer layers
    for i in range(depth):
        ffn_dim = int(d * config["mlp_ratio"][i])
        # Attention: QKV (with bias) + output projection
        params += 3 * d * d + 3 * d   # qkv weight + bias
        params += d * d + d            # out_proj weight + bias
        # FFN: fc1 + fc2
        params += d * ffn_dim + ffn_dim  # fc1
        params += ffn_dim * d + d        # fc2
        # LayerNorm x2 (pre-norm: before attn and before ffn)
        params += 4 * d                 # 2 * (weight + bias)

    # Final LayerNorm (pre-norm)
    params += 2 * d

    # MLM head: dense + LN + linear
    params += d * d + d              # mlm_dense
    params += 2 * d                  # mlm_ln
    params += d * vocab_size + vocab_size  # mlm_head

    # CLS head
    params += d * num_classes + num_classes

    return params


# ---------------------------------------------------------------------------
# Helper: build tokenizer from full data
# ---------------------------------------------------------------------------
def build_tokenizer(full_data, special_tokens):
    diag = full_data["ICD9_CODE"].values.tolist()
    med = full_data["NDC"].values.tolist()
    lab = full_data["LAB_TEST"].values.tolist()
    pro = full_data["PRO_CODE"].values.tolist()
    return EHRTokenizer(diag, med, lab, pro, special_tokens=special_tokens)


# ---------------------------------------------------------------------------
# Phase 1: Pretrain
# ---------------------------------------------------------------------------
def pretrain(args, tokenizer, pretrain_data, max_adm, device):
    print("\n" + "=" * 60)
    print(f"Phase 1: Pretrain supernet (MLM) — {args.hospital}")
    print("=" * 60)

    token_type = ["diag", "med", "lab", "pro"]

    full_dataset = PreTrainEHRDataset(
        ehr_pretrain_data=pretrain_data,
        tokenizer=tokenizer,
        token_type=token_type,
        mask_rate=0.15,
        max_adm_num=max_adm,
    )

    # 90/10 train/val split
    n_total = len(full_dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Pretrain split: {n_train} train / {n_val} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=batcher(tokenizer), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=batcher(tokenizer), shuffle=False)

    vocab_size = len(tokenizer.vocab.id2word)
    model = TransformerSuper(
        num_classes=0,
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        mlp_ratio=args.mlp_ratio,
        depth=args.depth,
        num_heads=args.num_heads,
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        pre_norm=True,
        max_adm_num=max_adm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Largest supernet config for validation
    max_depth = max(CHOICES["depth"])
    val_config = {
        "embed_dim": [args.embed_dim] * max_depth,
        "mlp_ratio": [max(CHOICES["mlp_ratio"])] * max_depth,
        "num_heads": [max(CHOICES["num_heads"])] * max_depth,
        "layer_num": max_depth,
    }

    best_val_loss = float("inf")
    best_model_sd = None
    epochs_without_improve = 0

    for epoch in range(args.pretrain_epochs):
        metrics = train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch,
            choices=CHOICES, mode="super", task="mlm",
            max_grad_norm=args.max_grad_norm,
        )

        val_metrics = evaluate_mlm(val_loader, model, device, config=val_config)

        print(f"[Pretrain Epoch {epoch+1}/{args.pretrain_epochs}] "
              f"train_loss={metrics['loss']:.4f}  masked_acc={metrics.get('masked_acc', 0):.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['masked_acc']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_sd = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.pretrain_patience:
            print(f"Pretrain early stopping at epoch {epoch+1} "
                  f"(no val loss improvement for {args.pretrain_patience} epochs)")
            break

    # Restore best model
    if best_model_sd is not None:
        model.load_state_dict(best_model_sd)
    print(f"Best pretrain val_loss: {best_val_loss:.4f}")

    # save checkpoint
    save_dir = Path(args.output_dir) / args.hospital / "checkpoint_mlm"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "mlm_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "choices": CHOICES,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "vocab_size": vocab_size,
        "max_adm_num": max_adm,
    }, ckpt_path)
    print(f"Saved pretrain checkpoint: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Phase 2: Finetune sampled architectures
# ---------------------------------------------------------------------------
def finetune_and_evaluate(args, tokenizer, train_data, val_data, test_data,
                          ckpt_path, device):
    print("\n" + "=" * 60)
    print(f"Phase 2: Finetune & Evaluate — {args.hospital}")
    print("=" * 60)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab_size = ckpt["vocab_size"]
    max_adm = ckpt["max_adm_num"]
    token_type = ["diag", "med", "lab", "pro"]

    # Sample random architectures once (shared across tasks)
    set_random_seed(args.seed)
    sampled_configs = [sample_configs(CHOICES) for _ in range(args.num_archs)]

    all_results = []

    for task in args.tasks:
        print(f"\n--- Task: {task} ---")

        # Compute label entropy for this task
        if task == "death":
            labels = train_data["DEATH"].astype(float).values
        elif task == "stay":
            labels = (train_data["STAY_DAYS"] > 7).astype(float).values
        elif task == "readmission":
            labels = train_data["READMISSION_3M"].astype(float).values
        counts = np.bincount(labels.astype(int))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        label_entropy = float(-np.sum(probs * np.log2(probs)))
        print(f"  Label entropy: {label_entropy:.4f}")

        train_dataset = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, task)
        val_dataset = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, task)
        test_dataset = FineTuneEHRDataset(test_data, tokenizer, token_type, max_adm, task)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=batcher(tokenizer, mode="finetune"), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                collate_fn=batcher(tokenizer, mode="finetune"), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 collate_fn=batcher(tokenizer, mode="finetune"), shuffle=False)

        for arch_idx, config in enumerate(sampled_configs):
            print(f"\n  Arch {arch_idx+1}/{args.num_archs}: "
                  f"embed_dim={config['embed_dim'][0]}, "
                  f"depth={config['layer_num']}, "
                  f"mlp_ratio={config['mlp_ratio'][0]}, "
                  f"num_heads={config['num_heads'][0]}")

            # build fresh model and load pretrained weights
            model = TransformerSuper(
                num_classes=2,
                vocab_size=vocab_size,
                embed_dim=args.embed_dim,
                mlp_ratio=args.mlp_ratio,
                depth=args.depth,
                num_heads=args.num_heads,
                qkv_bias=True,
                drop_rate=args.drop_rate,
                attn_drop_rate=args.attn_drop_rate,
                drop_path_rate=args.drop_path_rate,
                pre_norm=True,
                max_adm_num=max_adm,
            ).to(device)

            # load pretrained encoder weights (skip cls head since num_classes differs)
            pretrained_sd = ckpt["model_state_dict"]
            model_sd = model.state_dict()
            filtered = {k: v for k, v in pretrained_sd.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(filtered)
            model.load_state_dict(model_sd)

            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            # Track per-epoch: (val_auprc, model_state_dict, test_metrics)
            epoch_records = []
            best_val_auprc = -1.0
            epochs_without_improve = 0

            for epoch in range(args.finetune_epochs):
                train_one_epoch(
                    model=model, criterion=criterion, data_loader=train_loader,
                    optimizer=optimizer, device=device, epoch=epoch,
                    mode="retrain", retrain_config=config, task="cls",
                    max_grad_norm=args.max_grad_norm,
                )
                val_metrics = evaluate(val_loader, model, device, retrain_config=config)
                test_metrics = evaluate(test_loader, model, device, retrain_config=config)

                val_auprc = val_metrics['auprc']
                epoch_records.append((val_auprc, test_metrics))

                print(f"    Epoch {epoch+1}: val_AUPRC={val_auprc:.4f}  "
                      f"test_AUPRC={test_metrics['auprc']:.4f}")

                # early stopping check
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epochs_without_improve >= args.finetune_patience:
                    print(f"    Early stopping at epoch {epoch+1} "
                          f"(no improvement for {args.finetune_patience} epochs)")
                    break

            # Top-k averaging: pick top-k epochs by val AUPRC, average their test metrics
            epoch_records.sort(key=lambda x: x[0], reverse=True)
            top_k = min(args.top_k, len(epoch_records))
            top_k_test = [rec[1] for rec in epoch_records[:top_k]]

            avg_test = {}
            for key in ["accuracy", "f1", "auroc", "auprc"]:
                avg_test[key] = sum(m[key] for m in top_k_test) / top_k

            print(f"    Top-{top_k} avg test — Acc={avg_test['accuracy']:.4f}  "
                  f"F1={avg_test['f1']:.4f}  "
                  f"AUROC={avg_test['auroc']:.4f}  "
                  f"AUPRC={avg_test['auprc']:.4f}")

            # flatten architecture config into a single row
            row = {
                "hospital": args.hospital,
                "task": task,
                "arch_idx": arch_idx,
                "embed_dim": config["embed_dim"][0],
                "depth": config["layer_num"],
                "mlp_ratio": config["mlp_ratio"][0],
                "num_heads": config["num_heads"][0],
                "num_params": count_subnet_params(config, vocab_size, num_classes=2, max_adm=max_adm),
                "accuracy": avg_test["accuracy"],
                "f1": avg_test["f1"],
                "auroc": avg_test["auroc"],
                "auprc": avg_test["auprc"],
                "label_entropy": label_entropy,
            }

            all_results.append(row)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data paths ---
    data_root = Path(f"./data_process/{args.hospital}/{args.hospital}-processed")
    full_data_path = data_root / "mimic.pkl"
    pretrain_data_path = data_root / "mimic_pretrain.pkl"
    finetune_data_path = data_root / "mimic_downstream.pkl"

    # --- Load data ---
    special_tokens = ["[PAD]", "[CLS]", "[MASK]"]
    full_data = pickle.load(open(full_data_path, "rb"))
    tokenizer = build_tokenizer(full_data, special_tokens)
    max_adm = full_data.groupby("SUBJECT_ID")["HADM_ID"].nunique().max()
    print(f"Max admissions per patient: {max_adm}")
    pretrain_data = pickle.load(open(pretrain_data_path, "rb"))
    train_data, val_data, test_data = pickle.load(open(finetune_data_path, "rb"))

    # --- Phase 1: Pretrain ---
    ckpt_path = pretrain(args, tokenizer, pretrain_data, max_adm, device)

    # --- Phase 2: Finetune & Evaluate ---
    results = finetune_and_evaluate(
        args, tokenizer, train_data, val_data, test_data, ckpt_path, device
    )

    # --- Save metadata ---
    output_dir = Path(args.output_dir) / args.hospital
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata ({len(df)} rows) to {csv_path}")


if __name__ == "__main__":
    main()
