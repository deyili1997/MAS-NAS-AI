"""
Pretrain-Finetune Regression Study: AutoFormer vs Traditional
==============================================================
For a given hospital + task, randomly sample K architectures and evaluate each
under TWO pretrain-finetune schemes:

  (1) AutoFormer style — one supernet MLM pretrain at max dims, then finetune
      each sub-arch by routing through the corresponding supernet sub-path
      (weight sharing).
  (2) Traditional style — independently MLM-pretrain each sub-arch at its own
      dims (no weight sharing), then finetune.

Produces a 2x2 regression panel (Accuracy / F1 / AUROC / AUPRC) with Pearson r
annotated. The goal is to validate whether AutoFormer sub-path finetune is a
faithful proxy for the architecture's "true" performance under traditional
independent pretraining.

Usage:
    python run_regression.py --hospital MIMIC-IV --task death --k 10
"""

import argparse
import copy
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.seed import set_random_seed
from utils.dataset import PreTrainEHRDataset, FineTuneEHRDataset, batcher
from utils.engine import sample_configs, train_one_epoch, evaluate_mlm
from model.supernet_transformer import TransformerSuper
from run_pipeline import build_tokenizer, count_subnet_params, count_subnet_flops, CHOICES
from agents.experiment_agent import _finetune_one_arch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="AutoFormer vs Traditional pretrain-finetune regression study",
    )
    # Target
    p.add_argument("--hospital", type=str, required=True)
    p.add_argument("--task", type=str, required=True,
                   choices=["death", "stay", "readmission"])
    p.add_argument("--k", type=int, required=True,
                   help="Number of random architectures to evaluate under both schemes")

    # Pretrain (shared between both paths)
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--pretrain_patience", type=int, default=5)

    # Supernet max dims (AutoFormer path only)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=8)

    # Finetune (shared)
    p.add_argument("--finetune_epochs", type=int, default=20)
    p.add_argument("--finetune_patience", type=int, default=5)
    p.add_argument("--top_k_epochs", type=int, default=3)

    # Common training knobs
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--drop_rate", type=float, default=0.1)
    p.add_argument("--attn_drop_rate", type=float, default=0.1)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--flops_seq_len", type=int, default=512)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_scalar(cfg):
    """Internal per-layer config dict → scalar-form dict for display/CSV."""
    return {
        "embed_dim": cfg["embed_dim"][0],
        "depth": cfg["layer_num"],
        "mlp_ratio": cfg["mlp_ratio"][0],
        "num_heads": cfg["num_heads"][0],
    }


def _make_pretrain_loaders(pretrain_data, tokenizer, max_adm, batch_size, seed):
    """90/10 train/val split of pretrain data (mirrors run_pipeline.pretrain)."""
    dataset = PreTrainEHRDataset(
        ehr_pretrain_data=pretrain_data,
        tokenizer=tokenizer,
        token_type=["diag", "med", "lab", "pro"],
        mask_rate=0.15,
        max_adm_num=max_adm,
    )
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        collate_fn=batcher(tokenizer), shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        collate_fn=batcher(tokenizer), shuffle=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# AutoFormer-style supernet pretrain
# ---------------------------------------------------------------------------
def pretrain_supernet(args, vocab_size, max_adm, train_loader, val_loader, device):
    """Build TransformerSuper at max dims, train MLM with mode='super'
    (random config sampling each step). Returns in-memory ckpt dict."""
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

    max_depth = max(CHOICES["depth"])
    val_config = {
        "embed_dim": [args.embed_dim] * max_depth,
        "mlp_ratio": [max(CHOICES["mlp_ratio"])] * max_depth,
        "num_heads": [max(CHOICES["num_heads"])] * max_depth,
        "layer_num": max_depth,
    }

    best_val_loss = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(args.pretrain_epochs):
        metrics = train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch,
            choices=CHOICES, mode="super", task="mlm",
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate_mlm(val_loader, model, device, config=val_config)
        print(f"[SupernetPretrain Epoch {epoch+1}/{args.pretrain_epochs}] "
              f"train_loss={metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['masked_acc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_sd = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.pretrain_patience:
            print(f"[SupernetPretrain] early stop at epoch {epoch+1}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"[SupernetPretrain] best val_loss={best_val_loss:.4f}")

    ckpt = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "max_adm_num": max_adm,
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt


# ---------------------------------------------------------------------------
# Traditional-style per-architecture pretrain
# ---------------------------------------------------------------------------
def pretrain_single_arch(scalar_config, internal_config, args,
                         vocab_size, max_adm,
                         train_loader, val_loader, device):
    """Build TransformerSuper at the arch's exact dims (no weight sharing
    with any other arch). Train MLM with mode='retrain' fixed to that arch.
    Returns in-memory ckpt dict — NOT persisted to disk."""
    model = TransformerSuper(
        num_classes=0,
        vocab_size=vocab_size,
        embed_dim=scalar_config["embed_dim"],
        mlp_ratio=scalar_config["mlp_ratio"],
        depth=scalar_config["depth"],
        num_heads=scalar_config["num_heads"],
        qkv_bias=True,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        pre_norm=True,
        max_adm_num=max_adm,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(args.pretrain_epochs):
        metrics = train_one_epoch(
            model=model, criterion=criterion, data_loader=train_loader,
            optimizer=optimizer, device=device, epoch=epoch,
            mode="retrain", retrain_config=internal_config, task="mlm",
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate_mlm(val_loader, model, device, config=internal_config)
        print(f"  [TradPretrain Epoch {epoch+1}/{args.pretrain_epochs}] "
              f"train_loss={metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_acc={val_metrics['masked_acc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_sd = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.pretrain_patience:
            print(f"  [TradPretrain] early stop at epoch {epoch+1}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"  [TradPretrain] best val_loss={best_val_loss:.4f}")

    ckpt = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "vocab_size": vocab_size,
        "embed_dim": scalar_config["embed_dim"],
        "depth": scalar_config["depth"],
        "num_heads": scalar_config["num_heads"],
        "mlp_ratio": scalar_config["mlp_ratio"],
        "max_adm_num": max_adm,
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt


# ---------------------------------------------------------------------------
# 2x2 regression panel
# ---------------------------------------------------------------------------
def plot_regression_panel(df, output_path, hospital, task):
    metric_pairs = [("accuracy", "Accuracy"),
                    ("f1", "F1"),
                    ("auroc", "AUROC"),
                    ("auprc", "AUPRC")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    pearson_rows = []

    for ax, (key, label) in zip(axes.flat, metric_pairs):
        x = df[f"traditional_{key}"].values.astype(float)
        y = df[f"autoformer_{key}"].values.astype(float)
        n = len(x)

        ax.scatter(x, y, s=70, alpha=0.8, edgecolor="black", linewidth=0.6, zorder=3)

        # Reference y=x line
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = (hi - lo) * 0.05 if hi > lo else 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="gray", ls="--", lw=1, label="y = x", zorder=2)

        # Linear regression fit
        if n >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(lo - pad, hi + pad, 100)
            ax.plot(xs, slope * xs + intercept,
                    color="tab:red", lw=2, label="linear fit",
                    zorder=4)

        # Pearson r + p-value
        if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r, pv = pearsonr(x, y)
        else:
            r, pv = float("nan"), float("nan")

        ax.set_xlabel("Traditional pretrain-finetune")
        ax.set_ylabel("AutoFormer pretrain-finetune")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.text(
            0.05, 0.95,
            f"Pearson r = {r:.3f}\n(p = {pv:.3g}, n = {n})",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9),
        )

        pearson_rows.append({
            "metric": key,
            "pearson_r": r,
            "p_value": pv,
            "n": n,
        })

    fig.suptitle(
        f"AutoFormer vs Traditional Pretrain-Finetune — {hospital} / {task}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(pearson_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target: {args.hospital} / {args.task}   k = {args.k}")

    # --- Data ---
    data_root = Path(f"./data_process/{args.hospital}/{args.hospital}-processed")
    with open(data_root / "mimic.pkl", "rb") as f:
        full_data = pickle.load(f)
    with open(data_root / "mimic_pretrain.pkl", "rb") as f:
        pretrain_data = pickle.load(f)
    with open(data_root / "mimic_downstream.pkl", "rb") as f:
        train_data, val_data, _test_data = pickle.load(f)

    tokenizer = build_tokenizer(full_data, ["[PAD]", "[CLS]", "[MASK]"])
    max_adm = full_data.groupby("SUBJECT_ID")["HADM_ID"].nunique().max()
    vocab_size = len(tokenizer.vocab.id2word)
    print(f"vocab_size = {vocab_size}   max_adm = {max_adm}")

    # --- Pretrain loaders (shared by both paths) ---
    pretrain_train_loader, pretrain_val_loader = _make_pretrain_loaders(
        pretrain_data, tokenizer, max_adm, args.batch_size, args.seed,
    )

    # --- Finetune loaders ---
    token_type = ["diag", "med", "lab", "pro"]
    train_ds = FineTuneEHRDataset(train_data, tokenizer, token_type, max_adm, args.task)
    val_ds = FineTuneEHRDataset(val_data, tokenizer, token_type, max_adm, args.task)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        collate_fn=batcher(tokenizer, mode="finetune"), shuffle=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        collate_fn=batcher(tokenizer, mode="finetune"), shuffle=False,
    )

    # --- Sample K architectures (deterministic under seed) ---
    set_random_seed(args.seed)
    arch_configs = [sample_configs(CHOICES) for _ in range(args.k)]
    scalar_configs = [_to_scalar(c) for c in arch_configs]

    print(f"\nSampled {args.k} architectures:")
    for i, sc in enumerate(scalar_configs):
        print(f"  [{i+1}] embed={sc['embed_dim']:<4} depth={sc['depth']:<2} "
              f"mlp={sc['mlp_ratio']:<2} heads={sc['num_heads']}")

    # --- Output dir ---
    out_dir = Path(args.results_dir) / args.hospital / f"regression_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # PATH 1: AutoFormer — supernet pretrain + k sub-finetunes
    # =======================================================================
    print(f"\n{'='*60}")
    print("AUTOFORMER PATH — supernet pretrain (max dims)")
    print(f"{'='*60}")

    set_random_seed(args.seed)
    supernet_ckpt = pretrain_supernet(
        args, vocab_size, max_adm,
        pretrain_train_loader, pretrain_val_loader, device,
    )

    autoformer_rows = []
    for i, (scfg, icfg) in enumerate(zip(scalar_configs, arch_configs)):
        print(f"\n--- AutoFormer finetune arch {i+1}/{args.k}: {scfg} ---")
        set_random_seed(args.seed + 1 + i)
        avg_val, _ = _finetune_one_arch(
            config=icfg, ckpt=supernet_ckpt,
            train_loader=train_loader, val_loader=val_loader,
            device=device, args=args,
        )
        print(f"  AutoFormer val:  Acc={avg_val['accuracy']:.4f}  "
              f"F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  "
              f"AUPRC={avg_val['auprc']:.4f}")

        autoformer_rows.append({
            "arch_idx": i,
            **scfg,
            "num_params": count_subnet_params(icfg, vocab_size, max_adm=max_adm),
            "flops": count_subnet_flops(icfg, args.flops_seq_len),
            "autoformer_accuracy": avg_val["accuracy"],
            "autoformer_f1": avg_val["f1"],
            "autoformer_auroc": avg_val["auroc"],
            "autoformer_auprc": avg_val["auprc"],
        })

    del supernet_ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    af_df = pd.DataFrame(autoformer_rows)
    af_df.to_csv(out_dir / "autoformer_results.csv", index=False)
    print(f"\n→ Saved {out_dir / 'autoformer_results.csv'}")

    # =======================================================================
    # PATH 2: Traditional — per-arch pretrain + finetune
    # =======================================================================
    print(f"\n{'='*60}")
    print("TRADITIONAL PATH — independent per-arch pretrain + finetune")
    print(f"{'='*60}")

    traditional_rows = []
    for i, (scfg, icfg) in enumerate(zip(scalar_configs, arch_configs)):
        print(f"\n--- Traditional arch {i+1}/{args.k}: {scfg} ---")

        set_random_seed(args.seed)
        print("  Pretraining...")
        trad_ckpt = pretrain_single_arch(
            scfg, icfg, args, vocab_size, max_adm,
            pretrain_train_loader, pretrain_val_loader, device,
        )

        set_random_seed(args.seed + 1 + i)
        print("  Finetuning...")
        avg_val, _ = _finetune_one_arch(
            config=icfg, ckpt=trad_ckpt,
            train_loader=train_loader, val_loader=val_loader,
            device=device, args=args,
        )
        print(f"  Traditional val: Acc={avg_val['accuracy']:.4f}  "
              f"F1={avg_val['f1']:.4f}  "
              f"AUROC={avg_val['auroc']:.4f}  "
              f"AUPRC={avg_val['auprc']:.4f}")

        traditional_rows.append({
            "arch_idx": i,
            "traditional_accuracy": avg_val["accuracy"],
            "traditional_f1": avg_val["f1"],
            "traditional_auroc": avg_val["auroc"],
            "traditional_auprc": avg_val["auprc"],
        })

        # Free per-arch pretrain ckpt (delete, per user decision)
        del trad_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tr_df = pd.DataFrame(traditional_rows)
    tr_df.to_csv(out_dir / "traditional_results.csv", index=False)
    print(f"\n→ Saved {out_dir / 'traditional_results.csv'}")

    # =======================================================================
    # MERGE + PLOT
    # =======================================================================
    merged = af_df.merge(tr_df, on="arch_idx")
    merged.to_csv(out_dir / "results.csv", index=False)

    panel_path = out_dir / "regression_panel.png"
    pearson_df = plot_regression_panel(
        merged, panel_path, args.hospital, args.task
    )
    pearson_df.to_csv(out_dir / "pearson_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(pearson_df.to_string(index=False))
    print(f"\n→ Panel:   {panel_path}")
    print(f"→ Results: {out_dir / 'results.csv'}")
    print(f"→ Pearson: {out_dir / 'pearson_summary.csv'}")


if __name__ == "__main__":
    main()
