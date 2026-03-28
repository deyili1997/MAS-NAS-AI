import random
import torch
import numpy as np
from typing import Iterable
import math
import sys
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

def sample_configs(choices: dict):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        # 这里mlp_ratio和num_heads是每一层独立采样的
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    # 注意，这里所有层的 embed_dim 都是相同的
    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

# this function should be able to do both pretrain and retrain (finetuning)
def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, choices=None,
                    mode='super', retrain_config=None, task='mlm',
                    max_grad_norm=None, ignore_index=-100):

    model.train()
    random.seed(epoch) 
    
    if mode == 'retrain':
        config = retrain_config
        model.set_sample_config(config=config)

    running_loss = 0.0
    num_steps = 0

    # for mlm
    running_tokens = 0
    running_correct = 0

    # for cls
    running_samples = 0
        
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} [{mode}/{task}]", leave=True)
    for step, batch in enumerate(pbar, start=1):
        input_ids, token_types, adm_index, attn_mask, labels = batch
        input_ids = input_ids.to(device)
        token_types = token_types.to(device)
        adm_index = adm_index.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)
        
        if mode == 'super':
            config = sample_configs(choices=choices)
            model.set_sample_config(config=config)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            token_types=token_types,
            adm_index=adm_index,
            attn_mask=attn_mask,
            task=task,
        )

        if task == 'mlm':
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                labels.reshape(-1),
                ignore_index=ignore_index,
            )
        elif task == 'cls':
            loss = criterion(outputs, labels.long())
        else:
            raise ValueError(f"Unsupported task: {task}")

        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss_value
        num_steps += 1

        with torch.no_grad():
            if task == 'mlm':
                mask = labels.ne(ignore_index)
                pred = outputs.argmax(dim=-1)   # [B, L]
                running_correct += pred[mask].eq(labels[mask]).sum().item()
                running_tokens += mask.sum().item()

            elif task == 'cls':
                # 单标签分类: outputs [B, C], labels [B]
                pred = outputs.argmax(dim=-1)   # [B]
                running_correct += pred.eq(labels).sum().item()
                running_samples += labels.size(0)

        avg_loss = running_loss / max(1, num_steps)
        if task == 'mlm':
            acc = running_correct / max(1, running_tokens)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", masked_acc=f"{acc:.4f}")
        elif task == 'cls':
            acc = running_correct / max(1, running_samples)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", cls_acc=f"{acc:.4f}")

    avg_epoch_loss = running_loss / max(1, num_steps)
    metrics = {
        'loss': avg_epoch_loss,
    }

    if task == 'mlm':
        metrics['masked_acc'] = running_correct / max(1, running_tokens)
        metrics['masked_tokens'] = running_tokens

    elif task == 'cls':
        metrics['cls_acc'] = running_correct / max(1, running_samples)
        metrics['num_samples'] = running_samples

    return metrics


@torch.no_grad()
def evaluate_mlm(data_loader, model, device, config, ignore_index=-100):
    """Evaluate MLM loss on a validation set with a fixed subnet config."""
    model.eval()
    model.set_sample_config(config=config)

    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_tokens = 0

    for batch in tqdm(data_loader, desc="Val MLM", leave=False):
        input_ids, token_types, adm_index, attn_mask, labels = batch
        input_ids = input_ids.to(device)
        token_types = token_types.to(device)
        adm_index = adm_index.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            token_types=token_types,
            adm_index=adm_index,
            attn_mask=attn_mask,
            task='mlm',
        )

        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )
        total_loss += loss.item()
        total_steps += 1

        mask = labels.ne(ignore_index)
        pred = outputs.argmax(dim=-1)
        total_correct += pred[mask].eq(labels[mask]).sum().item()
        total_tokens += mask.sum().item()

    return {
        'loss': total_loss / max(1, total_steps),
        'masked_acc': total_correct / max(1, total_tokens),
    }


# this function should be able to do only retrain (finetuning)
@torch.no_grad()
def evaluate(data_loader, model, device, retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    if retrain_config is None:
        raise ValueError("retrain_config must be provided for evaluate()")
    model.set_sample_config(config=retrain_config)

    total_loss = 0.0
    total_steps = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids, token_types, adm_index, attn_mask, labels = batch
        input_ids = input_ids.to(device)
        token_types = token_types.to(device)
        adm_index = adm_index.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device).long()

        outputs = model(
            input_ids=input_ids,
            token_types=token_types,
            adm_index=adm_index,
            attn_mask=attn_mask,
            task='cls',
        )  # [B, num_classes]

        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=-1)
        preds = outputs.argmax(dim=-1)

        total_loss += loss.item()
        total_steps += 1

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # positive class probability (class 1)
    pos_probs = all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs[:, 0]

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auroc = roc_auc_score(all_labels, pos_probs) if len(np.unique(all_labels)) > 1 else 0.0
    auprc = average_precision_score(all_labels, pos_probs) if len(np.unique(all_labels)) > 1 else 0.0

    metrics = {
        'loss': total_loss / max(1, total_steps),
        'accuracy': acc,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'num_samples': len(all_labels),
    }
    return metrics