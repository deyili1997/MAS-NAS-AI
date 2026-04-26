"""
Device / DataLoader helpers
============================
Centralized utilities for GPU-friendly DataLoader configuration and for
taking device-agnostic snapshots of model state dicts.

Used by all entry scripts (run_pipeline, mas_search, run_regression,
baselines/*) so that a single change here propagates everywhere.
"""

import torch


def dataloader_kwargs(num_workers: int = 4) -> dict:
    """
    Return DataLoader kwargs tuned for the current device:
      - CUDA: pin_memory=True, num_workers=N (default 4), persistent_workers=True
      - MPS / CPU: num_workers=0, pin_memory=False (MPS fork/spawn issues +
                   pinned memory is a CUDA-only concept)

    Usage:
        loader = DataLoader(ds, batch_size=B, collate_fn=c,
                            shuffle=True, **dataloader_kwargs(args.num_workers))
    """
    on_cuda = torch.cuda.is_available()
    nw = int(num_workers) if on_cuda else 0
    return {
        "pin_memory": on_cuda,
        "num_workers": nw,
        "persistent_workers": nw > 0,
    }


def snapshot_sd_cpu(state_dict) -> dict:
    """
    Return a detached CPU copy of a model state_dict suitable for keeping
    around as a "best-so-far" snapshot without holding GPU memory.

    Each tensor is detached + moved to CPU + cloned so the snapshot is
    independent of the live model's GPU memory.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in state_dict.items()}
