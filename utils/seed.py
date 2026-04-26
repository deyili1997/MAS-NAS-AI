import os
import random
import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
    -----------
    seed : int
        The seed value to use for all relevant random number generators.
    deterministic : bool, default True
        If True, forces cuDNN into deterministic mode (reproducible but slower).
        If False, enables cuDNN benchmark mode (picks fastest kernel per shape,
        introduces run-to-run nondeterminism but 10-30% faster on typical
        Conv/MatMul workloads). Pass False on GPU when you prioritize speed
        over exact reproducibility.
    """

    # Set the Python built-in random module's seed
    random.seed(seed)

    # Set the NumPy random seed
    np.random.seed(seed)

    # Set the PyTorch random seed for CPU computations
    torch.manual_seed(seed)

    # If running on a machine with Apple Silicon (M1/M2/M3),
    # PyTorch uses Metal Performance Shaders (MPS) backend instead of CUDA.
    # CUDA seed calls are skipped in this case.
    if torch.backends.mps.is_available():
        print("[INFO] Detected Apple MPS backend (Mac with M1/M2/M3). Skipping CUDA seeds.")
    
    # If CUDA (NVIDIA GPU) is available — for external GPU cases on Mac/Linux/Windows
    elif torch.cuda.is_available():
        # Set CUDA random seed for the current device
        torch.cuda.manual_seed(seed)
        # Set CUDA random seed for all devices (multi-GPU training)
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic vs benchmark — trade-off between reproducibility and speed.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Set Python hash seed to ensure that any hash-based ops (like dict order)
    # are consistent across runs
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional (for logging/debugging)
    print(f"[INFO] Random seed set to {seed}")