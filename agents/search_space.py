"""Shared search-space definition for all MAS-NAS agents.

Finer scalar grid (1008 architectures) within the current supernet max dims
(embed_dim 256 / depth 8 / mlp_ratio 8 / num_heads 8). All embed_dim values are
divisible by 8 so every (embed_dim, num_heads) pairing satisfies the model's
``embed_dim % num_heads == 0`` constraint — no infeasible draws are wasted.

NOTE: this MUST stay in sync with the mirror in run_pipeline.py.
"""

CHOICES = {
    "embed_dim": [32, 48, 64, 96, 128, 192, 256],   # 7
    "depth":     [1, 2, 3, 4, 6, 8],                 # 6
    "mlp_ratio": [1, 2, 3, 4, 6, 8],                 # 6
    "num_heads": [1, 2, 4, 8],                       # 4
}   # = 1008 architectures
