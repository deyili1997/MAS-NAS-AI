"""Shared search-space definition for all MAS-NAS agents."""

CHOICES = {
    "mlp_ratio": [1, 2, 4, 8],
    "num_heads": [1, 2, 4, 8],
    "embed_dim": [32, 64, 128, 256],
    "depth": [1, 2, 4, 8],
}
